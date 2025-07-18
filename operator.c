#include "operator.h"
#include "types.h"
#include <math.h>
#include <stdlib.h>
#include "config.h"


void resdual_add(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}


// 核心是gemv , gemv优化可以设计TMA，使用异步内存拷贝实现同时计算和数据加载，从而隐藏数据延迟
void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized
    
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {

        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            // 组内计算，整数点积
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }

            // 组外计算，应用缩放因子
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            // 重置ival
            ival = 0;
        }

        xout[i] = val;
    }
}


/**
layerNorm存在的问题是，计算均值和方差需要对向量进行两次遍历，计算量相对较大
rmsnorm的优化是只需要一次遍历，计算均值和方差时可以复用平方和的结果
y = x / sqrt(mean(x^2) + epsilon)
为了保持模型的表达能力，引入了一个可学习的缩放参数 gamma (g)。
这里weight[j]作为gamma
output = g * y
*/

void rmsnorm(float* o, float* x, float* weight, int size, float epsilon) {
    float ss = 0.0f;
    // 计算平方和
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }

    // 计算均值
    ss /= size;
    // 加上一个小的常数以避免除零错误
    ss += epsilon; 
    // 计算标准差的倒数, 预先计算倒数，这样后续的除法就可以变成更高效的乘法
    ss = 1.0f / sqrtf(ss); 

    // 归一化并缩放
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * x[j] * ss;
    }
}


/**
需要3次循环，分别是计算最大值、计算指数和、归一化
*/
void softmax(float* x, int size) {

    float max_val= x[0];
    // 第一次循环，找到最大值
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // 第二次循环，计算指数和
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val); // 减去最大
        sum += x[i];
    }

    // 第三次循环，归一化
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}


void rope_encoding(int dim, int head_size, int pos, int kv_dim, float* q, float* k) {
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? q : k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }
}


void sigmoid(float *x) {
    // 使用更稳定的sigmoid实现
    *x = 1.0f / (1.0f + expf(-*x));
}


void swiglu(int dim, RunState *s) {
    for (int i = 0; i < dim; i++) {
        float val = s->hb[i];
        float x = val;
        sigmoid(&x);
        val = val * x;
        val = val * s->hb2[i]; // elementwise multiply with w3(x)
        s->hb[i] = val;
    }
}


// 仅需要两次循环
// TODO: fuse with top-k
void online_softmax(float* x, int size) {
    // 第一次循环，找到最大值
    float max_val = x[0];
    float d = 0.0f;

    for (int i = 0; i < size; i++) {
        float old_m = max_val;

        if (x[i] > max_val) {
            max_val = x[i];
        }
        d = d * expf(old_m - max_val) + expf(x[i] - max_val);
    }

    // 归一化
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val) / d;
    }
}


/**
 * @brief Implements "Algorithm 4: Online softmax and top-k" from the image.
 * 
 * This function performs a single pass over the input logits to find the top-K candidates
 * and simultaneously computes the necessary statistics for softmax in an online manner.
 *
 * @param logits      Input array of logits (size V).
 * @param V           Total size of the logits array (vocabulary size).
 * @param K           The number of top elements to find.
 * @param results     An output array of size K to store the final probabilities and indices.
 */
void online_softmax_with_topk(const float* logits, int V, int K, ProbIndex* results) {
    if (K <= 0 || V <= 0) {
        return;
    }
    if (K > V) K = V; // K 不能超过总大小

    // --- Initialization (Lines 1-4) ---
    // Line 1: m_0 <- -infinity
    float m = logits[0];
    // Line 2: d_0 <- 0
    float d = 0.0f;

    // Line 3: u <- {-inf, ..., -inf}, u is R^(K+1)
    // Line 4: p <- {-1, ..., -1}, p is Z^(K+1)
    float* u = (float*)malloc((K + 1) * sizeof(float));
    int* p = (int*)malloc((K + 1) * sizeof(int));
    for (int i = 0; i < K + 1; ++i) {
        u[i] = logits[0];
        p[i] = -1;
    }

    // --- Main Loop (Lines 5-16) ---
    // Line 5: for j <- 1, V do
    for (int j = 0; j < V; ++j) {
        // --- Online Softmax Update (Lines 6-7) ---
        float old_m = m;
        // Line 6: m_j <- max(m_{j-1}, x_j)
        if (logits[j] > m) {
            m = logits[j];
        }

        d = d * expf(old_m - m) + expf(logits[j] - m);

        // --- Online Top-K Update (Lines 8-15) ---
        // Line 8: u_{K+1} <- x_j
        u[K] = logits[j];
        // Line 9: p_{K+1} <- j
        p[K] = j;

        // Lines 10-15: Insertion sort for the last element
        // This loop bubbles the new element up to its correct sorted position.
        // It's more efficient than a full qsort every time.
        for (int k_idx = K; k_idx > 0; --k_idx) {
            if (u[k_idx] > u[k_idx - 1]) {
                // Swap values
                float temp_u = u[k_idx];
                u[k_idx] = u[k_idx - 1];
                u[k_idx - 1] = temp_u;
                // Swap indices
                int temp_p = p[k_idx];
                p[k_idx] = p[k_idx - 1];
                p[k_idx - 1] = temp_p;
            } else {
                // Element is in the correct position, no need to check further
                break;
            }
        }
    } // End of main loop

    // --- Finalization (Lines 17-20) ---
    // After the loop, 'm' is the final m_V, 'd' is the final d_V.
    // 'u' contains the top K logits, and 'p' contains their indices.

    // Line 17: for i <- 1, K do
    for (int i = 0; i < K; ++i) {
        // Line 18: v_i <- e^(u_i - m_V) / d_V
        results[i].prob = expf(u[i] - m) / d;
        // Line 19: z_i <- p_i
        results[i].index = p[i];
    }

    // --- Cleanup ---
    free(u);
    free(p);
}


