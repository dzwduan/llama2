#include <math.h>
#include "quant.h"

extern int GS; // group size global for quantization of the weights

// 按分组✖️量化因子
void dequantize(QuantizedTensor *qx, float *x, int n) {
  for (int i = 0; i < n; i++) {
    x[i] = qx->q[i] * qx->s[i / GS]; // 量化值乘以对应的缩放因子, i/GS是因为
                                     // 缩放因子是按组存储的
  }
}

// 存在的问题，n不是group_size的整数倍时，最后一组元素会被忽略，所以得香山取整，不够的补零
void quantize(QuantizedTensor *qx, float *x, int n) {
  int num_groups = (n + GS - 1) / GS; // 向上取整
  float Q_MAX = 127.0f;               // 量化的最大值

  for (int group = 0; group < num_groups; group++) {
    float wmax = 0.0;

    for (int i = 0; i < GS; i++) {
      float val = fabs(x[group * GS + i]);
      if (val > wmax) {
        wmax = val; // 找到当前组的最大绝对值
      }
    }

    float scale = wmax / Q_MAX; // 计算当前组的缩放因子
    qx->s[group] = scale;       // 存储缩放因子

    for (int i = 0; i < GS; i++) {
      float quant_value = x[group * GS + i] / scale; // 量化值
      int8_t quantized = fmaxf(fminf(quant_value, 127.0f), -127.0f);
      qx->q[group * GS + i] = quantized; // 存储量化后的值
    }
  }
}