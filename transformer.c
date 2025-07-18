
#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "quant.h"
#include "types.h"
#include "transformer.h"
#include "operator.h"
#include "config.h"
#include "tokenizer.h"


// 预先分配 RunState 的内存空间
void malloc_run_state(RunState *s, Config *p) {
  // we calloc instead of malloc to keep valgrind happy
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  s->x = calloc(p->dim, sizeof(float));
  s->xb = calloc(p->dim, sizeof(float));
  s->xb2 = calloc(p->dim, sizeof(float));
  s->hb = calloc(p->hidden_dim, sizeof(float));
  s->hb2 = calloc(p->hidden_dim, sizeof(float));
  s->xq.q = calloc(p->dim, sizeof(int8_t));
  s->xq.s = calloc(p->dim, sizeof(float));
  s->hq.q = calloc(p->hidden_dim, sizeof(int8_t));
  s->hq.s = calloc(p->hidden_dim, sizeof(float));
  s->q = calloc(p->dim, sizeof(float));
  s->k = calloc(kv_dim, sizeof(float));
  s->v = calloc(kv_dim, sizeof(float));
  s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
  s->logits = calloc(p->vocab_size, sizeof(float));

  // allocate kv cache
  s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));

  // ensure all mallocs went fine
  if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k ||
      !s->v || !s->att || !s->logits || !s->key_cache || !s->value_cache) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
}

void free_run_state(RunState *s) {
  free(s->x);
  free(s->xb);
  free(s->xb2);
  free(s->hb);
  free(s->hb2);
  free(s->xq.q);
  free(s->xq.s);
  free(s->hq.q);
  free(s->hq.s);
  free(s->q);
  free(s->k);
  free(s->v);
  free(s->att);
  free(s->logits);
  free(s->key_cache);
  free(s->value_cache);
}

//------------------------------------Quantization functions------------------------------------

// TODO: change to AWQ


// initialize `n` x quantized tensor (with `size_each` elements), starting from
// memory pointed at *ptr size_each ：tensor里面有几个元素 假设size_each = 64,
// GS = 8, 那么每个tensor的q数组有64个int8_t元素，s数组有64/GS=8个float元素
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
  void *p = *ptr;
  // 这里n是layer层数
  QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
  for (int i = 0; i < n; i++) {
    res[i].q = (int8_t *)p;          // map quantized int8 values
    p = (int8_t *)p + size_each;     // advance pointer by size_each
    res[i].s = (float *)p;           // map scale factors
    p = (float *)p + size_each / GS; // advance pointer by size_each
  }
  *ptr = p;

  return res;
}

// memory_map_weights: 将Transformer的权重映射到内存中
// ptr: 指向内存映射的起始位置
// shared_classifier: 是否使用共享分类器
void memory_map_weights(TransformerWeights *w, Config *p, void *ptr,
                        uint8_t shared_classifier) {
  int head_size = p->dim / p->n_heads; // 每个head的维度大小

  // 维度一开始都保存在fp32格式的内存中 (rmsnorm 1D weights)
  float *fptr = (float *)ptr;   // cast our pointer to float*
  w->rms_att_weight = fptr;     // (layer, dim) rmsnorm weights
  fptr += p->n_layers * p->dim; // advance pointer by n_layers
  w->rms_ffn_weight = fptr;     // (layer, dim)
  fptr += p->n_layers * p->dim; // advance pointer by n_layers
  w->rms_final_weight = fptr;   // (dim,)
  fptr += p->dim;               // advance pointer by dim

  // quantized weights
  ptr = (void *)fptr; // cast back to void* for quantized tensors
  // 量化的token embedding table
  w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
  // 反量化的token embedding table
  w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
  // 结果保存到token_embedding_table
  int8_dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

  // 量化的权重矩阵
  w->wq = init_quantized_tensors(&ptr, p->n_layers,
                                 p->dim * p->n_heads * head_size);
  w->wk = init_quantized_tensors(&ptr, p->n_layers,
                                 p->dim * p->n_kv_heads * head_size);
  w->wv = init_quantized_tensors(&ptr, p->n_layers,
                                 p->dim * p->n_kv_heads * head_size);
  w->wo = init_quantized_tensors(&ptr, p->n_layers,
                                 p->n_heads * head_size * p->dim);

  // 量化的ffn权重矩阵
  w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
  w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
  w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

  // 如果使用共享分类器，则wcls的维度为(n_heads * head_size, vocab_size)
  // 否则为(dim, vocab_size)
  w->wcls = shared_classifier
                ? w->q_tokens
                : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

void read_checkpoint(const char *checkpoint, Config *config,
                     TransformerWeights *weights, int *fd, float **data,
                     ssize_t *file_size) {
  FILE *file = fopen(checkpoint, "rb");
  if (!file) {
    fprintf(stderr, "couldn't open checkpoint file %s\n", checkpoint);
    exit(EXIT_FAILURE);
  }
  // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
  uint32_t magic_number;
  if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  if (magic_number != 0x616b3432) {
    fprintf(stderr, "Bad magic number\n");
    exit(EXIT_FAILURE);
  }

  int version;
  if (fread(&version, sizeof(int), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  if (version != 2) {
    fprintf(stderr, "Bad version %d, need version 2\n", version);
    exit(EXIT_FAILURE);
  }
  /**
  
   0      4      8      12     ...    256
┌──────┬──────┬──────────────┬──────────────┐
│Magic │Version│ Config结构体  │   Flags     │  权重数据...
│(4B)  │ (4B)  │ (可变大小)    │(5B:1+4)     │
└──────┴──────┴──────────────┴──────────────┘
↑ 头部固定总大小 = 256字节 ↑
  
  */

  int header_size = 256; // the header size for version 2 in bytes
  // read in the Config
  if (fread(config, sizeof(Config), 1, file) != 1) {
    fprintf(stderr, "failed to read config from %s\n", checkpoint);
    exit(EXIT_FAILURE);
  }

  // read in flags
  uint8_t shared_classifier;
  if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) {
    fprintf(stderr, "failed to read shared_classifier from %s\n", checkpoint);
    exit(EXIT_FAILURE);
  }

  int group_size;
  if (fread(&group_size, sizeof(int), 1, file) != 1) {
    fprintf(stderr, "failed to read group_size from %s\n", checkpoint);
    exit(EXIT_FAILURE);
  }
  GS = group_size; // set the global group size for quantization

  // read in the file size
  fseek(file, 0, SEEK_END);
  *file_size = ftell(file);
  fclose(file);

  // mmap transformer weights into data pointer
  *fd = open(checkpoint, O_RDONLY);
  if (*fd == -1) {
    fprintf(stderr, "open failed!\n");
    exit(EXIT_FAILURE);
  }
  // mmap the file into memory
  // 这里将整个文件映射到内存中，返回的data指向文件的起始位置
  // TODO:GPU如何处理？暂用vx_malloc
  *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) {
    fprintf(stderr, "mmap failed!\n");
    exit(EXIT_FAILURE);
  }

  void *weights_ptr = ((char *)*data) + header_size; // skip header bytes. char is 1 byte
  memory_map_weights(weights, config, weights_ptr, shared_classifier);
}



void build_transformer(Transformer *t, const char *checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_weight(TransformerWeights *w) {
    if (!w) return;
    free(w->q_tokens);
    // free_transformer 里面释放了最初使用指针操作的地址空间，无需重复释放
    // free(w->rms_att_weight);
    // free(w->rms_ffn_weight);
    // free(w->rms_final_weight);
    free(w->wq);
    free(w->wk);
    free(w->wv);
    free(w->wo);
    free(w->w1);
    free(w->w2);
    free(w->w3);
    if (w->wcls != w->q_tokens) {
        free(w->wcls); // 如果wcls不是q_tokens，则释放它
    }
    free(w->token_embedding_table); // 反量化的token embedding table
}

void free_transformer(Transformer *t) {
    if (!t) return;
    free_weight(&t->weights);
    free_run_state(&t->state);

    if (t->data) {
        munmap(t->data, t->file_size);
    }
    if (t->fd != -1) {
        close(t->fd);
    }
}   



// pos是因为每个token都要调用一遍transformer.forward()，所以pos是token在序列中的位置
float* forward(Transformer* t, int token_idx, int pos) {

  Config *p = &t->config;
  TransformerWeights *w = &t->weights;
  RunState *s = &t->state;

  float *x = s->x;
  int dim = p->dim;
  int hidden_dim = p->hidden_dim;
  // 每个head的维度大小
  int head_size = p->dim / p->n_heads;
  // MHA n_heads == n_kv_heads, 而GQA n_kv_heads < n_heads , 从而后者的dim也减少
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  // kv_mul表示键值对的共享倍数
  int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery

  /** 内存布局示意图
  
  w->token_embedding_table ----> [  Vector for token 0  ][  Vector for token 1  ][  Vector for token 2  ] ...
                               <------ dim floats ---><------ dim floats ---><------ dim floats --->
  
  */

  // x 是 (dim,) 的浮点数数组 , 实际上代表token token_idx 的嵌入向量
  // 将token_idx对应的嵌入向量复制到x中
  memcpy(x, w->token_embedding_table + token_idx * dim, dim * sizeof(float));

  for (int l = 0; l < p->n_layers; l++) {

    // attention rmsnorm
    // rms_att_weight 大小是 n_layers * dim , 所以遍历的每一层起点是 rms_att_weight + l * dim
    rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim, 1e-5);

    // qkv matmul
    int8_quantize(&s->xq, s->xb, dim);
    // w->wq 大小是 n_layers * sizeof(QuantizedTensor) , 所以遍历的每一层起点是 w + l
    matmul(s->q, &s->xq, w->wq + l, dim, dim);
    matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
    matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);


    // RoPE
    rope_encoding(dim, head_size, pos, kv_dim, s->q, s->k);


    // save key value to kv cache

//  整个 KV Cache (一维物理内存)
// s->key_cache 指向这里
// |
// v
// [ ----- Layer 0 Data ----- | ----- Layer 1 Data ----- | ----- Layer 2 Data ----- | ... ]
//                           ^                          ^
//                           |                          |
//                           当 l=1 时, +loff 跳到这里  当 l=2 时, +loff 跳到这里

// 放大看 Layer 2 的数据块内部
// [ Pos 0 | Pos 1 | Pos 2 | ... | Pos 100 | ... ]  <-- 每个 Pos 块大小为 kv_dim
//                               ^
//                               |
//                               当 pos=100 时, +pos*kv_dim 跳到这里

// 最终的 key_cache_row 指针就指向这里
// |
// v
// [ f0, f1, f2, ..., f_{kv_dim-1} ] // 这里将存放 pos=100 的 Key 向量
     
     


    int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    float * key_cache_row = s->key_cache + loff + pos * kv_dim;
    float * value_cache_row = s->value_cache + loff + pos * kv_dim;
    // 将当前的 key 和 value 向量保存到 kv cache 中
    memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
    memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));


    // multi head attention
    // TODO: 考虑乱序优化，让多个head实现并行？不过simt本来也是并行的，每个core计算一个head
    // 但是在pos这个循环，每个core的压力也很大，目前的算法需要串行执行softmax和加权求和，
    // 即使使用flash-attention, 也只是变为分块设计，需要在dispatcher层面加入softmax的乱序
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < p->n_heads; h++) {
      float *q = s->q + h * head_size; // query vector for this head
      float *att = s->att + h * p->seq_len; // attention scores

      /**
      // 1. 起点
[ s->key_cache ]

// 2. + loff (层偏移) -> 到达第 l 层
[ Layer 0 | Layer 1 | ... | Layer l | ... ]
                              ^

// 3. + t * kv_dim (时间步偏移) -> 在第 l 层内，到达第 t 个时间步
[ T_0 | T_1 | ... | T_t | ... ]  (在 Layer l 内部)
                    ^

// 4. + (h / kv_mul) * head_size (头偏移) -> 在第 t 个时间步内，找到对应的 K 向量
[ K_head_0 | K_head_1 | K_head_2 | ... ] (在 T_t 内部)
               ^ (假设 h/kv_mul 映射到了 1)

// 指针 k 现在就指向这里
      
      */

      for (int t = 0; t <= pos; t++) {
        float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;

        float score = 0.0f;
        // q * k^T
        for (int i = 0; i < head_size; i++) {
          score += q[i] * k[i]; // 计算 q 和 k 的点积
        }

        float sqrt_head_size = sqrtf(head_size);
        score /= sqrt_head_size; // 缩放分数

        att[t] = score; // 保存分数到注意力分数数组
      }

      softmax(att, pos + 1);


      // weighted sum of the value, store back into xb
      // xb 是一个缓冲区，用于存储当前头的加权值
      float *xb = s->xb + h * head_size; // buffer for this head
      memset(xb, 0, head_size * sizeof(float)); // 清零

      for (int t = 0; t <= pos; t++) {
         float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;

         float a = att[t]; // 获取注意力权重
         // 累加加权值到 xb
         for (int i = 0; i < head_size; i++) {
           xb[i] += a * v[i]; // 将值向量乘以注意力
         }
      }
    }


    int8_quantize(&s->xq,s->xb, dim);
    matmul(s->xb2, &s->xq, w->wo + l, dim, dim);


    // residual connection back into x
    for (int i = 0; i < dim; i++) {
      x[i] += s->xb2[i]; // 将 xb2 加到 x 上
    }


    // ffn rmsnorm
    rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

    int8_quantize(&s->xq, s->xb, dim);
    matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
    matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);


    // quant + ffn x 2 + SwiGLU + quant + matmul + Residual add
    swiglu(hidden_dim, s);

    int8_quantize(&s->hq, s->hb, hidden_dim);
    // 注意这里和架构图不一样，涂黎曼是hb2
    matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);


    // residual connection back into x
    for (int i = 0; i < dim; i++) {
      x[i] += s->xb[i]; // 将 xb 加到 x 上, 图里面是 hb2
    }
  }


      // rmsnorm + quant + matmul

    //FIXME: 架构图里面的output是xb
    rmsnorm(x, x, w->rms_final_weight, dim);
  

    // classifier into logits
    int8_quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);

    return s->logits;
}



/**
 * @brief 使用批处理加速处理输入的 prompt，填充 KV 缓存。
 * @param pos 指向当前处理位置的指针，函数会更新它。
 * @param transformer 指向 Transformer 对象的指针，包含权重和状态。
 * @param prompt_tokens 输入的 prompt token 数组。
 * @param num_prompt_tokens prompt token 的总数。
 * @param tokenizer 指向 Tokenizer 对象的指针。
 * @param print_tokens 是否打印正在处理的 token。
 * @param BATCH_SIZE  此函数要处理的批次大小 (对应 Rust 版本中的 'A')。
 * @note 此函数依赖一个假设存在的 `transformer_forward_batch` 函数，
 * 该函数可以一次性处理一批 token。这是对标准 llama2.c 的一个扩展。
 */
void prefill_batch(int* pos,
                   Transformer* transformer,
                   int* prompt_tokens,
                   int num_prompt_tokens,
                  //  Tokenizer* tokenizer,
                  //  int print_tokens,
                  RunState *s,
                  TransformerWeights *w,
                  Config *p,
                  const int BATCH_SIZE) {

    // 只要剩余的 token 数量足够组成一个完整的批次，就继续循环
    while (*pos + BATCH_SIZE < num_prompt_tokens) {
        
        // 为当前批次准备 token 和 position 数组
        // C99+ 支持变长数组 (VLA)，非常适合此场景
        int tokens[BATCH_SIZE];
        int positions[BATCH_SIZE];

        // --- 阶段1: 准备和打包一个批次的数据 ---
        for (int i = 0; i < BATCH_SIZE; i++) {
            // 获取当前位置的输入 token。对于位置 0，它是特殊的 BOS token (值为 1)。
            // 否则，它是 prompt 中的上一个 token。
            int input_token = (*pos == 0) ? 1 : prompt_tokens[*pos - 1];

            positions[i] = *pos;
            tokens[i] = input_token;

            // 如果需要，打印当前正在“生成”的 token (即 prompt 中的下一个 token)
            // if (print_tokens) {
            //     // bpe_decode 用于获取 token 的字符串表示
            //     char* token_str = bpe_decode(tokenizer, input_token, prompt_tokens[*pos]);
            //     printf("%s", token_str);
            //     fflush(stdout); // 确保立即输出
            // }

            // 更新全局位置
            (*pos)++;
        }

        // --- 阶段2: 一次性处理整个批次 ---
        // 调用一个假设存在的、支持批处理的 transformer 函数。
        // 这个函数会计算并更新 transformer->state 中的 KV 缓存。
        // 我们不需要它的 logits 输出，所以可以传 NULL。
        // transformer_forward_batch(transformer, tokens, positions, BATCH_SIZE);
        transformer_batched(tokens, positions, s, w, p, BATCH_SIZE);
    }
}



void transformer_batched(const int * tokens, const int * positions, RunState * s, TransformerWeights *w, Config * p, int B) {

  int dim = p->dim;
  
  float x[B][p->dim];
}
