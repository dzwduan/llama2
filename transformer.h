#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__

#include "sampler.h"
#include "types.h"
#include <unistd.h>

// 预先分配 RunState 的内存空间
void malloc_run_state(RunState *s, Config *p);

void free_run_state(RunState *s);

// initialize `n` x quantized tensor (with `size_each` elements), starting from
// memory pointed at *ptr size_each ：tensor里面有几个元素 假设size_each = 64,
// GS = 8, 那么每个tensor的q数组有64个int8_t元素，s数组有64/GS=8个float元素
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each);

// memory_map_weights: 将Transformer的权重映射到内存中
// ptr: 指向内存映射的起始位置
// shared_classifier: 是否使用共享分类器
void memory_map_weights(TransformerWeights *w, Config *p, void *ptr,
                        uint8_t shared_classifier);

void read_checkpoint(const char *checkpoint, Config *config,
                     TransformerWeights *weights, int *fd, float **data,
                     ssize_t *file_size);

void build_transformer(Transformer *t, const char *checkpoint_path);

void free_weight(TransformerWeights *w);

void free_transformer(Transformer *t);

float *forward(Transformer *transformer, int token, int pos);


void transformer_batched(const int * tokens, const int * positions, RunState * s, TransformerWeights *w, int BATCH_SIZE) ;

#endif // __TRANSFORMER_H__