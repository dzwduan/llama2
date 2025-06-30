#ifndef __OPERATOR_H__
#define __OPERATOR_H__

#include "types.h"

void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d);

void rmsnorm(float* o, float* x, float* weight, int size) ;

void softmax(float* x, int size) ;

void online_softmax(float* x, int size) ;

void online_softmax_with_topk(const float* logits, int V, int K, ProbIndex* results) ;

void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d);

void rope_encoding(int dim, int head_size, int pos, int kv_dim, float* q, float* k);

// void self_attention(RunState * s, Config *p, int head, int loff, int kv_dim, int kv_mul, int head_size, int pos);

void sigmoid(float *x);

void swiglu(int dim, RunState *s);

// q_hq * q_w2 + hb2
void residual_add();

//TODO: add vx_malloc and vx_free

#endif // __OPERATOR_H__