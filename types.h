#ifndef __TYPES_H__
#define __TYPES_H__

#include <stdint.h>
#include <stdio.h>

//------------------------------------ TOKENIZER ------------------------------------

// 用于在词汇表中匹配
typedef struct {
  char *str; // token对应的字符串
  int id;    // token的索引
} TokenIndex;



typedef struct {
  char **vocab; // 词汇表
  float *vocab_scores; // BPE 合并分数。在编码时，用于决定合并哪一对相邻的 token。
  TokenIndex *sorted_vocab; // 按字符串排序的词汇表，用于二分查找。
  int vocab_size;           // 词汇表大小
  unsigned int max_token_length; // 词汇表中最长 token 的长度，用于分配缓冲区
  unsigned char byte_pieces[256 * 2]; // 预先存储了所有 256 个单字节字符的字符串形式
} Tokenizer;



typedef struct {
  int dim;
  int hidden_dim; // for ffn layer
  int n_layers;   // number of layers
  int n_heads;    // number of query heads
  int n_kv_heads; // number of key/value heads (can be < query heads because of
                  // multiquery)
  int vocab_size; // useally 256 (byte-level)
  int seq_len;    // max sequence length
} Config;



//-----------------------------------Transformer------------------------------------------


typedef struct {
  int8_t *q; // 量化后的整数值
  float *s;  // 缩放因子数组，长度为
} QuantizedTensor;

typedef struct {
  // token embedding talbe
  QuantizedTensor *q_tokens;    // (vocab_size, dim)
  float *token_embedding_table; // same, but dequantized

  // weights for rmsnorms
  float *rms_att_weight; // (layer, dim) rmsnorm weights
  float *rms_ffn_weight; // (layer, dim)

  // weights for matmuls .note dim == n_heads * head_size
  QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
  QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
  QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
  QuantizedTensor *wo; // (layer, n_heads * head_size, dim)

  // weights for ffn
  QuantizedTensor *w1; // (layer, hidden_dim, dim)
  QuantizedTensor *w2; // (layer, dim, hidden_dim)
  QuantizedTensor *w3; // (layer, hidden_dim, dim)

  // final rmsnorm
  float *rms_final_weight; // (dim,)

  // classifier weights for the logits, on the last layer
  QuantizedTensor
      *wcls; // (dim, vocab_size) if shared_classifier == 0
             // (n_heads * head_size, vocab_size) if shared_classifier == 1
} TransformerWeights;



// Transformer state, used during the forward pass
typedef struct {
  float *x;           // activation at current time stamp (dim,)
  float *xb;          // same, but inside a residual branch (dim,)
  float *xb2;         // an additional buffer just for convenience (dim,)
  float *hb;          // buffer for hidden dimension in the ffn (hidden_dim,)
  float *hb2;         // buffer for hidden dimension in the ffn (hidden_dim,)
  QuantizedTensor xq; // quantized x (dim,)
  QuantizedTensor hq; // quantized hb (hidden_dim,)
  float *q;           // query (dim,)  dim = n_head * head_size
  float *k;           // key (dim,)
  float *v;           // value (dim,)
  float *att;         // buffer for scores/attention values (n_heads, seq_len)
  float *logits;      // output logits
  // kvcache
  float *key_cache;   // (layer, seq_len, dim)
  float *value_cache; // (layer, seq_len, dim)
} RunState;


typedef struct {
  Config config;
  TransformerWeights weights; // the weights of the model
  RunState state; // buffers for the "wave" of activations in the forward pass

  int fd;            // file descriptor for mmap
  float *data;       // memory mapped data pointer
  ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;


typedef struct {
    int index;
    float prob;
} ProbIndex;



#endif // __TYPES_H__