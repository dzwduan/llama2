
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void build_tokenizer(Tokenizer *t, const char *tokenizer_path, int vocab_size) {
  // 给vocab 和 scores 分配内存
  t->vocab = (char **)malloc(vocab_size * sizeof(char *));
  t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
  // 初始化为 NULL，稍后会懒加载
  t->sorted_vocab = NULL;

  // 这是一个巧妙的优化：预先生成所有 256
  // 个字节值对应的字符串，无需动态malloc创建 例如，对于字节值 65
  // ('A')，它会创建字符串 "A" (即 [65, 0])
  for (int i = 0; i < 256; i++) {
    t->byte_pieces[i * 2] = (unsigned char)i; // 字符
    t->byte_pieces[i * 2 + 1] = '\0';         // 字符串终止符
  }

  // 读取 tokenizer 文件
  FILE *file = fopen(tokenizer_path, "rb");
  if (!file) {
    fprintf(stderr, "couldn't load %s\n", tokenizer_path);
    exit(EXIT_FAILURE);
  }

  // 从文件中读取最大 token 长度
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
    fprintf(stderr, "couldn't read max_token_length from %s\n", tokenizer_path);
    exit(EXIT_FAILURE);
  }

  int len;

  // 循环 vocab_size 次，读取每个 token 的信息
  for (int i = 0; i < vocab_size; i++) {
    // 读取token的BPE分数
    if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
      fprintf(stderr, "failed to read vocab score\n");
      exit(EXIT_FAILURE);
    }

    // 读取token字符串的长度
    if (fread(&len, sizeof(int), 1, file) != 1) {
      fprintf(stderr, "failed to read token length\n");
      exit(EXIT_FAILURE);
    }

    // 为字符串分配内存， +1是为了末尾存放\0
    t->vocab[i] = (char *)malloc(len + 1);

    // 从文件中读取token字符串
    if (fread(t->vocab[i], len, 1, file) != 1) {
      fprintf(stderr, "failed to read token string\n");
      exit(EXIT_FAILURE);
    }

    // 添加\0
    t->vocab[i][len] = '\0';
  }

  fclose(file);
}

void free_tokenizer(Tokenizer *t) {
  for (int i = 0; i < t->vocab_size; i++) {
    free(t->vocab[i]); // 释放每个 token 的字符串
  }
  free(t->vocab);        // 释放词汇表指针数组
  free(t->vocab_scores); // 释放 BPE 分数数组
  free(t->sorted_vocab); // 释放排序后的词汇表
}

int compare_tokens(const void *a, const void *b) {
  // 用于 qsort 的比较函数
  return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
  // 在已排序的词汇表中查找字符串 str，返回其索引或 -1 如果未找到
  TokenIndex tok = {.str = str}; // 创建一个临时的 TokenIndex 作为查找键
  TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex),
                            compare_tokens);
  return res != NULL ? res->id : -1; // 如果找到，返回其 id，否则返回 -1
}

// encode the string text (input) into an upper-bound preallocated tokens[]
// array bos != 0 means prepend the BOS token (=1), eos != 0 means append the
// EOS token (=2)
void encode(Tokenizer *t, const char *text, int8_t bos, int8_t eos, int *tokens,
            int *n_tokens) {
  if (text == NULL) {
    fprintf(stderr, "cannot encode NULL text\n");
    exit(EXIT_FAILURE);
  }

  // 惰性排序
  if (t->sorted_vocab == NULL) {
    t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++) {
      // vocab[i]是在build_tokenizer时初始化的
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }

    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }

  // 创建临时字符串缓冲， 在合并阶段存放拼接后的token字符串
  char *str_buffer = (char *)malloc((t->max_token_length * 2 + 1 + 2) *
                                    sizeof(char)); // +1 for null terminator
  size_t str_len = 0;                              // 当前字符串长度

  // 重置token计数器
  *n_tokens = 0;

  // 如果需要，添加 BOS (=1) token
  if (bos) {
    tokens[(*n_tokens)++] = 1;
  }

  // 为了模拟 SentencePiece 的行为，它通常会在字符串前加一个空格来规范化
  // 这里完成后是 BOS ' '
  if (text[0] != '\0') {
    int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
    tokens[(*n_tokens)++] = dummy_prefix;
  }

  // 遍历输入字符串的每个字节，检查是否是 UTF-8 编码的字符
  // 如果是UTF-8，添加到缓冲区，然后查找其对应ID，给token编码
  for (char *c = text; *c != '\0'; c++) {
    // UTF-8 字符边界检测
    if ((*c & 0xC0) != 0x80) {
      str_len = 0;
    }

    str_buffer[str_len++] = *c; // 添加当前字符到缓冲区
    str_buffer[str_len] = '\0'; // 确保字符串以 null 结尾

    if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
        continue;
    }

    // 在词汇表中查找当前缓冲区的完整字符
    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
    if (id != -1) {
      tokens[(*n_tokens)++] = id;
    } else {
      // 如果未找到，使用字符的 UTF-8 编码值加 3 作为 token ID
      for (size_t i = 0; i < str_len; i++) {
        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
      }
    }

    // 重置缓冲区，为下一个字符做准备
    str_len = 0;
  }

  // 反复迭代，直到找不到可以合并的 token 对为止
  while (1) {
    float best_score = -1e10; // 初始化最优分数
    int best_id = -1;         // 最优 token ID
    int best_idx = -1;        // 最优 token 对的索引

    for (int i = 0; i < (*n_tokens) - 1; i++) {
      // 尝试合并相邻的两个 token
      sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score) {
        // 如果找到更好的合并对，记录其分数和位置
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    // 如果没找到可以合并的对，结束合并
    if (best_idx == -1) {
      break;
    }

    tokens[best_idx] = best_id; // 合并相邻的 token

    // 删除合并后的 token 的下一个位置

    for (int i = best_idx + 1; i < (*n_tokens) - 1; i++) {
      tokens[i] = tokens[i + 1]; // 整体向前移动 token
    }

    (*n_tokens)--; // 减少 token 数量
  }

  if (eos) {
    tokens[(*n_tokens)++] = 2; // 添加 EOS (=2) token
  }

  // 释放临时字符串缓冲区
  free(str_buffer);
}

char *decode(Tokenizer *t, int prev_token, int token) {

  char *piece = t->vocab[token];

  // BOS + 空格后面，说明正在解码句子的第一个词
  if (prev_token == 1 && piece[0] == ' ')
    piece++;

  // 处理那些不代表普通文本，而是代表单个原始字节的特殊词元
  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char *)t->byte_pieces + byte_val * 2;
  }

  return piece; // 返回 token 对应的字符串
}