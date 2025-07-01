#ifndef __TOKENIZER_H__
#define __TOKENIZER_H__

#include "types.h"

int compare_tokens(const void *a, const void *b) ;

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) ;

void build_tokenizer(Tokenizer *t, const char *tokenizer_path, int vocab_size) ;

void free_tokenizer(Tokenizer *t) ;


// encode the string text (input) into an upper-bound preallocated tokens[]
// array bos != 0 means prepend the BOS token (=1), eos != 0 means append the
// EOS token (=2)
void encode(Tokenizer *t, const char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);

char *decode(Tokenizer *t, int prev_token, int token);

#endif // __TOKENIZER_H__