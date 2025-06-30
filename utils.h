#ifndef __UTILS_H__
#define __UTILS_H__

#include "types.h"
#include "config.h"
#include "sampler.h"
#include "transformer.h"
#include "tokenizer.h"
#include "config.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

unsigned int random_u32(unsigned long long *state) ;
float random_f32(unsigned long long *state) ;
long time_in_ms() ;



void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) ;


void read_stdin(const char* guide, char* buffer, size_t bufsize) ;

// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps);


void error_usage() ;

void parse_args(int argc, char *argv[], Llama2Config *config) ;


#endif // __UTILS_H__