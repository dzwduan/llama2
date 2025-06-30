#include "config.h"
#include "operator.h"
#include "sampler.h"
#include "tokenizer.h"
#include "transformer.h"
#include "types.h"
#include "utils.h"
#include <stdio.h>



int main(int argc, char *argv[]) {

  parse_args(argc, argv, &g_config);

  Tokenizer tokenizer;
  Transformer transformer;
  Sampler sampler;

  const char * checkpoint_path = g_config.checkpoint_path;
  const char * tokenizer_path = g_config.tokenizer_path;
  float temperature = g_config.temperature;
  float topp = g_config.topp;
  unsigned long long rng_seed = g_config.rng_seed;


  build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
  build_transformer(&transformer, checkpoint_path);
  build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  // memory and file handles cleanup
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);
  return 0;
}