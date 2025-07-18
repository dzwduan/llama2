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


  const char *checkpoint_path = g_config.checkpoint_path;
  const char *tokenizer_path = g_config.tokenizer_path;
  // const char *mode = g_config.mode;
  const char *prompt = g_config.prompt;
  const char *system_prompt = g_config.system_prompt;
  int steps = g_config.steps;
  float temperature = g_config.temperature;
  float topp = g_config.topp;
  unsigned long long rng_seed = g_config.rng_seed;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);
  // run!
  // if (strcmp(mode, "generate") == 0) {
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
  // } else if (strcmp(mode, "chat") == 0) {
  //   //chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
  // } else {
  //   fprintf(stderr, "unknown mode: %s\n", mode);
  //   error_usage();
  // }

  // memory and file handles cleanup
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);
  return 0;
}