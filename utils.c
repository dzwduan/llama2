#include "utils.h"
#include "ctype.h"
#include <sys/time.h>


void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}


void parse_args(int argc, char *argv[], Llama2Config *config) {
     if (argc >= 2) { config->checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { config->temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { config->topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { config->rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { config->steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { config->prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { config->tokenizer_path = argv[i + 1]; }
        // else if (argv[i][1] == 'm') { config->mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { config->system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (config->rng_seed <= 0) config->rng_seed = (unsigned int)time(NULL);
    if (config->temperature < 0.0) config->temperature = 0.0;
    if (config->topp < 0.0 || 1.0 < config->topp) config->topp = 0.9;
    if (config->steps < 0) config->steps = 0;
}


void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}


unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}


float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}


void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}



void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, const char *prompt, int steps) {
    char * empty_prompt = "";
    if (prompt == NULL || strlen(prompt) == 0) {
        prompt = empty_prompt;
    }

    int num_prompt_tokens = 0;
    int * prompt_tokens = (int *)malloc((strlen(prompt) +3) * sizeof(int));  // +3 for \0 BOS EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    if (num_prompt_tokens < 1) {
        fprintf(stderr, "Error: prompt is empty or could not be tokenized.\n");
        free(prompt_tokens);
        exit(EXIT_FAILURE);
    }    


    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;

    while (pos < steps) {
        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            // TODO: 分离prefill，使用gemm提速
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }

        pos++;

        if (next == 1) break; // EOS token, stop generation

        // decode the next token and print it
        char *piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        // 现在生成的输出，会作为下一个输入
        // TODO: 改 forward batch 支持 gemm prefill，需要重构这里generate
        if (start == 0) {
            start = time_in_ms(); // 记录开始时间
        }
        token = next;
    }

    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}
