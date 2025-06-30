#include "utils.h"


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
        else if (argv[i][1] == 'm') { config->mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { config->system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (config->rng_seed <= 0) config->rng_seed = (unsigned int)time(NULL);
    if (config->temperature < 0.0) config->temperature = 0.0;
    if (config->topp < 0.0 || 1.0 < config->topp) config->topp = 0.9;
    if (config->steps < 0) config->steps = 0;
}


inline unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}


inline float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

inline long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

