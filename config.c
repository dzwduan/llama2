#include "config.h"


// 在这里定义全局配置变量的实体，并可以直接初始化为默认值
Llama2Config g_config = {
    .checkpoint_path = NULL,
    .tokenizer_path = "tokenizer.bin",
    .temperature = 1.0F,
    .topp = 0.9F,
    .steps = 256,
    .prompt = NULL,
    .rng_seed = 0, // 稍后在初始化函数中用时间覆盖
    .mode = "generate",
    .system_prompt = NULL,
};


int GS; // 量化权重的组大小，替代原来的 GZ 和多文件中定义的 GS