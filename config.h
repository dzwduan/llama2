#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <stddef.h>

// default parameters

typedef struct {
    const char *checkpoint_path; // 使用 const char* 更安全
    const char *tokenizer_path;
    float temperature;
    float topp;
    int steps;
    const char *prompt;
    unsigned long long rng_seed;
    // const char *mode;
    const char *system_prompt;
} Llama2Config;


// 使用 extern 声明一个全局唯一的配置实例
// 这行代码告诉所有包含此头文件的文件："有一个名为 g_config 的全局变量，
// 它的实体在别处定义，你们可以安全地使用它。"
extern Llama2Config g_config;

extern int GS;

#endif // __CONFIG_H__