#include "sampler.h"
#include "operator.h"
#include "utils.h"
#include <stdlib.h>


// 返回具有最高概率的索引
int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

// 从概率分布中采样索引
int sample_mult(float* probabilities, int n, float coin) {

    // coin 是一个在 [0, 1) 范围内的随机数，通常来自 random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // 处理舍入误差的情况
}


int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}


// 设立门槛 (topp)：假设 topp = 0.9 (90%)。
// 筛选精英奖券：抽奖组织者从概率最高的奖券开始，把它们一张张放进一个“精英抽奖箱”，同时累加它们的中奖概率。
// 停止筛选：当“精英抽奖箱”里所有奖券的概率总和刚好超过 90% 时，就停止放入。这个精英奖券集合就是所谓的“原子核 (Nucleus)”。
// 最终抽奖：最后，只在这个小小的、高质量的“精英抽奖箱”里进行抽奖。这样既保证了多样性（因为不止一张奖券），又排除了那些质量极差的选项。

// 从概率分布中进行 top-p 采样
int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}


void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}



int sample(Sampler* sampler, float* logits) {
    int next;

    // 没有随机性，直接选择最大概率的词
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // scaling
        // temperature > 1.0 (e.g., 1.5): logits 数组中的每个值都会变小。值与值之间的差距也随之缩小。
        // 这会使得概率分布变得更“平坦”，高分词的优势被削弱，低分词的概率被提升。
        // 效果：增加随机性，模型更可能选择不那么常见的词，更有创造力，但也更容易“胡说八道”。

        // temperature < 1.0 (e.g., 0.7): logits 数组中的每个值都会变大。值与值之间的差距被拉大。
        // 这会使得概率分布变得更“尖锐”，高分词的优势被强化。效果：减少随机性，
        // 模型更倾向于选择它最有信心的那几个词，输出更稳定、更聚焦。
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }

        // softmax 函数会处理数组，使得：所有元素都变成正数。所有元素的总和等于 1.0。
        softmax(logits, sampler->vocab_size);

        float coin = random_f32(&sampler->rng_state); // flip a (float) coin (this is our source of entropy for sampling)

        // topp 参数被禁用或设置为无效值。这个函数在整个词汇表的概率分布上进行采样（多项式分布采样）。每个词的扇区大小由其概率决定，然后用 coin 来决定指针停在哪里

        if (sampler->topp <= 0 || sampler->topp >= 1) { 
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        }
        else {
             // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }

    }

    return next;
}