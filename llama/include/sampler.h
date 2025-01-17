#ifndef SAMPLER_H
#define SAMPLER_H

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
typedef struct
{
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct
{
    int vocab_size;
    ProbIndex *probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n);
int sample_mult(float *probabilities, int n, float coin);
int compare(const void *a, const void *b);
int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin);
void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler *sampler);
unsigned int random_u32(unsigned long long *state);
float random_f32(unsigned long long *state);
int sample(Sampler *sampler, float *logits);

#endif