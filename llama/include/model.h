#ifndef MODEL_H
#define MODEL_H

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>
#include <time.h>

#include "tokenizer.h"
#include "sampler.h"

// 超参数配置
typedef struct Config {
    // gdb: {dim = 288, hidden_dim = 768, n_layers = 6, n_heads = 6, n_kv_heads = 6, vocab_size = 32000, seq_len = 256}
    int dim;        // embedding层的输出维度
    int hidden_dim; // fnn层隐藏层维度
    int n_layers;   // Transformer blocks的数量
    int n_heads;    // 多头注意力机制中的头数
    int n_kv_heads; // key/value的头数，通常小于等于n_heads
    int vocab_size; // 词汇表大小
    int seq_len;    // 最大序列长度
}Config;

// Transformer模型中的权重参数, 注意是权重
typedef struct
{
    // token embedding table
    float *token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    float *rms_att_weight; // (layer, dim) rmsnorm weights
    float *rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float *wq; // (layer, dim, n_heads * head_size) 用于生成Q矩阵的权重矩阵，形状为[dim,head_size]
    float *wk; // (layer, dim, n_kv_heads * head_size)
    float *wv; // (layer, dim, n_kv_heads * head_size)
    float *wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float *w1; // (layer, hidden_dim, dim)
    float *w2; // (layer, dim, hidden_dim)
    float *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float *rms_final_weight; // (dim,)
    // 用于最后一层的分类器权重 (dim, vocab_size)，当模型需要进行分类时使用
    float *wcls;
} TransformerWeights;

// Transformer模型在推理时的中间计算结果的缓冲区
typedef struct
{
    // current wave of activations
    float *x;      // activation at current time stamp (dim,)
    float *xb;     // same, but inside a residual branch (dim,)
    float *xb2;    // an additional buffer just for convenience (dim,)
    float *hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q;      // query (dim,)
    float *k;      // key (dim,)
    float *v;      // value (dim,)
    float *att;    // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // 存储模型的最终输出
    // kv cache
    float *key_cache;   // (layer, seq_len, dim)
    float *value_cache; // (layer, seq_len, dim)
} RunState;

// Transformer模型的数据结构
typedef struct
{
    Config config;              // 模型的超参数配置，用于定义模型架构
    TransformerWeights weights; // 模型的权重，包含所有层的参数
    RunState state;             // 用于前向传播过程中激活状态的缓冲区
    int fd;                     // 文件描述符，用于内存映射时打开权重文件
    float *data;                // 内存映射的数据指针，指向映射的文件数据
    ssize_t file_size;          // 检查点文件的大小（字节数）
} Transformer;


void malloc_run_state(RunState *s, Config *p);
void free_run_state(RunState *s);
void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights);
void softmax(float *x, int size);
void rmsnorm(float *o, float *x, float *weight, int size);
void read_checkpoint(char *checkpoint, Config *config, TransformerWeights *weights,
                     int *fd, float **data, ssize_t *file_size);
void build_transformer(Transformer *t, char *checkpoint_path);
void free_transformer(Transformer *t);
void matmul(float *xout, float *x, float *w, int n, int d);
float *forward(Transformer *transformer, int token, int pos);
void safe_printf(char *piece);
long time_in_ms();
void read_stdin(const char *guide, char *buffer, size_t bufsize);
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps);

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps);
          
#endif