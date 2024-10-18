#include "model.h"

/**
 * @brief 为推理过程中的运行状态分配内存空间。
 *
 * 该函数为 `RunState` 结构体中的各个缓冲区分配内存，主要包括用于存储激活值、注意力权重、FFN缓冲区等。
 * 内存分配使用 `calloc` 而不是 `malloc`，以确保分配的内存块都被初始化为0，避免未初始化数据问题（尤其是在使用Valgrind等工具时）。
 * 如果任意一个内存分配失败，程序会输出错误信息并退出。
 *
 * @param s 指向 `RunState` 结构体的指针，该结构体用于存储推理过程中的中间状态。
 * @param p 指向 `Config` 结构体的指针，包含模型的配置参数（如层数、嵌入维度等）。
 */
void malloc_run_state(RunState *s, Config *p)
{
    // 根据配置参数计算 key/value 的维度（用于注意力机制中）
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads; // kv_dim = 288

    // 为每个需要的缓冲区分配内存，使用calloc保证初始化为0
    s->x = calloc(p->dim, sizeof(float));                  // 激活值缓存 (维度：dim)
    s->xb = calloc(p->dim, sizeof(float));                 // 残差连接中的缓存 (维度：dim)
    s->xb2 = calloc(p->dim, sizeof(float));                // 额外的缓冲区 (维度：dim)
    s->hb = calloc(p->hidden_dim, sizeof(float));          // FFN中的隐藏层缓存 (维度：hidden_dim)
    s->hb2 = calloc(p->hidden_dim, sizeof(float));         // FFN中的第二隐藏层缓存 (维度：hidden_dim)
    s->q = calloc(p->dim, sizeof(float));                  // 查询向量缓存 (维度：dim)

    // 为 key 和 value 缓存分配内存，按序列长度和层数计算
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));  // 键缓存
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float)); // 值缓存

    // 为注意力得分分配缓存空间 (维度：n_heads * seq_len)
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));

    // 为logits（模型输出）分配内存 (维度：vocab_size)
    s->logits = calloc(p->vocab_size, sizeof(float));

    // 检查是否所有内存分配都成功，若任一分配失败则终止程序
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
        || !s->key_cache || !s->value_cache || !s->att || !s->logits)
    {
        // 如果内存分配失败，输出错误信息并退出程序
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}
// 释放RunState结构体的资源
void free_run_state(RunState *s)
{
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

/**
 * @brief 将 Transformer 的权重映射到内存中。
 *
 * 该函数将模型的各层权重依次映射到 `TransformerWeights` 结构体中，利用 `ptr` 指针依次遍历内存中的权重数据，并根据模型配置来计算权重在内存中的分布。
 * 
 * @param w 指向 `TransformerWeights` 结构体的指针，用于存储映射的权重。
 * @param p 指向 `Config` 结构体的指针，包含模型的配置参数（如层数、头数等）。
 * @param ptr 指向模型权重的内存数据指针，指向被映射的权重区域。
 * @param shared_weights 指示是否共享权重，如果为 1，分类器权重将与嵌入表共享。
 */
void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights)
{
    int head_size = p->dim / p->n_heads;  // 48 = 288 / 6

    // 为了适应大模型（例如13B参数模型）的参数数量，使用64位无符号整数
    unsigned long long n_layers = p->n_layers; // 6

    // 映射 token 嵌入表，大小为 (vocab_size, dim)
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim; // 32000 * 288 ,将 ptr 向前移动，跳过嵌入表所占用的内存

    // 映射每层的 rmsnorm 权重，大小为 (n_layers, dim)
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;

    // 映射查询 (wq) 矩阵，大小为 (n_layers, dim, n_heads * head_size)
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);

    // 映射键 (wk) 矩阵，大小为 (n_layers, dim, n_kv_heads * head_size)
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);

    // 映射值 (wv) 矩阵，大小为 (n_layers, dim, n_kv_heads * head_size)
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);

    // 映射输出 (wo) 矩阵，大小为 (n_layers, n_heads * head_size, dim)
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;

    // 映射 FFN 层的 rmsnorm 权重，大小为 (n_layers, dim)
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;

    // 映射 FFN 层的 w1 矩阵，大小为 (n_layers, hidden_dim, dim)
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;

    // 映射 FFN 层的 w2 矩阵，大小为 (n_layers, dim, hidden_dim)
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;

    // 映射 FFN 层的 w3 矩阵，大小为 (n_layers, hidden_dim, dim)
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;

    // 映射最后一层的 rmsnorm 权重，大小为 (dim)
    w->rms_final_weight = ptr;
    ptr += p->dim;

    // 跳过频率编码 (RoPE) 的部分内存，假设每个序列有 head_size / 2 的大小
    ptr += p->seq_len * head_size / 2;  // 跳过实部 (freq_cis_real)
    ptr += p->seq_len * head_size / 2;  // 跳过虚部 (freq_cis_imag)

    // 映射分类器权重，如果 `shared_weights` 为 1，则与嵌入表共享；否则映射新区域
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}


/**
 * @brief 读取检查点文件并将权重映射到内存中。
 *
 * 该函数从检查点文件中读取 Transformer 模型的配置和权重数据，并将它们映射到内存中，
 * 以便在模型推理时使用。通过内存映射可以高效地加载大型模型。
 *
 * @param checkpoint 指向检查点文件路径的指针。
 * @param config 指向存储模型配置的 Config 结构体。
 * @param weights 指向存储模型权重的 TransformerWeights 结构体。
 * @param fd 指向文件描述符的指针，用于存储打开的文件句柄。
 * @param data 指向内存映射数据的指针，用于将文件数据映射到内存。
 * @param file_size 指向存储文件大小的变量，用于保存检查点文件的字节大小。
 */
void read_checkpoint(char *checkpoint, Config *config, TransformerWeights *weights,
                     int *fd, float **data, ssize_t *file_size)
{
    // 打开检查点文件（只读模式）
    FILE *file = fopen(checkpoint, "rb"); //model.bin
    if (!file)
    {
        // 如果文件无法打开，打印错误并退出
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }

    // 读取配置头信息（Config 结构体）
    if (fread(config, sizeof(Config), 1, file) != 1)
    {
        // 如果读取失败，退出程序
        exit(EXIT_FAILURE);
    }

    // 如果词汇表大小为负值，则表示权重未共享（这是一个标记方式）
    // 这里通过判断词汇表大小的正负来决定是否共享权重
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);  // 将 vocab_size 处理为正数 --> 32000

    // 获取文件大小
    fseek(file, 0, SEEK_END);      // 将文件指针移动到文件末尾
    *file_size = ftell(file);      // 获取文件的字节大小
    fclose(file);                  // 关闭文件

    // 内存映射Transformer权重
    *fd = open(checkpoint, O_RDONLY);  // 以只读模式打开文件
    if (*fd == -1)
    {
        // 如果文件打开失败，打印错误并退出
        fprintf(stderr, "open failed!\n");
        exit(EXIT_FAILURE);
    }

    // 使用 mmap 将文件映射到内存中（只读权限）
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED)
    {
        // 如果 mmap 失败，打印错误并退出
        fprintf(stderr, "mmap failed!\n");
        exit(EXIT_FAILURE);
    }

    // 计算权重数据的起始位置（跳过 Config 部分）
    float *weights_ptr = *data + sizeof(Config) / sizeof(float);

    // 映射模型的权重到 TransformerWeights 结构体中
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}


void build_transformer(Transformer *t, char *checkpoint_path)
{
    // 从model.bin中读取权重和配置信息
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // 分配RUN_STATE
    malloc_run_state(&t->state, &t->config);
}

// 释放transformer结构体
void free_transformer(Transformer *t)
{
    // close the memory mapping
    if (t->data != MAP_FAILED)
    {
        munmap(t->data, t->file_size);
    }
    if (t->fd != -1)
    {
        close(t->fd);
    }
    // free the RunState buffers
    free_run_state(&t->state);
}
// 归一化
void rmsnorm(float *o, float *x, float *weight, int size)
{
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++)
    {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++)
    {
        o[j] = weight[j] * (ss * x[j]);
    }
}
// 激活函数
void softmax(float *x, int size)
{
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++)
    {
        if (x[i] > max_val)
        {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++)
    {
        x[i] /= sum;
    }
}

/**
 * @brief 执行矩阵和向量的乘法操作：xout = W * x
 *
 * 该函数执行矩阵-向量乘法，其中矩阵 `W` 的维度是 `(d, n)`，向量 `x` 的维度是 `(n)`，
 * 结果向量 `xout` 的维度是 `(d)`。此函数的计算过程是按行遍历矩阵 `W`，并与输入向量 `x` 相乘。
 *
 * @param xout 指向输出向量的指针，大小为 d。存储矩阵与向量乘法的结果。
 * @param x 指向输入向量的指针，大小为 n。输入的向量。
 * @param w 指向权重矩阵 `W` 的指针，大小为 d * n。存储矩阵的权重。
 * @param n 输入向量的维度，也对应矩阵 `W` 的列数。
 * @param d 输出向量的维度，也对应矩阵 `W` 的行数。
 */
void matmul(float *xout, float *x, float *w, int n, int d)
{
    int i;  // 用于循环的局部变量

    // 使用 OpenMP 进行并行计算，每个线程独立计算输出向量 xout 的一部分。
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++)  // 遍历输出向量的每个元素
    {
        float val = 0.0f;  // 初始化结果值
        for (int j = 0; j < n; j++)  // 对矩阵的每一行执行点积计算
        {
            // 执行矩阵的第 i 行与向量 x 的点积
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;  // 将结果存入输出向量的第 i 个位置
    }
}

/**
 * @brief Transformer 模型的前向传播函数，基于给定的 token 和位置，执行多层 Transformer 的计算。
 *
 * 该函数实现了 Transformer 的前向传播操作，包括嵌入查询、注意力计算、残差连接、前馈网络和归一化等。
 * 最终返回的是输出层的 logits，用于模型的预测任务（如分类或生成任务）。
 *
 * @param transformer 指向 Transformer 模型的指针，包含模型的权重、配置和运行状态。
 * @param token 当前输入的 token。
 * @param pos 当前 token 在序列中的位置。
 *
 * @return 返回计算后的 logits，大小为 (vocab_size)，用于模型的输出预测。
 */
float *forward(Transformer *transformer, int token, int pos)
{

    // 定义一些方便使用的变量
    Config *p = &transformer->config;          // 模型配置
    TransformerWeights *w = &transformer->weights; // 模型权重
    RunState *s = &transformer->state;         // 前向传播中的运行状态
    float *x = s->x;                           // 当前时间步的激活值
    int dim = p->dim;                          // 嵌入维度
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads; // 键/值的维度
    int kv_mul = p->n_heads / p->n_kv_heads;   // 多查询机制的键/值头的倍数
    int hidden_dim = p->hidden_dim;            // FFN的隐藏层维度
    int head_size = dim / p->n_heads;          // 每个注意力头的维度

    // 将输入的 token 嵌入表复制到 x 中
    float *content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));

    // 遍历所有层，执行前向传播
    for (unsigned long long l = 0; l < p->n_layers; l++)
    {

        // 对当前层的激活值执行 rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // 键和值指向注意力机制中的缓存位置
        int loff = l * p->seq_len * kv_dim; // 键值缓存层的偏移
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // 计算当前时间步的查询、键和值（Q, K, V）
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        // 计算相对位置编码 RoPE，对查询和键进行旋转操作
        for (int i = 0; i < dim; i += 2)
        {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // 2 表示同时旋转 Q 和 K，1 表示仅旋转 Q
            for (int v = 0; v < rotn; v++)
            {
                float *vec = v == 0 ? s->q : s->k; // 旋转查询或键向量
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // 多头注意力计算，遍历所有头
        int h;
#pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++)
        {
            // 获取当前头的查询向量
            float *q = s->q + h * head_size;
            // 存储当前头的注意力得分
            float *att = s->att + h * p->seq_len;
            // 遍历所有时间步，计算注意力得分
            for (int t = 0; t <= pos; t++)
            {
                // 获取当前时间步的键向量
                float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // 计算查询向量和键向量的点积（即注意力得分）
                float score = 0.0f;
                for (int i = 0; i < head_size; i++)
                {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size); // 归一化
                // 将得分存储到注意力缓冲区
                att[t] = score;
            }

            // 使用 softmax 计算注意力权重
            softmax(att, pos + 1);

            // 使用注意力权重对值向量进行加权求和，存储到 xb 中
            float *xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++)
            {
                // 获取当前时间步的值向量
                float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // 获取当前时间步的注意力权重
                float a = att[t];
                // 加权求和值
                for (int i = 0; i < head_size; i++)
                {
                    xb[i] += a * v[i];
                }
            }
        }

        // 最终通过线性层计算注意力的输出
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        // 残差连接，将结果加回到 x
        for (int i = 0; i < dim; i++)
        {
            x[i] += s->xb2[i];
        }

        // 对前馈网络执行 rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // 计算前馈网络中的两层（w1 和 w3）
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        // 应用 SwiGLU 非线性激活函数
        for (int i = 0; i < hidden_dim; i++)
        {
            float val = s->hb[i];
            // silu(x)=x*σ(x), 其中 σ(x) 是 logistic sigmoid 函数
            val *= (1.0f / (1.0f + expf(-val)));
            // 与 w3(x) 元素相乘
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // 通过线性层计算前馈网络的输出
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        // 残差连接，将结果加回到 x
        for (int i = 0; i < dim; i++)
        {
            x[i] += s->xb[i];
        }
    }

    // 执行最终的 rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // 将结果传递给输出层（logits），进行分类或生成任务
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}


void safe_printf(char *piece)
{
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL)
    {
        return;
    }
    if (piece[0] == '\0')
    {
        return;
    }
    if (piece[1] == '\0')
    {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val)))
        {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}
long time_in_ms()
{
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}
// generate：基于输入提示词逐步生成文本
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps)
{
    char *empty_prompt = "";
    if (prompt == NULL)
    {
        prompt = empty_prompt;
    }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1)
    {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;               // used to time our code, only initialized after first iteration
    int next;                     // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;                  // position in the sequence
    while (pos < steps)
    {

        // forward the transformer to get logits for the next token
        float *logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1)
        {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        }
        else
        {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1)
        {
            break;
        }

        // print the token as string, decode it with the Tokenizer object
        char *piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0)
        {
            start = time_in_ms();
        }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1)
    {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char *guide, char *buffer, size_t bufsize)
{
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL)
    {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n')
        {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps)
{

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;             // will store the next token in the sequence
    int token;            // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0; // position in the sequence
    while (pos < steps)
    {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn)
        {
            // get the (optional) system prompt at position 0
            if (pos == 0)
            {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL)
                {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                }
                else
                {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL)
            {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            }
            else
            {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0')
            {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            }
            else
            {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens)
        {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        }
        else
        {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2)
        {
            user_turn = 1;
        }

        // forward the transformer to get logits for the next token
        float *logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2)
        {
            // the Assistant is responding, so print its output
            char *piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2)
        {
            printf("\n");
        }
    }
    printf("\n");
    free(prompt_tokens);
}


