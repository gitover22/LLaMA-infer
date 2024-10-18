#include "model.h"
#include "tokenizer.h"
#include "sampler.h"

#ifndef TESTING

void error_usage()
{
    fprintf(stderr, "Usage:   ./llama-infer <checkpoint> [options]\n");
    fprintf(stderr, "Example: ./llama-infer model.bin -n 256 -i \"Once upon a time\"\n");
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

int main(int argc, char *argv[])
{

    // 默认参数
    char *checkpoint_path = NULL; // .bin权重文件
    char *tokenizer_path = "../tokenizer.bin";
    float temperature = 1.0f;        // 温度
    float topp = 0.9f;               // top-p
    int steps = 256;                 // number of steps to run for
    char *prompt = NULL;             // prompt
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";         // generate|chat
    char *system_prompt = NULL;      // the (optional) system prompt to use in chat mode

    // 检查配置
    if (argc >= 2)
    {
        checkpoint_path = argv[1];  // "stories42M.bin"
    }else{
        error_usage();
    }
    for (int i = 2; i < argc; i += 2)
    {
        // 规则检查
        if (i + 1 >= argc || argv[i][0] != '-' || strlen(argv[i]) !=2 )
        {
            error_usage();
        }
        // 温度
        if (argv[i][1] == 't')
        {
            temperature = atof(argv[i + 1]);
        }
        // top-p
        else if (argv[i][1] == 'p')
        {
            topp = atof(argv[i + 1]);
        }
        else if (argv[i][1] == 's')
        {
            rng_seed = atoi(argv[i + 1]);
        }
        else if (argv[i][1] == 'n')
        {
            steps = atoi(argv[i + 1]);
        }
        // prompt
        else if (argv[i][1] == 'i')
        {
            prompt = argv[i + 1];
        }
        else if (argv[i][1] == 'z')
        {
            tokenizer_path = argv[i + 1];
        }
        else if (argv[i][1] == 'm')
        {
            mode = argv[i + 1];
        }
        else if (argv[i][1] == 'y')
        {
            system_prompt = argv[i + 1];
        }
        else
        {
            error_usage();
        }
    }

    // 参数检查
    if (rng_seed <= 0)
        rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0)
        temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp)
        topp = 0.9;
    if (steps < 0)
        steps = 0;

    // 通过模型的.bin权重文件构建模型
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len)
        steps = transformer.config.seq_len;

    // 通过tokenizer.bin文件构建分词器
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // 构建采样器
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    if (strcmp(mode, "generate") == 0)
    {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    }
    else if (strcmp(mode, "chat") == 0)
    {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    }
    else
    {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // free
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif