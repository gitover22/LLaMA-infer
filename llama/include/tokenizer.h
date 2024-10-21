#ifndef TOKENIZER_H
#define TOKENIZER_H
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>

typedef struct
{
    char *str;
    int id;
} TokenIndex;

typedef struct
{
    char **vocab; // 存储所有的tokens string
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b);
void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer *t);
char *decode(Tokenizer *t, int prev_token, int token);
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size);
void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
#endif