#include "tokenizer.h"

int compare_tokens(const void *a, const void *b)
{
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

/**
 * @brief 初始化并构建分词器，从文件中加载词汇表和对应的分数。
 *
 * 该函数从指定路径的分词器文件中加载词汇表（vocabulary）及其对应的得分，并将它们存储在分词器结构体中。
 * 分词器用于将文本转换为 token，或将 token 转换为文本。
 *
 * @param t 指向分词器结构体的指针。
 * @param tokenizer_path 分词器文件的路径，用于加载词汇表和相关数据。
 * @param vocab_size 词汇表的大小（即词汇表中 token 的数量）。
 */
void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size)
{
    // 本应将 vocab_size 写入分词器文件中（遗憾没有这么做）
    t->vocab_size = vocab_size;

    // 分配空间来存储词汇表中的字符串和它们的分数
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));   // 分配词汇表指针数组
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float)); // 分配词汇表对应的分数
    t->sorted_vocab = NULL; // 懒加载，稍后再初始化排序的词汇表

    // 初始化 byte_pieces，用于字节级别的分词（将前256个字节的字符初始化为可打印的字符）
    for (int i = 0; i < 256; i++)
    {
        t->byte_pieces[i * 2] = (unsigned char)i;   // 存储字节字符
        t->byte_pieces[i * 2 + 1] = '\0';           // 以 '\0' 作为字符串终止符
    }

    // 打开分词器文件（以二进制只读模式）
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file)
    {
        fprintf(stderr, "couldn't load %s\n", tokenizer_path); // 文件打开失败
        exit(EXIT_FAILURE);
    }

    // 读取分词器文件中的最大 token 长度
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE); // 读取失败，程序退出
    }

    int len; // 用于存储字符串长度
    // 遍历词汇表，读取每个词汇及其对应的分数
    for (int i = 0; i < vocab_size; i++)
    {
        // 读取词汇表中每个 token 的分数
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1)
        {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }

        // 读取 token 对应的字符串长度
        if (fread(&len, sizeof(int), 1, file) != 1)
        {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }

        // 为每个 token 的字符串分配内存，并读取字符串
        t->vocab[i] = (char *)malloc(len + 1); // +1 是为 '\0' 留出空间
        if (fread(t->vocab[i], len, 1, file) != 1)
        {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }

        // 为字符串添加终止符
        t->vocab[i][len] = '\0';
    }

    // 关闭文件
    fclose(file);
}


void free_tokenizer(Tokenizer *t)
{
    for (int i = 0; i < t->vocab_size; i++)
    {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}


int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size)
{
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = {.str = str}; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens)
{
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL)
    {
        fprintf(stderr, "cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }

    if (t->sorted_vocab == NULL)
    {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++)
        {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char *str_buffer = malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos)
        tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0')
    {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++)
    {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80)
        {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4)
        {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1)
        {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        }
        else
        {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i = 0; i < str_len; i++)
            {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1)
    {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++)
        {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score)
            {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1)
        {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++)
        {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos)
        tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}
