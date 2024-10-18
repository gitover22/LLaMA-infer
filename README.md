## LLaMA-infer
A Inference Framework for LLaMA Models (cpu-only).
this project is a great practice for newers in LLM inference.
### Requirements
Download model weight files
```shell
cd ./llama
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
```
### Build & Run
```shell
cd ./llama/build
cmake ..
make
./llama-infer ../stories15M.bin (or ../stories42M.bin )
```


### Others
this project is based on [llama2.c](https://github.com/karpathy/llama2.c)
