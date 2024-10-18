cmake_minimum_required(VERSION 3.10)

project(LLaMA-infer VERSION 1.0)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

# include
include_directories(include)

# source
set(SOURCES
    src/main.c
    src/model.c
    src/sampler.c
    src/tokenizer.c
)
# debug
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "-g")

# exe
add_executable(llama-infer ${SOURCES})
# math lib
target_link_libraries(llama-infer m)
