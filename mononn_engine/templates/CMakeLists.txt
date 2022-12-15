cmake_minimum_required(VERSION 3.20)
project(template CUDA)

set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS "-w -use_fast_math")
set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(/home/zhuangdonglin.zdl/tensorflow/cutlass/include/)

add_executable(reduce reduce.cu reduce_impl.h)
add_executable(numeric_limits numeric_limits.cu)
add_executable(prefetch prefetch.cu)
add_executable(main main.cu)
