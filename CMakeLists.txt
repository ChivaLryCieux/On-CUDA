cmake_minimum_required(VERSION 3.20)

project(NsightProgram LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 2080 SUPER = Turing = compute capability 7.5.
# Pin it explicitly so nvcc does not spend time JIT-ing from a virtual arch.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# -lineinfo lets Nsight Compute correlate SASS back to source lines.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo --expt-relaxed-constexpr")

find_package(CUDAToolkit REQUIRED)

add_subdirectory(src/matmul)
add_subdirectory(src/conv2d)
