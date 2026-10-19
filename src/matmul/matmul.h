#pragma once
// Matrix multiplication launchers.
//
// Every matrix is row-major:
//   A: M x K,  B: K x N,  C: M x N,  with C = A * B.
//
// The four implementations form the classic optimization ladder:
//   v0 naive    - one thread per output, zero data reuse
//   v1 tiled    - shared memory blocking, reuse loaded tiles
//   v2 reg-tile - register blocking, each thread computes a TM x TN micro-tile
//   v3 wmma     - FP16 Tensor Cores via the WMMA API
#include <cuda_runtime.h>
#include <cuda_fp16.h>

void matmulNaive(const float* A, const float* B, float* C, int M, int N, int K,
                 cudaStream_t stream = nullptr);

template <int TILE>
void matmulTiled(const float* A, const float* B, float* C, int M, int N, int K,
                 cudaStream_t stream = nullptr);

void matmulRegTile(const float* A, const float* B, float* C, int M, int N, int K,
                   cudaStream_t stream = nullptr);

// FP16 inputs, FP32 accumulation. Handles arbitrary M, N, K (zero-padded tiles).
void matmulWmma(const __half* A, const __half* B, float* C, int M, int N, int K,
                cudaStream_t stream = nullptr);
