#pragma once
// Convolution launchers.
//
// Layout is NCHW, row-major:
//   in:   N*C*H*W
//   w:    OC*C*KH*KW          (standard conv)
//   bias: OC                  (nullable)
//   out:  N*OC*OH*OW
//
// The ladder mirrors matmul, plus the two tricks real frameworks use:
//   v0 naive      - direct convolution, no reuse
//   v1 smem       - input tile with halo staged in shared memory
//   v2 im2col     - reshape to GEMM, then reuse the optimized matmul
//   v3 dw-sep     - depthwise + pointwise, far fewer FLOPs
#include <cuda_runtime.h>

#include "reference.h"   // ConvShape

void convNaive(const float* in, const float* w, const float* bias, float* out,
               const ConvShape& s, cudaStream_t stream = nullptr);

void convSmem(const float* in, const float* w, const float* bias, float* out,
              const ConvShape& s, cudaStream_t stream = nullptr);

// workspace needs at least N * (C*KH*KW) * (OH*OW) floats.
void convIm2col(const float* in, const float* w, const float* bias, float* out,
                const ConvShape& s, float* workspace, cudaStream_t stream = nullptr);

// Depthwise-separable variant with its own weights:
//   dwW: C*KH*KW (one filter per channel), pwW: OC*C (1x1 conv), bias: OC.
// workspace needs at least N * C * (OH*OW) floats.
void convDepthwiseSeparable(const float* in, const float* dwW, const float* pwW,
                            const float* bias, float* out, const ConvShape& s,
                            float* workspace, cudaStream_t stream = nullptr);
