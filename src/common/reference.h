#pragma once
// CPU reference implementations + comparison helpers used to prove that every
// GPU kernel is numerically correct before we start optimizing it.
#include <cuda_runtime.h>
#include <cmath>

// NCHW convolution description.
struct ConvShape
{
    int N, C, H, W;      // input:  N images, C channels, H rows, W cols
    int OC, KH, KW;      // filter: OC output channels, KH x KW spatial
    int OH, OW;          // output spatial size
    int pad, stride;
};

// C = A * B, all row-major. A: M x K, B: K x N, C: M x N.
void matmulCpu(const float* A, const float* B, float* C, int M, int N, int K);

// Standard convolution, NCHW, row-major.
// in: N*C*H*W, w: OC*C*KH*KW, bias: OC (may be null), out: N*OC*OH*OW.
void conv2dCpu(const float* in, const float* w, const float* bias, float* out, const ConvShape& s);

// Depthwise-separable convolution reference (different weights, fewer FLOPs).
// dwW: C*KH*KW (one filter per channel), pwW: OC*C (1x1 conv), bias: OC.
void convDwSepCpu(const float* in, const float* dwW, const float* pwW, const float* bias,
                  float* out, const ConvShape& s);

struct DiffStats
{
    double relL2;   // ||a-b|| / ||a||
    double maxAbs;  // largest absolute difference
};
DiffStats compareArrays(const float* a, const float* b, size_t count);

inline double matmulFlops(int M, int N, int K)
{
    return 2.0 * static_cast<double>(M) * N * K;
}

inline double convFlops(const ConvShape& s)
{
    return 2.0 * s.N * s.OC * s.C * s.KH * s.KW * s.OH * s.OW;
}

// Depthwise (C*KH*KW per pixel) + pointwise (OC*C per pixel).
inline double convDwSepFlops(const ConvShape& s)
{
    double dw = 2.0 * s.N * s.C * s.KH * s.KW * s.OH * s.OW;
    double pw = 2.0 * s.N * s.OC * s.C * s.OH * s.OW;
    return dw + pw;
}
