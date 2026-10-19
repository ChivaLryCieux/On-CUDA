#include "reference.h"

#include <vector>

void matmulCpu(const float* A, const float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[static_cast<size_t>(i) * K + k] * B[static_cast<size_t>(k) * N + j];
            C[static_cast<size_t>(i) * N + j] = acc;
        }
    }
}

void conv2dCpu(const float* in, const float* w, const float* bias, float* out, const ConvShape& s)
{
    for (int n = 0; n < s.N; ++n) {
        for (int oc = 0; oc < s.OC; ++oc) {
            for (int oh = 0; oh < s.OH; ++oh) {
                for (int ow = 0; ow < s.OW; ++ow) {
                    float acc = bias ? bias[oc] : 0.0f;
                    for (int c = 0; c < s.C; ++c) {
                        for (int kh = 0; kh < s.KH; ++kh) {
                            int ih = oh * s.stride - s.pad + kh;
                            if (ih < 0 || ih >= s.H) continue;
                            for (int kw = 0; kw < s.KW; ++kw) {
                                int iw = ow * s.stride - s.pad + kw;
                                if (iw < 0 || iw >= s.W) continue;
                                float v = in[(static_cast<size_t>(n) * s.C + c) * s.H * s.W
                                             + static_cast<size_t>(ih) * s.W + iw];
                                float f = w[(static_cast<size_t>(oc) * s.C + c) * s.KH * s.KW
                                            + kh * s.KW + kw];
                                acc += v * f;
                            }
                        }
                    }
                    out[(static_cast<size_t>(n) * s.OC + oc) * s.OH * s.OW
                        + static_cast<size_t>(oh) * s.OW + ow] = acc;
                }
            }
        }
    }
}

void convDwSepCpu(const float* in, const float* dwW, const float* pwW, const float* bias,
                  float* out, const ConvShape& s)
{
    const int P = s.OH * s.OW;
    std::vector<float> tmp(static_cast<size_t>(s.N) * s.C * P);

    // Stage 1: depthwise. Each channel is convolved with its own KH x KW filter.
    for (int n = 0; n < s.N; ++n) {
        for (int c = 0; c < s.C; ++c) {
            for (int oh = 0; oh < s.OH; ++oh) {
                for (int ow = 0; ow < s.OW; ++ow) {
                    float acc = 0.0f;
                    for (int kh = 0; kh < s.KH; ++kh) {
                        int ih = oh * s.stride - s.pad + kh;
                        if (ih < 0 || ih >= s.H) continue;
                        for (int kw = 0; kw < s.KW; ++kw) {
                            int iw = ow * s.stride - s.pad + kw;
                            if (iw < 0 || iw >= s.W) continue;
                            acc += in[(static_cast<size_t>(n) * s.C + c) * s.H * s.W
                                      + static_cast<size_t>(ih) * s.W + iw]
                                   * dwW[static_cast<size_t>(c) * s.KH * s.KW + kh * s.KW + kw];
                        }
                    }
                    tmp[(static_cast<size_t>(n) * s.C + c) * P + static_cast<size_t>(oh) * s.OW + ow] = acc;
                }
            }
        }
    }

    // Stage 2: pointwise. 1x1 conv == GEMM [OC x C] * [C x P].
    for (int n = 0; n < s.N; ++n) {
        for (int oc = 0; oc < s.OC; ++oc) {
            for (int p = 0; p < P; ++p) {
                float acc = bias ? bias[oc] : 0.0f;
                for (int c = 0; c < s.C; ++c)
                    acc += pwW[static_cast<size_t>(oc) * s.C + c]
                           * tmp[(static_cast<size_t>(n) * s.C + c) * P + p];
                out[(static_cast<size_t>(n) * s.OC + oc) * P + p] = acc;
            }
        }
    }
}

DiffStats compareArrays(const float* a, const float* b, size_t count)
{
    double num = 0.0;
    double den = 0.0;
    double maxAbs = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += d * d;
        den += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        double ad = std::fabs(d);
        if (ad > maxAbs) maxAbs = ad;
    }
    DiffStats s;
    s.relL2 = (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
    s.maxAbs = maxAbs;
    return s;
}
