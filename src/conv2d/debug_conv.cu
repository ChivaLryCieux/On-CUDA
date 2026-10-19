// Focused debug harness for the convolution ladder.
// Isolates whether the v2 error comes from im2col, the GEMM, or the bias add
// by comparing against references computed with and without bias.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "cuda_utils.h"
#include "reference.h"
#include "conv.h"

int main(int argc, char** argv)
{
    ConvShape s;
    s.N = 1;
    s.C = (argc > 1) ? std::atoi(argv[1]) : 64;
    s.H = (argc > 2) ? std::atoi(argv[2]) : 224;
    s.W = (argc > 3) ? std::atoi(argv[3]) : 224;
    s.OC = (argc > 4) ? std::atoi(argv[4]) : 64;
    s.KH = s.KW = (argc > 5) ? std::atoi(argv[5]) : 3;
    s.pad = (argc > 6) ? std::atoi(argv[6]) : 1;
    s.stride = (argc > 7) ? std::atoi(argv[7]) : 1;
    s.OH = (s.H + 2 * s.pad - s.KH) / s.stride + 1;
    s.OW = (s.W + 2 * s.pad - s.KW) / s.stride + 1;

    std::printf("debug conv C=%d %dx%d OC=%d K=%d pad=%d stride=%d -> %dx%d\n",
                s.C, s.H, s.W, s.OC, s.KH, s.pad, s.stride, s.OH, s.OW);

    const size_t szIn = static_cast<size_t>(s.N) * s.C * s.H * s.W;
    const size_t szW = static_cast<size_t>(s.OC) * s.C * s.KH * s.KW;
    const size_t szOut = static_cast<size_t>(s.N) * s.OC * s.OH * s.OW;
    const size_t szWs = static_cast<size_t>(s.N) * s.C * s.KH * s.KW * s.OH * s.OW;

    std::vector<float> hIn(szIn), hW(szW), hBias(s.OC), hOut(szOut);
    std::vector<float> refBias(szOut), refNoBias(szOut);
    fillRandomHost(hIn.data(), szIn, 11u);
    fillRandomHost(hW.data(), szW, 22u);
    fillRandomHost(hBias.data(), s.OC, 33u);

    conv2dCpu(hIn.data(), hW.data(), hBias.data(), refBias.data(), s);
    conv2dCpu(hIn.data(), hW.data(), nullptr, refNoBias.data(), s);

    float *dIn, *dW, *dBias, *dOut, *dWs;
    CUDA_CHECK(cudaMalloc(&dIn, szIn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW, szW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBias, s.OC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, szOut * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWs, szWs * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), szIn * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, hW.data(), szW * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBias, hBias.data(), s.OC * sizeof(float), cudaMemcpyHostToDevice));

    auto report = [&](const char* tag) {
        CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, szOut * sizeof(float), cudaMemcpyDeviceToHost));
        DiffStats dB = compareArrays(hOut.data(), refBias.data(), szOut);
        DiffStats dN = compareArrays(hOut.data(), refNoBias.data(), szOut);
        std::printf("%-28s relL2 vs ref+bias = %.4e   relL2 vs ref-nobias = %.4e\n",
                    tag, dB.relL2, dN.relL2);
    };

    convIm2col(dIn, dW, dBias, dOut, s, dWs);
    CUDA_CHECK(cudaDeviceSynchronize());
    report("v2 im2col (with bias)");

    convIm2col(dIn, dW, nullptr, dOut, s, dWs);
    CUDA_CHECK(cudaDeviceSynchronize());
    report("v2 im2col (no bias)");

    convSmem(dIn, dW, dBias, dOut, s);
    CUDA_CHECK(cudaDeviceSynchronize());
    report("v1 smem (with bias)");

    convNaive(dIn, dW, dBias, dOut, s);
    CUDA_CHECK(cudaDeviceSynchronize());
    report("v0 naive (with bias)");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaFree(dBias));
    CUDA_CHECK(cudaFree(dOut));
    CUDA_CHECK(cudaFree(dWs));
    return 0;
}
