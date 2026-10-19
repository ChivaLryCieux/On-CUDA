// Benchmark driver for the convolution ladder.
//
// Usage: bench_conv [C] [H] [W] [OC] [K] [pad] [stride] [--noverify]
//
//   bench_conv                       # C=64 224x224 OC=64 3x3 (ResNet-ish)
//   bench_conv 3 512 512 8 7 3 2     # first-layer style: 7x7, stride 2
//   bench_conv 256 56 56 256 3 1 1   # deeper layer style
//   bench_conv --noverify ...        # skip the slow CPU reference
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cuda_utils.h"
#include "reference.h"
#include "conv.h"

static const int kIters = 20;
static const int kWarmup = 5;

struct Row
{
    const char* name;
    float ms;
    double gflops;
    double relL2;
};

static void printTable(const std::vector<Row>& rows, float baseMs)
{
    std::printf("%-26s %10s %12s %10s %12s\n", "implementation", "ms", "GFLOP/s", "speedup", "relL2 err");
    for (const Row& r : rows) {
        std::printf("%-26s %10.4f %12.2f %9.2fx %12.2e\n",
                    r.name, r.ms, r.gflops, baseMs / r.ms, r.relL2);
    }
    std::printf("\n");
}

int main(int argc, char** argv)
{
    bool verify = true;
    if (argc > 1 && std::strcmp(argv[1], "--noverify") == 0) {
        verify = false;
        --argc;
        ++argv;
    }

    ConvShape s;
    s.N = 1;
    s.C = 64; s.H = 224; s.W = 224;
    s.OC = 64; s.KH = 3; s.KW = 3;
    s.pad = 1; s.stride = 1;

    if (argc > 1) s.C = std::atoi(argv[1]);
    if (argc > 2) s.H = std::atoi(argv[2]);
    if (argc > 3) s.W = std::atoi(argv[3]);
    if (argc > 4) s.OC = std::atoi(argv[4]);
    if (argc > 5) { s.KH = s.KW = std::atoi(argv[5]); }
    if (argc > 6) s.pad = std::atoi(argv[6]);
    if (argc > 7) s.stride = std::atoi(argv[7]);

    s.OH = (s.H + 2 * s.pad - s.KH) / s.stride + 1;
    s.OW = (s.W + 2 * s.pad - s.KW) / s.stride + 1;

    std::printf("=== conv2d: N=%d C=%d %dx%d OC=%d K=%dx%d pad=%d stride=%d -> %dx%d ===\n",
                s.N, s.C, s.H, s.W, s.OC, s.KH, s.KW, s.pad, s.stride, s.OH, s.OW);
    std::printf("standard conv FLOPs: %.3f GFLOP\n", convFlops(s) / 1e9);

    const size_t szIn  = static_cast<size_t>(s.N) * s.C * s.H * s.W;
    const size_t szW   = static_cast<size_t>(s.OC) * s.C * s.KH * s.KW;
    const size_t szOut = static_cast<size_t>(s.N) * s.OC * s.OH * s.OW;
    const size_t szWsIm2col = static_cast<size_t>(s.N) * s.C * s.KH * s.KW * s.OH * s.OW;
    const size_t szWsDwSep  = static_cast<size_t>(s.N) * s.C * s.OH * s.OW;

    std::vector<float> hIn(szIn), hW(szW), hBias(s.OC), hOut(szOut), hRef(szOut);
    fillRandomHost(hIn.data(), szIn, 11u);
    fillRandomHost(hW.data(), szW, 22u);
    fillRandomHost(hBias.data(), s.OC, 33u);

    float *dIn = nullptr, *dW = nullptr, *dBias = nullptr, *dOut = nullptr, *dWs = nullptr;
    CUDA_CHECK(cudaMalloc(&dIn, szIn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW, szW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBias, s.OC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, szOut * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWs, (szWsIm2col > szWsDwSep ? szWsIm2col : szWsDwSep) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), szIn * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, hW.data(), szW * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBias, hBias.data(), s.OC * sizeof(float), cudaMemcpyHostToDevice));

    if (verify) {
        std::printf("running CPU reference (single threaded, please wait)...\n");
        conv2dCpu(hIn.data(), hW.data(), hBias.data(), hRef.data(), s);
    }

    std::vector<Row> rows;
    GpuTimer timer;
    const double flops = convFlops(s);

    // `ref` is passed in because depthwise-separable is a different operator
    // with its own weights and therefore its own reference output.
    auto bench = [&](const char* name, const float* ref, auto&& launch) {
        for (int i = 0; i < kWarmup; ++i) launch();
        // Catch silent launch failures: a failed kernel leaves the previous
        // result in dOut and would look "fast and correct".
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.start();
        for (int i = 0; i < kIters; ++i) launch();
        float ms = timer.stop() / static_cast<float>(kIters);

        double relL2 = 0.0;
        if (verify) {
            CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, szOut * sizeof(float), cudaMemcpyDeviceToHost));
            relL2 = compareArrays(hOut.data(), ref, szOut).relL2;
        }
        rows.push_back(Row{ name, ms, flops / (static_cast<double>(ms) * 1e-3) / 1e9, relL2 });
    };

    bench("v0 naive direct", hRef.data(), [&] { convNaive(dIn, dW, dBias, dOut, s); });
    bench("v1 shared-memory", hRef.data(), [&] { convSmem(dIn, dW, dBias, dOut, s); });
    bench("v2 im2col + GEMM", hRef.data(), [&] { convIm2col(dIn, dW, dBias, dOut, s, dWs); });

    const float baseMs = rows.front().ms;
    printTable(rows, baseMs);

    // ---- depthwise-separable: different operator, so its own weights + ref ----
    const size_t szDw = static_cast<size_t>(s.C) * s.KH * s.KW;
    const size_t szPw = static_cast<size_t>(s.OC) * s.C;
    std::vector<float> hDw(szDw), hPw(szPw), hDwBias(s.C), hRef2(szOut);
    fillRandomHost(hDw.data(), szDw, 44u);
    fillRandomHost(hPw.data(), szPw, 55u);
    fillRandomHost(hDwBias.data(), s.C, 66u);

    float *dDw = nullptr, *dPw = nullptr, *dDwBias = nullptr;
    CUDA_CHECK(cudaMalloc(&dDw, szDw * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dPw, szPw * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDwBias, s.C * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dDw, hDw.data(), szDw * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPw, hPw.data(), szPw * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDwBias, hDwBias.data(), s.C * sizeof(float), cudaMemcpyHostToDevice));

    if (verify) {
        std::printf("running CPU reference for depthwise-separable...\n");
        convDwSepCpu(hIn.data(), hDw.data(), hPw.data(), hDwBias.data(), hRef2.data(), s);
    }

    const double dwFlops = convDwSepFlops(s);
    std::printf("--- depthwise-separable (%.3f GFLOP, %.1fx fewer than standard) ---\n",
                dwFlops / 1e9, convFlops(s) / dwFlops);

    std::vector<Row> rows2;
    bench("v3 depthwise-separable", hRef2.data(), [&] {
        convDepthwiseSeparable(dIn, dDw, dPw, dDwBias, dOut, s, dWs);
    });
    rows2.push_back(rows.back());
    rows.pop_back();

    // Compare against the standard-conv naive baseline on an equal-FLOP basis.
    for (const Row& r : rows2) {
        double equivalent = baseMs * (dwFlops / convFlops(s));
        std::printf("%-26s %10.4f %12.2f %9.2fx %12.2e   (equal-FLOP baseline %.4f ms)\n",
                    r.name, r.ms, r.gflops, equivalent / r.ms, r.relL2, equivalent);
    }
    std::printf("\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaFree(dBias));
    CUDA_CHECK(cudaFree(dOut));
    CUDA_CHECK(cudaFree(dWs));
    CUDA_CHECK(cudaFree(dDw));
    CUDA_CHECK(cudaFree(dPw));
    CUDA_CHECK(cudaFree(dDwBias));
    return 0;
}
