// Benchmark driver for the matmul optimization ladder.
//
// Usage: bench_matmul [M] [N] [K]
//
// Every implementation is checked against a CPU reference before its timing
// is printed, so you can never fool yourself with a fast wrong answer.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "cuda_utils.h"
#include "reference.h"
#include "matmul.h"

// cuBLAS returns cublasStatus_t, not cudaError_t, so it needs its own check.
static void cublasCheck(cublasStatus_t st, const char* file, int line)
{
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[cuBLAS ERROR] %s:%d  status=%d\n", file, line, static_cast<int>(st));
        std::exit(EXIT_FAILURE);
    }
}
#define CUBLAS_CHECK(call) cublasCheck((call), __FILE__, __LINE__)

static const int kIters = 20;
static const int kWarmup = 5;

struct Row
{
    const char* name;
    float ms;
    double gflops;
    double relL2;
};

int main(int argc, char** argv)
{
    int M = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int N = (argc > 2) ? std::atoi(argv[2]) : 1024;
    int K = (argc > 3) ? std::atoi(argv[3]) : 1024;

    std::printf("=== matmul: M=%d N=%d K=%d  (row-major, C=A*B, FP32) ===\n", M, N, K);

    const double flops = matmulFlops(M, N, K);
    const size_t szA = static_cast<size_t>(M) * K;
    const size_t szB = static_cast<size_t>(K) * N;
    const size_t szC = static_cast<size_t>(M) * N;

    std::vector<float> hA(szA), hB(szB), hC(szC), hRef(szC);
    fillRandomHost(hA.data(), szA, 1u);
    fillRandomHost(hB.data(), szB, 2u);
    matmulCpu(hA.data(), hB.data(), hRef.data(), M, N, K);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, szA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, szB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, szC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), szA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), szB * sizeof(float), cudaMemcpyHostToDevice));

    // FP16 copies for the WMMA path.
    std::vector<__half> hAh(szA), hBh(szB);
    for (size_t i = 0; i < szA; ++i) hAh[i] = __float2half(hA[i]);
    for (size_t i = 0; i < szB; ++i) hBh[i] = __float2half(hB[i]);
    __half *dAh = nullptr, *dBh = nullptr;
    CUDA_CHECK(cudaMalloc(&dAh, szA * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dBh, szB * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dAh, hAh.data(), szA * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBh, hBh.data(), szB * sizeof(__half), cudaMemcpyHostToDevice));

    cublasHandle_t blas = nullptr;
    CUBLAS_CHECK(cublasCreate(&blas));
    const float alpha = 1.0f;
    const float beta = 0.0f;

    std::vector<Row> rows;
    GpuTimer timer;

    // Times launch() over kIters back-to-back kernels (throughput, not latency),
    // then verifies the result left in dC.
    auto bench = [&](const char* name, auto&& launch) {
        for (int i = 0; i < kWarmup; ++i) launch();
        // A failed launch (bad config, too many resources) leaves the previous
        // result in dC and would silently look "fast and correct" - catch it.
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.start();
        for (int i = 0; i < kIters; ++i) launch();
        float ms = timer.stop() / static_cast<float>(kIters);

        CUDA_CHECK(cudaMemcpy(hC.data(), dC, szC * sizeof(float), cudaMemcpyDeviceToHost));
        DiffStats d = compareArrays(hC.data(), hRef.data(), szC);
        rows.push_back(Row{ name, ms, flops / (static_cast<double>(ms) * 1e-3) / 1e9, d.relL2 });
    };

    bench("v0 naive          ", [&] { matmulNaive(dA, dB, dC, M, N, K); });
    bench("v1 tiled 16x16    ", [&] { matmulTiled<16>(dA, dB, dC, M, N, K); });
    bench("v1 tiled 32x32    ", [&] { matmulTiled<32>(dA, dB, dC, M, N, K); });
    bench("v2 reg-tile 64x64 ", [&] { matmulRegTile(dA, dB, dC, M, N, K); });
    bench("v3 wmma fp16      ", [&] { matmulWmma(dAh, dBh, dC, M, N, K); });

    // cuBLAS as the "what a tuned library achieves" reference point.
    // Row-major C = A*B is evaluated as the column-major product C^T = B^T * A^T.
    bench("cuBLAS (library) ", [&] {
        CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                &alpha, dB, N, dA, K, &beta, dC, N));
    });

    std::printf("%-20s %10s %12s %10s %12s\n", "implementation", "ms", "GFLOP/s", "speedup", "relL2 err");
    const float base = rows.front().ms;
    for (const Row& r : rows) {
        std::printf("%-20s %10.4f %12.1f %9.1fx %12.2e\n",
                    r.name, r.ms, r.gflops, base / r.ms, r.relL2);
    }

    // RTX 2080 SUPER: 48 SMs x 64 FP32 lanes x 2 x ~1.81 GHz ~= 11.2 TFLOP/s FP32.
    std::printf("\nreference points: FP32 peak ~11.2 TFLOP/s, HBM ~496 GB/s\n");
    std::printf("note: v3 relL2 is larger because inputs are FP16 (expect ~1e-3)\n");

    CUBLAS_CHECK(cublasDestroy(blas));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dAh));
    CUDA_CHECK(cudaFree(dBh));
    return 0;
}
