// Focused debug harness: run one matmul impl on a tiny problem and dump the
// exact pattern of wrong elements, so the bug can be located by inspection.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

#include "cuda_utils.h"
#include "reference.h"
#include "matmul.h"

int main(int argc, char** argv)
{
    int M = (argc > 1) ? std::atoi(argv[1]) : 64;
    int N = (argc > 2) ? std::atoi(argv[2]) : 64;
    int K = (argc > 3) ? std::atoi(argv[3]) : 64;
    const char* which = (argc > 4) ? argv[4] : "regtile";

    std::printf("debug matmul M=%d N=%d K=%d impl=%s\n", M, N, K, which);

    const size_t szA = static_cast<size_t>(M) * K;
    const size_t szB = static_cast<size_t>(K) * N;
    const size_t szC = static_cast<size_t>(M) * N;

    std::vector<float> hA(szA), hB(szB), hC(szC), hRef(szC);
    fillRandomHost(hA.data(), szA, 1u);
    fillRandomHost(hB.data(), szB, 2u);
    matmulCpu(hA.data(), hB.data(), hRef.data(), M, N, K);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, szA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, szB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, szC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), szA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), szB * sizeof(float), cudaMemcpyHostToDevice));

    if (std::string(which) == "regtile") {
        matmulRegTile(dA, dB, dC, M, N, K);
    } else {
        matmulNaive(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, szC * sizeof(float), cudaMemcpyDeviceToHost));

    // Report the worst element and a histogram of where errors live.
    double worst = 0.0;
    int worstI = -1, worstJ = -1;
    int wrongCount = 0;
    std::vector<int> wrongPerRow(M, 0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double d = std::fabs(hC[static_cast<size_t>(i) * N + j] - hRef[static_cast<size_t>(i) * N + j]);
            if (d > 1e-3) { ++wrongCount; ++wrongPerRow[i]; }
            if (d > worst) { worst = d; worstI = i; worstJ = j; }
        }
    }
    std::printf("wrong elements: %d / %zu   worst abs diff %.6f at (%d,%d)\n",
                wrongCount, szC, worst, worstI, worstJ);

    // Which rows have wrong elements? Print a compact map.
    std::printf("rows containing wrong elements (i: count): ");
    for (int i = 0; i < M; ++i)
        if (wrongPerRow[i] > 0) std::printf("%d:%d ", i, wrongPerRow[i]);
    std::printf("\n");

    // Dump a small window around the worst element.
    if (worstI >= 0) {
        int i0 = (worstI > 2) ? worstI - 2 : 0;
        int j0 = (worstJ > 2) ? worstJ - 2 : 0;
        for (int i = i0; i < i0 + 5 && i < M; ++i) {
            std::printf("row %2d: ", i);
            for (int j = j0; j < j0 + 8 && j < N; ++j)
                std::printf("(%d,%d) got=%9.4f ref=%9.4f  ", i, j,
                            hC[static_cast<size_t>(i) * N + j],
                            hRef[static_cast<size_t>(i) * N + j]);
            std::printf("\n");
        }
    }

    // Cross-check: is the error explained by a missing k-range? Compare
    // against a partial product that stops at k = K - BK.
    if (K >= 8) {
        std::vector<float> partial(szC);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float acc = 0.0f;
                for (int k = 0; k < K - 8; ++k)
                    acc += hA[static_cast<size_t>(i) * K + k] * hB[static_cast<size_t>(k) * N + j];
                partial[static_cast<size_t>(i) * N + j] = acc;
            }
        double dPart = 0.0, nPart = 0.0;
        for (size_t i = 0; i < szC; ++i) {
            double d = hC[i] - partial[i];
            dPart += d * d;
            nPart += partial[i] * partial[i];
        }
        std::printf("relL2(got, partial-sum-without-last-8-k) = %.6e\n",
                    std::sqrt(dPart / nPart));
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
