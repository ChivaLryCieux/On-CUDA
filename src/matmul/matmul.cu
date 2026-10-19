#include "matmul.h"
#include "cuda_utils.h"

#include <mma.h>

using namespace nvcuda;

// ===========================================================================
// v0: naive.
// One thread per output element. Correct, and the slowest thing you can write.
// Problem: no reuse at all. The kernel issues M*N*K loads for A and M*N*K for
// B, so global memory traffic scales with the FLOP count. GEMM is arithmetic
// intensity ~N, so this design is permanently memory-bound.
// ===========================================================================
__global__ void kMatmulNaive(const float* A, const float* B, float* C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
        sum += A[static_cast<size_t>(row) * K + k] * B[static_cast<size_t>(k) * N + col];
    C[static_cast<size_t>(row) * N + col] = sum;
}

void matmulNaive(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    kMatmulNaive<<<grid, block, 0, s>>>(A, B, C, M, N, K);
}

// ===========================================================================
// v1: shared-memory tiling.
// A TILE x TILE block of C is computed by TILE*TILE threads. Per k-step the
// block loads one A tile and one B tile into shared memory, so each global
// element is reused TILE times instead of once.
//
// Bank-conflict note: with this layout a warp only ever touches two rows of As
// and one row of Bs, so As reads are broadcast and Bs reads are broadcast too.
// No padding needed here. Change the access pattern (e.g. transpose As) and
// you will immediately need the classic As[TILE][TILE+1] padding trick.
// ===========================================================================
template <int TILE>
__global__ void kMatmulTiled(const float* A, const float* B, float* C, int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        // Coalesced: consecutive threads read consecutive global addresses.
        As[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[static_cast<size_t>(row) * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[static_cast<size_t>(bRow) * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        // Needed: another block must not overwrite the tiles while we still read them.
        __syncthreads();
    }

    if (row < M && col < N) C[static_cast<size_t>(row) * N + col] = sum;
}

template <int TILE>
void matmulTiled(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    kMatmulTiled<TILE><<<grid, block, 0, s>>>(A, B, C, M, N, K);
}

template void matmulTiled<16>(const float*, const float*, float*, int, int, int, cudaStream_t);
template void matmulTiled<32>(const float*, const float*, float*, int, int, int, cudaStream_t);

// ===========================================================================
// v2: register tiling (the big win).
// Block computes BM x BN = 64 x 64 with 256 threads; each thread owns a
// TM x TN = 4 x 4 micro-tile held entirely in registers. Per k-step the block
// loads BK x BM + BK x BN elements, and each one is used TM + TN times from
// registers instead of shared memory. Arithmetic intensity per thread jumps
// from 1 to TM + TN.
//
// Minor known inefficiency: the Bs reads hit each bank twice (threadCol only
// spans multiples of 4 with period 32). Fixing it needs a thread->data remap;
// left as an exercise.
// ===========================================================================
#define BM 64
#define BN 64
#define BK 8
#define TM 4
#define TN 4

__global__ void kMatmulRegTile(const float* A, const float* B, float* C, int M, int N, int K)
{
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;   // 0..255
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int threadRow = (tid / (BN / TN)) * TM;   // 0,4,8,...,60
    const int threadCol = (tid % (BN / TN)) * TN;   // 0,4,8,...,60

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Cooperative load: 2*(BK*BM)/256 = 4 global loads per thread per k-step.
        // Consecutive threads take consecutive indices -> coalesced.
        #pragma unroll
        for (int i = 0; i < (2 * BK * BM) / 256; ++i) {
            int idx = tid + i * 256;
            if (idx < BK * BM) {
                // As is [k][m]. The flat index is taken m-major (idx = m*BK + k)
                // so that consecutive threads read consecutive k in global A,
                // which is the coalesced direction for a row-major M x K matrix.
                int m = idx / BK;   // M index
                int k = idx % BK;   // K index
                int aRow = blockRow + m;
                int aCol = k0 + k;
                As[k][m] = (aRow < M && aCol < K) ? A[static_cast<size_t>(aRow) * K + aCol] : 0.0f;
            } else {
                // Bs is [k][n]; idx = k*BN + n, so consecutive threads read
                // consecutive n in global B - also coalesced.
                int j = idx - BK * BM;
                int k = j / BN;
                int n = j % BN;
                int bRow = k0 + k;
                int bCol = blockCol + n;
                Bs[k][n] = (bRow < K && bCol < N) ? B[static_cast<size_t>(bRow) * N + bCol] : 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM];
            float b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[k][threadRow + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][threadCol + j];

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gr = blockRow + threadRow + i;
        if (gr >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gc = blockCol + threadCol + j;
            if (gc < N) C[static_cast<size_t>(gr) * N + gc] = acc[i][j];
        }
    }
}

void matmulRegTile(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s)
{
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    kMatmulRegTile<<<grid, 256, 0, s>>>(A, B, C, M, N, K);
}

// ===========================================================================
// v3: WMMA on Tensor Cores (FP16 in, FP32 accumulate).
// Turing tensor cores do FP16 only - no TF32 and no BF16 on SM 7.5, so this
// is the only "tensor core" option available on this GPU.
//
// One block computes WM x WN = 64 x 64 using 512 threads = 16 warps, each
// warp owning a 16 x 16 output tile. Per k-step the block stages a
// WK x WM A tile and a WK x WN B tile in shared memory, then every warp does
// load_matrix_sync / mma_sync.
// ===========================================================================
#define WM 64
#define WN 64
#define WSK 64   // shared-memory k extent per stage; must be a multiple of 16

__global__ void kMatmulWmma(const __half* A, const __half* B, float* C, int M, int N, int K)
{
    // As is [m][k] row-major over the M x k tile: that is exactly the layout
    // wmma::load_matrix_sync expects for a row_major matrix_a, with ldm = WSK.
    // Bs is [k][n] row-major over the k x N tile, matching row_major matrix_b.
    // Staging WSK = 64 k-values at once also makes both global loads fully
    // coalesced (128 bytes per row) instead of 32-byte fragments.
    __shared__ __half As[WM][WSK];
    __shared__ __half Bs[WSK][WN];
    __shared__ float Cs[WM][WN];

    const int tid = threadIdx.x;   // 0..511
    const int warpId = tid / 32;
    const int warpRow = warpId / (WN / 16);
    const int warpCol = warpId % (WN / 16);
    const int blockRow = blockIdx.y * WM;
    const int blockCol = blockIdx.x * WN;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);

    const int aTile = WM * WSK;         // 4096 halves
    const int total = 2 * aTile;        // A tile + B tile

    for (int k0 = 0; k0 < K; k0 += WSK) {
        // (2*WM*WSK)/512 = 16 half loads per thread per stage.
        // Deliberately NOT unrolled: 16x unrolling pushes this kernel past the
        // 128 registers/thread that 512 threads can fit in an SM, and the
        // launch then fails with "too many resources requested for launch".
        for (int i = 0; i < total / 512; ++i) {
            int idx = tid + i * 512;
            if (idx < aTile) {
                int m = idx / WSK;      // M index
                int k = idx % WSK;      // K index
                int aRow = blockRow + m;
                int aCol = k0 + k;
                As[m][k] = (aRow < M && aCol < K)
                               ? A[static_cast<size_t>(aRow) * K + aCol]
                               : __float2half(0.0f);
            } else {
                int j = idx - aTile;
                int k = j / WN;         // K index
                int n = j % WN;         // N index
                int bRow = k0 + k;
                int bCol = blockCol + n;
                Bs[k][n] = (bRow < K && bCol < N)
                               ? B[static_cast<size_t>(bRow) * N + bCol]
                               : __float2half(0.0f);
            }
        }
        __syncthreads();

        // Four 16x16x16 tensor-core ops per staged tile.
        #pragma unroll
        for (int kk = 0; kk < WSK / 16; ++kk) {
            wmma::load_matrix_sync(fa, &As[warpRow * 16][kk * 16], WSK);
            wmma::load_matrix_sync(fb, &Bs[kk * 16][warpCol * 16], WN);
            wmma::mma_sync(fc, fa, fb, fc);
        }
        __syncthreads();
    }

    // Fragments cannot be written straight to global memory; stage in shared.
    wmma::store_matrix_sync(&Cs[warpRow * 16][warpCol * 16], fc, WN, wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < WM * WN; i += 512) {
        int m = i / WN;
        int n = i % WN;
        int gr = blockRow + m;
        int gc = blockCol + n;
        if (gr < M && gc < N) C[static_cast<size_t>(gr) * N + gc] = Cs[m][n];
    }
}

void matmulWmma(const __half* A, const __half* B, float* C, int M, int N, int K, cudaStream_t s)
{
    dim3 grid((N + WN - 1) / WN, (M + WM - 1) / WM);
    kMatmulWmma<<<grid, 512, 0, s>>>(A, B, C, M, N, K);
}
