#include "conv.h"
#include "cuda_utils.h"
#include "matmul.h"   // convIm2col / dw-separable reuse the register-tiled GEMM

// ===========================================================================
// v0: naive direct convolution.
// One thread per output element. Every thread walks the whole (C, KH, KW)
// volume, and neighbouring threads re-read overlapping input patches, so the
// same input pixel is pulled from global memory OC*KH*KW times.
// ===========================================================================
__global__ void kConvNaive(const float* in, const float* w, const float* bias, float* out,
                           int C, int H, int W, int OC, int KH, int KW,
                           int OH, int OW, int pad, int stride)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z % OC;
    int n  = blockIdx.z / OC;
    if (oh >= OH || ow >= OW) return;

    const float* inN = in + static_cast<size_t>(n) * C * H * W;
    const float* wOC = w + static_cast<size_t>(oc) * C * KH * KW;
    float acc = bias ? bias[oc] : 0.0f;

    for (int c = 0; c < C; ++c) {
        const float* inC = inN + static_cast<size_t>(c) * H * W;
        const float* wC  = wOC + static_cast<size_t>(c) * KH * KW;
        for (int kh = 0; kh < KH; ++kh) {
            int ih = oh * stride - pad + kh;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < KW; ++kw) {
                int iw = ow * stride - pad + kw;
                if (iw < 0 || iw >= W) continue;
                acc += inC[static_cast<size_t>(ih) * W + iw] * wC[kh * KW + kw];
            }
        }
    }
    out[(static_cast<size_t>(n) * OC + oc) * OH * OW + static_cast<size_t>(oh) * OW + ow] = acc;
}

void convNaive(const float* in, const float* w, const float* bias, float* out,
               const ConvShape& s, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((s.OW + 15) / 16, (s.OH + 15) / 16, s.N * s.OC);
    kConvNaive<<<grid, block, 0, stream>>>(in, w, bias, out, s.C, s.H, s.W, s.OC,
                                           s.KH, s.KW, s.OH, s.OW, s.pad, s.stride);
}

// ===========================================================================
// v1: shared-memory tiling with halo.
// A tileH x tileW block of outputs needs a larger input patch that includes
// the halo rows/cols produced by padding. The whole patch for one channel is
// loaded once and reused by every (kh, kw) tap, which is exactly the reuse
// the naive version was throwing away.
//
// Note the missing early-returns: every thread must reach both __syncthreads()
// or the block deadlocks.
// ===========================================================================
__global__ void kConvSmem(const float* in, const float* w, const float* bias, float* out,
                          int C, int H, int W, int OC, int KH, int KW,
                          int OH, int OW, int pad, int stride, int tileH, int tileW)
{
    // Dynamic shared memory holds two regions:
    //   [0 .. smH*smW)          the input patch for the current channel
    //   [smH*smW .. +KH*KW)     the filter taps for the current channel
    // Staging the taps matters: without it every thread issues C*KH*KW global
    // loads for the filter, which dominates the kernel.
    extern __shared__ float smem[];
    const int smH = (tileH - 1) * stride + KH;
    const int smW = (tileW - 1) * stride + KW;
    float* tile = smem;
    float* fw = smem + smH * smW;

    const int oc  = blockIdx.z % OC;
    const int n   = blockIdx.z / OC;
    const int oh0 = blockIdx.y * tileH;
    const int ow0 = blockIdx.x * tileW;
    const int oh  = oh0 + threadIdx.y;
    const int ow  = ow0 + threadIdx.x;
    const bool inside = (oh < OH && ow < OW);

    const float* inN = in + static_cast<size_t>(n) * C * H * W;
    const float* wOC = w + static_cast<size_t>(oc) * C * KH * KW;
    float acc = bias ? bias[oc] : 0.0f;

    for (int c = 0; c < C; ++c) {
        const float* inC = inN + static_cast<size_t>(c) * H * W;

        // Cooperative load of the input patch, zero-padded at the borders.
        for (int i = threadIdx.y; i < smH; i += blockDim.y) {
            int ih = oh0 * stride - pad + i;
            for (int j = threadIdx.x; j < smW; j += blockDim.x) {
                int iw = ow0 * stride - pad + j;
                float v = 0.0f;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    v = inC[static_cast<size_t>(ih) * W + iw];
                tile[i * smW + j] = v;
            }
        }
        // Cooperative load of this channel's KH*KW filter taps.
        for (int i = threadIdx.x; i < KH * KW; i += blockDim.x)
            fw[i] = wOC[static_cast<size_t>(c) * KH * KW + i];
        __syncthreads();

        if (inside) {
            for (int kh = 0; kh < KH; ++kh)
                for (int kw = 0; kw < KW; ++kw)
                    acc += tile[(threadIdx.y * stride + kh) * smW + (threadIdx.x * stride + kw)]
                           * fw[kh * KW + kw];
        }
        __syncthreads();   // guard the tiles before the next channel overwrites them
    }

    if (inside)
        out[(static_cast<size_t>(n) * OC + oc) * OH * OW + static_cast<size_t>(oh) * OW + ow] = acc;
}

void convSmem(const float* in, const float* w, const float* bias, float* out,
              const ConvShape& s, cudaStream_t stream)
{
    const int tileH = 16;
    const int tileW = 16;
    const int smH = (tileH - 1) * s.stride + s.KH;
    const int smW = (tileW - 1) * s.stride + s.KW;
    const size_t smem = (static_cast<size_t>(smH) * smW + s.KH * s.KW) * sizeof(float);

    // Only opt in to >48KB dynamic shared memory if this shape actually needs it.
    static int smemOptIn = 0;
    if (smem > static_cast<size_t>(smemOptIn)) {
        CUDA_CHECK(cudaFuncSetAttribute(kConvSmem, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem)));
        smemOptIn = static_cast<int>(smem);
    }

    dim3 block(tileW, tileH);
    dim3 grid((s.OW + tileW - 1) / tileW, (s.OH + tileH - 1) / tileH, s.N * s.OC);
    kConvSmem<<<grid, block, smem, stream>>>(in, w, bias, out, s.C, s.H, s.W, s.OC,
                                             s.KH, s.KW, s.OH, s.OW, s.pad, s.stride,
                                             tileH, tileW);
}

// ===========================================================================
// v2: im2col + GEMM.
// im2col expands each (C, KH, KW) receptive field into one column of a
// (C*KH*KW) x (OH*OW) matrix, which turns the convolution into a single GEMM.
// That lets the whole matmul optimization ladder be reused directly, and it is
// what cuDNN actually does for most convolution shapes.
//
// Trade-off: the im2col matrix duplicates every border pixel, so it uses more
// memory bandwidth than a fused kernel. The GEMM win usually dwarfs that.
// ===========================================================================
__global__ void kIm2col(const float* in, float* col, int C, int H, int W, int KH, int KW,
                        int OH, int OW, int pad, int stride)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;   // output pixel index
    int k = blockIdx.y * blockDim.y + threadIdx.y;   // index inside the patch
    if (p >= OH * OW || k >= C * KH * KW) return;

    int n  = blockIdx.z;
    int c  = k / (KH * KW);
    int rem = k % (KH * KW);
    int kh = rem / KW;
    int kw = rem % KW;
    int oh = p / OW;
    int ow = p % OW;

    int ih = oh * stride - pad + kh;
    int iw = ow * stride - pad + kw;
    float v = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
        v = in[(static_cast<size_t>(n) * C + c) * H * W + static_cast<size_t>(ih) * W + iw];

    col[(static_cast<size_t>(n) * C * KH * KW + k) * (OH * OW) + p] = v;
}

__global__ void kAddBias(float* out, const float* bias, long long total, int OC, int P)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int oc = static_cast<int>((idx / P) % OC);
    out[idx] += bias[oc];
}

void convIm2col(const float* in, const float* w, const float* bias, float* out,
                const ConvShape& s, float* workspace, cudaStream_t stream)
{
    const int CK = s.C * s.KH * s.KW;
    const int P  = s.OH * s.OW;

    dim3 block(16, 16);
    dim3 grid((P + 15) / 16, (CK + 15) / 16, s.N);
    kIm2col<<<grid, block, 0, stream>>>(in, workspace, s.C, s.H, s.W, s.KH, s.KW,
                                        s.OH, s.OW, s.pad, s.stride);

    // out[oc][p] = sum_k w[oc][k] * col[k][p]  ->  GEMM with M=OC, N=P, K=CK.
    for (int n = 0; n < s.N; ++n)
        matmulRegTile(w, workspace + static_cast<size_t>(n) * CK * P,
                      out + static_cast<size_t>(n) * s.OC * P, s.OC, P, CK, stream);

    if (bias) {
        long long total = static_cast<long long>(s.N) * s.OC * P;
        const int threads = 256;
        long long blocks = (total + threads - 1) / threads;
        kAddBias<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(out, bias, total, s.OC, P);
    }
}

// ===========================================================================
// v3: depthwise-separable convolution.
// The MobileNet insight: a standard KxK conv over C channels costs OC*C*K*K
// per pixel, but you can split it into
//   depthwise: each channel filtered on its own      -> C*K*K per pixel
//   pointwise: 1x1 conv mixing channels              -> OC*C per pixel
// For K=3 that is 9x fewer FLOPs for C=64, OC=64. Quality drops a little,
// but the arithmetic saving is enormous - this is why every mobile network
// uses it. The pointwise stage is a GEMM, so it reuses matmulRegTile again.
// ===========================================================================
__global__ void kConvDepthwise(const float* in, const float* w, float* out,
                               int C, int H, int W, int KH, int KW, int OH, int OW,
                               int pad, int stride)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c  = blockIdx.z % C;
    int n  = blockIdx.z / C;
    if (oh >= OH || ow >= OW) return;

    const float* inC = in + (static_cast<size_t>(n) * C + c) * H * W;
    const float* wC  = w + static_cast<size_t>(c) * KH * KW;
    float acc = 0.0f;

    for (int kh = 0; kh < KH; ++kh) {
        int ih = oh * stride - pad + kh;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < KW; ++kw) {
            int iw = ow * stride - pad + kw;
            if (iw < 0 || iw >= W) continue;
            acc += inC[static_cast<size_t>(ih) * W + iw] * wC[kh * KW + kw];
        }
    }
    out[(static_cast<size_t>(n) * C + c) * OH * OW + static_cast<size_t>(oh) * OW + ow] = acc;
}

void convDepthwiseSeparable(const float* in, const float* dwW, const float* pwW,
                            const float* bias, float* out, const ConvShape& s,
                            float* workspace, cudaStream_t stream)
{
    const int P = s.OH * s.OW;

    dim3 block(16, 16);
    dim3 grid((s.OW + 15) / 16, (s.OH + 15) / 16, s.N * s.C);
    kConvDepthwise<<<grid, block, 0, stream>>>(in, dwW, workspace, s.C, s.H, s.W,
                                               s.KH, s.KW, s.OH, s.OW, s.pad, s.stride);

    for (int n = 0; n < s.N; ++n)
        matmulRegTile(pwW, workspace + static_cast<size_t>(n) * s.C * P,
                      out + static_cast<size_t>(n) * s.OC * P, s.OC, P, s.C, stream);

    if (bias) {
        long long total = static_cast<long long>(s.N) * s.OC * P;
        const int threads = 256;
        long long blocks = (total + threads - 1) / threads;
        kAddBias<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(out, bias, total, s.OC, P);
    }
}
