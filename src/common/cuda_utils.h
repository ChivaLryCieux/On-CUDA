#pragma once
// Common CUDA helpers.
//
// ASCII-ONLY FILE ON PURPOSE. On a Chinese Windows (codepage 936) cl.exe
// misreads UTF-8 non-ASCII bytes in .cu/.h sources and the byte misalignment
// breaks parsing of <<<>>> kernel launches. Keep every source file ASCII.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

inline void cudaCheckImpl(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n", file, line, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(call) cudaCheckImpl((call), __FILE__, __LINE__)

// GPU-side timer built on CUDA events. No CPU clock, no OS jitter.
class GpuTimer
{
public:
    GpuTimer()
    {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(start_, stream)); }

    // Elapsed milliseconds between start() and stop(); blocks until done.
    float stop(cudaStream_t stream = nullptr)
    {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

// Deterministic LCG fill in [-1, 1). Deterministic so every run is comparable.
inline void fillRandomHost(float* data, size_t count, unsigned seed = 12345u)
{
    unsigned state = seed;
    for (size_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        data[i] = static_cast<float>((state >> 8) & 0xFFFFFFu) / 8388608.0f * 2.0f - 1.0f;
    }
}
