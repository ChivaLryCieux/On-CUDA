// Minimal CUDA smoke test: verify toolchain + query GPU properties
// NOTE: source files are ASCII-only on purpose. On a Chinese Windows
// (codepage 936) cl.exe misreads UTF-8 Chinese comments and the byte
// misalignment breaks <<<>>> kernel launch parsing.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel(int* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) out[0] = 42;   // only thread 0 writes
}

int main()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    printf("CUDA device count: %d\n", device_count);

    for (int d = 0; d < device_count; ++d) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);
        printf("  [%d] %s\n", d, p.name);
        printf("      compute capability : %d.%d\n", p.major, p.minor);
        printf("      SMs               : %d\n", p.multiProcessorCount);
        printf("      global memory     : %.1f GB\n", p.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("      shared mem/block  : %zu KB\n", p.sharedMemPerBlock / 1024);
        printf("      max threads/block : %d\n", p.maxThreadsPerBlock);
    }

    int* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(int));
    hello_kernel<<<1, 256>>>(d_out);
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) printf("launch error: %s\n", cudaGetErrorString(launch_err));
    cudaDeviceSynchronize();

    int h_out = -1;
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("kernel result = %d\n", h_out);

    cudaFree(d_out);
    printf("sync error: %s\n", cudaGetErrorString(cudaGetLastError()));
    return h_out == 42 ? 0 : 1;
}
