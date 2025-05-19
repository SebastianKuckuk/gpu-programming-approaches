#include "increase-util.h"

#include "../cuda-util.h"


__global__ void increase(double *data, size_t nx) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx) {
        data[i0] += 1;
    }
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    double *data;
    checkCudaError(cudaMallocHost((void **)&data, sizeof(double) * nx));

    double *d_data;
    checkCudaError(cudaMalloc((void **)&d_data, sizeof(double) * nx));

    // init
    initIncrease(data, nx);

    checkCudaError(cudaMemcpy(d_data, data, sizeof(double) * nx, cudaMemcpyHostToDevice));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        increase<<<ceilingDivide(nx, 256), 256>>>(d_data, nx);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        increase<<<ceilingDivide(nx, 256), 256>>>(d_data, nx);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(double) + sizeof(double), 1);

    checkCudaError(cudaMemcpy(data, d_data, sizeof(double) * nx, cudaMemcpyDeviceToHost));

    // check solution
    checkSolutionIncrease(data, nx, nIt + nItWarmUp);

    checkCudaError(cudaFree(d_data));

    checkCudaError(cudaFreeHost(data));

    return 0;
}
