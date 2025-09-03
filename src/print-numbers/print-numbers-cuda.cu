#include "print-numbers-util.h"

#include "../cuda-util.h"


__global__ void printNumbers(size_t nx) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx) {
        printf("%ld\n", i0);
    }
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        printNumbers<<<ceilingDivide(nx, 256), 256>>>(nx);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        printNumbers<<<ceilingDivide(nx, 256), 256>>>(nx);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx);

    return 0;
}
