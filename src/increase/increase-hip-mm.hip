#include "increase-util.h"

#include "../../hip-util.h"


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
    checkHipError(hipMallocManaged((void **)&data, sizeof(double) * nx));

    // init
    initIncrease(data, nx);

    checkHipError(hipMemPrefetchAsync(data, sizeof(double) * nx, 0));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        increase<<<ceilingDivide(nx, 256), 256>>>(data, nx);
    }
    checkHipError(hipDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        increase<<<ceilingDivide(nx, 256), 256>>>(data, nx);
    }
    checkHipError(hipDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(double) + sizeof(double), 1);

    checkHipError(hipMemPrefetchAsync(data, sizeof(double) * nx, hipCpuDeviceId));

    // check solution
    checkSolutionIncrease(data, nx, nIt + nItWarmUp);

    checkHipError(hipFree(data));

    return 0;
}
