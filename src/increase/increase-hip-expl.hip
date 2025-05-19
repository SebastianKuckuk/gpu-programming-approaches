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
    checkHipError(hipHostMalloc((void **)&data, sizeof(double) * nx));

    double *d_data;
    checkHipError(hipMalloc((void **)&d_data, sizeof(double) * nx));

    // init
    initIncrease(data, nx);

    checkHipError(hipMemcpy(d_data, data, sizeof(double) * nx, hipMemcpyHostToDevice));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        increase<<<ceilingDivide(nx, 256), 256>>>(d_data, nx);
    }
    checkHipError(hipDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        increase<<<ceilingDivide(nx, 256), 256>>>(d_data, nx);
    }
    checkHipError(hipDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(double) + sizeof(double), 1);

    checkHipError(hipMemcpy(data, d_data, sizeof(double) * nx, hipMemcpyDeviceToHost));

    // check solution
    checkSolutionIncrease(data, nx, nIt + nItWarmUp);

    checkHipError(hipFree(d_data));

    checkHipError(hipHostFree(data));

    return 0;
}
