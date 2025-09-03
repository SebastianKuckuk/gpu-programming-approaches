#include "matrix-add-util.h"

#include "../cuda-util.h"


__global__ void matrixAdd(const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ c, size_t nx, size_t ny) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 < nx && i1 < ny) {
        c[i0 + i1 * nx] = a[i0 + i1 * nx] + b[i0 + i1 * nx];
    }
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    double *a;
    checkCudaError(cudaMallocHost((void **)&a, sizeof(double) * nx * ny));
    double *b;
    checkCudaError(cudaMallocHost((void **)&b, sizeof(double) * nx * ny));
    double *c;
    checkCudaError(cudaMallocHost((void **)&c, sizeof(double) * nx * ny));

    double *d_a;
    checkCudaError(cudaMalloc((void **)&d_a, sizeof(double) * nx * ny));
    double *d_b;
    checkCudaError(cudaMalloc((void **)&d_b, sizeof(double) * nx * ny));
    double *d_c;
    checkCudaError(cudaMalloc((void **)&d_c, sizeof(double) * nx * ny));

    // init
    initMatrixAdd(a, b, c, nx, ny);

    checkCudaError(cudaMemcpy(d_a, a, sizeof(double) * nx * ny, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_b, b, sizeof(double) * nx * ny, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_c, c, sizeof(double) * nx * ny, cudaMemcpyHostToDevice));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        matrixAdd<<<dim3(ceilingDivide(nx, 16), ceilingDivide(ny, 16)), dim3(16, 16)>>>(d_a, d_b, d_c, nx, ny);
        std::swap(d_c, d_a);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        matrixAdd<<<dim3(ceilingDivide(nx, 16), ceilingDivide(ny, 16)), dim3(16, 16)>>>(d_a, d_b, d_c, nx, ny);
        std::swap(d_c, d_a);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx * ny, sizeof(double) + sizeof(double) + sizeof(double), 1);

    checkCudaError(cudaMemcpy(a, d_a, sizeof(double) * nx * ny, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(b, d_b, sizeof(double) * nx * ny, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(c, d_c, sizeof(double) * nx * ny, cudaMemcpyDeviceToHost));

    // check solution
    checkSolutionMatrixAdd(a, b, c, nx, ny, nIt + nItWarmUp);

    checkCudaError(cudaFree(d_a));
    checkCudaError(cudaFree(d_b));
    checkCudaError(cudaFree(d_c));

    checkCudaError(cudaFreeHost(a));
    checkCudaError(cudaFreeHost(b));
    checkCudaError(cudaFreeHost(c));

    return 0;
}
