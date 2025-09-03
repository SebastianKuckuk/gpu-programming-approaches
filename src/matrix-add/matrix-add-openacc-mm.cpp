#include "matrix-add-util.h"


inline void matrixAdd(const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ c, size_t nx, size_t ny) {
#pragma acc parallel loop present(a [0:nx * ny], b [0:nx * ny], c [0:nx * ny]) collapse(2)
    for (size_t i1 = 0; i1 < ny; ++i1) {
        for (size_t i0 = 0; i0 < nx; ++i0) {
            c[i0 + i1 * nx] = a[i0 + i1 * nx] + b[i0 + i1 * nx];
        }
    }
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    double *a;
    a = new double[nx * ny];
    double *b;
    b = new double[nx * ny];
    double *c;
    c = new double[nx * ny];

    // init
    initMatrixAdd(a, b, c, nx, ny);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        matrixAdd(a, b, c, nx, ny);
        std::swap(c, a);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        matrixAdd(a, b, c, nx, ny);
        std::swap(c, a);
    }

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx * ny, sizeof(double) + sizeof(double) + sizeof(double), 1);

    // check solution
    checkSolutionMatrixAdd(a, b, c, nx, ny, nIt + nItWarmUp);

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
