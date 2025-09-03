#pragma once

#include "../util.h"


inline void initMatrixAdd(double *__restrict__ a, double *__restrict__ b, double *__restrict__ c, size_t nx, size_t ny) {
    for (size_t i1 = 0; i1 < ny; ++i1) {
        for (size_t i0 = 0; i0 < nx; ++i0) {
            a[i0 + i1 * nx] = (double)1;
            b[i0 + i1 * nx] = (double)2;
            c[i0 + i1 * nx] = (double)0;
        }
    }
}

inline void checkSolutionMatrixAdd(const double *__restrict__ a, const double *__restrict__ b, const double *__restrict__ c, size_t nx, size_t ny, size_t nIt) {
    for (size_t i1 = 0; i1 < ny; ++i1) {
        for (size_t i0 = 0; i0 < nx; ++i0) {
            if ((double)(2 * nIt + 1) != a[i0 + i1 * nx]) {
                std::cerr << "MatrixAdd check failed for element " << i0 << ", " << i1 << " (expected " << 2 * nIt + 1 << " but got " << a[i0 + i1 * nx] << ")" << std::endl;
                return;
            }
        }
    }
}

inline void parseCLA_2D(int argc, char **argv, size_t &nx, size_t &ny, size_t &nItWarmUp, size_t &nIt) {
    // default values
    nx = 4096;
    ny = 4096;

    nItWarmUp = 2;
    nIt = 10;

    // override with command line arguments
    int i = 1;
    if (argc > i)
        nx = atoi(argv[i]);
    ++i;
    if (argc > i)
        ny = atoi(argv[i]);
    ++i;

    if (argc > i)
        nItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i)
        nIt = atoi(argv[i]);
    ++i;
}
