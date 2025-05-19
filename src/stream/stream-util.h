#pragma once

#include "../util.h"


inline void initStream(double *__restrict__ dest, double *__restrict__ src, size_t nx) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        src[i0] = i0;
        dest[i0] = 0;
    }
}

inline void checkSolutionStream(const double *const __restrict__ dest, const double *const __restrict__ src, size_t nx, size_t nIt) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        if ((i0 + nIt) != src[i0]) {
            std::cerr << "Stream check failed for element " << i0 << " (expected " << i0 + nIt << " but got " << src[i0] << ")" << std::endl;
            return;
        }
    }
}

inline void parseCLA_1d(int argc, char **argv, size_t &nx, size_t &nItWarmUp, size_t &nIt) {
    // default values
    nx = 67108864;

    nItWarmUp = 2;
    nIt = 10;

    // override with command line arguments
    int i = 1;
    if (argc > i)
        nx = atoi(argv[i]);
    ++i;

    if (argc > i)
        nItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i)
        nIt = atoi(argv[i]);
    ++i;
}
