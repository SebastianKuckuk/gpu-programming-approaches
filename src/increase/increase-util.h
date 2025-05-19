#pragma once

#include "../util.h"


inline void initIncrease(double *data, size_t nx) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        data[i0] = (double)i0;
    }
}

inline void checkSolutionIncrease(double *data, size_t nx, size_t nIt) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        if ((double)(i0 + nIt) != data[i0]) {
            std::cerr << "Increase check failed for element " << i0 << " (expected " << i0 + nIt << " but got " << data[i0] << ")" << std::endl;
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

inline void printStats(const std::chrono::duration<double> elapsedSeconds, size_t nIt, size_t nCells, size_t numBytesPerCell, size_t numFlopsPerCell) {
    std::cout << "  #cells / #it:  " << nCells << " / " << nIt << "\n";
    std::cout << "  elapsed time:  " << 1e3 * elapsedSeconds.count() << " ms\n";
    std::cout << "  per iteration: " << 1e3 * elapsedSeconds.count() / nIt << " ms\n";
    std::cout << "  MLUP/s:        " << 1e-6 * nCells * nIt / elapsedSeconds.count() << "\n";
    std::cout << "  bandwidth:     " << 1e-9 * numBytesPerCell * nCells * nIt / elapsedSeconds.count() << " GB/s\n";
    std::cout << "  compute:       " << 1e-9 * numFlopsPerCell * nCells * nIt / elapsedSeconds.count() << " GFLOP/s\n";
}
