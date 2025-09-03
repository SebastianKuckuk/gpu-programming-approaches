#pragma once

#include "../util.h"


inline void parseCLA_1d(int argc, char **argv, size_t &nx, size_t &nItWarmUp, size_t &nIt) {
    // default values
    nx = 10;

    nItWarmUp = 0;
    nIt = 1;

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


void printStats(const std::chrono::duration<double> elapsedSeconds, size_t nIt, size_t nCells) {
    std::cout << "  #cells / #it:  " << nCells << " / " << nIt << "\n";
    std::cout << "  elapsed time:  " << 1e3 * elapsedSeconds.count() << " ms\n";
    std::cout << "  per iteration: " << 1e3 * elapsedSeconds.count() / nIt << " ms\n";
}
