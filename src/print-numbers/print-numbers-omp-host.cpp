#include "print-numbers-util.h"


inline void printNumbers(size_t nx) {
#pragma omp parallel for schedule(static)
    for (size_t i0 = 0; i0 < nx; ++i0) {
        printf("%ld\n", i0);
    }
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        printNumbers(nx);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        printNumbers(nx);
    }

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx);

    return 0;
}
