#include "increase-util.h"


inline void increase(double *data, size_t nx) {
#pragma acc parallel loop present(data[0 : nx], data[0 : nx])
    for (size_t i0 = 0; i0 < nx; ++i0) {
        data[i0] += 1;
    }
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    double *data;
    data = new double[nx];

    // init
    initIncrease(data, nx);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        increase(data, nx);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        increase(data, nx);
    }

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(double) + sizeof(double), 1);

    // check solution
    checkSolutionIncrease(data, nx, nIt + nItWarmUp);

    delete[] data;

    return 0;
}
