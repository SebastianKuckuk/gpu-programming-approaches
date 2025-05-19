#include "stream-util.h"


inline void stream(const double *const __restrict__ src, double *__restrict__ dest, size_t nx) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        dest[i0] = src[i0] + 1;
    }
}


int main(int argc, char *argv[]) {
    char *doubleName;
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    double *dest;
    dest = new double[nx];
    double *src;
    src = new double[nx];

    // init
    initStream(dest, src, nx);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stream(src, dest, nx);
        std::swap(src, dest);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stream(src, dest, nx);
        std::swap(src, dest);
    }

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(double) + sizeof(double), 1);

    // check solution
    checkSolutionStream(dest, src, nx, nIt + nItWarmUp);

    delete[] dest;
    delete[] src;

    return 0;
}
