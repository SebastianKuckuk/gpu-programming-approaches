#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>


#include "print-numbers-util.h"


inline void printNumbers(size_t nx) {
    thrust::for_each(thrust::device, thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(nx),
                     [=] __device__ (const auto &i0) {
                        printf("%ld\n", i0);
                     });
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
