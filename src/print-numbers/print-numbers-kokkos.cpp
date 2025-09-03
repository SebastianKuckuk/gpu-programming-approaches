#include <Kokkos_Core.hpp>

#include "print-numbers-util.h"


inline void printNumbers(size_t nx) {
    Kokkos::parallel_for(                //
        Kokkos::RangePolicy<>(0, nx),    //
        KOKKOS_LAMBDA(const size_t i0) { //
            printf("%ld\n", i0);
        });
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    int c = 1;
    Kokkos::initialize(c, argv);
    {
        // warm-up
        for (size_t i = 0; i < nItWarmUp; ++i) {
            printNumbers(nx);
        }
        Kokkos::fence();

        // measurement
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nIt; ++i) {
            printNumbers(nx);
        }
        Kokkos::fence();

        auto end = std::chrono::steady_clock::now();

        printStats(end - start, nIt, nx);
    }
    Kokkos::finalize();

    return 0;
}
