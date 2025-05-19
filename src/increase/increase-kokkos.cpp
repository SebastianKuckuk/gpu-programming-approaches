#include <Kokkos_Core.hpp>

#include "increase-util.h"


inline void increase(Kokkos::View<double *> &data, size_t nx) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(0, nx),
        KOKKOS_LAMBDA(const size_t i0) {
            data(i0) += 1;
        });
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    int c = 1;
    Kokkos::initialize(c, argv);
    {
        Kokkos::View<double *> data("data", nx);

        auto h_data = Kokkos::create_mirror_view(data);

        // init
        initIncrease(h_data.data(), nx);

        Kokkos::deep_copy(data, h_data);

        // warm-up
        for (size_t i = 0; i < nItWarmUp; ++i) {
            increase(data, nx);
        }
        Kokkos::fence();

        // measurement
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nIt; ++i) {
            increase(data, nx);
        }
        Kokkos::fence();

        auto end = std::chrono::steady_clock::now();

        printStats(end - start, nIt, nx, sizeof(double) + sizeof(double), 1);

        Kokkos::deep_copy(h_data, data);

        // check solution
        checkSolutionIncrease(h_data.data(), nx, nIt + nItWarmUp);
    }
    Kokkos::finalize();

    return 0;
}
