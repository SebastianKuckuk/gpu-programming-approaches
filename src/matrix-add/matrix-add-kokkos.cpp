#include <Kokkos_Core.hpp>

#include "matrix-add-util.h"


inline void matrixAdd(const Kokkos::View<double **> &a, const Kokkos::View<double **> &b, Kokkos::View<double **> &c, size_t nx, size_t ny) {
    Kokkos::parallel_for(                                                                           //
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::Schedule<Kokkos::Static>>({0, 0}, {nx, ny}), //
        KOKKOS_LAMBDA(const size_t i0, const size_t i1) {                                           //
            c(i0, i1) = a(i0, i1) + b(i0, i1);
        });
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    int c = 1;
    Kokkos::initialize(c, argv);
    {
        Kokkos::View<double **> a("a", nx, ny);
        Kokkos::View<double **> b("b", nx, ny);
        Kokkos::View<double **> c("c", nx, ny);

        auto h_a = Kokkos::create_mirror_view(a);
        auto h_b = Kokkos::create_mirror_view(b);
        auto h_c = Kokkos::create_mirror_view(c);

        // init
        initMatrixAdd(h_a.data(), h_b.data(), h_c.data(), nx, ny);

        Kokkos::deep_copy(a, h_a);
        Kokkos::deep_copy(b, h_b);
        Kokkos::deep_copy(c, h_c);

        // warm-up
        for (size_t i = 0; i < nItWarmUp; ++i) {
            matrixAdd(a, b, c, nx, ny);
            std::swap(c, a);
        }
        Kokkos::fence();

        // measurement
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nIt; ++i) {
            matrixAdd(a, b, c, nx, ny);
            std::swap(c, a);
        }
        Kokkos::fence();

        auto end = std::chrono::steady_clock::now();

        printStats(end - start, nIt, nx * ny, sizeof(double) + sizeof(double) + sizeof(double), 1);

        Kokkos::deep_copy(h_a, a);
        Kokkos::deep_copy(h_b, b);
        Kokkos::deep_copy(h_c, c);

        // check solution
        checkSolutionMatrixAdd(h_a.data(), h_b.data(), h_c.data(), nx, ny, nIt + nItWarmUp);
    }
    Kokkos::finalize();

    return 0;
}
