#include "matrix-add-util.h"

#include "../sycl-util.h"


inline void matrixAdd(sycl::queue &q, sycl::buffer<double> &b_a, sycl::buffer<double> &b_b, sycl::buffer<double> &b_c, size_t nx, size_t ny) {
    q.submit([&](sycl::handler &h) {
        auto a = b_a.get_access(h, sycl::read_only);
        auto b = b_b.get_access(h, sycl::read_only);
        auto c = b_c.get_access(h, sycl::write_only);

        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(ceilToMultipleOf(ny, 16), ceilToMultipleOf(nx, 16)), sycl::range<2>(16, 16)), [=](sycl::nd_item<2> item) {
            const auto i0 = item.get_global_id(1);
            const auto i1 = item.get_global_id(0);

            if (i0 < nx && i1 < ny) {
                c[i0 + i1 * nx] = a[i0 + i1 * nx] + b[i0 + i1 * nx];
            }
        });
    });
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    sycl::queue q(sycl::property::queue::in_order{}); // in-order queue to remove need for waits after each kernel

    double *a;
    a = new double[nx * ny];
    double *b;
    b = new double[nx * ny];
    double *c;
    c = new double[nx * ny];

    // init
    initMatrixAdd(a, b, c, nx, ny);

    {
        sycl::buffer b_a(a, sycl::range(nx * ny));
        sycl::buffer b_b(b, sycl::range(nx * ny));
        sycl::buffer b_c(c, sycl::range(nx * ny));

        // warm-up
        for (size_t i = 0; i < nItWarmUp; ++i) {
            matrixAdd(q, b_a, b_b, b_c, nx, ny);
            std::swap(b_c, b_a);
        }
        q.wait();

        // measurement
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nIt; ++i) {
            matrixAdd(q, b_a, b_b, b_c, nx, ny);
            std::swap(b_c, b_a);
        }
        q.wait();

        auto end = std::chrono::steady_clock::now();

        printStats(end - start, nIt, nx * ny, sizeof(double) + sizeof(double) + sizeof(double), 1);
    } // implicit D-H copy of destroyed buffers

    // check solution
    checkSolutionMatrixAdd(a, b, c, nx, ny, nIt + nItWarmUp);

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
