#include "matrix-add-util.h"

#include "../sycl-util.h"


inline void matrixAdd(sycl::queue &q, const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ c, size_t nx, size_t ny) {
    q.submit([&](sycl::handler &h) {
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
    a = sycl::malloc_host<double>(nx * ny, q);
    double *b;
    b = sycl::malloc_host<double>(nx * ny, q);
    double *c;
    c = sycl::malloc_host<double>(nx * ny, q);

    double *d_a;
    d_a = sycl::malloc_device<double>(nx * ny, q);
    double *d_b;
    d_b = sycl::malloc_device<double>(nx * ny, q);
    double *d_c;
    d_c = sycl::malloc_device<double>(nx * ny, q);

    // init
    initMatrixAdd(a, b, c, nx, ny);

    q.memcpy(d_a, a, sizeof(double) * nx * ny);
    q.wait();
    q.memcpy(d_b, b, sizeof(double) * nx * ny);
    q.wait();
    q.memcpy(d_c, c, sizeof(double) * nx * ny);
    q.wait();

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        matrixAdd(q, d_a, d_b, d_c, nx, ny);
        std::swap(d_c, d_a);
    }
    q.wait();

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        matrixAdd(q, d_a, d_b, d_c, nx, ny);
        std::swap(d_c, d_a);
    }
    q.wait();

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx * ny, sizeof(double) + sizeof(double) + sizeof(double), 1);

    q.memcpy(a, d_a, sizeof(double) * nx * ny);
    q.wait();
    q.memcpy(b, d_b, sizeof(double) * nx * ny);
    q.wait();
    q.memcpy(c, d_c, sizeof(double) * nx * ny);
    q.wait();

    // check solution
    checkSolutionMatrixAdd(a, b, c, nx, ny, nIt + nItWarmUp);

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);

    return 0;
}
