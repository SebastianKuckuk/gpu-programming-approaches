#include "increase-util.h"

#include "../sycl-util.h"


inline void increase(sycl::queue &q, double *data, size_t nx) {
    q.submit([&](sycl::handler &h) {
        h.parallel_for(nx, [=](auto i0) {
            if (i0 < nx) {
                data[i0] += 1;
            }
        });
    });
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    sycl::queue q(sycl::property::queue::in_order{}); // in-order queue to remove need for waits after each kernel

    double *data;
    data = sycl::malloc_shared<double>(nx, q);

    // init
    initIncrease(data, nx);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        increase(q, data, nx);
    }
    q.wait();

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        increase(q, data, nx);
    }
    q.wait();

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(double) + sizeof(double), 1);

    // check solution
    checkSolutionIncrease(data, nx, nIt + nItWarmUp);

    sycl::free(data, q);

    return 0;
}
