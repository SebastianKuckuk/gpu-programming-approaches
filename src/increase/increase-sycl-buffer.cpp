#include "increase-util.h"

#include "../sycl-util.h"


inline void increase(sycl::queue &q, sycl::buffer<double> &b_data, size_t nx) {
    q.submit([&](sycl::handler &h) {
        auto data = b_data.get_access(h, sycl::read_write);

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
    data = new double[nx];

    // init
    initIncrease(data, nx);

    {
        sycl::buffer b_data(data, sycl::range(nx));

        // warm-up
        for (size_t i = 0; i < nItWarmUp; ++i) {
            increase(q, b_data, nx);
        }
        q.wait();

        // measurement
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nIt; ++i) {
            increase(q, b_data, nx);
        }
        q.wait();

        auto end = std::chrono::steady_clock::now();

        printStats(end - start, nIt, nx, sizeof(double) + sizeof(double), 1);
    } // implicit D-H copy of destroyed buffers

    // check solution
    checkSolutionIncrease(data, nx, nIt + nItWarmUp);

    delete[] data;

    return 0;
}
