#include "print-numbers-util.h"

#include "../sycl-util.h"


inline void printNumbers(sycl::queue &q, size_t nx) {
    q.submit([&](sycl::handler &h) {
        sycl::stream str(8192, 1024, h); // use sycl stream instead of printf
        h.parallel_for(nx, [=](auto i0) {
            if (i0 < nx) {
                // printf("%ld\n", i0);
                str << (size_t)i0 << sycl::endl; // use sycl endl
            }
        });
    });
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    sycl::queue q(sycl::property::queue::in_order{}); // in-order queue to remove need for waits after each kernel
    {
        // warm-up
        for (size_t i = 0; i < nItWarmUp; ++i) {
            printNumbers(q, nx);
        }
        q.wait();

        // measurement
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nIt; ++i) {
            printNumbers(q, nx);
        }
        q.wait();

        auto end = std::chrono::steady_clock::now();

        printStats(end - start, nIt, nx);
    }

    return 0;
}
