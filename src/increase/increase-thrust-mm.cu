#include <algorithm>
#include <execution>


#include <thrust/universal_vector.h>

#include <thrust/execution_policy.h>

#include <thrust/for_each.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>


#include "increase-util.h"


inline void increase(thrust::universal_vector<double> &data, size_t nx) {
    // // version 1: thrust transform
    // thrust::transform(thrust::device, data.begin(), data.end(), data.begin(),
    //                   [=] __host__ __device__ (double data_elem) {
    //                       return data_elem + 1;
    //                   });

    // // version 2: thrust for_each + counting iterator
    // double *data_ptr = thrust::raw_pointer_cast(data.data());
    // thrust::for_each(thrust::device, thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(nx),
    //                  [=] __host__ __device__ (size_t i0) {
    //                      data_ptr[i0] += 1;
    //                  });

    // version 3: thrust tabulate
    double *data_ptr = thrust::raw_pointer_cast(data.data());
    thrust::tabulate(thrust::device, data.begin(), data.end(),
                     [=] __host__ __device__ (size_t i0) {
                         return data_ptr[i0] + 1;
                     });
}

int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    thrust::universal_vector<double> data(nx);

    // init
    initIncrease(thrust::raw_pointer_cast(data.data()), nx);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        increase(data, nx);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        increase(data, nx);
    }

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(double) + sizeof(double), 1);

    // check solution
    checkSolutionIncrease(thrust::raw_pointer_cast(data.data()), nx, nIt + nItWarmUp);

    return 0;
}
