#include <algorithm>
#include <execution>


#include <thrust/universal_vector.h>

#include <thrust/execution_policy.h>

#include <thrust/for_each.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>


#include "matrix-add-util.h"


inline void matrixAdd(thrust::universal_vector<double> &a, thrust::universal_vector<double> &b, thrust::universal_vector<double> &c, size_t nx, size_t ny) {
    // // version 1: thrust transform
    // thrust::transform(thrust::device, a.begin(), a.end(), b.begin(), c.begin(),
    //                   [=] __host__ __device__ (double a_elem, double b_elem) {
    //                       return a_elem + b_elem;
    //                   });

    // // version 2: thrust for_each + counting iterator
    // double *a_ptr = thrust::raw_pointer_cast(a.data());
    // double *b_ptr = thrust::raw_pointer_cast(b.data());
    // double *c_ptr = thrust::raw_pointer_cast(c.data());
    // thrust::for_each(thrust::device, thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(nx * ny),
    //                  [=] __host__ __device__ (size_t i) {
    //                     // size_t i0 = i % nx;
    //                     // size_t i1 = i / nx;
    //                     c_ptr[i] = a_ptr[i] + b_ptr[i];
    //                  });

    // version 3: thrust tabulate
    double *a_ptr = thrust::raw_pointer_cast(a.data());
    double *b_ptr = thrust::raw_pointer_cast(b.data());
    double *c_ptr = thrust::raw_pointer_cast(c.data());
    thrust::tabulate(thrust::device, c.begin(), c.end(),
                     [=] __host__ __device__ (size_t i) {
                        // size_t i0 = i % nx;
                        // size_t i1 = i / nx;
                        return a_ptr[i] + b_ptr[i];
                     });
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    thrust::universal_vector<double> a(nx * ny);
    thrust::universal_vector<double> b(nx * ny);
    thrust::universal_vector<double> c(nx * ny);

    // init
    initMatrixAdd(thrust::raw_pointer_cast(a.data()),
                  thrust::raw_pointer_cast(b.data()),
                  thrust::raw_pointer_cast(c.data()),
                  nx, ny);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        matrixAdd(a, b, c, nx, ny);
        std::swap(c, a);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        matrixAdd(a, b, c, nx, ny);
        std::swap(c, a);
    }

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx * ny, sizeof(double) + sizeof(double) + sizeof(double), 1);

    // check solution
    checkSolutionMatrixAdd(thrust::raw_pointer_cast(a.data()),
                           thrust::raw_pointer_cast(b.data()),
                           thrust::raw_pointer_cast(c.data()),
                           nx, ny, nIt + nItWarmUp);

    return 0;
}
