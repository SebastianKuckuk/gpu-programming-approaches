#include <algorithm>
#include <execution>


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/execution_policy.h>

#include <thrust/for_each.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>


#include "matrix-add-util.h"


inline void matrixAdd(thrust::device_vector<double> &a, thrust::device_vector<double> &b, thrust::device_vector<double> &c, size_t nx, size_t ny) {
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

    thrust::host_vector<double> a(nx * ny);
    thrust::device_vector<double> d_a(nx * ny);

    thrust::host_vector<double> b(nx * ny);
    thrust::device_vector<double> d_b(nx * ny);

    thrust::host_vector<double> c(nx * ny);
    thrust::device_vector<double> d_c(nx * ny);

    // init
    initMatrixAdd(a.data(), b.data(), c.data(), nx, ny);

    thrust::copy(a.begin(), a.end(), d_a.begin());
    thrust::copy(b.begin(), b.end(), d_b.begin());
    thrust::copy(c.begin(), c.end(), d_c.begin());

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        matrixAdd(d_a, d_b, d_c, nx, ny);
        std::swap(d_c, d_a);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        matrixAdd(d_a, d_b, d_c, nx, ny);
        std::swap(d_c, d_a);
    }

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx * ny, sizeof(double) + sizeof(double) + sizeof(double), 1);

    thrust::copy(d_a.begin(), d_a.end(), a.begin());
    thrust::copy(d_b.begin(), d_b.end(), b.begin());
    thrust::copy(d_c.begin(), d_c.end(), c.begin());

    // check solution
    checkSolutionMatrixAdd(a.data(), b.data(), c.data(), nx, ny, nIt + nItWarmUp);

    return 0;
}
