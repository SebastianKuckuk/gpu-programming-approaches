#pragma once

#include <sycl/sycl.hpp>


template<typename tpe>
inline void setFieldToZero(sycl::queue &q, size_t nx, tpe *field) {
    q.memset(field, 0, nx * sizeof(tpe));
}

template<typename tpe>
inline void setFieldToZero(sycl::queue &q, size_t nx, size_t ny, tpe *field) {
    setFieldToZero(q, nx * ny, field);
}

template<typename tpe>
inline void setFieldToZero(sycl::queue &q, size_t nx, size_t ny, size_t nz, tpe *field) {
    setFieldToZero(q, nx * ny * nz, field);
}

template<typename tpe, int numDims>
inline void setBufferToZero(sycl::queue &q, size_t nx, sycl::buffer <tpe, numDims> &b_field) {
    q.submit([&](sycl::handler &h) {
        auto field = b_field.get_access(h, sycl::write_only, sycl::no_init);

        h.parallel_for(sycl::range<1>(nx),
                       [=](auto item) {
                           const auto i = item[0];

                           field[i] = 0.;
                       });
    });
}

template<typename tpe, int numDims>
inline void setBufferToZero(sycl::queue &q, size_t nx, size_t ny, size_t nz, sycl::buffer <tpe, numDims> &b_field) {
    q.submit([&](sycl::handler &h) {
        auto field = b_field.get_access(h, sycl::write_only, sycl::no_init);

        h.parallel_for(sycl::range<3>(nz, ny, nx),
                       [=](auto item) {
                           const auto i = item[2];
                           const auto j = item[1];
                           const auto k = item[0];

                           field[k][j][i] = 0.;
                       });
    });
}

template<typename tpe, int numDims>
inline void setBufferToZero(sycl::queue &q, size_t nx, size_t ny, sycl::buffer <tpe, numDims> &b_field) {
    q.submit([&](sycl::handler &h) {
        auto field = b_field.get_access(h, sycl::write_only, sycl::no_init);

        h.parallel_for(sycl::range<2>(ny, nx),
                       [=](auto item) {
                           const auto i = item[1];
                           const auto j = item[0];

                           field[j][i] = 0.;
                       });
    });
}

template<typename tpe, int numDims>
inline void copyBuffer(sycl::queue &q, sycl::buffer <tpe, numDims> &b_dest, sycl::buffer <tpe, numDims> &b_src) {
    q.submit([&](sycl::handler &h) {
        auto src = b_src.get_access(h, sycl::read_only);
        auto dest = b_dest.get_access(h, sycl::write_only, sycl::no_init);
        h.copy(src, dest);
    });
};
