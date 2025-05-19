#pragma once

#include <cstring>

#include "../util.h"


void initConjugateGradient(double *__restrict__ u, double *__restrict__ rhs, const size_t nx, const size_t ny) {
    constexpr auto app = 1;

    switch (app) {
        case 0:
        // version 0 - zero boundary, unit inner values
        for (size_t i1 = 0; i1 < ny; ++i1) {
            for (size_t i0 = 0; i0 < nx; ++i0) {
                if (0 == i0 || 0 == i1 || nx - 1 == i0 || ny - 1 == i1)
                    u[i1 * nx + i0] = 0;
                else
                    u[i1 * nx + i0] = 1;

                rhs[i1 * nx + i0] = 0;
            }
        }

        break;

        case 1:
        // version 1 - simple trigonometric test problem
        for (size_t i1 = 0; i1 < ny; ++i1) {
            for (size_t i0 = 0; i0 < nx; ++i0) {
                auto x = i0 - 1 + 0.5;
                auto y = i1 - 1 + 0.5;

                if (0 == i0 || 0 == i1 || nx - 1 == i0 || ny - 1 == i1)
                    u[i1 * nx + i0] = sin(1e-3 * x) * cos(2e-3 * y);
                else
                    u[i1 * nx + i0] = 1;
                rhs[i1 * nx + i0] = 5e-6 * sin(1e-3 * x) * cos(2e-3 * y);
            }
        }
        break;
    }
}

void checkSolutionConjugateGradient(const double *const __restrict__ u, const double *const __restrict__ rhs, const size_t nx, const size_t ny) {
    double res = 0;
    for (size_t i1 = 1; i1 < ny - 1; ++i1) {
        for (size_t i0 = 1; i0 < nx - 1; ++i0) {
            const double localRes = rhs[i1 * nx + i0] - (
                    4 * u[i1 * nx + i0] - (u[i1 * nx + i0 - 1] + u[i1 * nx + i0 + 1] + u[(i1 - 1) * nx + i0] + u[(i1 + 1) * nx + i0]));

            res += localRes * localRes;
        }
    }

    res = sqrt(res);

    std::cout << "  Final residual is " << res << std::endl;
}

inline void parseCLA_2d(int argc, char **argv, size_t &nx, size_t &ny, size_t &nItWarmUp, size_t &nIt) {
    // default values
    nx = 4096;
    ny = 4096;

    nItWarmUp = 2;
    nIt = 10;

    // override with command line arguments
    int i = 1;
    if (argc > i)
        nx = atoi(argv[i]);
    ++i;
    if (argc > i)
        ny = atoi(argv[i]);
    ++i;

    if (argc > i)
        nItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i)
        nIt = atoi(argv[i]);
    ++i;
}
