#pragma once

#include <iostream>

#include "hip/hip_runtime.h"


inline void checkHipError(hipError_t code, bool checkGetLastError = false) {
    if (hipSuccess != code) {
        std::cerr << "HIP Error: " << hipGetErrorString(code) << std::endl;
        exit(1);
    }

    if (checkGetLastError)
        checkHipError(hipGetLastError(), false);
}
