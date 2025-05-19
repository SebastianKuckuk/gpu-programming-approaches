#pragma once

#include <iostream>


#define checkCudaError(...) \
    checkCudaErrorImpl(__FILE__, __LINE__, __VA_ARGS__)

inline void checkCudaErrorImpl(const std::string &file, int line, cudaError_t code, bool checkGetLastError = false) {
    if (cudaSuccess != code) {
        std::cerr << "CUDA Error (" << file << " : " << line << ") --- " << cudaGetErrorString(code) << std::endl;
        exit(1);
    }

    if (checkGetLastError)
        checkCudaErrorImpl(file, line, cudaGetLastError(), false);
}
