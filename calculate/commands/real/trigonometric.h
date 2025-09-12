#ifndef CALCULATE_REAL_TRIGONOMETRIC_H
#define CALCULATE_REAL_TRIGONOMETRIC_H

#include "command.h"

#define TEMPLATE_TRIGONOMETRIC(fnname, name) \
__global__ void f_##fnname(const double* inA, double* out, size_t size) { \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; \
    uint stride = gridDim.x * blockDim.x; \
    for (size_t i = idx; i < size; i += stride) { \
        out[i] = std::fnname(inA[i]); \
    } \
} \
class name : public SingleCommand<name> { \
public: \
    using SingleCommand::SingleCommand;      \
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override { \
        if (inA >= space.size() || out >= space.size()) { \
            std::cerr << "Invalid indices for " << #name << "." << std::endl; \
            return false; \
        } \
        for (size_t i = 0; i < size_space; ++i) { \
            space[out][i] = std::fnname(space[inA][i]); \
        } \
        return true; \
    } \
    IMPLEMENT_CUDA_SINGLE_COMMAND(f_##fnname, #name) \
    string exp() const override { \
        return string(#fnname) + "(" + FORMAT_MEMORY_LOCATION(inA) + ")"; \
    } \
};
#define TEMPLATE_TRIGONOMETRIC_VAR2(fnname, name) \
__global__ void f_##fnname(const double* inA, const double* inB, double* out, size_t size) { \
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; \
    uint stride = gridDim.x * blockDim.x; \
    for (size_t i = idx; i < size; i += stride) { \
        out[i] = std::fnname(inA[i], inB[i]); \
    } \
} \
class name : public NormalCommand<name> { \
public: \
    using NormalCommand::NormalCommand;           \
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override { \
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) { \
            std::cerr << "Invalid indices for " << #name << "." << std::endl; \
            return false; \
        } \
        for (size_t i = 0; i < size_space; ++i) { \
            space[out][i] = std::fnname(space[inA][i], space[inB][i]); \
        } \
        return true; \
    } \
    IMPLEMENT_CUDA_NORMAL_COMMAND(f_##fnname, #name) \
    string exp() const override { \
        return string(#fnname) + "(" + FORMAT_MEMORY_LOCATION(inA) + FORMAT_MEMORY_LOCATION(inB) + ")"; \
    } \
};

TEMPLATE_TRIGONOMETRIC(sin, SinCommand)
TEMPLATE_TRIGONOMETRIC(cos, CosCommand)
TEMPLATE_TRIGONOMETRIC(tan, TanCommand)
TEMPLATE_TRIGONOMETRIC(asin, ASinCommand)
TEMPLATE_TRIGONOMETRIC(acos, ACosCommand)
TEMPLATE_TRIGONOMETRIC(atan, ATanCommand)
TEMPLATE_TRIGONOMETRIC(sinh, SinhCommand)
TEMPLATE_TRIGONOMETRIC(cosh, CoshCommand)
TEMPLATE_TRIGONOMETRIC(tanh, TanhCommand)
TEMPLATE_TRIGONOMETRIC(asinh, ASinhCommand)
TEMPLATE_TRIGONOMETRIC(acosh, ACoshCommand)
TEMPLATE_TRIGONOMETRIC(atanh, ATanhCommand)

TEMPLATE_TRIGONOMETRIC_VAR2(atan2, ATan2Command)


#endif // CALCULATE_REAL_TRIGONOMETRIC_H