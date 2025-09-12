#ifndef CALCULATE_REAL_POW_H
#define CALCULATE_REAL_POW_H

#include "command.h"
#include <cmath>
__global__ void f_pow(const double* inA, const double* inB, double* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        out[i] = std::pow(inA[i], inB[i]);
    }
}
class PowCommand : public NormalCommand<PowCommand> {
public:
    using NormalCommand::NormalCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for DivCommand." << std::endl;
            return false;
        }
        for (size_t i = 0; i < size_space; ++i) {
            space[out][i] = std::pow(space[inA][i], space[inB][i]);
        }
        return true;
    }
    IMPLEMENT_CUDA_NORMAL_COMMAND(f_pow, "PowCommand")
    string exp() const override {
        return FORMAT_MEMORY_LOCATION(inA) + "^" + FORMAT_MEMORY_LOCATION(inB);
    }
};


__global__ void f_pow_base_c(double* in, double ref, double* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        out[i] = std::pow(in[i], ref);
    }
}
class PowBaseConstantCommand : public SimpleConstantCommand<PowBaseConstantCommand> {
public:
    using SimpleConstantCommand::SimpleConstantCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for DivCommand." << std::endl;
            return false;
        }
        for (size_t i = 0; i < size_space; ++i) {
            space[out][i] = std::pow(space[inA][i], this->ref);
        }
        return true;
    }
    IMPLEMENT_CUDA_CONSTANT_COMMAND(f_pow_base_c, "PowBaseConstantCommand");
    string exp() const override {
        return FORMAT_MEMORY_LOCATION(inA) + "^" + std::to_string(ref);
    }
};
__global__ void f_pow_power_c(double* in, double ref, double* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        out[i] = std::pow(ref, in[i]);
    }
}
class PowPowerConstantCommand : public SimpleConstantCommand<PowPowerConstantCommand> {
public:
    using SimpleConstantCommand::SimpleConstantCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for DivCommand." << std::endl;
            return false;
        }
        for (size_t i = 0; i < size_space; ++i) {
            space[out][i] = std::pow(this->ref, space[inA][i]);
        }
        return true;
    }
    IMPLEMENT_CUDA_CONSTANT_COMMAND(f_pow_power_c, "PowPowerConstantCommand");
    string exp() const override {
        return std::to_string(ref) + "^" + FORMAT_MEMORY_LOCATION(inA);
    }
};

#endif // CALCULATE_REAL_POW_H