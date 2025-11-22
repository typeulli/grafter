#ifndef CALCULATE_REAL_POW_H
#define CALCULATE_REAL_POW_H

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "command.h"

__global__ void f_pow(const double* __restrict__ inA, const double* __restrict__ inB, double* __restrict__ out, size_t size) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        out[i] = pow(inA[i], inB[i]);
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
            space[out][i] = pow(space[inA][i], space[inB][i]);
        }
        return true;
    }
    IMPLEMENT_CUDA_NORMAL_COMMAND(f_pow, "PowCommand")
    string exp() const override {
        return FORMAT_MEMORY_LOCATION(inA) + "^" + FORMAT_MEMORY_LOCATION(inB);
    }
};


__global__ void f_pow_base_c(double* __restrict__ in, double ref, double* __restrict__ out, size_t size) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        out[i] = pow(in[i], ref);
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
        if (ref == 0.0) {
            std::fill_n(space[out], size_space, 1.0);
            return true;
        }
        if (ref == 1.0) {
            std::copy_n(space[inA], size_space, space[out]);
            return true;
        }
        for (size_t i = 0; i < size_space; ++i) {
            space[out][i] = pow(space[inA][i], this->ref);
        }
        return true;
    }
    bool runCommandCUDA(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for " << "PowBaseConstantCommand" << "." << std::endl;
            return false;
        }
        if (ref == 0.0) {
            thrust::device_ptr<double> out_ptr(space[out]);
            thrust::fill(out_ptr, out_ptr + size_space, 1.0);
            return true;
        }
        if (ref == 1.0) {
            cudaMemcpy(space[out], space[inA], size_space * sizeof(double), cudaMemcpyDeviceToDevice);
            return true;
        }
        f_pow_base_c<<<16, 16>>>(space[inA], ref, space[out], size_space);
        IMPLEMENT_CUDA_SYNCRONIZE("PowBaseConstantCommand")\
        return true;
    }
    string exp() const override {
        return FORMAT_MEMORY_LOCATION(inA) + "^" + std::to_string(ref);
    }
};
__global__ void f_pow_power_c(double* in, double ref, double* out, size_t size) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        out[i] = pow(ref, in[i]);
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
            space[out][i] = pow(this->ref, space[inA][i]);
        }
        return true;
    }
    IMPLEMENT_CUDA_CONSTANT_COMMAND(f_pow_power_c, "PowPowerConstantCommand");
    string exp() const override {
        return std::to_string(ref) + "^" + FORMAT_MEMORY_LOCATION(inA);
    }
};

#endif // CALCULATE_REAL_POW_H