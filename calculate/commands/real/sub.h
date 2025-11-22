#ifndef CALCULATE_REAL_SUB_H
#define CALCULATE_REAL_SUB_H

#include "command.h"
__global__ void f_sub(const double* inA, const double* inB, double* out, size_t size) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        out[i] = inA[i] - inB[i];
    }
}
class SubCommand : public NormalCommand<SubCommand> {
public:
    using NormalCommand::NormalCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for SubCommand." << std::endl;
            return false;
        }
        for (size_t i = 0; i < size_space; ++i) {
            space[out][i] = space[inA][i] - space[inB][i];
        }
        return true;
    }
    IMPLEMENT_CUDA_NORMAL_COMMAND(f_sub, "SubCommand")
    string exp() const override {
        return FORMAT_MEMORY_LOCATION(inA) + "-" + FORMAT_MEMORY_LOCATION(inB);
    }
};


__global__ void f_sub_c(const double* in, double ref, double* out, size_t size) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        out[i] = in[i] - ref;
    }
}
class SubConstantCommand : public SimpleConstantCommand<SubConstantCommand> {
public:
    using SimpleConstantCommand::SimpleConstantCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for SubConstantCommand." << std::endl;
            return false;
        }
        for (size_t i = 0; i < size_space; ++i) {
            space[out][i] = space[inA][i] - this->ref;
        }
        return true;
    }
    IMPLEMENT_CUDA_CONSTANT_COMMAND(f_sub_c, "SubConstantCommand")
    string exp() const override {
        return FORMAT_MEMORY_LOCATION(inA) + "-" + std::to_string(ref);
    }
};

#endif // CALCULATE_REAL_SUB_H