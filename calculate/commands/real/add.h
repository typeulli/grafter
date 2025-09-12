#ifndef CALCULATE_REAL_ABS_H
#define CALCULATE_REAL_ABS_H

#include "command.h"
__global__ void f_add(const double* inA, const double* inB, double* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        out[i] = inA[i] + inB[i];
    }
}
class AddCommand : public NormalCommand<AddCommand> {
public:
    using NormalCommand::NormalCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for AddCommand." << std::endl;
            return false;
        }
        for (size_t i = 0; i < size_space; ++i) {
            space[out][i] = space[inA][i] + space[inB][i];
        }
        return true;
    }
    IMPLEMENT_CUDA_NORMAL_COMMAND(f_add, "AddCommand")
    string exp() const override {
        return FORMAT_MEMORY_LOCATION(inA) + "+" + FORMAT_MEMORY_LOCATION(inB);
    }

};


__global__ void f_add_c(const double* in, double ref, double* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        out[i] = in[i] + ref;
    }
}
class AddConstantCommand : public SimpleConstantCommand<AddConstantCommand> {
public:
    using SimpleConstantCommand::SimpleConstantCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for AddConstantCommand." << std::endl;
            return false;
        }
        double* in = space[inA];
        double* outp = space[out];
        for (size_t i = 0; i < size_space; ++i) {
            outp[i] = in[i] + ref;
        }
        return true;
    }
    IMPLEMENT_CUDA_CONSTANT_COMMAND(f_add_c, "AddConstantCommand")
    string exp() const override {
        return FORMAT_MEMORY_LOCATION(inA) + "+" + std::to_string(ref);
    }
};

#endif //CALCULATE_REAL_ABS_H