#ifndef GRAFTER_DIV_H
#define GRAFTER_DIV_H

#include "command.h"
#include <cmath>
__global__ void f_div(const double* inA, const double* inB, double* out, size_t size) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        out[i] = inA[i] / inB[i];

    }
}
class DivCommand : public NormalCommand<DivCommand> {
public:
    using NormalCommand::NormalCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for DivCommand." << std::endl;
            return false;
        }
        for (size_t i = 0; i < size_space; ++i) {
            space[out][i] = space[inA][i] / space[inB][i];
        }
        return true;
    }
    IMPLEMENT_CUDA_NORMAL_COMMAND(f_div, "DivCommand")
    string exp() const override {
        return FORMAT_MEMORY_LOCATION(inA) + "/" + FORMAT_MEMORY_LOCATION(inB);
    }
};


__global__ void f_div_c(const double* in, double ref, double* out, size_t size) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        out[i] = ref / in[i];
    }
}
class DivConstantCommand : public SimpleConstantCommand<DivConstantCommand> {
public:
    using SimpleConstantCommand::SimpleConstantCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for DivCommand." << std::endl;
            return false;
        }
        for (size_t i = 0; i < size_space; ++i) {
            space[out][i] = this->ref / space[inA][i];
        }
        return true;
    }
    IMPLEMENT_CUDA_CONSTANT_COMMAND(f_div_c, "DivConstantCommand");
    string exp() const override {
        return std::to_string(ref) + "/" + FORMAT_MEMORY_LOCATION(inA);
    }
};

#endif //GRAFTER_DIV_H