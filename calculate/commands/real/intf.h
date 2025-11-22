#ifndef CALCULATE_REAL_INTF_H
#define CALCULATE_REAL_INTF_H

#include "command.h"

__global__ void f_floor(const double* inA, double* out, size_t size) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        out[i] = floor(inA[i]);
    }
}
class FloorCommand : public SingleCommand<FloorCommand> {
public:
    using SingleCommand::SingleCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for DivCommand." << std::endl;
            return false;
        }
        for (size_t i = 0; i < size_space; ++i) {
            space[out][i] = floor(space[inA][i]);
        }
        return true;
    }
    IMPLEMENT_CUDA_SINGLE_COMMAND(f_floor, "FloorCommand")
    [[nodiscard]] string exp() const override {
        return "floor(" + FORMAT_MEMORY_LOCATION(inA) + ")";
    }
};

__global__ void f_ceil(const double* inA, double* out, size_t size) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        out[i] = ceil(inA[i]);
    }
}
class CeilCommand : public SingleCommand<CeilCommand> {
public:
    using SingleCommand::SingleCommand;

    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for DivCommand." << std::endl;
            return false;
        }
        for (size_t i = 0; i < size_space; ++i) {
            space[out][i] = ceil(space[inA][i]);
        }
        return true;
    }
    IMPLEMENT_CUDA_SINGLE_COMMAND(f_ceil, "CeilCommand")
    string exp() const override {
        return "ceil(" + FORMAT_MEMORY_LOCATION(inA) + ")";
    }
};

#endif // CALCULATE_REAL_INTF_H