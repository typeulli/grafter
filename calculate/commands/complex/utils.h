#ifndef GRAFTER_CALCULATE_COMPLEX_UTILS_H
#define GRAFTER_CALCULATE_COMPLEX_UTILS_H

#include "command.h"
__global__ void f_magnitude(const double* inA, const double* inB, double* out, size_t size) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        out[i] = sqrt(inA[i] * inA[i] + inB[i] * inB[i]);
    }
}
class MagnitudeCommand : public NormalCommand<MagnitudeCommand> {
public:
    using NormalCommand::NormalCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for MagnitudeCommand." << std::endl;
            return false;
        }
        for (size_t i = 0; i < size_space; ++i) {
            space[out][i] = sqrt(space[inA][i] * space[inA][i] + space[inB][i] * space[inB][i]);
        }
        return true;
    }

    IMPLEMENT_CUDA_NORMAL_COMMAND(f_magnitude, "MagnitudeCommand")
    string exp() const override {
        return "magnitude(" + FORMAT_MEMORY_LOCATION(inA) + ", " + FORMAT_MEMORY_LOCATION(inB) + ")";
    }
};

#endif //GRAFTER_CALCULATE_COMPLEX_UTILS_H