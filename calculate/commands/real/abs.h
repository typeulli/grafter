#ifndef GRAFTER_REAL_ABS_H
#define GRAFTER_REAL_ABS_H


#include "command.h"
__global__ void f_abs(const double* inA, double* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        out[i] = std::abs(inA[i]);
    }
}
class AbsCommand : public SingleCommand<AbsCommand> {
public:
    using SingleCommand::SingleCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for AbsCommand." << std::endl;
            return false;
        }
        double* in_ptr = space[inA];
        double* out_ptr = space[out];
        size_t total_size = scalers[size_scalers - 1].cnt;
        for (size_t i = 0; i < total_size; ++i) {
            out_ptr[i] = std::abs(in_ptr[i]);
        }
        return true;
    }
    IMPLEMENT_CUDA_SINGLE_COMMAND(f_abs, "AbsCommand")
    string exp() const override {
        return "|" + FORMAT_MEMORY_LOCATION(inA) + "|";
    }

};

#endif // GRAFTER_REAL_ABS_H