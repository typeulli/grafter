#ifndef CALCULATE_REAL_DIFF_H
#define CALCULATE_REAL_DIFF_H

#include "command.h"

__global__ void _ediff(const double* inA, const double* inB, double* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        if (i == 0) {
            out[i] = (inA[1] - inA[0]) / (inB[1] - inB[0]);
            continue;
        }
        out[i] = (inA[i] - inA[i-1]) / (inB[i] - inB[i-1]);
    }
}
class ExplicitDiffCommand : public NormalCommand<ExplicitDiffCommand> {
public:
    using NormalCommand::NormalCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for ExplicitDiffCommand." << std::endl;
            return false;
        }
        if (size_space < 2) {
            std::cerr << "Size must be at least 2 for ExplicitDiffCommand." << std::endl;
            return false;
        }
        // CPU implementation
        double* inA_ptr = space[inA];
        double* inB_ptr = space[inB];
        double* out_ptr = space[out];

        out_ptr[0] = (inA_ptr[1] - inA_ptr[0]) / (inB_ptr[1] - inB_ptr[0]);
        for (size_t i = 1; i < size_space; ++i) {
            out_ptr[i] = (inA_ptr[i] - inA_ptr[i - 1]) / (inB_ptr[i] - inB_ptr[i - 1]);
        }
        return true;
    }
    IMPLEMENT_CUDA_NORMAL_COMMAND(_ediff, "ExplicitDiffCommand")
    [[nodiscard]] string exp() const override {
        return "d(" + FORMAT_MEMORY_LOCATION(inA) + ")/d(" + FORMAT_MEMORY_LOCATION(inB) + ")";
    }
    [[nodiscard]] string stmt() const override {
        return FORMAT_MEMORY_LOCATION(out) + " = " + exp();
    }
};

//__global__ void _idiff(const double* inA, const double* inB, double* out, double ratio, double xSize, size_t size) {
//    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//    uint stride = gridDim.x * blockDim.x;
//
//
//
//    for (size_t i = idx; i < size; i += stride) {
//        if (i == 0) {
//            out[i] = (inA[1] - inA[0]) / (inB[1] - inB[0]);
//            continue;
//        }
//        out[i] = (inA[i] - inA[i-1]) / (inB[i] - inB[i-1]);
//    }
//}
//
//class ImplicitDiffCommand : public NormalCommand<ImplicitDiffCommand> {
//public:
//    using NormalCommand::NormalCommand;
//    IMPLEMENT_FULL_COMMAND(_idiff, ImplicitDiffCommand)
//};

#endif // CALCULATE_REAL_DIFF_H