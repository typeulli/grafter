#ifndef CALCULATE_REAL_LOG_H
#define CALCULATE_REAL_LOG_H

#include "../command.h"

__global__ void f_ln(const double* inA, double* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        if (inA[i] <= 0.0) {
            out[i] = NAN; // Logarithm of non-positive numbers is undefined
        } else {
            out[i] = std::log(inA[i]);
        }
    }
}

class LnCommand : public SingleCommand<LnCommand> {
public:
    using SingleCommand::SingleCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for LnCommand." << std::endl;
            return false;
        }
        for (size_t i = 0; i < size_space; ++i) {
            if (space[inA][i] <= 0.0) {
                space[out][i] = NAN; // Logarithm of non-positive numbers is undefined
            } else {
                space[out][i] = std::log(space[inA][i]);
            }
        }
        return true;
    }
    IMPLEMENT_CUDA_SINGLE_COMMAND(f_ln, "LnCommand");
    string exp() const override {
        return "ln(" + FORMAT_MEMORY_LOCATION(inA) + ")";
    }
};


__global__ void f_log(const double* inA, const double* inB, double* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        if (inA[i] <= 0.0 || inB[i] <= 0.0) {
            out[i] = NAN; // Logarithm of non-positive numbers is undefined
        } else {
            out[i] = std::log(inA[i]) / std::log(inB[i]);
        }
    }
}
class LogCommand : public NormalCommand<LogCommand> {
public:
    using NormalCommand::NormalCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for LogCommand." << std::endl;
            return false;
        }
        for (size_t i = 0; i < size_space; ++i) {
            if (space[inA][i] <= 0.0 || space[inB][i] <= 0.0) {
                space[out][i] = NAN; // Logarithm of non-positive numbers is undefined
            } else {
                space[out][i] = std::log(space[inA][i]) / std::log(space[inB][i]);
            }
        }
        return true;
    }
    IMPLEMENT_CUDA_NORMAL_COMMAND(f_log, "LogCommand");
    string exp() const override {
        return "log(" + FORMAT_MEMORY_LOCATION(inA) + ", " + FORMAT_MEMORY_LOCATION(inB) + ")";
    }
};


__global__ void f_log_base_c(double* in, double ref, double* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        if (in[i] <= 0.0 || ref <= 0.0) {
            out[i] = NAN; // Logarithm of non-positive numbers is undefined
        } else {
            out[i] = std::log(ref) / std::log(in[i]);
        }
    }
}
class LogBaseConstantCommand : public SimpleConstantCommand<LogBaseConstantCommand> {
public:
    using SimpleConstantCommand::SimpleConstantCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for LogBaseConstantCommand." << std::endl;
            return false;
        }
        for (size_t i = 0; i < size_space; ++i) {
            if (space[inA][i] <= 0.0 || ref <= 0.0) {
                space[out][i] = NAN; // Logarithm of non-positive numbers is undefined
            } else {
                space[out][i] = std::log(ref) / std::log(space[inA][i]);
            }
        }
        return true;
    }
    IMPLEMENT_CUDA_CONSTANT_COMMAND(f_log_base_c, "LogBaseConstantCommand");
    string exp() const override {
        return "log(" + std::to_string(ref) + ", " + FORMAT_MEMORY_LOCATION(inA) + ")";
    }
};
__global__ void f_log_arg_c(double* in, double ref, double* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        if (in[i] <= 0.0 || ref <= 0.0) {
            out[i] = NAN; // Logarithm of non-positive numbers is undefined
        } else {
            out[i] = std::log(in[i]) / std::log(ref);
        }
    }
}
class LogArgConstantCommand : public SimpleConstantCommand<LogArgConstantCommand> {
public:
    using SimpleConstantCommand::SimpleConstantCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for LogArgConstantCommand." << std::endl;
            return false;
        }
        for (size_t i = 0; i < size_space; ++i) {
            if (space[inA][i] <= 0.0 || ref <= 0.0) {
                space[out][i] = NAN; // Logarithm of non-positive numbers is undefined
            } else {
                space[out][i] = std::log(space[inA][i]) / std::log(ref);
            }
        }
        return true;
    }
    IMPLEMENT_CUDA_CONSTANT_COMMAND(f_log_arg_c, "LogArgConstantCommand");
    string exp() const override {
        return "log(" + FORMAT_MEMORY_LOCATION(inA) + ", " + std::to_string(ref) + ")";
    }
};

#endif // CALCULATE_REAL_LOG_H