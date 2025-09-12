#ifndef GRAFTER_CALCULATE_VAR_H
#define GRAFTER_CALCULATE_VAR_H
#include <memory>

#include "command.h"
#include "utils/utils.h"


__global__ void _var(double* out, Scaler* scalers, size_t ref, ull size_space) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;


    ull size_block = 1;
    for (size_t i = 0; i < ref; ++i) {
        size_block *= scalers[i].cnt;
    }


    for (size_t i = idx; i < size_space; i += stride) {
        out[i] = scalers[ref].start + scalers[ref].sep * ((i / size_block) % scalers[ref].cnt);
    }
}

class VarCommand : public ExecuteCommand<VarCommand> {
public:
    VarCommand(size_t dim, size_t out)
        : ExecuteCommand(dim, 0, out, 0.0) {}
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for VarCommand." << std::endl;
            return false;
        }

        const size_t dim = this->inA;
        if (dim >= size_scalers) {
            std::cerr << "Dimension index out of range in VarCommand." << std::endl;
            return false;
        }

        ull size_block = 1;
        for (size_t i = 0; i < dim; ++i) {
            size_block *= scalers[i].cnt;
        }

        double* out_ptr = space[out];
        for (size_t i = 0; i < size_space; ++i) {
            out_ptr[i] = scalers[dim].start + scalers[dim].sep * ((i / size_block) % scalers[dim].cnt);
        }
        return true;
    }
    bool runCommandCUDA(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {

        if (inA >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for VarCommand." << std::endl;
            return false;
        }

        const size_t dim = this->inA;
        _var<<<16, 16>>>(space[out], scalers, dim, size_space);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in VarCommand: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        cudaDeviceSynchronize();
        return true;
    }

    static unique_ptr<VarCommand> fromString(const string& str) {
        std::regex pattern(R"((\d+) (\d+))");
        std::smatch match;
        if (std::regex_match(str, match, pattern)) {
            return std::make_unique<VarCommand>(std::stoul(match[1]), std::stoul(match[2]));
        } else {
            std::cerr << "Invalid command string: " << str << std::endl;
            throw std::invalid_argument("Invalid command string format");
        }
    }
    string exp() const override {
        return "var(dim=" + std::to_string(inA) + ")";
    }
};

class CopyCommand : public ExecuteCommand<CopyCommand> {
public:
    CopyCommand(size_t inA, size_t out)
        : ExecuteCommand(inA, 0, out, 0.0) {}
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for CopyCommand." << std::endl;
            return false;
        }

        std::copy_n(space[inA], size_space, space[out]);
        return true;
    }

    bool runCommandCUDA(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for CopyCommand." << std::endl;
            return false;
        }

        cudaMemcpy(space[out], space[inA], size_space * sizeof(double), cudaMemcpyDeviceToDevice);
        return true;
    }

    static unique_ptr<CopyCommand> fromString(const string& str) {
        std::regex pattern(R"((\d+) (\d+))");
        std::smatch match;
        if (std::regex_match(str, match, pattern)) {
            return std::make_unique<CopyCommand>(std::stoul(match[1]), std::stoul(match[2]));
        } else {
            std::cerr << "Invalid command string: " << str << std::endl;
            throw std::invalid_argument("Invalid command string format");
        }
    }

    string exp() const override {
        return FORMAT_MEMORY_LOCATION(inA);
    }


};

#endif //GRAFTER_CALCULATE_VAR_H