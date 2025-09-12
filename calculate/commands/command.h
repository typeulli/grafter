#ifndef COMMAND_H
#define COMMAND_H


#include <iostream>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

#include "device.h"
#include "../scaler.h"
#include "../utils/utils.h"
#include "../images/draw.h"

using std::string;
using std::unique_ptr;
using std::vector;
using std::pair;
struct ICommand {
    virtual ~ICommand() = default;
    virtual bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) = 0;
    virtual bool runCommandCUDA(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) = 0;
    virtual bool run(DeviceInfo device, vector<pair<cv::Mat, MatInfo>> mats, vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) = 0;
    virtual string exp() const = 0;
    virtual string stmt() const = 0;

};

struct IExecuteCommand : public ICommand {
    bool run(DeviceInfo device, vector<pair<cv::Mat, MatInfo>> mats, vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {

        switch (device.type) {
            case DeviceType::CPU:
                return runCommandCPU(space, scalers, size_space, size_scalers);
            case DeviceType::CUDA:
                return runCommandCUDA(space, scalers, size_space, size_scalers);
            case DeviceType::UNKNOWN:
                std::cerr << "Unknown device type." << std::endl;
                return false;
        }
        return false;
    }

};
#define FORMAT_MEMORY_LOCATION(idx) ("$" + std::to_string(idx))

template <typename Derived>
class ExecuteCommand : public IExecuteCommand {
public:
    size_t inA, inB, out;
    double ref;


    explicit ExecuteCommand(size_t inA = 0, size_t inB = 0, size_t out = 0, double ref = 0.0)
            : inA(inA), inB(inB), out(out), ref(ref) {}


    string stmt() const override {
        return FORMAT_MEMORY_LOCATION(this->out) + " = " + this->exp();
    }
};


#define IMPLEMENT_CUDA_FULL_COMMAND(__fn, name)\
    bool runCommandCUDA(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {\
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {\
            std::cerr << "Invalid indices for " << name << "." << std::endl;\
            return false;\
        }\
        __fn<<<16, 16>>>(space[inA], space[inB], space[out], ref, size_space);\
        cudaError_t err = cudaGetLastError();\
        if (err != cudaSuccess) {\
            std::cerr << "CUDA error in " << name << ": " << cudaGetErrorString(err) << std::endl;\
            return false;\
        }\
        cudaDeviceSynchronize();\
        return true;\
    }


#define IMPLEMENT_CUDA_SINGLE_COMMAND(__fn, name)\
    bool runCommandCUDA(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {\
        if (inA >= space.size() || out >= space.size()) {\
            std::cerr << "Invalid indices for " << name << "." << std::endl;\
            return false;\
        }\
        __fn<<<16, 16>>>(space[inA], space[out], size_space);\
        cudaError_t err = cudaGetLastError();\
        if (err != cudaSuccess) {\
            std::cerr << "CUDA error in " << name << ": " << cudaGetErrorString(err) << std::endl;\
            return false;\
        }\
        cudaDeviceSynchronize();\
        return true;\
    }


#define IMPLEMENT_CUDA_NORMAL_COMMAND(__fn, name)\
    bool runCommandCUDA(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {\
        if (inA >= space.size() || inB >= space.size() || out >= space.size()) {\
            std::cerr << "Invalid indices for " << name << "." << std::endl;\
            return false;\
        }\
        __fn<<<16, 16>>>(space[inA], space[inB], space[out], size_space);\
        cudaError_t err = cudaGetLastError();\
        if (err != cudaSuccess) {\
            std::cerr << "CUDA error in " << name << ": " << cudaGetErrorString(err) << std::endl;\
            return false;\
        }\
        cudaDeviceSynchronize();\
        return true;\
    }



#define IMPLEMENT_CUDA_CONSTANT_COMMAND(__fn, name)\
    bool runCommandCUDA(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {\
        if (inA >= space.size() || out >= space.size()) {\
            std::cerr << "Invalid indices for " << name << "." << std::endl;\
            return false;\
        }\
        __fn<<<16, 16>>>(space[inA], ref, space[out], size_space);\
        cudaError_t err = cudaGetLastError();\
        if (err != cudaSuccess) {\
            std::cerr << "CUDA error in " << name << ": " << cudaGetErrorString(err) << std::endl;\
            return false;\
        }\
        cudaDeviceSynchronize();\
        return true;\
    }


#define AUTOFILL_CPU_COMMAND()\
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {\
        return this->run({DeviceType::CPU, 0}, {}, space, scalers, size_space, size_scalers);\
    }
#define AUTOFILL_CUDA_COMMAND()\
    bool runCommandCUDA(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {\
        return this->run({DeviceType::CUDA, 0}, {}, space, scalers, size_space, size_scalers);\
    }
#define AUTOFILL_COMMAND() \
    AUTOFILL_CPU_COMMAND() \
    AUTOFILL_CUDA_COMMAND()

template <typename Derived>
class SingleCommand : public ExecuteCommand<Derived> {
public:
    SingleCommand(size_t in, size_t out) {
        this->inA = in;
        this->inB = 0; // SingleCommand does not use inB
        this->out = out;
        this->ref = 0.0; // SingleCommand does not use ref
    }

    static unique_ptr<Derived> fromString(const string& str) {
        std::regex pattern(R"((\d+) (\d+))");
        std::smatch match;
        if (std::regex_match(str, match, pattern)) {
            return std::unique_ptr<Derived>(new Derived(std::stoul(match[1]), std::stoul(match[2])));
        } else {
            std::cerr << "Invalid command string: " << str << std::endl;
            throw std::invalid_argument("Invalid command string format");
        }
    }
};
template<typename Derived>
class SimpleConstantCommand : public ExecuteCommand<Derived> {
public:
    SimpleConstantCommand(size_t in, size_t out, double ref) {
        this->inA = in;
        this->inB = 0; // SimpleConstantCommand does not use inB
        this->out = out;
        this->ref = ref;
    }

    static unique_ptr<Derived> fromString(const string& str) {
        std::regex pattern(R"((\d+) (\d+) ([+-]?\d*\.?\d+))");
        std::smatch match;
        if (std::regex_match(str, match, pattern)) {
            return std::unique_ptr<Derived>(new Derived(std::stoul(match[1]), std::stoul(match[2]), std::stod(match[3]) ));
        } else {
            std::cerr << "Invalid command string: " << str << std::endl;
            throw std::invalid_argument("Invalid command string format");
        }
    }

};


template<typename Derived>
class FullCommand : public ExecuteCommand<Derived> {
public:
    FullCommand(size_t inA, size_t inB, size_t out, double ref)
            : ExecuteCommand<Derived>(inA, inB, out, ref) {}


    static unique_ptr<Derived> fromString(const string& str) {
        std::regex pattern(R"((\d+) (\d+) (\d+) ([+-]?\d*\.?\d+))");
        std::smatch match;
        if (std::regex_match(str, match, pattern)) {
            return std::unique_ptr<Derived>(new Derived(std::stoul(match[1]), std::stoul(match[2]), std::stoul(match[3]), std::stod(match[4])));
        } else {
            std::cerr << "Invalid command string: " << str << std::endl;
            throw std::invalid_argument("Invalid command string format");
        }
    }
};

template<typename Derived>
class NormalCommand : public ExecuteCommand<Derived> {
public:
    NormalCommand(size_t inA, size_t inB, size_t out)
            : ExecuteCommand<Derived>(inA, inB, out, 0.0) {}


    static unique_ptr<Derived> fromString(const string& str) {
        std::regex pattern(R"((\d+) (\d+) (\d+))");
        std::smatch match;
        if (std::regex_match(str, match, pattern)) {
            return std::unique_ptr<Derived>(new Derived(std::stoul(match[1]), std::stoul(match[2]), std::stoul(match[3])));
        } else {
            std::cerr << "Invalid command string: \"" << str << "\"" <<std::endl;
            throw std::invalid_argument("Invalid command string format");
        }
    }


};
__global__ void f_constant(double* out, double ref, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        out[i] = ref;
    }
}
class ConstantCommand : public ExecuteCommand<ConstantCommand> {
public:
    ConstantCommand(size_t out, double ref) {
        this->inA = 0; // ConstantCommand does not use inA
        this->inB = 0; // ConstantCommand does not use inB
        this->out = out;
        this->ref = ref;
    }
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t) override {
        if (out >= space.size()) {
            std::cerr << "Invalid index for SimpleConstantCommand.\n";
            return false;
        }
        std::fill_n(space[out], size_space, ref);
        return true;
    }
    bool runCommandCUDA(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (out >= space.size()) {
            std::cerr << "Invalid index for SimpleConstantCommand." << std::endl;
            return false;
        }
        f_constant<<<16, 16>>>(space[out], ref, size_space);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in SimpleConstantCommand: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        cudaDeviceSynchronize();
        return true;
    }
    string exp() const override {
        return "const(" + std::to_string(ref) + ")";
    }
    string stmt() const override {
        return FORMAT_MEMORY_LOCATION(out) + " = " + exp();
    }
    static unique_ptr<ConstantCommand> fromString(const string& str) {
        std::regex pattern(R"((\d+) ([+-]?\d*\.?\d+))");
        std::smatch match;
        if (std::regex_match(str, match, pattern)) {
            return std::make_unique<ConstantCommand>(std::stoul(match[1]), std::stod(match[2]));
        } else {
            std::cerr << "Invalid command string: " << str << std::endl;
            throw std::invalid_argument("Invalid command string format");
        }
    }
};
#endif //COMMAND_H