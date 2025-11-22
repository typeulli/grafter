#ifndef CALCULATE_REAL_COMP_H
#define CALCULATE_REAL_COMP_H

#include <memory>

#include "command.h"

__global__ void f_eq0_2d(const double* inA, double ref, double* out, size_t size, size_t sizex) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        double cur = inA[i];
        double abs_cur = fabs(cur);


        if (abs_cur > ref) {
            out[i] = 0;
            continue;
        }
        if (cur > 0) {
            if (i >= sizex) {
                size_t pos_up = i - sizex;
                if (inA[pos_up] < 0 && abs_cur < fabs(inA[pos_up])) {
                    out[i] = 1;
                    continue;
                }
            }

            size_t pos_down = i + sizex;
            if (pos_down < size) {
                if (inA[pos_down] < 0 && abs_cur < fabs(inA[pos_down])) {
                    out[i] = 1;
                    continue;
                }
            }

            uint rx = i % sizex;

            if (rx != 0) {
                uint pos_left = i - 1;
                if (inA[pos_left] < 0 && abs_cur < fabs(inA[pos_left])) {
                    out[i] = 1;
                    continue;
                }
            }

            if (rx + 1 != sizex) {
                uint pos_right = i + 1;
                if (inA[pos_right] < 0 && abs_cur < fabs(inA[pos_right])) {
                    out[i] = 1;
                    continue;
                }
            }


        } else {


            if (i >= sizex) {
                size_t pos_up = i - sizex;
                if (inA[pos_up] > 0 && abs_cur < fabs(inA[pos_up])) {
                    out[i] = 1;
                    continue;
                }
            }

            size_t pos_down = i + sizex;
            if (pos_down < size) {
                if (inA[pos_down] > 0 && abs_cur < fabs(inA[pos_down])) {
                    out[i] = 1;
                    continue;
                }
            }

            uint rx = i % sizex;

            if (rx != 0) {
                uint pos_left = i - 1;
                if (inA[pos_left] > 0 && abs_cur < fabs(inA[pos_left])) {
                    out[i] = 1;
                    continue;
                }
            }

            if (rx + 1 != sizex) {
                uint pos_right = i + 1;
                if (inA[pos_right] > 0 && abs_cur < fabs(inA[pos_right])) {
                    out[i] = 1;
                    continue;
                }
            }


        }
        out[i] = 0;
    }
}

class EqualZero2DCommand : public SimpleConstantCommand<EqualZero2DCommand> {
public:
    using SimpleConstantCommand::SimpleConstantCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for EqualZero2DCommand." << std::endl;
            return false;
        }
        Scaler scaler_x = scalers[0];
        ull sizex = scaler_x.cnt;
        double *in = space[inA];
        double *outp = space[out];
        for (size_t i = 0; i < size_space; ++i) {
            double cur = in[i];
            double abs_cur = fabs(cur);


            if (abs_cur > ref) {
                outp[i] = 0;
                continue;
            }
            if (cur > 0) {
                if (i >= sizex) {
                    size_t pos_up = i - sizex;
                    if (in[pos_up] < 0 && abs_cur < fabs(in[pos_up])) {
                        outp[i] = 1;
                        continue;
                    }
                }

                size_t pos_down = i + sizex;
                if (pos_down < size_space) {
                    if (in[pos_down] < 0 && abs_cur < fabs(in[pos_down])) {
                        outp[i] = 1;
                        continue;
                    }
                }

                uint rx = i % sizex;

                if (rx != 0) {
                    uint pos_left = i - 1;
                    if (in[pos_left] < 0 && abs_cur < fabs(in[pos_left])) {
                        outp[i] = 1;
                        continue;
                    }
                }

                if (rx + 1 != sizex) {
                    uint pos_right = i + 1;
                    if (in[pos_right] < 0 && abs_cur < fabs(in[pos_right])) {
                        outp[i] = 1;
                        continue;
                    }
                }


            } else {


                if (i >= sizex) {
                    size_t pos_up = i - sizex;
                    if (in[pos_up] > 0 && abs_cur < fabs(in[pos_up])) {
                        outp[i] = 1;
                        continue;
                    }
                }

                size_t pos_down = i + sizex;
                if (pos_down < size_space) {
                    if (in[pos_down] > 0 && abs_cur < fabs(in[pos_down])) {
                        outp[i] = 1;
                        continue;
                    }
                }

                uint rx = i % sizex;

                if (rx != 0) {
                    uint pos_left = i - 1;
                    if (in[pos_left] > 0 && abs_cur < fabs(in[pos_left])) {
                        outp[i] = 1;
                        continue;
                    }
                }

                if (rx + 1 != sizex) {
                    uint pos_right = i + 1;
                    if (in[pos_right] > 0 && abs_cur < fabs(in[pos_right])) {
                        outp[i] = 1;
                        continue;
                    }
                }


            }
            outp[i] = 0;
        }
    }
    bool runCommandCUDA(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for EqualZero2DCommand." << std::endl;
            return false;
        }
        Scaler scaler_x{};
        cudaMemcpy(&scaler_x, &scalers[0], sizeof(Scaler), cudaMemcpyDeviceToHost);
        ull sizex = scaler_x.cnt;
        f_eq0_2d<<<16, 16>>>(space[inA], ref, space[out], size_space, sizex);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in EqualZero2DCommand: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        cudaDeviceSynchronize();
        return true;
    }
    string exp() const override {
        return FORMAT_MEMORY_LOCATION(inA) + " == 0 with tolerance " + std::to_string(ref);
    }
    string stmt() const override {
        return FORMAT_MEMORY_LOCATION(out) + " = " + exp();
    }
};


__global__ void f_eq0(const double* inA, double ref, double* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        if (inA[i] < ref && inA[i] > -ref) {
            out[i] = abs(inA[i]) / ref;
        } else {
            out[i] = 1;
        }
    }
}
class EqualZeroCommand : public SimpleConstantCommand<EqualZeroCommand> {
public:
    using SimpleConstantCommand::SimpleConstantCommand;
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || out >= space.size()) {
            std::cerr << "Invalid indices for EqualZeroCommand." << std::endl;
            return false;
        }
        double* in = space[inA];
        double* outp = space[out];
        for (size_t i = 0; i < size_space; ++i) {
            if (in[i] < ref && in[i] > -ref) {
                outp[i] = abs(in[i]) / ref;
            } else {
                outp[i] = 1;
            }
        }
        return true;
    }
    IMPLEMENT_CUDA_CONSTANT_COMMAND(f_eq0, "EqualZeroCommand")
    string exp() const override {
        return FORMAT_MEMORY_LOCATION(inA) + " == 0 with tolerance " + std::to_string(ref);
    }
    string stmt() const override {
        return FORMAT_MEMORY_LOCATION(out) + " = " + exp();
    }
};
//
//__global__ void _eq(const double* inA, const double* inB, double* out, double epsilon, double delta, size_t size) {
//    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//    uint stride = gridDim.x * blockDim.x;
//    double diff, abs_diff;
//    for (size_t i = idx; i < size; i += stride) {
//        diff = inA[i] - inB[i];
//        abs_diff = abs(diff);
//        if (abs_diff < delta) {
//            out[i] = abs_diff;
//        } else {
//            if (abs_diff >= delta) {
//                out[i] = -1;
//                continue;
//            }
//
//            if (diff > 0) {
//                out[i] = (abs_diff - delta) / (abs(inA[i]) + abs(inB[i]) + 1e-10) / epsilon;
//            } else {
//                out[i] = (abs_diff - delta) / (abs(inA[i]) + abs(inB[i]) + 1e-10) / epsilon;
//            }
//        }
//    }
//}
//class EqualCommand : public FullCommand<EqualCommand> {
//public:
//    using FullCommand::FullCommand;
//    IMPLEMENT_CUDA_FULL_COMMAND(_eq, "EqualCommand")
//    static unique_ptr<EqualCommand> fromString(const string& str) {
//        std::regex pattern(R"((\d+) (\d+) (\d+) ([+-]?\d*\.?\d+))");
//        std::smatch match;
//        if (std::regex_match(str, match, pattern)) {
//            return std::make_unique<EqualCommand>(std::stoul(match[1]), std::stoul(match[2]), std::stoul(match[3]), std::stod(match[4]));
//        } else {
//            std::cerr << "Invalid command string: " << str << std::endl;
//            throw std::invalid_argument("Invalid command string format");
//        }
//    }
//    string exp() const override {
//        return FORMAT_MEMORY_LOCATION(inA) + "==" + FORMAT_MEMORY_LOCATION(inB) + " with tolerance " + std::to_string(ref);
//    }
//    string stmt() const override {
//        return FORMAT_MEMORY_LOCATION(out) + " = " + exp();
//    }
//};

#endif // CALCULATE_REAL_COMP_H