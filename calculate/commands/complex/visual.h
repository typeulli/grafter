#ifndef GRAFTER_CALCULATE_COMPLEX_VISUAL_H
#define GRAFTER_CALCULATE_COMPLEX_VISUAL_H

#include <memory>

#include "command.h"

__global__ void f_complex_rgb(const double* inA, const double* inB, double* outR, double* outG, double* outB, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        double real = inA[i];
        double imag = inB[i];

        // Magnitude
        double magnitude = sqrt(real * real + imag * imag);

        // Hue
        double angle = atan2(imag, real); // -π ~ π
        double H = (angle + M_PI) / (2 * M_PI) * 360.0; // 0 ~ 360

        // 채도와 명도 설정
        double S = 1.0;
        double V = atan(magnitude) * 2 / M_PI; // 조절 가능






        double c = V * S;
        double x = c * (1 - fabs(fmod(H / 60.0, 2.0) - 1));
        double m = V - c;

        double r, g, b;
        if (0 <= H && H < 60)      r = c, g = x, b = 0;
        else if (60 <= H && H < 120)  r = x, g = c, b = 0;
        else if (120 <= H && H < 180) r = 0, g = c, b = x;
        else if (180 <= H && H < 240) r = 0, g = x, b = c;
        else if (240 <= H && H < 300) r = x, g = 0, b = c;
        else                          r = c, g = 0, b = x;

        outR[i] = (r + m) * 255.0;
        outG[i] = (g + m) * 255.0;
        outB[i] = (b + m) * 255.0;
    }
}
class ComplexRGBCommand : public ExecuteCommand<ComplexRGBCommand> {
public:
    size_t outR, outG, outB;
    ComplexRGBCommand(size_t inA, size_t inB, size_t outR, size_t outG, size_t outB)
            : ExecuteCommand<ComplexRGBCommand>(inA, inB, 0, 0.0), outR(outR), outG(outG), outB(outB) {}
    bool runCommandCPU(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || outR >= space.size() || outG >= space.size() || outB >= space.size()) {
            std::cerr << "Invalid indices for ComplexRGBCommand." << std::endl;
            return false;
        }
        double* inA_ptr = space[inA];
        double* inB_ptr = space[inB];
        double* outR_ptr = space[outR];
        double* outG_ptr = space[outG];
        double* outB_ptr = space[outB];

        for (size_t i = 0; i < size_space; ++i) {
            double real = inA_ptr[i];
            double imag = inB_ptr[i];

            // Magnitude
            double magnitude = sqrt(real * real + imag * imag);

            // Hue
            double angle = atan2(imag, real); // -π ~ π
            double H = (angle + M_PI) / (2 * M_PI) * 360.0; // 0 ~ 360

            // 채도와 명도 설정
            double S = 1.0;
            double V = atan(magnitude) * 2 / M_PI; // 조절 가능

            double c = V * S;
            double x = c * (1 - fabs(fmod(H / 60.0, 2.0) - 1));
            double m = V - c;

            double r, g, b;
            if (0 <= H && H < 60)      r = c, g = x, b = 0;
            else if (60 <= H && H < 120)  r = x, g = c, b = 0;
            else if (120 <= H && H < 180) r = 0, g = c, b = x;
            else if (180 <= H && H < 240) r = 0, g = x, b = c;
            else if (240 <= H && H < 300) r = x, g = 0, b = c;
            else                          r = c, g = 0, b = x;

            outR_ptr[i] = (r + m) * 255.0;
            outG_ptr[i] = (g + m) * 255.0;
            outB_ptr[i] = (b + m) * 255.0;
        }
        return true;
    }
    bool runCommandCUDA(vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (inA >= space.size() || inB >= space.size() || outR >= space.size() || outG >= space.size() || outB >= space.size()) {
            std::cerr << "Invalid indices for ComplexRGBCommand." << std::endl;
            return false;
        }
        f_complex_rgb<<<16, 16>>>(space[inA], space[inB], space[outR], space[outG], space[outB], size_space);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in ComplexRGBCommand: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        cudaDeviceSynchronize();
        return true;
    }

    static unique_ptr<ComplexRGBCommand> fromString(const string& str) {
        std::regex pattern(R"((\d+) (\d+) (\d+) (\d+) (\d+))");
        std::smatch match;
        if (std::regex_match(str, match, pattern)) {
            return std::make_unique<ComplexRGBCommand>(
                std::stoul(match[1]), std::stoul(match[2]),
                std::stoul(match[3]), std::stoul(match[4]), std::stoul(match[5])
            );
        } else {
            std::cerr << "Invalid command string: " << str << std::endl;
            throw std::invalid_argument("Invalid command string format");
        }
    }
    [[nodiscard]] string exp() const override {
        return "ComplexToRGB(" + FORMAT_MEMORY_LOCATION(inA) +
               "+" + FORMAT_MEMORY_LOCATION(inB) + "i)";
    }
    [[nodiscard]] string stmt() const override {
        return "r(" + FORMAT_MEMORY_LOCATION(outR) + "), g(" + FORMAT_MEMORY_LOCATION(outG) + "), b(" + FORMAT_MEMORY_LOCATION(outB) + ") = "
          + exp();
    }
};

#endif //GRAFTER_CALCULATE_COMPLEX_VISUAL_H