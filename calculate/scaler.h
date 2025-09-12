#ifndef SCALER_H
#define SCALER_H

#include <iostream>
#include <regex>
#include <vector>
using std::vector;

struct Scaler {
    double start, sep;
    size_t cnt;
};



Scaler fromLinspace(double start, double stop, size_t cnt) {
    Scaler scaler{};
    scaler.start = start;
    scaler.sep = (stop - start) / cnt;
    scaler.cnt = cnt+1;
    return scaler;
}

Scaler parseScaler(const std::string& line) {
    std::regex pattern_scaler(R"((fill|linspace)@([+-]?\d+\.?\d*)@([+-]?\d+\.?\d*)@(\d+))");
    std::smatch match;
    if (!std::regex_match(line, match, pattern_scaler)) {
        std::cerr << "Invalid scaler format: " << line << std::endl;
        throw std::invalid_argument("Invalid scaler format");
    }
    if (match[1] == "fill") {
        return {std::stod(match[2]), std::stod(match[3]), std::stoul(match[4])};
    }
    if (match[1] == "linspace") {
        return fromLinspace(std::stod(match[2]), std::stod(match[3]), std::stoul(match[4]));
    }
    std::cerr << "Unknown scaler type: " << match[1] << std::endl;
    throw std::invalid_argument("Unknown scaler type");
}
__global__ void fillScaler(Scaler* scaler, double* target) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    double last = scaler->start + scaler->sep * idx;
    double step = scaler->sep * stride;

    for (size_t i = idx; i < scaler->cnt; i += stride) {
        target[i] = last;
        last += step;
    }

}

#endif
