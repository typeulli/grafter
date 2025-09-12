#ifndef GRAPH_H
#define GRAPH_H
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>

#include "command.h"
using std::string;
using std::vector;
using std::unique_ptr;

#include "utils/formats.h"

struct CommandInfo {
    ull line;
    string name;
    unique_ptr<ICommand> command;

    [[nodiscard]] string summary() const {
        return "[" + name + ":" + std::to_string(line) + "]" + command->stmt();
    }
};


Scaler* allocate_cuda_scaler(vector<Scaler> scalers) {
    Scaler* scaler_cuda;
    cudaError_t err = cudaMalloc(&scaler_cuda, sizeof(Scaler) * scalers.size());
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    err = cudaMemcpy(scaler_cuda, scalers.data(), sizeof(Scaler) * scalers.size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(scaler_cuda);
        return nullptr;
    }
    return scaler_cuda;
}

vector<double*> allocate_cuda_space(size_t size, size_t space_width) {
    vector<double*> space(size);
    cudaError_t err;
    for (size_t i = 0; i < size; ++i) {
        err = cudaMalloc(&space[i], sizeof(double) * space_width);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            for (size_t j = 0; j < i; ++j) {
                cudaFree(space[j]);
            }
            return {};
        }
    }
    return space;
}

class Graph {
public:
    [[nodiscard]] string summary() const {
        std::ostringstream oss;
        oss << "Formula_Size=" << size << "\n";
        oss << "Mats_Count=" << mats.size() << "\n";
        for (const auto& mat : mats) {
            std::cout << " - " << mat.width << "x" << mat.height << " [" << mat.x_min << ", " << mat.x_max << "] x [" << mat.y_min << ", " << mat.y_max << "]" << std::endl;

        }
        oss << "Scalers_Count=" << scalers.size() << "\n";
        for (const auto& scaler : scalers) {
            oss << " - Start: " << scaler.start << " Sep: " << scaler.sep << " Cnt: " << scaler.cnt << "\n";
        }
        oss << "Commands_Count=" << commands.size() << "\n";
        for (const auto& command : commands) {
            oss << " - " << command.summary() << "\n";
        }


        return oss.str();
    }
    size_t size;
    vector<CommandInfo> commands;
    vector<MatInfo> mats;
    vector<Scaler> scalers;
    explicit Graph(size_t size) : size(size) {}

    [[nodiscard]] vector<cv::Mat> execute(DeviceInfo device, bool verbose = false) const {
        vector<pair<cv::Mat, MatInfo>> imgs;
        for (const auto& mat : mats) {
            cv::Mat img(mat.height, mat.width, CV_8UC4, cv::Scalar(0, 0, 0, 0));
            imgs.emplace_back(img, mat);
        }

        if (verbose) printf("Mats are ready.\n");


        Scaler* scaler;
        switch (device.type) {
            case DeviceType::CPU:
                scaler = new Scaler[scalers.size()];
                std::copy(scalers.begin(), scalers.end(), scaler);
                break;
            case DeviceType::CUDA:
                scaler = allocate_cuda_scaler(scalers);
                break;
            case DeviceType::UNKNOWN:
                throw std::runtime_error("Unknown device type");
        }
        if (verbose) {
            printf("Allocated %lluB for scaler on device\n", sizeof(Scaler) * scalers.size());
        }


        vector<double*> space(size);
        ull space_width = 1;
        for (const auto& s : scalers) {
            space_width *= s.cnt;
        }

        auto start = std::chrono::high_resolution_clock::now();
        switch (device.type) {
            case DeviceType::CPU:
                for (auto& ptr : space) {
                    ptr = new double[space_width];
                }
                break;
            case DeviceType::CUDA:
                space = allocate_cuda_space(size, space_width);
                if (space.empty()) {
                    cudaFree(scaler);
                    throw std::runtime_error("Failed to allocate space on CUDA device");
                }
                break;
            case DeviceType::UNKNOWN:
                throw std::runtime_error("Unknown device type");
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        if (verbose) {
            std::cout << "Elapsed time: " << diff.count() << " s\n";
        }
        cudaDeviceSynchronize();

        if (verbose) {
            std::cout << "Allocated " << memory(sizeof(double) * space_width * size) << " for calculating on device" << std::endl;
            printf("Executing commands...\n");
        }


        for (auto& command : commands) {
            start = std::chrono::high_resolution_clock::now();
            if (!command.command->run(device, imgs, space, scaler, space_width, scalers.size())) {
                std::cerr << "Command execution failed." << std::endl;
                break;
            }
            end = std::chrono::high_resolution_clock::now();
            diff = end - start;

            if (verbose) {
                std::cout << "Elapsed time: " << diff.count() << " s\n";
            }
        }


        if (verbose) printf("Execution finished.\n");
        switch (device.type) {
            case DeviceType::CPU:
                std::for_each(space.begin(), space.end(), [](auto ptr){ delete[] ptr; });
                delete[] scaler;
                break;
            case DeviceType::CUDA:
                std::for_each(space.begin(), space.end(), [](auto ptr){ cudaFree(ptr); });
                cudaFree(scaler);
                break;
            case DeviceType::UNKNOWN:
                throw std::runtime_error("Unknown device type");
        }
        if (verbose) printf("Cleanup finished.\n");

        vector<cv::Mat> result;
        for (const auto& img : imgs) {
            cv::Mat src_image = img.first;


            double min_x = img.second.x_min;
            double max_x = img.second.x_max;
            double min_y = img.second.y_min;
            double max_y = img.second.y_max;
            plotAxis(src_image, "X-axis", "Y-axis", min_x, max_x, min_y, max_y);


            result.push_back(src_image);
        }

        return result;
    }
};


#endif // GRAPH_H