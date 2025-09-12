#ifndef GRAFTER_RGB2D_H
#define GRAFTER_RGB2D_H

#include "command.h"

cv::Mat drawRGB2D(
        cv::Mat& img,
        Scaler scaler_x,
        Scaler scaler_y,
        const std::vector<double>& r,
        const std::vector<double>& g,
        const std::vector<double>& b,
        double min_x = -10.0,
        double max_x = 10.0,
        double min_y = -10.0,
        double max_y = 10.0
) {
    if (r.size() != g.size() || r.size() != b.size() || r.empty() || r.size() != scaler_x.cnt * scaler_y.cnt) {
        std::cerr << "r, g, b must have the same non-zero size and match the scaler dimensions." << std::endl;
        return {};
    }
    int width = img.cols;
    int height = img.rows;

    double x_range = max_x - min_x;
    double y_range = max_y - min_y;
    if (x_range == 0) x_range = 1; // 분모 0 방지
    if (y_range == 0) y_range = 1;
    int pad = 50;
    auto to_pixel = [&](double x_val, double y_val) -> cv::Point {
        int px = pad + static_cast<int>(((x_val - min_x) / x_range) * (width - 2 * pad));
        int py = height - pad - static_cast<int>(((y_val - min_y) / y_range) * (height - 2 * pad));
        return cv::Point(px, py);
    };
    for (size_t i = 0; i < r.size(); ++i) {
        double x_val = scaler_x.start + scaler_x.sep * (i % scaler_x.cnt);
        double y_val = scaler_y.start + scaler_y.sep * (i / scaler_x.cnt);
        cv::Point p = to_pixel(x_val, y_val);
        img.at<cv::Vec3b>(p) = cv::Vec3b(static_cast<uchar>(b[i]), static_cast<uchar>(g[i]), static_cast<uchar>(r[i]));
    }
    return img;
}

class DrawRGB2D : public ICommand {
public:
    size_t mat, real, imaginary;
    int r, g, b;
    DrawRGB2D(size_t mat, size_t real, size_t imaginary, int r, int g, int b) : mat(mat), real(real), imaginary(imaginary), r(r), g(g), b(b) {}
    static unique_ptr<DrawRGB2D> fromString(const string& str) {
        std::regex pattern(R"((\d+) (\d+) (\d+) (\d+) (\d+) (\d+))");
        std::smatch match;
        if (std::regex_match(str, match, pattern)) {
            return std::make_unique<DrawRGB2D>(
                    std::stoul(match[1]), std::stoul(match[2]), std::stoul(match[3]),
                    std::stoi(match[4]),  std::stoi(match[5]), std::stoi(match[6])
            );
        } else {
            std::cerr << "Invalid command string: " << str << std::endl;
            throw std::invalid_argument("Invalid command string format");
        }
    }

    AUTOFILL_COMMAND()
    bool run(DeviceInfo device, vector<pair<cv::Mat, MatInfo>> mats, vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (real >= space.size() || imaginary >= space.size()) {
            std::cerr << "Invalid indices for DrawExplicitCommand." << std::endl;
            return false;
        }

        Scaler scaler_x{};
        Scaler scaler_y{};
        if (device.type == DeviceType::CUDA) {
            cudaMemcpy(&scaler_x, &scalers[real], sizeof(Scaler), cudaMemcpyDeviceToHost);
            cudaMemcpy(&scaler_y, &scalers[imaginary], sizeof(Scaler), cudaMemcpyDeviceToHost);
        }

        vector<double> rVec(size_space);
        vector<double> gVec(size_space);
        vector<double> bVec(size_space);
        if (device.type == DeviceType::CUDA) {
            cudaMemcpy(rVec.data(), space[r], size_space * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(gVec.data(), space[g], size_space * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(bVec.data(), space[b], size_space * sizeof(double), cudaMemcpyDeviceToHost);
        }

        cv::Mat result = drawRGB2D(mats[mat].first, scaler_x, scaler_y,
                                    rVec, gVec, bVec,
                                     mats[mat].second.x_min, mats[mat].second.x_max,
                                     mats[mat].second.y_min, mats[mat].second.y_max);
        if (result.empty()) {
            std::cerr << "Failed to draw explicit function." << std::endl;
            return false;
        }
        return true;
    }
    string exp() const override {
        return "drawRGB2D(" + FORMAT_MEMORY_LOCATION(real) + ", " + FORMAT_MEMORY_LOCATION(imaginary) + ", " +
               std::to_string(r) + ", " + std::to_string(g) + ", " + std::to_string(b) + ")";
    }
    string stmt() const override {
        return "Draw RGB2D real(" + FORMAT_MEMORY_LOCATION(real) + "), imaginary(" + FORMAT_MEMORY_LOCATION(imaginary) + ") with color (" +
               std::to_string(r) + ", " + std::to_string(g) + ", " + std::to_string(b) + ")";
    }
};

#endif //GRAFTER_RGB2D_H