#ifndef GRAFTER_EXPLICIT_H
#define GRAFTER_EXPLICIT_H

#include <memory>

#include "command.h"


cv::Mat drawExplicit(
        cv::Mat& img,
        const std::vector<double>& x,
        const std::vector<double>& y,
        double min_x = -10.0,
        double max_x = 10.0,
        double min_y = -10.0,
        double max_y = 10.0,
        int r = 0, int g = 0, int b = 255
) {
    if (x.size() != y.size() || x.empty()) {
        std::cerr << "x and y must have the same non-zero size." << std::endl;
        return {};
    }

    int width = img.cols;
    int height = img.rows;


    double x_range = max_x - min_x;
    double y_range = max_y - min_y;
    if (x_range == 0) x_range = 1; // 분모 0 방지
    if (y_range == 0) y_range = 1;

    // padding
    int pad = 50;

    // 좌표 변환 함수
    auto to_pixel  = [&](double x_val, double y_val) -> cv::Point {
        int px = pad + static_cast<int>(((x_val - min_x) / x_range) * (width - 2 * pad));
        int py = height - pad - static_cast<int>(((y_val - min_y) / y_range) * (height - 2 * pad));
        return {px, py};
    };

    //draw points
    for (size_t i = 1; i < x.size(); ++i) {
        cv::circle(img, to_pixel(x[i], y[i]), 1, cv::Scalar(b, g, r), cv::FILLED); // 빨간색 점
    }
    // Draw the first point
    cv::circle(img, to_pixel(x[0], y[0]), 1, cv::Scalar(b, g, r), cv::FILLED);



    return img;
}


class DrawExplicitCommand : public ICommand {
public:
    size_t mat, inA, inB;
    int r, g, b;
    DrawExplicitCommand(size_t mat, size_t x, size_t y, int r, int g, int b) : mat(mat), inA(x), inB(y), r(r), g(g), b(b) {}
    static unique_ptr<DrawExplicitCommand> fromString(const string& str) {
        std::regex pattern(R"((\d+) (\d+) (\d+) (\d+) (\d+) (\d+))");
        std::smatch match;
        if (std::regex_match(str, match, pattern)) {
            return std::make_unique<DrawExplicitCommand>(
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
        if (inA >= space.size() || inB >= space.size()) {
            std::cerr << "Invalid indices for DrawExplicitCommand." << std::endl;
            return false;
        }
        vector<double> x(size_space);
        vector<double> y(size_space);

        cudaMemcpy(x.data(), space[inA], size_space * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(y.data(), space[inB], size_space * sizeof(double), cudaMemcpyDeviceToHost);

        cv::Mat result = drawExplicit(mats[mat].first, x, y,
                                      mats[mat].second.x_min, mats[mat].second.x_max,
                                      mats[mat].second.y_min, mats[mat].second.y_max,
                                      r, g, b);
        if (result.empty()) {
            std::cerr << "Failed to draw explicit function." << std::endl;
            return false;
        }

        return true;
    }
    string exp() const override {
        return "DrawExplicitCommand(mat=" + std::to_string(mat) +
               ", x=" + std::to_string(inA) +
               ", y=" + std::to_string(inB) +
               ", r=" + std::to_string(r) +
               ", g=" + std::to_string(g) +
               ", b=" + std::to_string(b) + ")";
    }
    string stmt() const override {
        return "Draw Explicit Command on mat(" + std::to_string(mat) + ") with x(" + FORMAT_MEMORY_LOCATION(inA) +
               "), y(" + FORMAT_MEMORY_LOCATION(inB) + ") and color (" +
               std::to_string(r) + ", " + std::to_string(g) + ", " + std::to_string(b) + ")";
    }
};

#endif // GRAFTER_EXPLICIT_H