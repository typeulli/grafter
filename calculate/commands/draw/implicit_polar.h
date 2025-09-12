#ifndef GRAFTER_IMPLICIT_POLAR_H
#define GRAFTER_IMPLICIT_POLAR_H

#include <memory>
#include "command.h"
#include "visual.h"

cv::Mat drawImplicit(
        cv::Mat& img,
        const std::vector<double>& grid,
        size_t r_size,
        double min_x, double max_x,
        double min_y, double max_y,
        double min_theta, double max_theta,
        int r, int g, int b,
        double ref
) {

    if (grid.size() % r_size != 0 || grid.empty()) {
        std::cerr << "grid must be a multiple of x_size" << std::endl;
        return {};
    }

    int width = img.cols;
    int height = img.rows;

    size_t grid_size_y = grid.size() / r_size;

    double x_range = max_x - min_x;
    double y_range = max_y - min_y;

    double r_min = std::min(std::min(
            sqrt(min_x * min_x + min_y * min_y),
            sqrt(max_x * max_x + max_y * max_y)
    ), std::min(
            sqrt(max_x * max_x + min_y * min_y),
            sqrt(min_x * min_x + max_y * max_y)
    ));
    double r_max = std::max(std::max(
            sqrt(min_x * min_x + min_y * min_y),
            sqrt(max_x * max_x + max_y * max_y)
    ), std::max(
            sqrt(max_x * max_x + min_y * min_y),
            sqrt(min_x * min_x + max_y * max_y)
    ));
    double r_range = r_max - r_min;




    int pad = 50;

    auto to_pixel = [&](double x_val, double y_val) -> cv::Point {
        int px = pad + static_cast<int>(((x_val - min_x) / x_range) * (width - 2 * pad));
        int py = height - pad - static_cast<int>(((y_val - min_y) / y_range) * (height - 2 * pad));
        return {px, py};
    };

    for (size_t i = 0; i < grid_size_y; ++i) {
        for (size_t j = 0; j < r_size; ++j) {
            size_t idx = i * r_size + j;
            if (grid[idx] == 1) continue;



//            double rate = abs(grid[idx]) / ref;
//
//            int nr = r * rate + (1 - rate) * 255;
//            int ng = g * rate + (1 - rate) * 255;
//            int nb = b * rate + (1 - rate) * 255;

            double radius = r_min + r_range * j / (r_size - 1);
            double theta = min_theta + (max_theta - min_theta) * i / (grid_size_y - 1);
            double x = radius * cos(theta);
            double y = radius * sin(theta);
            cv::Point p = to_pixel(x, y);
            cv::circle(img, p, 1, cv::Scalar(b, g, r), cv::FILLED); // 경계점 표시

        }
    }

    return img;
}


class DrawImplicitPolarCommand : public ICommand {
public:
    size_t mat, in, rIdx;
    double ref;
    int r, g, b;
    DrawImplicitPolarCommand(size_t mat, size_t in, size_t rIdx, double ref, int r, int g, int b) : mat(mat), in(in), rIdx(rIdx), ref(ref), r(r), g(g), b(b) {}
    static unique_ptr<DrawImplicitPolarCommand> fromString(const string& str) {
        std::regex pattern(R"((\d+) (\d+) ([+-]?\d*\.?\d+) (\d+) (\d+) (\d+))");
        std::smatch match;
        if (std::regex_match(str, match, pattern)) {
            return std::make_unique<DrawImplicitPolarCommand>(
                std::stoul(match[1]), std::stoul(match[2]), std::stoul(match[3]), std::stod(match[4]),
                std::stoi(match[5]), std::stoi(match[6]), std::stoi(match[7])
            );
        } else {
            std::cerr << "Invalid command string: " << str << std::endl;
            throw std::invalid_argument("Invalid command string format");
        }
    }

    AUTOFILL_COMMAND()

    bool run(DeviceInfo device, vector<pair<cv::Mat, MatInfo>> mats, vector<double*> space, Scaler* scalers, size_t size_space, size_t size_scalers) override {
        if (mat >= space.size() || in >= space.size()) {
            std::cerr << "Invalid indices for DrawImplicitPolarCommand." << std::endl;
            return false;
        }
        vector<double> grid(size_space);
        cudaMemcpy(grid.data(), space[in], size_space * sizeof(double), cudaMemcpyDeviceToHost);

        Scaler scaler_r{};
        cudaMemcpy(&scaler_r, &scalers[rIdx], sizeof(Scaler), cudaMemcpyDeviceToHost);

        cv::Mat result = drawImplicit(mats[mat].first, grid, scaler_r.cnt,
                                      mats[mat].second.x_min, mats[mat].second.x_max,
                                      mats[mat].second.y_min, mats[mat].second.y_max,
                                      0, 2 * M_PI,
                                      r, g, b, ref);


        return true;
    }

    string exp() const override {
        return "DrawImplicitPolarCommand(mat=" + std::to_string(mat) +
               ", in=" + std::to_string(in) +
               ", rIdx=" + std::to_string(rIdx) +
               ", ref=" + std::to_string(ref) +
               ", r=" + std::to_string(r) +
               ", g=" + std::to_string(g) +
               ", b=" + std::to_string(b) + ")";
    }
    string stmt() const override {
        return "Draw Implicit Command: " + exp();
    }
};

#endif //GRAFTER_IMPLICIT_POLAR_H