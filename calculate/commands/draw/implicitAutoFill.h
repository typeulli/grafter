#ifndef GRAFTER_IMPLICIT_H
#define GRAFTER_IMPLICIT_H


#include <memory>
#include "command.h"
cv::Mat drawImplicit(
        cv::Mat& img,
        const std::vector<double>& grid,
        size_t x_size,
        double min_x, double max_x,
        double min_y, double max_y,
        int r, int g, int b,
        double ref
) {

    if (grid.size() % x_size != 0 || grid.empty()) {
        std::cerr << "grid must be a multiple of x_size" << std::endl;
        return {};
    }

    int width = img.cols;
    int height = img.rows;

    size_t grid_size_y = grid.size() / x_size;

    double x_range = max_x - min_x;
    double y_range = max_y - min_y;

    double x_gap = x_range / (x_size - 1);
    double y_gap = y_range / (grid_size_y - 1);

    int pad = 50;


    vector<pair<int, int>> positions;
    for (size_t i = 0; i < grid_size_y; ++i) {
        for (size_t j = 0; j < x_size; ++j) {
            size_t idx = i * x_size + j;
            if (grid[idx]== 0) continue;


            double x = min_x + x_gap * j;
            double y = min_y + y_gap * i;

            int px = pad + static_cast<int>(((x - min_x) / x_range) * (width - 2 * pad));
            int py = height - pad - static_cast<int>(((y - min_y) / y_range) * (height - 2 * pad));


            positions.emplace_back(px, py);
        }
    }
    if (positions.empty()) return img;

    std::sort(positions.begin(), positions.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
        return a.first < b.first;
    });

    vector<pair<int, int>> last;

    int cur_x = positions[0].first;
    size_t del_size = 0;
    for (int i = 1; i < positions.size(); ++i) {
        if (positions[i].first == cur_x) {
            last.emplace_back(positions[i]);
            img.at<cv::Vec4b>(positions[i][1], positions[i][0]) = cv::Vec4b(b, g, r, 255);
            del_size++;
        } else {
            break;
        }
    }
    if (del_size > 0) {
        positions.erase(positions.begin() + 1, positions.begin() + 1 + del_size);
    }


    return img;
}


class DrawImplicitCommand : public ICommand {
public:
    size_t mat, in, xIdx;
    double ref;
    int r, g, b;
    DrawImplicitCommand(size_t mat, size_t in, size_t xIdx, double ref, int r, int g, int b) : mat(mat), in(in), xIdx(xIdx), ref(ref), r(r), g(g), b(b) {}

    static unique_ptr<DrawImplicitCommand> fromString(const string& str) {
        std::regex pattern(R"((\d+) (\d+) ([+-]?\d*\.?\d+) (\d+) (\d+) (\d+))");
        std::smatch match;
        if (std::regex_match(str, match, pattern)) {
            return std::make_unique<DrawImplicitCommand>(
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
            std::cerr << "Invalid indices for DrawImplicitCommand." << std::endl;
            return false;
        }
        if (device.type == DeviceType::CPU) {
            vector<double> grid(size_space);
            memcpy(grid.data(), space[in], size_space * sizeof(double));
            cv::Mat result = drawImplicit(mats[mat].first, grid, scalers[xIdx].cnt,
                                          mats[mat].second.x_min, mats[mat].second.x_max,
                                          mats[mat].second.y_min, mats[mat].second.y_max,
                                          r, g, b, ref);


            return true;
        }
        else if (device.type == DeviceType::CUDA) {

            vector<double> grid(size_space);
            cudaMemcpy(grid.data(), space[in], size_space * sizeof(double), cudaMemcpyDeviceToHost);

            Scaler scaler_x{};
            cudaMemcpy(&scaler_x, &scalers[xIdx], sizeof(Scaler), cudaMemcpyDeviceToHost);

            cv::Mat result = drawImplicit(mats[mat].first, grid, scaler_x.cnt,
                                          mats[mat].second.x_min, mats[mat].second.x_max,
                                          mats[mat].second.y_min, mats[mat].second.y_max,
                                          r, g, b, ref);


            return true;
        }
        return false;
    }

    [[nodiscard]] string exp() const override {
        return "DrawImplicitCommand(mat=" + std::to_string(mat) +
               ", in=" + std::to_string(in) +
               ", xIdx=" + std::to_string(xIdx) +
               ", ref=" + std::to_string(ref) +
               ", r=" + std::to_string(r) +
               ", g=" + std::to_string(g) +
               ", b=" + std::to_string(b) + ")";
    }
    [[nodiscard]] string stmt() const override {
        return "Draw Implicit Command: " + exp();
    }
};
#endif //GRAFTER_IMPLICIT_H
