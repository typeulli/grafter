#ifndef GRAFTER_DRAW_H
#define GRAFTER_DRAW_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>


struct MatInfo {
    int width;
    int height;
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    MatInfo(int width, int height, double x_min, double x_max, double y_min, double y_max)
            : width(width), height(height), x_min(x_min), x_max(x_max), y_min(y_min), y_max(y_max) {}
};

string MatInfoString(const MatInfo& mat) {
    return "w" + std::to_string(mat.width) + "h" + std::to_string(mat.height) +
           "x" + std::to_string(mat.x_min) + "~" + std::to_string(mat.x_max) +
           "y" + std::to_string(mat.y_min) + "~" + std::to_string(mat.y_max);
}

MatInfo parseMatInfo(const std::string& line) {

    std::regex pattern_mat(R"(\d+@\d+@[+-]?\d+(\.\d+)?@[+-]?\d+(\.\d+)?@[+-]?\d+(\.\d+)?@[+-]?\d+(\.\d+)?)");
    std::smatch match_mat;
    if (!std::regex_match(line, match_mat, pattern_mat)) {
        std::cerr << "Invalid matrix format: " << line << std::endl;
        throw std::invalid_argument("Invalid matrix format");
    }
    vector<string> splited_by_space(1, "");
    for (const auto& part : line) {
        if (part == '@') {
            splited_by_space.emplace_back("");
            continue;
        }
        splited_by_space[splited_by_space.size()-1] += part;
    }

    int width = std::stoi(splited_by_space[0]);
    int height = std::stoi(splited_by_space[1]);
    double x_min = std::stod(splited_by_space[2]);
    double x_max = std::stod(splited_by_space[3]);
    double y_min = std::stod(splited_by_space[4]);
    double y_max = std::stod(splited_by_space[5]);
    return {
        width, height, x_min, x_max, y_min, y_max
    };
}

inline double rescale(double x, double old_min, double old_max, double new_min, double new_max) {
    return (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min;
}

void plotAxis(
    cv::Mat& img,
    const std::string& x_label,
    const std::string& y_label,
    double min_x, double max_x,
    double min_y, double max_y
) {
    int width = img.cols;
    int height = img.rows;
//   axis_x_height - 0 : height - axis_x_height = 0 - min_y : max_y - 0
//   axis_x_height *max_y= - min_y*height + min_y* axis_x_height
//  axis_x_height = min_y * height / (min_y + max_y)
    int axis_x_height = static_cast<int>(rescale(0, min_y, max_y, 0, height));
    // 축 그리기
    cv::line(img,
                cv::Point(0, height - axis_x_height),
                cv::Point(width, height - axis_x_height),
                cv::Scalar(0, 0, 0, 255), 1);
    int axis_y_width = static_cast<int>(rescale(0, min_x, max_x, 0, width));
    cv::line(img,
                cv::Point(axis_y_width, 0),
                cv::Point(axis_y_width, height),
                cv::Scalar(0, 0, 0, 255), 1);
}


#endif // GRAFTER_DRAW_H