#pragma once
#include "utils/parsers/jsonParser.h"
#include "../utils/utils.h"
#include "../utils/device/device.h"
#include "httplib.h"

#include <opencv2/opencv.hpp>
#include "assets.h"


void mount(httplib::Server &server, const string &mount_point, const string &filepath, const string &content_type) {
    string data = loadAssetFile(filepath);
    if (!data.empty()) {
        server.Get(mount_point, [data, content_type](const httplib::Request& req, httplib::Response& res) {
            res.set_content(data, content_type);
        });
    }
}


extern bool startServer(const string& host, int port, DeviceInfo device_default) {
    httplib::Server svr;

    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.status = 204;
    });

    mount(svr, "/", "index.html", "text/html");
    mount(svr, "/sortable.min.js", "sortable.min.js", "application/javascript");


    std::ifstream file_ico("grafter.ico", std::ios::binary);
    if (file_ico.is_open()) {
        std::ostringstream buffer;
        buffer << file_ico.rdbuf();
        file_ico.close();

        std::string data = buffer.str();
        svr.Get("/grafter.ico", [data](const httplib::Request&, httplib::Response& res) {
            res.set_content(data.data(), data.size(), "image/x-icon");
        });
    } else {
        printf("Could not open ico file\n");
    }

    svr.Get("/devices", [](const httplib::Request&, httplib::Response& res) {

        auto dataDeviceJson = getDeviceJson();

        res.set_content(dataDeviceJson, "application/json");
    });

    svr.Post("/draw", [device_default](const httplib::Request& req, httplib::Response& res) {

        try {


            string body = req.body;
            vector<cv::Mat> mats;
            try {
                mats = runJsonString(body, device_default);

            }
            catch (const std::exception& e) {
                res.status = 400;
                res.set_content(std::string("Invalid formula: ") + e.what(), "text/plain");
                return;
            }

            std::vector<uchar> buffer;
            if (!cv::imencode(".png", mats[0], buffer)) {
                res.status = 500;
                res.set_content("Failed to encode image", "text/plain");
                return;
            }
            res.set_content(reinterpret_cast<const char*>(buffer.data()), buffer.size(), "image/png");

            res.set_header("Cache-Control", "no-store");
            res.status = 200;
            return;
        } catch (const std::invalid_argument& e) {
            res.status = 400;
            res.set_content(std::string("Invalid formula: ") + e.what(), "text/plain");
            return;
        }
    });


    svr.set_logger([](const httplib::Request& req, const httplib::Response& res) {
        std::cout
            << req.remote_addr << ":" << std::right << std::setw(5) << to_string(req.remote_port)
            << " " << std::right << std::setw(6) << req.method
            << " " << std::left  << std::setw(3) << res.status
            << " " << req.path;
        if (!req.params.empty()) {
            std::cout << "?";
            bool is_first = true;
            for (const auto& param : req.params) {
                if (!is_first) {
                    std::cout << "&";
                }
                is_first = false;
                std::cout << param.first << "=" << param.second;
            }
        }
        std::cout << std::endl;
    });

    std::cout << "Starting server at http://" << host << ":" << port << std::endl;
    svr.listen(host, port);
    return true;
}