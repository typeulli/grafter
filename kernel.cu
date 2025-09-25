#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include "server/server.h"

#include <cuda_runtime.h>



#include "cxxopts.hpp"
#include <opencv2/opencv.hpp>
#include "calculate/graph.h"
#include "utils/parsers/formulaParser.h"
#include "utils/parsers/cgc.h"

#include "device/device.h"

#include "env.h"

using std::pair;
using std::unique_ptr;
using std::vector;
using std::string;


int handleCLI(int argc, char* argv[]) {
    std::cout << "Using opencv version: " << CV_VERSION << std::endl;

    cxxopts::Options options("Grafter", "A graphing tool for mathematical formulas");

    options.add_options()
            ("i,input", "Input cgc file name", cxxopts::value<std::string>())
            ("f,formula", "Formula to calculate", cxxopts::value<std::string>())
//            ("t,type", "Type of the formula", cxxopts::value<std::string>()->default_value("real"))
            ("v,verbose", "Verbose output")
            ("s,summary", "Print summary of the graph")
            ("w,window", "Show "
                         "window with the graph")
            ("d,device", "Device to use for calculations (default: cpu:0)", cxxopts::value<std::string>()->default_value("cpu:0"))
            ("o,output", "Output file name", cxxopts::value<std::string>()->default_value("result.png"))
            ("serve", "Serve the graph as a web page")
            ("host", "Host for the web server", cxxopts::value<std::string>()->default_value("localhost"))
            ("port", "Port for the web server", cxxopts::value<int>()->default_value("8080"))
            ("version", "Print version information")
            ("h,help", "Print usage");

    options.allow_unrecognised_options();

    auto result = options.parse(argc, argv);

    if (result.arguments().empty()) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (!result.unmatched().empty()) {
        std::cout << "Unrecognized options: ";
        for (const auto& opt : result.unmatched()) {
            std::cout << opt << " ";
        }
        std::cout << std::endl;
        std::cerr << options.help() << std::endl;
        return 1;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (result.count("version")) {
        std::cout << "Grafter version: " << MAJOR_VERSION << "." << MINOR_VERSION << "." << PATCH_VERSION << std::endl;
        std::cout << "SymEngine version: " << SYMENGINE_VERSION << std::endl;
    }
    DeviceInfo device = {DeviceType::CPU, 0};

    if (result.count("device")) {
        string deviceText = result["device"].as<string>();
        if (deviceText == "list") {
            std::cout << "Available devices:" << std::endl;
            auto deviceList = getDeviceList();
            for (const auto& deviceCategory : deviceList) {
                std::cout << deviceCategory.first;
                for (const auto& device : deviceCategory.second) {
                    std::cout << "|" << deviceCategory.first << ":" << device.first.ref << "/" << device.second;
                }
                std::cout << std::endl;
            }
            return 0;
        }
        device = parseDevice(deviceText);
        if (device.type == DeviceType::UNKNOWN) {
            std::cerr << "Invalid device type. Use 'list' to see available devices." << std::endl;
            return 1;
        }
        if (device.ref == -1) {
            std::cerr << "Invalid device selection. Use 'list' to see available devices." << std::endl;
            return 1;
        }
    }

    if (result.count("serve")) {
        const string host = result["host"].as<string>();
        const uchar port = result["port"].as<uchar>();
        startServer(host, port, device);
        return 0;
    }



    const bool verbose = result.count("verbose") > 0;

    Graph graph{0};

    if (result.count("formula")) {
        // graph = parseFormula(result["formula"].as<string>());
    }


    else if (result.count("input")) {
        string inputFile = "input.cgc";
        if (result.count("file")) {
            inputFile = result["file"].as<string>();
        }
        graph = fromCGCFile(inputFile);
    }




    if (graph.size == 0) {
        std::cerr << "Graph is empty. Please check your input." << std::endl;
        return 1;
    }
    if (result.count("summary")) {
        std::cout << graph.summary() << std::endl;
    }

    cv::Mat mat = graph.execute(device, verbose)[0];

    if (result.count("output")) {
        string outputFile = result["output"].as<string>();
        bool success = cv::imwrite(outputFile, mat);
        if (!success) {
            std::cerr << "Failed to save image: " << outputFile << std::endl;
        } else {
            std::cout << "Saved graph to " << outputFile << std::endl;
        }
    }

    if (result.count("window")) {
        cv::namedWindow("Graph", cv::WINDOW_AUTOSIZE);
        cv::imshow("Graph", mat);
        cv::waitKey(0);
    }
    return 0;
}