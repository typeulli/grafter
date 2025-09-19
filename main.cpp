#include <cxxopts.hpp>
#include <vector>
#include <iostream>
#include <ostream>
#include <string>
#include <nlohmann/json.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/mat.hpp>
#include <symengine/symengine_config.h>
#include <webview/webview.h>

#include "env.h"
#include "httplib.h"
using std::string;
using std::cout;
using std::endl;
using std::vector;

enum class DeviceType {
    UNKNOWN = 0,
    CPU = 1,
    CUDA = 2
};


struct DeviceInfo {
    DeviceType type;
    int ref;
};
extern DeviceInfo parseDevice(const string& deviceText);
extern bool startServer(const string& host, int port, DeviceInfo device_default, bool verbose);
extern "C" void* extern_runJsonString(const char* jsonString, DeviceInfo defaultDevice);
vector<cv::Mat> runJsonStringWrapper(const string& jsonString, DeviceInfo defaultDevice) {
    return *(vector<cv::Mat>*)extern_runJsonString(jsonString.c_str(), defaultDevice);
}
extern string getDeviceJson();





#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <tlhelp32.h>

DWORD GetParentProcessID()
{
    DWORD parentPID = 0;
    DWORD currentPID = GetCurrentProcessId();

    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnapshot == INVALID_HANDLE_VALUE)
        return 0;

    PROCESSENTRY32 pe;
    pe.dwSize = sizeof(PROCESSENTRY32);

    if (Process32First(hSnapshot, &pe)) {
        do {
            if (pe.th32ProcessID == currentPID) {
                parentPID = pe.th32ParentProcessID;
                break;
            }
        } while (Process32Next(hSnapshot, &pe));
    }

    CloseHandle(hSnapshot);
    return parentPID;
}
#endif

int main(int argc, char* argv[]) {
#if defined(_WIN32) || defined(_WIN64)
    HWND hWnd = GetConsoleWindow();
    HWND hWndFore = GetForegroundWindow();
    DWORD parentPID = GetParentProcessID();
    DWORD fgPID;
    GetWindowThreadProcessId(hWndFore, &fgPID);
    if (hWndFore != nullptr && parentPID != fgPID) {
        ShowWindow(hWnd, SW_HIDE);
    }
#endif



    cxxopts::Options options("Grafter", "A graphing tool for mathematical formulas");

    options.add_options()
            ("i,input", "Input cgc file name", cxxopts::value<std::string>())
            ("v,verbose", "Verbose output")
            ("s,summary", "Print summary of the graph")
            ("w,window", "Show window with the graph")
            ("d,device", "Device to use for calculations (default: cpu:0)", cxxopts::value<std::string>()->default_value("cpu:0"))
            ("o,output", "Output file name", cxxopts::value<std::string>()->default_value("result.png"))
            ("serve", "Serve the graph as a web page")
            ("host", "Host for the web server", cxxopts::value<std::string>()->default_value("localhost"))
            ("port", "Port for the web server", cxxopts::value<int>()->default_value("8080"))
            ("version", "Print version information")
            ("h,help", "Print usage")
            ("console", "Keep console window open");

    options.allow_unrecognised_options();

    auto result = options.parse(argc, argv);


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
        std::cout << "Opencv version: " << CV_VERSION << std::endl;
    }
    DeviceInfo device = {DeviceType::CPU, 0};

    if (result.count("device")) {
        string deviceText = result["device"].as<string>();
        if (deviceText == "list") {
            std::cout << "Available devices:" << std::endl;
            auto deviceList = getDeviceJson();
            cout << deviceList << std::endl;
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

    bool verbose = result.count("verbose") > 0;

    if (result.count("serve")) {
        const string host = result["host"].as<string>();
        const int port = result["port"].as<int>();
        startServer(host, port, device, verbose);
        return 0;
    }



    char exePath[MAX_PATH];
    GetModuleFileNameA(NULL, exePath, MAX_PATH);
    std::string exeDir = std::string(exePath);
    exeDir = exeDir.substr(0, exeDir.find_last_of("\\/")); // exe 폴더


    webview::webview w(true, nullptr);
    w.set_title("Grafter");
    w.set_size(800, 600, WEBVIEW_HINT_NONE);
    w.bind("__bind_callGrafter", [](const string &s) -> string {
        nlohmann::json j = nlohmann::json::parse(s);
        vector<cv::Mat> mats = runJsonStringWrapper(j[0].get<std::string>(), {DeviceType::CPU, 0});
        std::vector<uchar> buffer;
        if (!cv::imencode(".png", mats[0], buffer)) {
            return "error: Failed to encode image";
        }
        string data(buffer.begin(), buffer.end());
        string jstring = nlohmann::json::object({
            {"data", string("data:image/png;base64,") + httplib::detail::base64_encode(data)}
        }).dump();
        return jstring;
    });
    w.bind("__bind_getGrafterDevices", [](const string &s) {
        return getDeviceJson();
    });


    HWND hwnd = static_cast<HWND>(w.window().value());


    string icoPath = exeDir + "/grafter.ico";
    cout << icoPath << endl;
    auto hIcon = static_cast<HICON>(LoadImageW(
        nullptr,
        std::wstring(icoPath.begin(), icoPath.end()).c_str(),
        IMAGE_ICON,
        32, 32,
        LR_LOADFROMFILE
    ));

    if (hIcon) {
        SendMessage(hwnd, WM_SETICON, ICON_SMALL, (LPARAM)hIcon);
        SendMessage(hwnd, WM_SETICON, ICON_BIG, (LPARAM)hIcon);
    }

    cout << "Loading webview from: file:///" << exeDir + "/index.html" << endl;
    w.navigate("file:///" + exeDir + "/index.html");
    w.run();
}

