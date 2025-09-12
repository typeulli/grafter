#ifndef GRAFTER_DEVICE_H
#define GRAFTER_DEVICE_H
#include "cpu.h"
#include "../utils/formats.h"
#include <map>
#include <cuda_runtime.h>


using std::map;


enum class DeviceType {
    UNKNOWN = 0,
    CPU = 1,
    CUDA = 2
};


struct DeviceInfo {
    DeviceType type;
    int ref;
};

map<string, vector<pair<DeviceInfo, string>>> getDeviceList() {
    map<string, vector<pair<DeviceInfo, string>>> devices;
    devices["cpu"] = {{DeviceInfo{DeviceType::CPU, 0}, get_cpu_name()}};

    devices["cuda"] = {};
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error == cudaSuccess) {
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, i);
            devices["cuda"].emplace_back(DeviceInfo{DeviceType::CUDA, i}, prop.name);
        }
    }

    return devices;
}

extern DeviceInfo parseDevice(const string& deviceText) {

    if (deviceText.length() >= 5 && deviceText.substr(0, 5) == "cuda:") {
        uchar deviceId = std::stoi(deviceText.substr(5));
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceId < 0 || deviceId >= deviceCount) {
            return {DeviceType::CUDA, -1};
        }
        cudaSetDevice(deviceId);
        return {DeviceType::CUDA, deviceId};
    } else if (deviceText.length() >= 3 && deviceText.substr(0, 3) == "cpu") {
        return {DeviceType::CPU, 0};
    }
    return {DeviceType::UNKNOWN, 0};
}

extern string getDeviceJson() {


    string dataDeviceJson = "{";
    auto deviceList = getDeviceList();
    for (const auto& deviceCategory : deviceList) {
        dataDeviceJson += "\"" + deviceCategory.first + "\":{";
        for (const auto& device : deviceCategory.second) {
            dataDeviceJson += "\"" + deviceCategory.first + ":" + to_string(device.first.ref) + "\":\"" + device.second + "\",";
        }
        if (dataDeviceJson.back() == ',') dataDeviceJson.pop_back();
        dataDeviceJson += "},";
    }
    if (dataDeviceJson.back() == ',') dataDeviceJson.pop_back();
    dataDeviceJson += "}";

    return dataDeviceJson;
}

#endif //GRAFTER_DEVICE_H
