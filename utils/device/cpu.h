#include <iostream>
#include <string>


#if defined(_WIN32) || defined(_WIN64) // Windows
#include <array>
#include <intrin.h>
std::string get_cpu_name() {
    int cpuInfo[4] = { -1 };
    std::array<char, 0x40> cpuBrand{};

    __cpuid(cpuInfo, 0x80000000);
    unsigned int nExIds = cpuInfo[0];

    if (nExIds >= 0x80000004) {
        __cpuid(cpuInfo, 0x80000002);
        memcpy(cpuBrand.data(), cpuInfo, sizeof(cpuInfo));

        __cpuid(cpuInfo, 0x80000003);
        memcpy(cpuBrand.data() + 16, cpuInfo, sizeof(cpuInfo));

        __cpuid(cpuInfo, 0x80000004);
        memcpy(cpuBrand.data() + 32, cpuInfo, sizeof(cpuInfo));
    }

    return cpuBrand.data();
}

#elif defined(__linux__) // Linux
#include <fstream>
std::string get_cpu_name() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            return line.substr(line.find(":") + 2);
        }
    }
    return {};
}

#elif defined(__APPLE__) // macOS
#include <sys/types.h>
#include <sys/sysctl.h>
std::string get_cpu_name() {
    char buffer[256];
    size_t bufferlen = sizeof(buffer);
    if (sysctlbyname("machdep.cpu.brand_string", &buffer, &bufferlen, NULL, 0) == 0) {
        return std::string(buffer);
    }
    return {};
}

#else
std::string get_cpu_name() {
    return "Unknown CPU (unsupported OS)";
}
#endif