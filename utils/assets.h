
#ifndef GRAFTER_SERVER_ASSETS_H
#define GRAFTER_SERVER_ASSETS_H
#include "utils/utils.h"
string getRootPath() {
#if defined(_WIN32) || defined(_WIN64)
    char exePath[MAX_PATH];
    GetModuleFileNameA(NULL, exePath, MAX_PATH);
    std::string exeDir = std::string(exePath);
    exeDir = exeDir.substr(0, exeDir.find_last_of("\\/")); // exe 폴더
    return exeDir;
#else
    return "";
#endif
}

string getAssetPath(string filename) {

    if (std::filesystem::exists("assets")) {
        return getRootPath() + "/assets/" + filename;
    }
    return getRootPath() + "/" + filename;
}

string loadAssetFile(string filename) {

    std::ifstream file(getAssetPath(filename));

    string dataIndexHtml;
    if (file.is_open()) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        dataIndexHtml = buffer.str();
    } else {
        printf(("WARNING: " + filename + " is missing.\n").c_str());
    }
    file.close();
    return dataIndexHtml;
}


#endif //GRAFTER_SERVER_ASSETS_H
