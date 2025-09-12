//
// Created by USER on 25. 8. 19..
//

#ifndef GRAFTER_SERVER_ASSETS_H
#define GRAFTER_SERVER_ASSETS_H
#include "utils/utils.h"


string loadAssetFile(string filename) {
    std::ifstream file(filename);

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
