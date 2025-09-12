#ifndef UTILS_FORMATS_H
#define UTILS_FORMATS_H

#include "utils.h"
string memory(ull size) {
    if (size < 1024) {
        return to_string(size) + " B";
    }
    if (size < 1024 * 1024) {
        return to_string(size / 1024) + " KB";
    }
    if (size < 1024 * 1024 * 1024) {
        return to_string(size / (1024 * 1024)) + " MB";
    }
    return to_string(size / (1024 * 1024 * 1024)) + " GB";
}
#endif //UTILS_FORMATS_H