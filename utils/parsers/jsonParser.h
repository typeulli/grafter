#ifndef GRAFTER_JSONPARSER_H
#define GRAFTER_JSONPARSER_H
#include <nlohmann/json.hpp>

#include "formulaParser.h"
using json = nlohmann::json;

struct JsonConfig {
    DeviceInfo device;
    Graph graph;
};

JsonConfig parseJsonConfig(const string& jsonString, DeviceInfo defaultDevice={DeviceType::CPU, 0}) {
    DeviceInfo device = defaultDevice;
    vector<MatInfo> mats;
    vector<Scaler> scalers;
    json j;
    try {
        j = json::parse(jsonString);
    } catch (const json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        throw;
    }



    if (j.contains("device") && j["device"].is_string()) {
        device = parseDevice(j["device"].get<string>());
        if (device.type == DeviceType::UNKNOWN) {
            device = defaultDevice;
        }
    }



    FormulaType::fType fType = FormulaType::PREFAB_REAL_RECT_2D;
    if (j.contains("dtype") && j["dtype"].is_string()) {
        string dtypeStr = j["dtype"].get<string>();
        if (dtypeStr == "real"){
            fType = FormulaType::setDataType(fType, FormulaType::DATA_REAL);
        } else if (dtypeStr == "complex") {
            fType = FormulaType::setDataType(fType, FormulaType::DATA_COMPLEX);
        }
    }


    if (j.contains("coord") && j["coord"].is_string()) {
        string coordStr = j["coord"].get<string>();
        if (coordStr == "rect") {
            fType = FormulaType::setCoordType(fType, FormulaType::COORD_RECT);
        } else if (coordStr == "polar") {
            fType = FormulaType::setCoordType(fType, FormulaType::COORD_POLAR);
        }
    }

    if (j.contains("dimention") && j["dimention"].is_string()) {
        string dimStr = j["dimention"].get<string>();
        if (dimStr == "2d") {
            fType = FormulaType::setDimType(fType, FormulaType::DIM_2D);
        }
    }



    if (j.contains("mats") && j["mats"].is_array()) {
        for (const auto& matJson : j["mats"]) {
            if (matJson.contains("width") && matJson.contains("height") &&
                matJson.contains("x_min") && matJson.contains("x_max") &&
                matJson.contains("y_min") && matJson.contains("y_max")) {
                MatInfo mat{
                    matJson["width"].get<int>(),
                    matJson["height"].get<int>(),
                    matJson["x_min"].get<double>(),
                    matJson["x_max"].get<double>(),
                    matJson["y_min"].get<double>(),
                    matJson["y_max"].get<double>()
                };
                mats.push_back(mat);
            }
        }
    }


    if (j.contains("scalers") && j["scalers"].is_array()) {
        for (const auto& scalerJson : j["scalers"]) {
            if (scalerJson.contains("type") && scalerJson["type"].is_string()) {
                string type = scalerJson["type"].get<string>();


                if (type == "linspace" &&
                    scalerJson.contains("start") && scalerJson.contains("end") && scalerJson.contains("cnt")) {
                    double start = scalerJson["start"].get<double>();
                    double end = scalerJson["end"].get<double>();
                    int cnt = scalerJson["cnt"].get<int>();
                    if (cnt <= 0) {
                        std::cerr << "Invalid cnt value in scaler: " << cnt << std::endl;
                        continue;
                    }
                    Scaler scaler = fromLinspace(start, end, cnt);
                    scalers.push_back(scaler);
                }
            }
        }
    }


    map<string, uint> symbols;
    symbols.emplace("x", 0);
    symbols.emplace("y", 1);

    for (const auto& graphJson : j["graph"]) {
        for (const auto& expr : graphJson["expr"]) {
            RCPBasic parsed = parseFormulaExpression(expr.get<string>(), fType);
            applyNewSymbols(parsed, symbols);
        }
    }

    FormulaParser parser(symbols, mats, scalers);

    for (const auto& graphJson : j["graph"]) {
        string type = graphJson["type"].get<string>();
        if (type == "plot") {
            ull matId = graphJson["mat"].get<ull>();
            tuple<int, int, int> color = {
                graphJson["color"][0].get<int>(),
                graphJson["color"][1].get<int>(),
                graphJson["color"][2].get<int>()
            };
            RCPBasic lwh = parseFormulaExpression(graphJson["expr"][0].get<string>(), fType);
            RCPBasic rwh = parseFormulaExpression(graphJson["expr"][1].get<string>(), fType);
            parser.addGraph(lwh, rwh, fType, matId, std::get<0>(color), std::get<1>(color), std::get<2>(color));
        } else if (type == "var") {
            string varName = graphJson["name"].get<string>();
            RCPBasic expr = parseFormulaExpression(graphJson["expr"][0].get<string>(), fType);
            parser.addVar(varName, expr);
        }
    }


    return {
        device, parser.build()
    };;
}

vector<cv::Mat> runJsonString(const string& jsonString, DeviceInfo defaultDevice) {
    JsonConfig config = parseJsonConfig(jsonString, defaultDevice);
    return config.graph.execute(config.device, false);
}
extern "C" void* extern_runJsonString(const char* jsonString, DeviceInfo defaultDevice) {
    return new vector<cv::Mat>(runJsonString(string(jsonString), defaultDevice));
}

#endif //GRAFTER_JSONPARSER_H