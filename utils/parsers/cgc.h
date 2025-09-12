#ifndef GRAFTER_PARSER_CGC_H
#define GRAFTER_PARSER_CGC_H
#include "calculate/graph.h"

Graph parseCGC(function<bool(string&)> getline) {



    string line;
    ull line_index = 0;


    getline(line);
    std::regex pattern_setting_size(R"(size (\d+))");
    std::smatch match_setting_size;
    if (!std::regex_match(line, match_setting_size, pattern_setting_size)) {
        std::cerr << "Invalid file format: " << line << std::endl;
        throw std::invalid_argument("Invalid file format");
    }
    size_t size = std::stoul(match_setting_size[1]);




    getline(line);
    line_index++;
    std::regex pattern_setting_mat(R"(mat (\d+))");
    std::smatch match_setting_mat;
    if (!std::regex_match(line, match_setting_mat, pattern_setting_mat)) {
        std::cerr << "Invalid file format: " << line << std::endl;
        throw std::invalid_argument("Invalid file format");
    }
    size_t mat_count = std::stoul(match_setting_mat[1]);


    Graph graph(size);

    for (size_t i = 0; i < mat_count; ++i) {
        getline(line);
        line_index++;
        graph.mats.push_back(parseMatInfo(line));
    }








    getline(line);
    line_index++;
    std::regex pattern_setting_axis(R"(axis (\d+))");
    std::smatch match_setting_axis;
    if (!std::regex_match(line, match_setting_axis, pattern_setting_axis)) {
        std::cerr << "Invalid file format: " << line << std::endl;
        throw std::invalid_argument("Invalid file format");
    }
    size_t axis_size = std::stoul(match_setting_axis[1]);





    for (size_t i = 0; i < axis_size; ++i) {
        getline(line);
        line_index++;
        graph.scalers.push_back(parseScaler(line));
    }


    vector<CommandInfo> commands;

    // loop all left lines
    while (getline(line)) {
        line_index++;
        if (line.empty() || line[0] == '#') {
            continue; // skip empty lines and comments
        }
        string commandType = line.substr(0, line.find(' '));
        auto search = generators.find(commandType);
        if (search == generators.end()) {
            std::cerr << "Unknown command type: " << commandType << std::endl;
            throw std::invalid_argument("Unknown command type");
        }
        unique_ptr<ICommand> command = search->second(line.substr(commandType.length() + 1));
        graph.commands.push_back({line_index, search->first, std::move(command)});
    }

    return graph;
}




Graph fromCGCFile(const string& filepath) {
    std::ifstream file(filepath);

    if(!file.is_open()){
        std::cerr << "Failed to open file: " << filepath << std::endl;
        throw std::system_error (errno, std::generic_category(), "Failed to open file");
    }

    function<bool(string&)> getline = [&file](string& line) {
        return static_cast<bool>(std::getline(file, line));
    };
    Graph graph = parseCGC(getline);
    file.close();
    return graph;
}

#endif //GRAFTER_PARSER_CGC_H