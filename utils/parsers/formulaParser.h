#ifndef GRAFTER_PARSE_FORMULA_H
#define GRAFTER_PARSE_FORMULA_H


#include <memory>
#include <tuple>

#include "utils/utils.h"

#include <symengine/basic.h>
#include <symengine/symbol.h>
#include <symengine/integer.h>
#include <symengine/parser.h>
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/complex.h>
#include <symengine/real_double.h>
#include <symengine/eval_double.h>
#include "commands/__init.h"
#include "graph.h"
using std::tuple;

using namespace SymEngine;
using RCPBasic = RCP<const Basic>;

enum class NodeType {
    Integer, RealDouble, Symbol,
    Add, Multiply, Pow,
    Log,
    Sin, Cos, Tan,
    ATan2
};

struct FormulaNode {
    NodeType type;
    ull argCount;
    long long vInt;
    double vDouble;
};

void printRCPtree(const RCP<const Basic> &node,
                const std::string &prefix = "",
                bool is_last = true)
{
    // 노드 레이블: 표현식 문자열 (예: "x", "Add(x, y)", "sin(z)")
    std::string label;
    try {
        label = node->__str__(); // 대부분의 SymEngine 버전에서 사용 가능
    } catch (...) {
        label = "<expr>";
    }

    // 가지 그리기 문자
    std::cout << prefix;
    if (!prefix.empty()) {
        std::cout << (is_last ? "└─ " : "├─ ");
    }

    std::cout << label << "\n";

    // 자식들 가져오기
    auto args = node->get_args(); // vec_basic 형태, 각 원소는 RCP<const Basic>
    for (size_t i = 0; i < args.size(); ++i) {
        bool last = (i + 1 == args.size());
        // 다음 prefix 결정: 현재이면서 마지막이면 공백, 아니면 이어지는 세로선 표시
        std::string next_prefix = prefix;
        if (!prefix.empty()) next_prefix += (is_last ? "   " : "│  ");
        else next_prefix = (is_last ? "   " : "│  ");

        printRCPtree(args[i], next_prefix, last);
    }
}


void applyNewSymbols(const RCPBasic& expr, map<string, uint>& symbols) {
    if (is_a<Symbol>(*expr)) {
        const Symbol &sym = down_cast<const Symbol &>(*expr);
        if (symbols.find(sym.get_name()) == symbols.end()) {
            symbols.emplace(sym.get_name(), symbols.size());
        }
    } else {
        for (const auto& arg : expr->get_args()) {
            applyNewSymbols(arg, symbols);
        }
    }
}

void postorder_traversal(const RCPBasic& expr, std::vector<FormulaNode>& out, const map<string, uint>& symbols) {
    for (const auto& arg : expr->get_args()) {
        postorder_traversal(arg, out, symbols);
    }
    ull argCount = expr->get_args().size();
    if (is_a<Integer>(*expr)) {
        const Integer &intNode = down_cast<const Integer &>(*expr);
        out.push_back(FormulaNode{NodeType::Integer, argCount, intNode.as_int(), 0.0});
    } else if (is_a<RealDouble>(*expr)) {
        const RealDouble &realNode = down_cast<const RealDouble &>(*expr);
        out.push_back(FormulaNode{NodeType::RealDouble, argCount, 0, realNode.as_double()});
    } else if (is_a<Rational>(*expr)) {
        const Rational &ratNode = down_cast<const Rational &>(*expr);
        double value = SymEngine::eval_double(ratNode);
        out.push_back(FormulaNode{NodeType::RealDouble, argCount, 0, value});
    } else if (is_a<Symbol>(*expr)) {
        const Symbol &symNode = down_cast<const Symbol &>(*expr);
        auto it = symbols.find(symNode.get_name());
        if (it != symbols.end()) {
            out.push_back(FormulaNode{NodeType::Symbol, argCount, it->second, 0.0});
        } else {
            throw std::runtime_error("Symbol not found in symbols map: " + symNode.get_name());
        }
    } else if (is_a<Add>(*expr)) {
        for (int i = 0; i < expr->get_args().size()-1; ++i) {
            out.push_back(FormulaNode{NodeType::Add, argCount, 0, 0.0});
        }
    } else if (is_a<Mul>(*expr)) {
        for (int i = 0; i < expr->get_args().size()-1; ++i) {
            out.push_back(FormulaNode{NodeType::Multiply, argCount, 0, 0.0});
        }
    } else if (is_a<Pow>(*expr)) {
        out.push_back(FormulaNode{NodeType::Pow, argCount, 0, 0.0});
    } else if (is_a<Log>(*expr)) {
        out.push_back(FormulaNode{NodeType::Log, argCount, 0, 0.0});
    } else if (is_a<Sin>(*expr)) {
        out.push_back(FormulaNode{NodeType::Sin, argCount, 0, 0.0});
    } else if (is_a<Cos>(*expr)) {
        out.push_back(FormulaNode{NodeType::Cos, argCount, 0, 0.0});
    } else if (is_a<Tan>(*expr)) {
        out.push_back(FormulaNode{NodeType::Tan, argCount, 0, 0.0});
    } else if (is_a<ATan2>(*expr)) {
        out.push_back(FormulaNode{NodeType::ATan2, argCount, 0, 0.0});
    } else {
        printf("Unknown node type: %s\n", expr->__str__().c_str());
    }
}

#define UPDATE_MAX_INDEX(maxIdx, newIdx) \
    maxIdx = std::max(maxIndex, newIdx);

#define SYM_MACRO_1(ntype, command) \
    if (node.type == NodeType::ntype) {    \
        size_t rightIndex = (*index)--; \
        size_t leftIndex = (*index)--;  \
        size_t saveIndex = 0; \
        if (isLast && !useAutoSave) saveIndex = save_index; \
        else saveIndex = ++(*index);    \
        UPDATE_MAX_INDEX(maxIndex, saveIndex); \
        out.emplace_back(CommandInfo{ \
            0, \
            "<string>", \
            std::make_unique<command>(leftIndex, rightIndex, saveIndex) \
        }); \
        if (isLast) return maxIndex; \
        continue; \
    }
#define SYM_MACRO_2(ntype, command) \
    if (node.type == NodeType::ntype) {    \
        size_t inputIndex = (*index)--;    \
        size_t saveIndex = 0; \
        if (isLast && !useAutoSave) saveIndex = save_index; \
        else saveIndex = ++(*index);    \
        UPDATE_MAX_INDEX(maxIndex, saveIndex); \
        out.emplace_back(CommandInfo{ \
            0, \
            "<string>", \
            std::make_unique<command>(inputIndex, saveIndex) \
        }); \
        if (isLast) return maxIndex; \
        continue; \
    }
#define SYM_MACRO_3(node, name, type) \
    if (node.get_name() == name) {         \
        size_t inputIndex = (*index)--;    \
        size_t saveIndex = 0; \
        if (isLast && !useAutoSave) saveIndex = save_index; \
        else saveIndex = ++(*index);    \
        UPDATE_MAX_INDEX(maxIndex, saveIndex); \
        out.emplace_back(CommandInfo{ \
            0, \
            "<string>", \
            std::make_unique<type>(inputIndex, saveIndex) \
        }); \
        if (isLast) return maxIndex; \
        continue; \
    }
#define SYM_MACRO_4(node, name, type) \
    if (node.get_name() == name) {         \
        size_t rightIndex = (*index)--; \
        size_t leftIndex = (*index)--;  \
        size_t saveIndex = 0; \
        if (isLast && !useAutoSave) saveIndex = save_index; \
        else saveIndex = ++(*index);    \
        UPDATE_MAX_INDEX(maxIndex, saveIndex); \
        out.emplace_back(CommandInfo{ \
            0, \
            "<string>", \
            std::make_unique<type>(leftIndex, rightIndex, saveIndex) \
        }); \
        if (isLast) return maxIndex; \
        continue; \
    }
size_t sym2cmd(const RCPBasic& expr, std::vector<CommandInfo>& out, const map<string, uint>& symbols, size_t* index, size_t save_index = 0, bool useAutoSave = true) {
    size_t maxIndex = 0;

    vector<FormulaNode> postorder;
    postorder_traversal(expr, postorder, symbols);

    for (const auto& node : postorder) {
        bool isLast = (&node == &postorder.back());

        if (node.type == NodeType::Integer || node.type == NodeType::RealDouble) {
            // convert to double
            double value = 0.0;
            if (node.type == NodeType::Integer) {
                value = static_cast<double>(node.vInt);
            } else if (node.type == NodeType::RealDouble) {
                value = node.vDouble;
            }
            size_t saveIndex = 0;
            if (isLast && !useAutoSave) saveIndex = save_index;
            else saveIndex = ++(*index);
            UPDATE_MAX_INDEX(maxIndex, saveIndex);

            out.emplace_back(CommandInfo{
                0,
                "<string>",
                std::make_unique<ConstantCommand>(saveIndex, value)
            });
            if (isLast) return maxIndex;
            continue;

        }

        if (node.type == NodeType::Symbol) {
            size_t saveIndex = 0;
            if (isLast && !useAutoSave) saveIndex = save_index;
            else saveIndex = ++(*index);
            UPDATE_MAX_INDEX(maxIndex, saveIndex);

            out.emplace_back(CommandInfo{
                0,
                "<string>",
                std::make_unique<CopyCommand>(node.vInt, saveIndex)
            });

            if (isLast) return maxIndex;
            continue;
        }
        if (node.argCount == 1) {
            if (node.type == NodeType::Log) {
                size_t inputIndex = (*index)--;
                size_t saveIndex = 0; \
                if (isLast && !useAutoSave) saveIndex = save_index;
                else saveIndex = ++(*index);
                UPDATE_MAX_INDEX(maxIndex, saveIndex);
                out.emplace_back(CommandInfo{
                    0,
                    "<string>",
                    std::make_unique<LogArgConstantCommand>(inputIndex, saveIndex, M_E)
                });
                if (isLast) return maxIndex;
                continue;
            }
            SYM_MACRO_2(Sin, SinCommand)
            SYM_MACRO_2(Cos, CosCommand)
            SYM_MACRO_2(Tan, TanCommand)
        } else if (node.argCount == 2) {
            SYM_MACRO_1(Add, AddCommand)
            SYM_MACRO_1(Multiply, MulCommand)
            SYM_MACRO_1(Pow, PowCommand)
            SYM_MACRO_1(Log, LogCommand)
            SYM_MACRO_1(ATan2, ATan2Command)
        }

    }
    return maxIndex;
}

namespace FormulaType {
    typedef unsigned long long fType;


    const fType MASK_DATA_TYPE = 0b111'000'000;
    const fType MASK_COORD_TYPE = 0b000'111'000;
    const fType MASK_DIM_TYPE = 0b000'000'111;

    const fType DATA_REAL = 0b000'000'000;
    const fType DATA_COMPLEX = 0b001'000'000;

    const fType COORD_RECT = 0b000'000'000;
    const fType COORD_POLAR = 0b000'001'000;

    const fType DIM_2D = 0b000'000'000;

    const fType PREFAB_REAL_RECT_2D = DATA_REAL | COORD_RECT | DIM_2D;
    const fType PREFAB_COMPLEX_RECT_2D = DATA_COMPLEX | COORD_RECT | DIM_2D;
    const fType PREFAB_REAL_POLAR_2D = DATA_REAL | COORD_POLAR | DIM_2D;

    inline fType set(fType value, fType mask, fType newBits) {
        return (value & ~mask) | (newBits & mask);
    }
    inline fType setDataType(fType value, fType newBits) {
        return set(value, MASK_DATA_TYPE, newBits);
    }
    inline fType setCoordType(fType value, fType newBits) {
        return set(value, MASK_COORD_TYPE, newBits);
    }
    inline fType setDimType(fType value, fType newBits) {
        return set(value, MASK_DIM_TYPE, newBits);
    }
}



RCPBasic __substitutePolar(RCPBasic& expr) {
    map_basic_basic subs_map;
    subs_map[symbol("r")]     = parse("sqrt(x**2+y**2)");
    subs_map[symbol("theta")] = atan2(symbol("y"), symbol("x"));

    return expr->subs(subs_map);
}




RCPBasic parseFormulaExpression(const string& expression, FormulaType::fType type) {
    RCPBasic parsed;
    try {
        parsed = parse(expression);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing expression: " << expression << "\n" << e.what() << std::endl;
        throw;
    }




    if ((type & FormulaType::MASK_COORD_TYPE) == FormulaType::COORD_POLAR) {
        return __substitutePolar(parsed);
    }
    return parsed;
}


class FormulaParser {
    map<string, uint> symbols;
    vector<MatInfo> mats;
    vector<Scaler> scalers;

    size_t maxIndex;
    size_t index;
    double epsilon = 0.01;
    vector<CommandInfo> commands;
public:
    FormulaParser(map<string, uint> symbols, vector<MatInfo> mats, vector<Scaler> scalers) {
        this->index = symbols.size() - 1;
        this->maxIndex = index;


        this->symbols = std::move(symbols);
        this->mats = std::move(mats);
        this->scalers = std::move(scalers);

        size_t _dim = 0;
        double min_sep = this->scalers[0].sep;
        for (Scaler axis : this->scalers) {
            commands.emplace_back(CommandInfo{
                0,
                "<string>",
                std::make_unique<VarCommand>(_dim, _dim)
            });
            min_sep = std::min(min_sep, axis.sep);
            _dim++;
        }
        const double k = 10000.0;
        this->epsilon = min_sep * k;
    }

    void addVar(const string &var, RCPBasic expr) {
        ull idx = symbols.at(var);
        this->maxIndex = std::max(maxIndex, sym2cmd(expr, commands, symbols, &this->index, idx, false));
    }


    void addGraph(RCPBasic lwh, RCPBasic rwh, FormulaType::fType fType, ull matId=0, uchar r=0, uchar g=0, uchar b=0) {
        // size_t position_equal = formula.find('=');
        // string lwh = formula.substr(0, position_equal);
        // string rwh = formula.substr(position_equal + 1);
        //
        // RCPBasic parsed_lwh = parseFormulaExpression(lwh, fType);
        // RCPBasic parsed_rwh = parseFormulaExpression(rwh, fType);
        //
        // applyNewSymbols(parsed_lwh, symbols);
        // applyNewSymbols(parsed_rwh, symbols);
        auto l = sym2cmd(lwh, commands, symbols, &this->index);

        this->maxIndex = std::max(this->maxIndex,
            l
        );
        this->maxIndex = std::max(this->maxIndex,
            sym2cmd(rwh, commands, symbols, &this->index)
        );
        size_t leftIndex = this->index--;
        size_t rightIndex = this->index--;
        commands.emplace_back(CommandInfo{
                0,
                "<string>",
                std::make_unique<SubCommand>(leftIndex, rightIndex, ++this->index)
        });


        auto eq_source = this->index;
        auto eq_result = ++this->index;
        commands.emplace_back(CommandInfo{
            0,
            "<string>",
            std::make_unique<EqualZero2DCommand>(eq_source, eq_result, epsilon)
        });
        this->maxIndex = std::max(this->maxIndex, this->index);

        switch (fType){
            case FormulaType::PREFAB_REAL_RECT_2D: {
                commands.emplace_back(CommandInfo{
                        0,
                        "<string>",
                        std::make_unique<DrawImplicitCommand>(matId, this->index--, 0, epsilon, r, g, b)
                });
                break;
            }
            case FormulaType::PREFAB_REAL_POLAR_2D: {
                commands.emplace_back(CommandInfo{
                        0,
                        "<string>",
                        std::make_unique<DrawImplicitCommand>(matId, this->index--, 0, epsilon, r, g, b)
                });
                break;
            }
        }
    }
    Graph build() {
        Graph graph(this->maxIndex + 1);
        graph.commands = std::move(commands);
        graph.mats = std::move(mats);
        graph.scalers = std::move(scalers);
        return graph;
    }
};
// Graph parseFormula(const string& formula) {
//
//
//     string cleaned_formula;
//     cleaned_formula.reserve(formula.size() - std::count(formula.begin(), formula.end(), ' '));
//     for (char c : formula) {
//         if (!isspace(c)) {
//             cleaned_formula += c;
//         }
//     }
//     map<string, uint> symbols;
//     if (fType == FormulaType::RealRect) {
//         symbols.emplace("x", 0);
//         symbols.emplace("y", 1);
//     } else if (fType == FormulaType::RealPolar) {
//         symbols.emplace("r", 0);
//         symbols.emplace("theta", 1);
//     }
//
//
//
//     vector<MatInfo> mats;
//     vector<Scaler> scalers;
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//     vector<tuple<bool, long long, RCPBasic, RCPBasic, tuple<int, int, int>>> formulas;
//
//     for (const auto& line : lines) {
//         size_t position_sep = line.find('?');
//         if (position_sep == string::npos) {
//             std::cerr << "Invalid formula line: " << line << std::endl;
//             throw std::invalid_argument("Invalid formula line format");
//         }
//
//
//         string matInfo = line.substr(0, position_sep);
//         string expression = line.substr(position_sep + 1);
//
//         if (matInfo.empty()) {
//             size_t position_equal = expression.find('=');
//             string symbol = expression.substr(0, position_equal);
//             if (symbols.find(symbol) == symbols.end()) {
//                 symbols.emplace(symbol, symbols.size());
//             }
//             string value = expression.substr(position_equal + 1);
//
//
//
//             RCPBasic parsed = parseFormulaExpression(value, fType);
//             applyNewSymbols(parsed, symbols);
//
//             formulas.emplace_back(false, symbols.at(symbol), parsed, RCPBasic(), std::make_tuple(0, 0, 0));
//         }
//
//         else {
//             std::regex pattern_mat(R"((\d+)@(\d+)@(\d+)@(\d+))");
//             std::smatch match;
//             if (!std::regex_match(matInfo, match, pattern_mat)) {
//                 std::cerr << "Invalid matrix info format: " << matInfo << std::endl;
//                 throw std::invalid_argument("Invalid matrix info format");
//             }
//
//             size_t position_equal = expression.find('=');
//             string lwh = expression.substr(0, position_equal);
//             string rwh = expression.substr(position_equal + 1);
//
//             RCPBasic parsed_lwh = parseFormulaExpression(lwh, fType);
//             RCPBasic parsed_rwh = parseFormulaExpression(rwh, fType);
//
//             applyNewSymbols(parsed_lwh, symbols);
//             applyNewSymbols(parsed_rwh, symbols);
//
//             formulas.emplace_back(true, std::stoul(match[1]), parsed_lwh, parsed_rwh, std::make_tuple(
//             std::stoi(match[2]), std::stoi(match[3]), std::stoi(match[4])
//                 ));
//         }
//     }
//
//
//
//
//
//     vector<CommandInfo> commands;
//     size_t _dim = 0;
//     for (Scaler axis : scalers) {
//         commands.emplace_back(CommandInfo{
//             0,
//             "<string>",
//             std::make_unique<VarCommand>(_dim, _dim)
//         });
//         _dim++;
//     }
//
//
//     const double k = 10000.0;
//     double epsilon = 0.01;
//
//     switch (fType) {
//         case FormulaType::RealRect: {
//             Scaler x_scaler = scalers[symbols.at("x")];
//             Scaler y_scaler = scalers[symbols.at("y")];
//             epsilon = std::min(x_scaler.sep, y_scaler.sep) * k;
//         }
//     }
//
//
//
//
//
//
//
//     size_t index = symbols.size() - 1;
//
//     size_t maxIndex = index;
//     for (const auto& [is_equation, idx, lwh, rwh, color] : formulas) {
//         if (!is_equation) {
//             maxIndex = std::max(maxIndex, sym2cmd(lwh, commands, symbols, &index, idx, false));
//         } else {
//             maxIndex = std::max(maxIndex,
//                 sym2cmd(lwh, commands, symbols, &index)
//             );
//             maxIndex = std::max(maxIndex,
//                 sym2cmd(rwh, commands, symbols, &index)
//             );
//             size_t leftIndex = index--;
//             size_t rightIndex = index--;
//             commands.emplace_back(CommandInfo{
//                     0,
//                     "<string>",
//                     std::make_unique<SubCommand>(leftIndex, rightIndex, ++index)
//             });
//
//
//             auto eq_source = index;
//             auto eq_result = ++index;
//             commands.emplace_back(CommandInfo{
//                 0,
//                 "<string>",
//                 std::make_unique<EqualZero2DCommand>(eq_source, eq_result, epsilon)
//             });
//             maxIndex = std::max(maxIndex, index);
//             auto [r, g, b] = color;
//             switch (fType){
//                 case FormulaType::RealRect: {
//                     commands.emplace_back(CommandInfo{
//                             0,
//                             "<string>",
//                             std::make_unique<DrawImplicitCommand>(idx, index--, 0, epsilon, r, g, b)
//                     });
//                     break;
//                 }
//                 case FormulaType::RealPolar: {
//                     commands.emplace_back(CommandInfo{
//                             0,
//                             "<string>",
//                             std::make_unique<DrawImplicitPolarCommand>(idx, index--, 0, 0.01, r, g, b)
//                     });
//                     break;
//                 }
//             }
//         }
//     }
//
//     Graph graph(maxIndex + 1);
//     graph.commands = std::move(commands);
//     graph.mats = std::move(mats);
//     graph.scalers = std::move(scalers);
//     return graph;
//
// }

#endif //GRAFTER_PARSE_FORMULA_H