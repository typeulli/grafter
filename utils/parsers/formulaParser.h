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
    Add, Multiply, Pow, Pow_Power, Pow_Base,
    Log,
    Sin, Cos, Tan,
    ATan2
};

struct FormulaNode {
    NodeType type;
    ull argCount;
    long long vInt = 0;
    double vDouble = 0;
};

void printRCPtree(const RCP<const Basic> &node,
                const std::string &prefix = "",
                bool is_last = true)
{
    std::string label;
    try {
        label = node->__str__();
    } catch (...) {
        label = "<expr>";
    }

    std::cout << prefix;
    if (!prefix.empty()) {
        std::cout << (is_last ? "└─ " : "├─ ");
    }

    std::cout << label << "\n";

    auto args = node->get_args();
    for (size_t i = 0; i < args.size(); ++i) {
        bool last = (i + 1 == args.size());
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

bool tryParseDouble(const RCPBasic& expr, double* out) {
    if (is_a<Integer>(*expr)) {
        const Integer &intNode = down_cast<const Integer &>(*expr);
        *out = intNode.as_int();
        return true;
    }
    if (is_a<RealDouble>(*expr)) {
        const RealDouble &realNode = down_cast<const RealDouble &>(*expr);
        *out = realNode.as_double();
        return true;
    }
    if (is_a<Rational>(*expr)) {
        const Rational &ratNode = down_cast<const Rational &>(*expr);
        *out = SymEngine::eval_double(ratNode);
        return true;
    }
    return false;
}

void postorder_traversal(const RCPBasic& expr, std::vector<FormulaNode>& out, const map<string, uint>& symbols) {
    auto args = expr->get_args();
    ull argCount = args.size();

    if (is_a<Pow>(*expr) && argCount == 2) {
        double dValue;
        if (tryParseDouble(args[1], &dValue)) {
            postorder_traversal(args[0], out, symbols);
            out.push_back(FormulaNode{NodeType::Pow_Power, 1, 0, dValue});
            return;
        }
        if (tryParseDouble(args[0], &dValue)) {
            postorder_traversal(args[1], out, symbols);
            out.push_back(FormulaNode{NodeType::Pow_Base, 1, 0, dValue});
            return;
        }
    }

    for (const auto& arg : args) {
        postorder_traversal(arg, out, symbols);
    }
    if (is_a<Integer>(*expr)) {
        const Integer &intNode = down_cast<const Integer &>(*expr);
        out.push_back(FormulaNode{NodeType::Integer, argCount, intNode.as_int()});
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
            out.push_back(FormulaNode{NodeType::Symbol, argCount, it->second});
        } else {
            throw std::runtime_error("Symbol not found in symbols map: " + symNode.get_name());
        }
    } else if (is_a<Add>(*expr)) {
        for (int i = 0; i < expr->get_args().size()-1; ++i) {
            out.push_back(FormulaNode{NodeType::Add, argCount});
        }
    } else if (is_a<Mul>(*expr)) {
        for (int i = 0; i < expr->get_args().size()-1; ++i) {
            out.push_back(FormulaNode{NodeType::Multiply, argCount});
        }
    } else if (is_a<Pow>(*expr)) {
        out.push_back(FormulaNode{NodeType::Pow, argCount});
    } else if (is_a<Log>(*expr)) {
        out.push_back(FormulaNode{NodeType::Log, argCount});
    } else if (is_a<Sin>(*expr)) {
        out.push_back(FormulaNode{NodeType::Sin, argCount});
    } else if (is_a<Cos>(*expr)) {
        out.push_back(FormulaNode{NodeType::Cos, argCount});
    } else if (is_a<Tan>(*expr)) {
        out.push_back(FormulaNode{NodeType::Tan, argCount});
    } else if (is_a<ATan2>(*expr)) {
        out.push_back(FormulaNode{NodeType::ATan2, argCount});
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
            out.size(), \
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
            out.size(), \
            "<string>", \
            std::make_unique<command>(inputIndex, saveIndex) \
        }); \
        if (isLast) return maxIndex; \
        continue; \
    }
#define SYM_MACRO_3(ntype, command, ref) \
    if (node.type == NodeType::ntype) {    \
        size_t inputIndex = (*index)--;    \
        size_t saveIndex = 0; \
        if (isLast && !useAutoSave) saveIndex = save_index; \
        else saveIndex = ++(*index);    \
        UPDATE_MAX_INDEX(maxIndex, saveIndex); \
        out.emplace_back(CommandInfo{ \
            out.size(), \
            "<string>", \
            std::make_unique<command>(inputIndex, saveIndex, ref) \
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
                out.size(),
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
                out.size(),
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
                    out.size(),
                    "<string>",
                    std::make_unique<LogArgConstantCommand>(inputIndex, saveIndex, M_E)
                });
                if (isLast) return maxIndex;
                continue;
            }
            SYM_MACRO_3(Pow_Power, PowBaseConstantCommand, node.vDouble)
            SYM_MACRO_3(Pow_Base, PowPowerConstantCommand, node.vDouble)
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
                commands.size(),
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
            commands.size(),
            "<string>",
            std::make_unique<SubCommand>(leftIndex, rightIndex, ++this->index)
        });


        auto eq_source = this->index;
        auto eq_result = ++this->index;
        commands.emplace_back(CommandInfo{
            commands.size(),
            "<string>",
            std::make_unique<EqualZero2DCommand>(eq_source, eq_result, epsilon)
        });
        this->maxIndex = std::max(this->maxIndex, this->index);

        switch (fType){
            case FormulaType::PREFAB_REAL_RECT_2D: {
                commands.emplace_back(CommandInfo{
                        commands.size(),
                        "<string>",
                        std::make_unique<DrawImplicitCommand>(matId, this->index--, 0, epsilon, r, g, b)
                });
                break;
            }
            case FormulaType::PREFAB_REAL_POLAR_2D: {
                commands.emplace_back(CommandInfo{
                        commands.size(),
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

#endif //GRAFTER_PARSE_FORMULA_H