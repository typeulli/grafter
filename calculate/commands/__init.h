#ifndef CALCULATE_COMMANDS_INIT_H
#define CALCULATE_COMMANDS_INIT_H

#include "command.h"


#include <map>
#include <functional>
using std::map;
using std::function;



#include "commands/draw/explicit.h"
#include "commands/draw/implicit_polar.h"
#include "commands/draw/rgb2D.h"
#include "commands/draw/implicit.h"

#include "const.h"

#include "comp.h"


#include "commands/real/add.h"
#include "commands/real/sub.h"
#include "commands/real/mul.h"
#include "commands/real/div.h"

#include "commands/complex/utils.h"
#include "commands/complex/visual.h"

#include "commands/real/pow.h"
#include "commands/real/log.h"

#include "commands/real/trigonometric.h"

#include "commands/real/abs.h"

#include "commands/real/intf.h"

#include "diff.h"

#define addGenF(name, fn) \
    m[#name] = [](const string& s) { return fn::fromString(s); }

const static auto generators = []() {
    map<string, function<unique_ptr<ICommand>(const string&)>> m;
    addGenF(var, VarCommand);
    addGenF(copy, CopyCommand);
    addGenF(constant, ConstantCommand);

    addGenF(eq0_2d, EqualZero2DCommand);
    addGenF(eq0, EqualZeroCommand);

    addGenF(addGenF, AddCommand);
    addGenF(cadd, AddConstantCommand);

    addGenF(sub, SubCommand);
    addGenF(csub, SubConstantCommand);

    addGenF(mul, MulCommand);
    addGenF(cmul, MulConstantCommand);

    addGenF(div, DivCommand);
    addGenF(cdiv, DivConstantCommand);

    addGenF(magnitude, MagnitudeCommand);
    addGenF(complex_rgb, ComplexRGBCommand);

    addGenF(pow, PowCommand);
    addGenF(cpow_base, PowBaseConstantCommand);
    addGenF(cpow_power, PowPowerConstantCommand);

    addGenF(ln, LnCommand);
    addGenF(log, LogCommand);
    addGenF(clog_base, LogBaseConstantCommand);
    addGenF(clog_arg, LogArgConstantCommand);

    addGenF(sin, SinCommand);
    addGenF(cos, CosCommand);
    addGenF(tan, TanCommand);
    addGenF(asin, ASinCommand);
    addGenF(acos, ACosCommand);
    addGenF(atan, ATanCommand);
    addGenF(sinh, SinhCommand);
    addGenF(cosh, CoshCommand);
    addGenF(tanh, TanhCommand);
    addGenF(asinh, ASinhCommand);
    addGenF(acosh, ACoshCommand);
    addGenF(atanh, ATanhCommand);
    addGenF(atan2, ATan2Command);

    addGenF(floor, FloorCommand);
    addGenF(ceil, CeilCommand);


    addGenF(abs, AbsCommand);

    addGenF(ediff, ExplicitDiffCommand);

    addGenF(draw_explicit, DrawExplicitCommand);
    addGenF(draw_implicit, DrawImplicitCommand);
    addGenF(draw_implicit_polar, DrawImplicitPolarCommand);
    addGenF(draw_rgb2D, DrawRGB2D);

    return m;
}();

#undef addGenF

#endif // CALCULATE_COMMANDS_INIT_H