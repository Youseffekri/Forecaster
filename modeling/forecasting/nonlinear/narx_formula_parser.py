"""
NARX_FormulaParser: Parser for `NARX_Formula` Specification Strings

Extends `FormulaParser` (the linear grammar) with the free-parameter forms -- the Python
counterpart of ScalaTion's `nonlinear/NARX_FormulaParser.scala`:

    slot     := "?" [ NUM ]                        # a free parameter, optionally with its start value
    atom     |= FN "(" slot "+" atom ")"           # free shift inside a function:  log(? + x~1)
              | FN "(" slot "*" atom ")"           # free scale inside a function:  exp(? * x~1)
              | "pow" "(" atom "," slot ")"        # free exponent
              | "sat" | "hinge" | "ema" | "dq" "(" atom "," (slot | NUM) ")"
              | "trans" "(" atom "," (slot | NUM) "," (slot | NUM) ")"
              | "wave" "(" slot ")"                # one sin/cos pair with a free frequency

A lag range shares ONE parameter: pow(x~(3,1), ?) is three terms with one exponent.  A number
in a parameter position is a constant.  The same validity rules and rejects as the linear
parser apply, plus the ones of the forms above (e.g., powr keeps a fixed exponent).

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.5
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from modeling.forecasting.arx_formula import FUNCTIONS, Term, fn, powr
from modeling.forecasting.formula_parser import Ast, FormulaError, FormulaParser
from modeling.forecasting.nonlinear.narx_formula import (NARX_Formula, Par, ParRole, TransTerm, dq, ema, hinge, pow,
                                                        sat, wave)

#: The functions taking parameters after the term, with their number (ScalaTion's `NL_FUNCTIONS`).
NL_FUNCTIONS: Dict[str, int] = {"sat": 1, "hinge": 1, "trans": 2, "ema": 1, "dq": 1}


class NARX_FormulaParser(FormulaParser):
    """
    Parses formula specification strings with free parameters into `NARX_Formula` objects
    (ScalaTion's `NARX_FormulaParser` object).
    """

    @classmethod
    def parseFormula(cls, formulaStr: str, vars_: Sequence[str], maxLag: int = 8,
                     freqs: Optional[Sequence[float]] = None, maxTerms: Optional[int] = 40) -> NARX_Formula:
        """
        Parses a specification string into a `NARX_Formula` (parameters as in `FormulaParser.parseFormula`).

        Raises
        ------
        FormulaError
            With the reason, for a syntax error or a validity rule.
        """
        fail = cls.failer(formulaStr)
        lhs, elements, tform = cls.syntax(formulaStr, fail)
        vs = set(vars_)

        def parOf(role: ParRole, a) -> Par:                       # a parameter position
            if isinstance(a, Ast.Slot):
                return Par(role, a.w0)                             # free: start value and bounds from the role
            if isinstance(a, Ast.Num):
                return Par.const(a.v)                              # a constant
            fail("a parameter position must hold ? or a number")

        def exponent(e: float) -> float:
            if e < cls.P_MIN or e > cls.P_MAX:
                fail(f"exponent {e} outside [{cls.P_MIN},{cls.P_MAX}]")
            return e

        def build(ast, inProduct: bool) -> List[Term]:
            if isinstance(ast, Ast.WaveP):
                if inProduct:
                    fail("wave(?) not allowed inside interactions")
                return wave(Par(ParRole.FREQ, ast.slot.w0))
            if isinstance(ast, Ast.Call) and ast.fn in NL_FUNCTIONS and ast.num is not None:
                return buildP(ast.fn, ast.arg, [Ast.Num(ast.num)], inProduct)   # sat(x~1, 2.0): a constant in the slot
            if isinstance(ast, Ast.Call) and ast.fn in NL_FUNCTIONS and ast.fn != "dq" and ast.num is None:
                fail(f"{ast.fn} needs (term, {'threshold, width' if ast.fn == 'trans' else 'parameter'})")
            if isinstance(ast, Ast.CallP):
                return buildP(ast.fn, ast.arg, ast.ps, inProduct)
            if isinstance(ast, Ast.Call) and isinstance(ast.arg, (Ast.Shift, Ast.Scale)):
                return buildP(ast.fn, ast.arg, [] if ast.num is None else [Ast.Num(ast.num)], inProduct)
            return cls.buildLinear(ast, inProduct, vs, maxLag, freqs, fail, build)

        def buildP(f: str, arg, ps: list, inProduct: bool) -> List[Term]:
            if isinstance(arg, Ast.Shift):                         # the argument, shifted or scaled when asked
                role = ParRole.PHASE if f in ("sin", "cos") else ParRole.SHIFT
                ts = parOf(role, arg.slot) + build(arg.arg, inProduct)
            elif isinstance(arg, Ast.Scale):
                ts = parOf(ParRole.RATE, arg.slot) * build(arg.arg, inProduct)
            else:
                ts = build(arg, inProduct)
            n = len(ps)
            if f == "pow" and n == 1 and isinstance(ps[0], Ast.Slot):
                return pow(ts, Par(ParRole.POWER, ps[0].w0))
            if f == "pow" and n == 1 and isinstance(ps[0], Ast.Num):
                return pow(ts, exponent(ps[0].v))
            if f == "powr" and n == 1 and isinstance(ps[0], Ast.Num):
                return powr(ts, exponent(ps[0].v))
            if f == "powr" and n == 1 and isinstance(ps[0], Ast.Slot):
                fail("powr keeps a fixed exponent (not smooth in it): use pow(term, ?)")
            if f in ("pow", "powr") and n == 0:
                fail(f"{f} needs (term, exponent)")
            if f in ("pow", "powr"):
                fail(f"{f} takes one exponent")
            if f == "sat" and n == 1:
                return sat(ts, parOf(ParRole.HALF, ps[0]))
            if f == "hinge" and n == 1:
                return hinge(ts, parOf(ParRole.THRESH, ps[0]))
            if f == "trans" and n == 2:
                pc, pw = parOf(ParRole.THRESH, ps[0]), parOf(ParRole.WIDTH, ps[1])
                return [TransTerm(t, pc, pw) for t in ts]
            if f == "ema" and n == 1:
                return ema(ts, parOf(ParRole.DECAY, ps[0]))
            if f == "dq" and n == 1:
                return dq(ts, parOf(ParRole.CAP, ps[0]))
            if f == "dq" and n == 0:
                return dq(ts)
            if f in FUNCTIONS and n == 0:
                return fn(f, ts)
            if f in NL_FUNCTIONS:
                k = NL_FUNCTIONS[f]
                fail(f"{f} takes {k} parameter{'s' if k > 1 else ''} after the term")
            if f in FUNCTIONS:
                fail(f"{f} takes one argument")
            fail(f"unknown function '{f}'")

        resp, preds, tFormT = cls.assemble(lhs, elements, tform, vs, maxTerms, build, fail)
        return NARX_Formula(resp, preds, tFormT)

    @classmethod
    def tryParseFormula(cls, formulaStr: str, vars_: Sequence[str], maxLag: int = 8,
                        freqs: Optional[Sequence[float]] = None, maxTerms: Optional[int] = 40
                        ) -> Tuple[Optional[NARX_Formula], Optional[str]]:
        """Parses a specification, returning (formula, None) or (None, reason)."""
        try:
            return cls.parseFormula(formulaStr, vars_, maxLag, freqs, maxTerms), None
        except FormulaError as e:
            if cls.logRejects:
                print(f"tryParseFormula: {e}")
            return None, str(e)

    @staticmethod
    def grammarLines(x: str = "x1") -> str:
        """The free-parameter forms of the grammar as text for a prompt or a report (one line per
        form), in terms of the shown variable name."""
        return (
            "A number in a parameter position may be written ? (a FREE parameter, fitted on the training rows) or ?w\n"
            "(fitted, starting at w).  Free-parameter forms:\n"
            f"  pow({x}~1, ?)            free exponent (1.1 to 4)\n"
            f"  log(? + {x}~1)           free shift inside log, log1p, sqrt or pow (keeps the argument positive)\n"
            f"  exp(? * {x}~1)           free rate inside exp (also tanh); sin(? + ...) a free phase\n"
            f"  sat({x}~1, ?)            saturation x / (x + k), free half-saturation k\n"
            f"  hinge({x}~1, ?)          max(0, x - c), free threshold c\n"
            f"  trans({x}~1, ?, ?)       smooth step 1 / (1 + exp(-(x - c) / w)), free threshold c and width w\n"
            f"  ema({x}~1, ?)            exponentially weighted moving average, free decay in (0, 1)\n"
            f"  dq({x}~1, ?)             dampened quadratic, free cap\n"
            "  wave(?)                 one seasonal sin/cos pair with a free frequency\n"
            f"A lag range shares one parameter: pow({x}~(3,1), ?) is three terms with one exponent.  Each free parameter\n"
            "costs a penalty like a term, so use one only where a fixed number would be a guess."
        )
