"""
FormulaParser: Parser for `ARX_Formula` Specification Strings

Turns formula STRINGS (proposed by an LLM or written by hand) into `ARX_Formula` objects, so a
search can run propose (string) -> parse -> fit -> score -- the Python counterpart of
ScalaTion's `FormulaParser.scala`.  The grammar is the one `ARX_Formula.formulaStr` prints, so
`parseFormula(f.formulaStr, ...)` rebuilds f:

    input    := formula [ tform ]                  # tform on the same or the next line
    formula  := ID "~=" element { "+" element }
    tform    := "TFORM:" ID                        # e.g., TFORM: Log1p (any case)
    element  := atom { "*" atom }                  # at most two factors (no wave inside);
                                                   # a lag range distributes over the product
    atom     := ID "~" INT                         # lagged variable  y~3
              | ID "~(" INT ")"                    # same, parenthesized  y~(3)
              | ID "~(" INT "," INT ")"            # lag range        y~(5,1)
              | FN "(" atom ")"                    # FN in `ARX_Formula.FUNCTIONS`
              | "pow"  "(" atom "," NUM ")"        # plain power
              | "powr" "(" atom "," NUM ")"        # sign-safe power
              | "wave" "(" INT ")"                 # k seasonal sin/cos pairs; frequencies
                                                   # come from the caller (periodogram)
    -- free nonlinear parameters (built by `nonlinear.narx_formula_parser.NARX_FormulaParser`,
       rejected here with a reason):
    slot     := "?" [ NUM ]                        # a free parameter, optionally with its start value
    atom     |= FN "(" slot "+" atom ")"           # free shift inside a function:  log(? + x~1)
              | FN "(" slot "*" atom ")"           # free scale inside a function:  exp(? * x~1)
              | "pow" "(" atom "," slot ")"        # free exponent
              | FNP "(" atom "," (slot | NUM) { "," (slot | NUM) } ")"   # sat, hinge, trans, ema, dq
              | "wave" "(" slot ")"                # one sin/cos pair with a free frequency

The syntax is read by a recursive-descent parser into a small syntax tree (`Ast`); the validity
rules for a proposal (unknown variable, lag or exponent out of range, missing frequencies for
wave, too many terms) are checked on the tree, and each reject carries a reason, since the
validity rate is itself a reported metric.  Duplicate terms (same label) are collapsed rather
than rejected, as they would make the design matrix singular.

Author: Yousef Fekri Dabanloo
Date: 2026-10-13
Version: 1.3
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from Util.transform import TransformT, transform_t
from modeling.forecasting.arx_formula import (ARX_Formula, FUNCTIONS, FormulaVar, ProductTerm, Term, fn, pow as pow_,
                                              powr, wave)


class FormulaError(ValueError):
    """A formula string that does not parse or does not pass the validity rules (the reason)."""


# ---------------------------------------------------------------------------------------------
#  The syntax tree
# ---------------------------------------------------------------------------------------------

class Ast:
    """The syntax tree of an atom (ScalaTion's `FormulaParser.Ast`)."""

    @dataclass
    class Lag:
        name: str
        hi: int
        lo: int

    @dataclass
    class Wave:
        k: int

    @dataclass
    class Call:
        fn: str
        arg: object
        num: Optional[float]

    @dataclass
    class Slot:
        w0: float

    @dataclass
    class Num:
        v: float

    @dataclass
    class Shift:
        slot: object
        arg: object

    @dataclass
    class Scale:
        slot: object
        arg: object

    @dataclass
    class CallP:
        fn: str
        arg: object
        ps: list

    @dataclass
    class WaveP:
        slot: object


# ---------------------------------------------------------------------------------------------
#  Tokens and the syntax
# ---------------------------------------------------------------------------------------------

_TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<tform>TFORM\s*:)
  | (?P<tilde_eq>~=)
  | (?P<num>\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)
  | (?P<int>\d+)
  | (?P<id>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<sym>[~(),+*?-])
""", re.VERBOSE | re.IGNORECASE)


def _tokenize(s: str, fail: Callable[[str], None]) -> List[Tuple[str, str, int]]:
    """Splits a formula string into (kind, text, position) tokens."""
    out = []
    pos = 0
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if m is None:
            fail(f"syntax error (position {pos + 1}, found \"{s[pos:pos + 12]}\")")
        kind = m.lastgroup
        if kind != "ws":
            out.append((kind, m.group(kind), pos))
        pos = m.end()
    out.append(("eof", "", len(s)))
    return out


class _Syntax:
    """Recursive-descent reader of the syntax into (lhs, elements, tform) -- ScalaTion's `syntax`."""

    def __init__(self, text: str, fail: Callable[[str], None]) -> None:
        self.text = text
        self.fail = fail
        self.toks = _tokenize(text, fail)
        self.i = 0

    def _peek(self, k: int = 0):
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def _next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def _accept(self, kind: str, text: Optional[str] = None) -> bool:
        k, tx, _ = self._peek()
        if k == kind and (text is None or tx == text):
            self.i += 1
            return True
        return False

    def _expect(self, kind: str, text: Optional[str] = None, what: str = "") -> str:
        k, tx, pos = self._peek()
        if k == kind and (text is None or tx == text):
            self.i += 1
            return tx
        found = tx if k != "eof" else "end of formula"
        self.fail(f"syntax error (position {pos + 1}, expected {what or text or kind}, found \"{found}\")")

    def _number(self, what: str = "a number") -> float:
        k, tx, pos = self._peek()
        if k in ("num", "int"):
            self._next()
            return float(tx)
        if k == "sym" and tx == "-" and self._peek(1)[0] in ("num", "int"):
            self._next()
            return -float(self._next()[1])
        self.fail(f"syntax error (position {pos + 1}, expected {what}, found \"{tx if k != 'eof' else 'end of formula'}\")")

    # -- grammar ------------------------------------------------------------------------------

    def formula(self) -> Tuple[str, List[list], Optional[str]]:
        lhs = self._expect("id", what="the response variable")
        self._expect("tilde_eq", what="~=")
        elements = [self.element()]
        while self._accept("sym", "+"):
            elements.append(self.element())
        tform = None
        if self._accept("tform"):
            tform = self._expect("id", what="the transform name (e.g., Log1p)")
        if self._peek()[0] != "eof":
            k, tx, pos = self._peek()
            self.fail(f"syntax error (position {pos + 1}, found \"{tx}\")")
        return lhs, elements, tform

    def element(self) -> list:
        factors = [self.atom()]
        while self._accept("sym", "*"):
            factors.append(self.atom())
        return factors

    def slot(self) -> Ast.Slot:
        """`?`, `?1.5`, `?-0.5` (the ? was consumed by the caller)."""
        k, tx, _ = self._peek()
        if k in ("num", "int") or (k == "sym" and tx == "-" and self._peek(1)[0] in ("num", "int")):
            return Ast.Slot(self._number())
        return Ast.Slot(math.nan)

    def par(self):
        """A parameter position: ? or a number."""
        if self._accept("sym", "?"):
            return self.slot()
        return Ast.Num(self._number("? or a number"))

    def arg(self):
        """A function argument: ? + atom, ? * atom, NUM + atom, NUM * atom, or atom."""
        if self._accept("sym", "?"):
            s = self.slot()
            if self._accept("sym", "+"):
                return Ast.Shift(s, self.atom())
            if self._accept("sym", "*"):
                return Ast.Scale(s, self.atom())
            _, tx, pos = self._peek()
            self.fail(f"syntax error (position {pos + 1}, expected + or * after ?, found \"{tx}\")")
        k, tx, _ = self._peek()
        if k in ("num", "int"):
            c = Ast.Num(self._number())
            if self._accept("sym", "+"):
                return Ast.Shift(c, self.atom())
            if self._accept("sym", "*"):
                return Ast.Scale(c, self.atom())
            _, tx2, pos = self._peek()
            self.fail(f"syntax error (position {pos + 1}, expected + or * after a number, found \"{tx2}\")")
        return self.atom()

    def atom(self):
        k, tx, pos = self._peek()
        if k != "id":
            found = tx if k != "eof" else "end of formula"
            self.fail(f"syntax error (position {pos + 1}, found \"{found}\")")
        name = tx
        nxt = self._peek(1)
        if nxt[0] == "sym" and nxt[1] == "~":                   # a lagged variable
            self._next(); self._next()
            if self._accept("sym", "("):
                hi = int(self._expect("int", what="a lag"))
                lo = hi
                if self._accept("sym", ","):
                    lo = int(self._expect("int", what="a lag"))
                self._expect("sym", ")")
                return Ast.Lag(name, hi, lo)
            hi = int(self._expect("int", what="a lag"))
            return Ast.Lag(name, hi, hi)
        if nxt[0] == "sym" and nxt[1] == "(":                   # a call
            self._next(); self._next()
            lname = name.lower()
            if lname == "wave":
                if self._accept("sym", "?"):
                    s = self.slot()
                    self._expect("sym", ")")
                    return Ast.WaveP(s)
                kk = int(self._expect("int", what="the number of sin/cos pairs"))
                self._expect("sym", ")")
                return Ast.Wave(kk)
            a = self.arg()
            ps = []
            while self._accept("sym", ","):
                ps.append(self.par())
            self._expect("sym", ")")
            if not ps:
                return Ast.Call(lname, a, None)
            if len(ps) == 1 and isinstance(ps[0], Ast.Num):
                return Ast.Call(lname, a, ps[0].v)
            return Ast.CallP(lname, a, ps)
        if k == "id":
            self.fail(f"syntax error (position {pos + 1}, found \"{name}\": a variable needs a lag, e.g., {name}~1)")


# ---------------------------------------------------------------------------------------------
#  The parser
# ---------------------------------------------------------------------------------------------

class FormulaParser:
    """
    Parses formula specification strings into `ARX_Formula` objects (ScalaTion's `FormulaParser`
    object); a formula using free parameters (?) is rejected with a reason.
    """

    logRejects = True                                            # whether `tryParseFormula` prints the reason
    P_MIN = 0.2                                                  # minimum allowed exponent
    P_MAX = 3.0                                                  # maximum allowed exponent

    @staticmethod
    def failer(formulaStr: str) -> Callable[[str], None]:
        """The function that rejects a specification with a reason."""
        def fail(reason: str):
            raise FormulaError(f"parse: {reason} in formula: {' '.join(formulaStr.split())}")
        return fail

    @staticmethod
    def syntax(formulaStr: str, fail: Callable[[str], None]) -> Tuple[str, List[list], Optional[str]]:
        """The syntax of a specification: its response name, elements (each a list of atoms) and
        optional TFORM name; a syntax error is a reject."""
        return _Syntax(" ".join(formulaStr.strip().split()), fail).formula()

    @classmethod
    def buildLinear(cls, ast, inProduct: bool, vars_: set, maxLag: int, freqs: Optional[Sequence[float]],
                    fail: Callable[[str], None], recur: Callable) -> List[Term]:
        """
        Builds the terms of an atom of the LINEAR grammar (lagged variables, lag ranges, functions,
        powers, waves) with the validity rules; the free-parameter forms are rejected with a reason
        (`NARX_FormulaParser` builds them).  The argument of a function is built by `recur`, so a
        parser that extends the grammar passes its own builder and still gets the linear forms here.
        """
        if isinstance(ast, Ast.Lag):
            nm, hi, lo = ast.name, ast.hi, ast.lo
            if nm not in vars_:
                fail(f"variable '{nm}' not in dataset {{{','.join(sorted(vars_))}}}")
            if lo < 1 or hi > maxLag or hi < lo:
                if hi == lo:
                    fail(f"lag {hi} outside [1,{maxLag}]")
                fail(f"lag range ({hi},{lo}) outside [1,{maxLag}]")
            return FormulaVar(nm).lags(hi, lo)
        if isinstance(ast, Ast.Wave):
            if inProduct:
                fail("wave(k) not allowed inside interactions")
            if freqs is None:
                fail("wave(k) used but no frequencies supplied")
            if ast.k < 1 or ast.k > len(freqs):
                fail(f"wave({ast.k}) outside [1,{len(freqs)}]")
            return wave(ast.k, freqs)
        if isinstance(ast, Ast.Call):
            f, num = ast.fn, ast.num
            if f in ("pow", "powr"):
                if num is None:
                    fail(f"{f} needs (term, exponent)")
                if num < cls.P_MIN or num > cls.P_MAX:
                    fail(f"{f} exponent {num} outside [{cls.P_MIN},{cls.P_MAX}]")
                ts = recur(ast.arg, inProduct)
                return pow_(ts, num) if f == "pow" else powr(ts, num)
            if f in FUNCTIONS:
                if num is not None:
                    fail(f"{f} takes one argument")
                return fn(f, recur(ast.arg, inProduct))
            fail(f"unknown function '{f}'")
        fail("free parameters (?) need nonlinear.NARX_FormulaParser")

    @staticmethod
    def assemble(lhs: str, elements: List[list], tform: Optional[str], vars_: set, maxTerms: int,
                 build: Callable, fail: Callable[[str], None]) -> Tuple[FormulaVar, List[Term], Optional[TransformT]]:
        """
        Assembles a parsed specification: checks the response and the TFORM, builds the terms of
        every element (a product of two atoms distributes a lag range), applies the term cap and
        returns the response, the terms and the transform type.
        """
        if lhs not in vars_:
            fail(f"response '{lhs}' not a dataset variable")
        tFormT = None
        if tform is not None:
            try:
                tFormT = transform_t(tform)
            except ValueError:
                fail(f"unknown TFORM '{tform}'")
        preds: List[Term] = []
        for el in elements:
            if len(el) == 1:
                preds += build(el[0], False)
            elif len(el) == 2:
                as_, bs = build(el[0], True), build(el[1], True)
                preds += [ProductTerm(a, b) for a in as_ for b in bs]   # a range distributes over the product
            else:
                fail(f"interaction with {len(el)} factors (at most two)")
        if maxTerms is not None and len(preds) > maxTerms:
            fail(f"{len(preds)} terms exceeds maxTerms = {maxTerms}")
        return FormulaVar(lhs), preds, tFormT

    @classmethod
    def parseFormula(cls, formulaStr: str, vars_: Sequence[str], maxLag: int = 8,
                     freqs: Optional[Sequence[float]] = None, maxTerms: Optional[int] = 40) -> ARX_Formula:
        """
        Parses a specification string into an `ARX_Formula`.

        Parameters
        ----------
        formulaStr : str
            The specification, e.g., "y ~= y~(3,1) + log1p(icu~1) + wave(1) TFORM: Log1p".
        vars_ : sequence of str
            The legal variable names (the dataset's keys).
        maxLag : int, default=8
            The maximum allowed lag.
        freqs : sequence of float, optional
            The wave frequencies (cycles per time unit) from the TRAINING-slice periodogram;
            required only if the formula uses wave(k).
        maxTerms : int, default=40
            The maximum number of expanded terms (parsimony guard); None => no cap.

        Returns
        -------
        ARX_Formula

        Raises
        ------
        FormulaError
            With the reason, for a syntax error or a validity rule.
        """
        fail = cls.failer(formulaStr)
        lhs, elements, tform = cls.syntax(formulaStr, fail)
        vs = set(vars_)

        def build(ast, inProduct: bool) -> List[Term]:
            return cls.buildLinear(ast, inProduct, vs, maxLag, freqs, fail, build)

        resp, preds, tFormT = cls.assemble(lhs, elements, tform, vs, maxTerms, build, fail)
        return ARX_Formula(resp, preds, tFormT)

    @classmethod
    def tryParseFormula(cls, formulaStr: str, vars_: Sequence[str], maxLag: int = 8,
                        freqs: Optional[Sequence[float]] = None, maxTerms: Optional[int] = 40
                        ) -> Tuple[Optional[ARX_Formula], Optional[str]]:
        """
        Parses a specification, returning (formula, None) or (None, reason) -- the entry point for
        search loops, where invalid proposals are counted and skipped, never fatal.
        """
        try:
            return cls.parseFormula(formulaStr, vars_, maxLag, freqs, maxTerms), None
        except FormulaError as e:
            if cls.logRejects:
                print(f"tryParseFormula: {e}")
            return None, str(e)
