"""
Parser for NARX Formula Specification Strings

Turns formula STRINGS (proposed by an LLM or written by hand) into `Formula` objects,
so a structure search can run propose (string) -> parse -> fit -> score.  The grammar is
the one `Formula.formula_str` prints, so parse_formula(f.formula_str) rebuilds f:

    input    := formula [ tform ]                      # tform on the same or the next line
    formula  := ID "~=" element { "+" element }
    tform    := "TFORM:" ID                            # e.g., TFORM: Log1p (any case)
    element  := atom { "*" atom }                      # at most two factors (no wave inside);
                                                       # a lag range distributes over the product
    atom     := ID "~" INT                             # lagged variable  y~3
              | ID "~(" INT ")"                        # same, parenthesized  y~(3)
              | ID "~(" INT "," INT ")"                # lag range  y~(5,1): one term per lag
              | FN "(" atom ")"                        # FN in `formula.FUNCTIONS`
              | FN "(" slot "+" atom ")"               # free shift inside a function:  log(? + x~1)
              | FN "(" slot "*" atom ")"               # free scale inside a function:  exp(? * x~1)
              | "pow"  "(" atom "," (NUM | slot) ")"   # power with a fixed or a free exponent
              | "powr" "(" atom "," NUM ")"            # sign-safe power (fixed exponent only)
              | "sat" | "hinge" | "ema" | "dq"  "(" atom "," (slot | NUM) ")"
              | "trans" "(" atom "," (slot | NUM) "," (slot | NUM) ")"
              | "wave" "(" INT ")"                     # k seasonal sin/cos pairs at the caller's frequencies
              | "wave" "(" slot ")"                    # one sin/cos pair with a free frequency
    slot     := "?" [ NUM ]                            # a free parameter, optionally with its start value

A lag range shares ONE parameter: pow(x~(3,1), ?) is three terms with one exponent.  The
validity rules (unknown variable, lag out of range, missing frequencies for wave, too many
terms) are checked on the syntax tree and each reject carries a reason, since the validity
rate of a proposer is itself a reported metric.  Duplicate terms are collapsed, not rejected.

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple

from modeling.forecasting.formula import (FUNCTIONS, Formula, Par, ParRole, Term, Var, fn,
                                          pow_, powr, sat, hinge, trans, ema, dq, wave)


class FormulaError(ValueError):
    """A formula string that does not parse or does not pass the validity rules."""


# ------------------------------------------------------------------------------------
#  Tokens
# ------------------------------------------------------------------------------------

_TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<tform>TFORM\s*:)
  | (?P<tilde_eq>~=)
  | (?P<num>\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)
  | (?P<int>\d+)
  | (?P<id>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<sym>[~(),+*?])
""", re.VERBOSE | re.IGNORECASE)


def _tokenize(s: str) -> List[Tuple[str, str, int]]:
    """Splits a formula string into (kind, text, position) tokens."""
    out = []
    pos = 0
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if m is None:
            raise FormulaError(f"syntax error (position {pos + 1}, found \"{s[pos:pos + 12]}\")")
        kind = m.lastgroup
        if kind != "ws":
            out.append((kind, m.group(kind), pos))
        pos = m.end()
    out.append(("eof", "", len(s)))
    return out


#: Functions taking one parameter after the term and the role of that parameter.
_ONE_PAR = {"sat": ParRole.HALF, "hinge": ParRole.THRESH, "ema": ParRole.DECAY, "dq": ParRole.CAP}


class _Parser:
    """Recursive-descent parser over the token list (one formula string)."""

    def __init__(self, text: str, vars_: Sequence[str], max_lag: int,
                 freqs: Optional[Sequence[float]], max_terms: Optional[int]) -> None:
        self.text = text
        self.toks = _tokenize(text)
        self.i = 0
        self.vars = set(vars_)
        self.max_lag = max_lag
        self.freqs = list(freqs) if freqs is not None else None
        self.max_terms = max_terms

    # -- token helpers ------------------------------------------------------------------

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
        raise FormulaError(f"syntax error (position {pos + 1}, expected {what or text or kind}, found \"{found}\")")

    def _fail(self, msg: str):
        raise FormulaError(msg)

    # -- grammar ------------------------------------------------------------------------

    def parse(self) -> Formula:
        resp = self._expect("id", what="the response variable")
        self._expect("tilde_eq", what="~=")
        elements: List[Term] = []
        elements += self._element()
        while self._accept("sym", "+"):
            elements += self._element()
        tform = None
        if self._accept("tform"):
            tform = self._expect("id", what="the transform name (e.g., Log1p)")
        if self._peek()[0] != "eof":
            k, tx, pos = self._peek()
            self._fail(f"syntax error (position {pos + 1}, found \"{tx}\")")
        if resp not in self.vars:
            self._fail(f"response '{resp}' not in dataset {{{','.join(sorted(self.vars))}}}")
        try:
            f = Formula(resp, elements, tform)
        except ValueError as e:
            self._fail(str(e))
        if self.max_terms is not None and f.n_terms > self.max_terms:
            self._fail(f"{f.n_terms} terms, at most {self.max_terms} allowed")
        return f

    def _element(self) -> List[Term]:
        factors = [self._atom()]
        while self._accept("sym", "*"):
            factors.append(self._atom())
        if len(factors) == 1:
            return factors[0]
        if len(factors) > 2:
            self._fail("an interaction has at most two factors")
        left, right = factors
        from modeling.forecasting.formula import Product, Wave, ParWave
        for side in (left, right):
            if any(isinstance(t, (Wave, ParWave)) for t in side):
                self._fail("wave(?) not allowed inside interactions" if any(isinstance(t, ParWave) for t in side)
                           else "wave(k) not allowed inside interactions")
        if len(left) == 1:
            return [Product(left[0], r) for r in right]
        if len(right) == 1:
            return [Product(l, right[0]) for l in left]
        return [Product(l, r) for l in left for r in right]            # two ranges: every pair

    def _slot_or_num(self, role: ParRole) -> Par:
        """A parameter position: `?`, `?w` (free) or a number (a constant)."""
        if self._accept("sym", "?"):
            k, tx, _ = self._peek()
            if k in ("num", "int"):
                self._next()
                return Par(role, float(tx))
            return Par(role)
        k, tx, _ = self._peek()
        if k in ("num", "int"):
            self._next()
            return Par.fixed(float(tx))
        _, tx, pos = self._peek()
        self._fail(f"syntax error (position {pos + 1}, expected ? or a number, found \"{tx}\")")

    def _atom(self) -> List[Term]:
        k, tx, pos = self._peek()
        if k != "id":
            found = tx if k != "eof" else "end of formula"
            self._fail(f"syntax error (position {pos + 1}, found \"{found}\")")
        name = tx
        nxt = self._peek(1)
        # -- a lagged variable ---------------------------------------------------------
        if nxt[0] == "sym" and nxt[1] == "~":
            self._next()
            self._next()
            if self._accept("sym", "("):
                l1 = int(self._expect("int", what="a lag"))
                if self._accept("sym", ","):
                    l2 = int(self._expect("int", what="a lag"))
                    self._expect("sym", ")")
                    return self._lags(name, l1, l2)
                self._expect("sym", ")")
                return self._lags(name, l1, l1)
            l1 = int(self._expect("int", what="a lag"))
            return self._lags(name, l1, l1)
        # -- a function application ----------------------------------------------------
        if nxt[0] == "sym" and nxt[1] == "(":
            self._next()
            self._next()
            lname = name.lower()
            if lname == "wave":
                terms = self._wave()
                self._expect("sym", ")")
                return terms
            if lname in ("pow", "powr"):
                src = self._fn_arg(lname)
                self._expect("sym", ",", what="the exponent")
                if lname == "powr":
                    if self._peek()[0] == "sym" and self._peek()[1] == "?":
                        self._fail("powr keeps a fixed exponent (not smooth in it): use pow(term, ?)")
                    q = float(self._expect("num" if self._peek()[0] == "num" else "int", what="the exponent"))
                    self._expect("sym", ")")
                    return powr(src, q)
                p = self._slot_or_num(ParRole.POWER)
                self._expect("sym", ")")
                return pow_(src, p if p.free else p.value)
            if lname == "trans":
                src = self._fn_arg(lname)
                if not self._accept("sym", ","):
                    self._fail("trans takes 2 parameters after the term")
                c = self._slot_or_num(ParRole.THRESH)
                if not self._accept("sym", ","):
                    self._fail("trans takes 2 parameters after the term")
                w = self._slot_or_num(ParRole.WIDTH)
                self._expect("sym", ")")
                return trans(src, c, w)
            if lname in _ONE_PAR:
                src = self._fn_arg(lname)
                if not self._accept("sym", ","):
                    self._fail(f"{lname} needs (term, parameter)")
                p = self._slot_or_num(_ONE_PAR[lname])
                if self._peek()[0] == "sym" and self._peek()[1] == ",":
                    self._fail(f"{lname} takes 1 parameter after the term")
                self._expect("sym", ")")
                return {"sat": sat, "hinge": hinge, "ema": ema, "dq": dq}[lname](src, p)
            if lname in FUNCTIONS:
                arg = self._fn_arg(lname)
                self._expect("sym", ")")
                return fn(lname, arg)
            self._fail(f"unknown function '{name}'")
        # -- anything else ----------------------------------------------------------------
        if name in self.vars:
            self._fail(f"variable '{name}' needs a lag >= 1 (write {name}~1)")
        self._fail(f"syntax error (position {pos + 1}, found \"{name}\")")

    def _fn_arg(self, lname: str) -> List[Term]:
        """The argument of a function: an atom, or a free shift / scale of an atom."""
        from modeling.forecasting.formula import Shift, Scale
        if self._accept("sym", "?"):
            k, tx, _ = self._peek()
            w0 = math.nan
            if k in ("num", "int"):
                self._next()
                w0 = float(tx)
            if self._accept("sym", "+"):
                role = ParRole.PHASE if lname in ("sin", "cos") else ParRole.SHIFT
                p = Par(role, w0)
                return [Shift(t, p) for t in self._atom()]
            if self._accept("sym", "*"):
                p = Par(ParRole.RATE, w0)
                return [Scale(t, p) for t in self._atom()]
            _, tx, pos = self._peek()
            self._fail(f"syntax error (position {pos + 1}, expected + or * after ?, found \"{tx}\")")
        k, tx, _ = self._peek()
        if k in ("num", "int"):                                            # a constant shift: log(2 + x~1)
            self._next()
            c = float(tx)
            if self._accept("sym", "+"):
                return [Shift(t, Par.fixed(c)) for t in self._atom()]
            if self._accept("sym", "*"):
                return [Scale(t, Par.fixed(c)) for t in self._atom()]
            _, tx2, pos = self._peek()
            self._fail(f"syntax error (position {pos + 1}, expected + or * after a number, found \"{tx2}\")")
        return self._atom()

    def _lags(self, name: str, l1: int, l2: int) -> List[Term]:
        if name not in self.vars:
            self._fail(f"variable '{name}' not in dataset {{{','.join(sorted(self.vars))}}}")
        if l1 < l2:
            self._fail(f"lag range ({l1},{l2}) for '{name}' must be decreasing")
        if l2 < 1:
            self._fail(f"lag {l2} for '{name}' must be >= 1 (only past values are known at forecast time)")
        if l1 > self.max_lag:
            self._fail(f"lag {l1} for '{name}' exceeds the maximum lag {self.max_lag}")
        return Var(name).lags(l1, l2)

    def _wave(self) -> List[Term]:
        if self._accept("sym", "?"):
            k, tx, _ = self._peek()
            w0 = math.nan
            if k in ("num", "int"):
                self._next()
                w0 = float(tx)
            return wave(Par(ParRole.FREQ, w0))
        kk = int(self._expect("int", what="the number of sin/cos pairs"))
        if kk < 1:
            self._fail("wave(k) needs k >= 1")
        if self.freqs is None or len(self.freqs) < kk:
            self._fail(f"wave({kk}) needs {kk} frequencies, {0 if self.freqs is None else len(self.freqs)} given")
        return wave(kk, self.freqs)


# ------------------------------------------------------------------------------------
#  Public API
# ------------------------------------------------------------------------------------

def parse_formula(text: str, vars_: Sequence[str], max_lag: int = 6,
                  freqs: Optional[Sequence[float]] = None, max_terms: Optional[int] = None) -> Formula:
    """
    Parses a formula string into a `Formula`.

    Parameters
    ----------
    text : str
        The formula, e.g., "y ~= y~(3,1) + log(? + icu~1) + hinge(temp~1, ?) TFORM: Log1p".
    vars_ : sequence of str
        The legal variable names (the keys of the dataset).
    max_lag : int, default=6
        The largest lag allowed.
    freqs : sequence of float, optional
        The frequencies (cycles per step) `wave(k)` draws on, e.g., `eda.top_freqs`.
    max_terms : int, optional
        A cap on the number of terms (None => no cap).

    Returns
    -------
    Formula

    Raises
    ------
    FormulaError
        With the reason, for a syntax error or a validity rule.
    """
    text = " ".join(text.strip().split())
    return _Parser(text, vars_, max_lag, freqs, max_terms).parse()


def try_parse_formula(text: str, vars_: Sequence[str], max_lag: int = 6,
                      freqs: Optional[Sequence[float]] = None,
                      max_terms: Optional[int] = None) -> Tuple[Optional[Formula], Optional[str]]:
    """
    Parses a formula string, returning (formula, None) or (None, reason).  The reason
    carries the formula, so a proposer can be told what was wrong.
    """
    try:
        return parse_formula(text, vars_, max_lag, freqs, max_terms), None
    except FormulaError as e:
        return None, f"parse: {e} in formula: {text.strip()}"


def grammar_lines(x: str = "x1") -> str:
    """
    The free-parameter forms, as lines for a prompt (the LLM is told the grammar).

    Parameters
    ----------
    x : str, default="x1"
        The variable name used in the examples.
    """
    return (
        "A number in a parameter position may be written ? (a FREE parameter, fitted on the training rows) or ?w\n"
        "(fitted, starting at w).  Free-parameter forms:\n"
        f"  pow({x}~1, ?)            free exponent (0.5 to 3)\n"
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
