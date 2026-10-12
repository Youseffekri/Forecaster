"""
FormulaScorer: the Fit-and-Score Engine of LLM-NARX-SR

One engine takes a formula (a string or a `Formula`), fits it as a direct multi-horizon NARX
model (`NARX_SR_D.from_formula`: coefficients by ridge, free nonlinear parameters by VarPro,
L-BFGS-B or a joint fit) and scores it with the selection score J used by every search arm:

    J_h = sMAPE_h (validation rows) * (1 + 2 p / n),     p = terms + 1 + fitted parameters

the FPE-form parsimony adjustment of the out-of-sample sMAPE, averaged over the horizons
(or taken at h = 1 when hp["hSel"] = 1).  The validation rows are the last hp["valFrac"] of
the TRAINING rows, split into hp["valFolds"] rolling-origin blocks: fold i refits the
candidate on the rows before block i and scores it on the block, so J never sees the testing
rows (the final rolling validation on the testing rows is `tnt`).  With hp["valFrac"] = 0, J
is the in-sample FPE-sMAPE on the training rows.

Rejection rules (the reason travels with the `Candidate`, so a proposer can be told):
more than hp["maxFree"] free parameters; a fitted parameter within hp["boundTol"] of its
box; an exponent within 0.05 of 1 (a plain lag); a free-parameter column with a VIF above
hp["maxVIF"] on the fit rows (collinear with the others).

The Python counterpart of ScalaTion's `llm_narx_sr.FormulaScorer` (core part: no term
libraries / CEM stage here, that is the search layer).

Author: Yousef Fekri Dabanloo
Date: 2026-10-12
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from Util.tools import diagnose
from modeling.forecasting.formula import Formula, Par, ParRole
from modeling.forecasting.formula_parser import parse_formula, FormulaError
from modeling.forecasting.narx_sr_d import NARX_SR_D


#: The hyper-parameters of the engine (one dict every arm and the harness read).
DEFAULT_HP: Dict[str, Any] = {
    "hh": 6,              # maximum forecasting horizon (h = 1..hh)
    "maxLag": 6,          # maximum lag for proposals and the parser
    "skip": 6,            # rows skipped by every in-sample diagnosis (the same for all candidates)
    "rc": 5,              # retraining cycle of rollValidate on the testing rows
    "hSel": 0,            # 0 => score = mean FPE-sMAPE over the horizons, 1 => h = 1 only
    "lambda": 1.0,        # ridge penalty carried by every model
    "nFreqs": 2,          # periodogram frequencies offered to wave(k)
    "valFrac": 0.25,      # share of the training rows held out for the selection score (0 => in-sample J)
    "valFolds": 3,        # rolling-origin blocks of the validation rows (k fits per candidate; 1 => one block)
    "maxFree": 2,         # free nonlinear parameters allowed per candidate (-1 => no cap)
    "boundTol": 0.01,     # a fitted parameter within this fraction of its box of a bound is rejected
    "maxVIF": 10.0,       # a free-parameter column with a larger VIF is rejected (0 => off)
    "optimizer": "varpro",  # the optimizer of the nonlinear parameters: varpro, lbfgsb, joint
    "nneg": True,         # forecasts clipped at zero (count data); set False for series that go negative
}

QOF_KEYS = ("MSE", "MAE", "R2", "R2Bar", "SMAPE", "m", "dfr")


@dataclass
class Candidate:
    """
    A scored formula: the unit a search population holds.

    Attributes
    ----------
    formula : Optional[Formula]
        The formula (its parameters hold the fitted values); None for a failed parse.
    score : float
        The selection score J (lower is better); inf for a failed or rejected fit.
    qof_train : list of dict
        The QoF per horizon on the fit rows (MSE, MAE, R2, R2Bar, SMAPE, m, dfr).
    n_terms : int
        The number of terms.
    qof_val : list of dict, optional
        The QoF per horizon on the validation blocks (mean over the folds); None => in-sample.
    reason : str, optional
        Why the candidate failed or was rejected.
    label : str
        A display label for a candidate without a formula.
    extra_df : int
        Extra degrees of freedom charged (e.g., parameters fitted outside the model).
    """
    formula: Optional[Formula]
    score: float
    qof_train: List[Dict[str, float]] = field(default_factory=list)
    n_terms: int = 0
    qof_val: Optional[List[Dict[str, float]]] = None
    reason: Optional[str] = None
    label: str = ""
    extra_df: int = 0

    @property
    def failed(self) -> bool:
        return not math.isfinite(self.score)

    @property
    def key(self) -> str:
        return self.formula.key if self.formula is not None else self.label

    @property
    def formula_str(self) -> str:
        return self.formula.formula_str if self.formula is not None else self.label

    @property
    def proposal_str(self) -> str:
        return self.formula.proposal_str if self.formula is not None else self.label

    @property
    def n_free(self) -> int:
        return self.formula.n_free if self.formula is not None else 0

    @property
    def smape_h(self) -> np.ndarray:
        return np.array([q["SMAPE"] for q in self.qof_train]) if self.qof_train else np.array([])

    @property
    def smape(self) -> float:
        return float(self.smape_h.mean()) if self.qof_train else math.inf

    @property
    def qof_v(self) -> List[Dict[str, float]]:
        return self.qof_val if self.qof_val is not None else self.qof_train

    @property
    def smape_val(self) -> float:
        return float(np.mean([q["SMAPE"] for q in self.qof_v])) if self.qof_train else math.inf

    @property
    def score_h(self) -> np.ndarray:
        return FormulaScorer.j_of(self.qof_train, self.qof_v, self.extra_df) if self.qof_train else np.array([])

    @property
    def why(self) -> str:
        return f" -- {self.reason}" if self.reason else ""

    def __repr__(self) -> str:
        s = f"Candidate({self.formula_str}, score={self.score:.4f}" if not self.failed else f"Candidate({self.formula_str}, FAILED"
        return s + f", n_terms={self.n_terms}, n_free={self.n_free}{self.why})"


def qof_table(qof: Sequence[Dict[str, float]], title: str = "") -> str:
    """Formats a per-horizon QoF list as the table `Util.tools.fit_map` prints."""
    rows = {"horizon": [f"{h + 1}" for h in range(len(qof))]}
    for k in ("MSE", "MAE", "R2", "R2Bar", "SMAPE", "m", "dfr"):
        rows[k] = [f"{q[k]:.4f}" if isinstance(q[k], float) else f"{q[k]}" for q in qof]
    out = (title + "\n") if title else ""
    for k, v in rows.items():
        out += f"{k}\t->\t" + "\t".join(v) + "\n"
    return out


class FormulaScorer:
    """
    The fit-and-score engine over one named-column dataset.

    Parameters
    ----------
    data : dict of str -> np.ndarray
        The named columns (the response under its name, usually "y"), all of one length.
    hh : int, optional
        The maximum horizon (defaults to hp["hh"]).
    hp : dict, optional
        Overrides of `DEFAULT_HP`.
    scaled : bool, default=True
        Whether the models standardize their columns (also inside VarPro).
    freqs : sequence of float, optional
        The frequencies `wave(k)` draws on (defaults to the top hp["nFreqs"] of the training
        response's periodogram).

    Attributes
    ----------
    vars : list of str
        The legal variable names.
    m, tr_size, fit_size : int
        Rows in the series, training rows (80 %), fit rows (training rows minus the validation rows).
    blocks : list of range
        The rolling-origin validation blocks (empty when hp["valFrac"] = 0).
    evaluations : int
        The number of model fits made so far.
    """

    def __init__(self, data: Dict[str, np.ndarray], hh: Optional[int] = None,
                 hp: Optional[Dict[str, Any]] = None, scaled: bool = True,
                 freqs: Optional[Sequence[float]] = None) -> None:
        self.hp = dict(DEFAULT_HP)
        if hp:
            self.hp.update(hp)
        self.data = {k: np.asarray(v, dtype=float) for k, v in data.items()}
        self.vars = list(self.data.keys())
        self.hh = int(hh if hh is not None else self.hp["hh"])
        self.scaled = scaled
        self.m = len(next(iter(self.data.values())))
        self.tr_size = int((1.0 - NARX_SR_D._TE_RATIO) * self.m)
        self.te_size = self.m - self.tr_size
        self.tr_rows = range(0, self.tr_size)
        self.te_rows = range(self.tr_size, self.m)
        self.skip = int(self.hp["skip"])
        self.h_sel = int(self.hp["hSel"])
        self.rc = int(self.hp["rc"])
        self.ridge = float(self.hp["lambda"])
        self.val_frac = float(self.hp["valFrac"])
        self.val_folds = max(1, int(self.hp["valFolds"]))
        self.n_val = 0 if self.val_frac <= 0.0 else max(self.val_folds * (self.hh + 1), int(round(self.val_frac * self.tr_size)))
        self.fit_size = self.tr_size - self.n_val
        self.fit_rows = range(0, self.fit_size)
        self.val_rows = range(self.fit_size, self.tr_size)
        if self.n_val == 0:
            self.blocks: List[range] = []
        else:
            b = self.n_val // self.val_folds
            self.blocks = [range(self.fit_size + i * b, self.tr_size if i == self.val_folds - 1 else self.fit_size + (i + 1) * b)
                           for i in range(self.val_folds)]
        self.max_free = int(self.hp["maxFree"])
        self.bound_tol = float(self.hp["boundTol"])
        self.max_vif = float(self.hp["maxVIF"])
        self.optimizer = str(self.hp["optimizer"])
        self.evaluations = 0
        if self.tr_size <= self.skip + 2:
            raise ValueError(f"FormulaScorer: tr_size = {self.tr_size} too small for skip = {self.skip}")
        if self.n_val > 0 and self.fit_size <= self.skip + self.hh + 1:
            raise ValueError(f"FormulaScorer: fit rows = {self.fit_size} too few for skip = {self.skip} and hh = {self.hh}: lower valFrac")
        if freqs is not None:
            self.freqs = list(freqs)
        else:
            from modeling.forecasting.llm_narx_sr.eda import top_freqs
            resp = "y" if "y" in self.data else self.vars[0]
            self.freqs = list(top_freqs(self.data[resp][: self.tr_size], int(self.hp["nFreqs"])))

    # -- parsing and fitting -------------------------------------------------------------

    def parse(self, text: str) -> Formula:
        """Parses a formula string against this dataset (variables, maxLag, frequencies)."""
        return parse_formula(text, self.vars, int(self.hp["maxLag"]), self.freqs)

    def _formula(self, f: Union[str, Formula]) -> Formula:
        return self.parse(f) if isinstance(f, str) else f

    def build(self, f: Union[str, Formula], size: Optional[int] = None, fit_params: bool = True) -> NARX_SR_D:
        """
        Builds the model of a formula with its nonlinear parameters fitted on the rows whose
        targets lie in the first `size` rows (default: the fit rows); untrained.
        """
        size = self.fit_size if size is None else size
        self.evaluations += 1
        return NARX_SR_D.from_formula(self._formula(f), self.data, self.hh, tr_size=size, lam=self.ridge,
                                      scaled=self.scaled, optimizer=self.optimizer,
                                      max_lag=int(self.hp["maxLag"]), freqs=self.freqs,
                                      nneg=bool(self.hp["nneg"]), fit_params=fit_params)

    def train(self, mod: NARX_SR_D, rows: Optional[range] = None) -> NARX_SR_D:
        """Trains the coefficients on the given rows (default: the training rows)."""
        rows = self.tr_rows if rows is None else rows
        sl = slice(rows.start, rows.stop)
        mod.train(mod.Y[sl], mod.X[sl])
        return mod

    def fit(self, f: Union[str, Formula]) -> NARX_SR_D:
        """The candidate's model: parameters and coefficients fitted on the fit rows."""
        return self.train(self.build(f), self.fit_rows)

    def fit_full(self, f: Union[str, Formula]) -> NARX_SR_D:
        """The final model: parameters and coefficients fitted on all the training rows."""
        return self.train(self.build(f, self.tr_size), self.tr_rows)

    # -- scoring -----------------------------------------------------------------------------

    def score(self, f: Union[str, Formula]) -> Candidate:
        """
        Parses (if a string), fits and scores a formula; a reject or a failed fit gives a
        failed candidate with the reason.
        """
        if isinstance(f, str):
            try:
                f = self.parse(f)
            except FormulaError as e:
                return Candidate(None, math.inf, reason=f"parse: {e}", label=f.strip())
        nf = f.n_free
        if 0 <= self.max_free < nf:
            return Candidate(f, math.inf, n_terms=f.n_terms, reason=f"{nf} free parameters, at most {self.max_free} allowed")
        try:
            mod = self.fit(f)
            why = self.rejection(f, mod)
            if why is not None:
                return Candidate(f, math.inf, n_terms=f.n_terms, reason=why)
            q = self.qof_fit(mod)
            qv = self.qof_validation_all(mod, lambda size: self.build(f.fresh(), size))   # own copies: the candidate keeps its fit
            return Candidate(f, self.score_of(q, qv), q, f.n_terms, qof_val=qv)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as e:
            return Candidate(f, math.inf, n_terms=f.n_terms, reason=f"fit failed: {e}")

    def rejection(self, f: Formula, mod: NARX_SR_D) -> Optional[str]:
        """The reason a fitted formula is rejected, or None."""
        pars = [p for p in f.params() if p.free]
        for p in pars:
            rng = p.ub - p.lb
            if math.isfinite(rng) and rng > 0.0:
                tol = self.bound_tol * rng
                if p.value - p.lb <= tol:
                    return f"{p.role.value} {p.value:.4g} at its lower bound {p.lb:.4g}"
                if p.ub - p.value <= tol:
                    return f"{p.role.value} {p.value:.4g} at its upper bound {p.ub:.4g}"
            if p.role is ParRole.POWER and abs(p.value - 1.0) < 0.05:
                return f"exponent {p.value:.4g} is a plain lag"
        if self.max_vif > 0.0 and pars and f.n_terms > 1:
            X = mod.X[: self.fit_size]
            Xc = X - X.mean(axis=0)
            for j, t in enumerate(f.preds):
                if any(p.free for p in t.params()):
                    v = self.vif(Xc, j)
                    if v > self.max_vif:
                        return f"term {t.label()} is collinear with the others (VIF {v:.1f})"
        return None

    @staticmethod
    def vif(Xc: np.ndarray, j: int) -> float:
        """The variance inflation factor of column j of a centered matrix."""
        if Xc.shape[1] < 2:
            return 1.0
        xj = Xc[:, j]
        sst = float(xj @ xj)
        if sst <= 0.0:
            return math.inf
        Xo = np.delete(Xc, j, axis=1)
        b = np.linalg.lstsq(Xo, xj, rcond=None)[0]
        sse = float(np.sum((xj - Xo @ b) ** 2))
        r2 = 1.0 - sse / sst
        return math.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)

    # -- QoF -----------------------------------------------------------------------------------

    def _true_pred(self, mod: NARX_SR_D, rows: range, h: int, no_skip: bool):
        start = rows.start if no_skip else rows.start + self.skip
        stop = min(rows.stop, self.m - h + 1)
        if stop <= start:
            return None, None
        idx = np.arange(start, stop)
        y_true = mod.y[idx + h - 1]
        y_pred = mod.predict_rows(idx)[:, h - 1]
        if mod._yForm is not None:
            y_true = mod._yForm.inverse_transform(y_true.reshape(-1, 1)).ravel()
            y_pred = mod._yForm.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        return y_true, y_pred

    def diagnose_h(self, mod: NARX_SR_D, rows: range, h: int, no_skip: bool = False) -> Dict[str, float]:
        """The QoF of horizon h (1-based) over the forecasts made at the given rows (the first
        `skip` rows dropped unless `no_skip`), in the original scale."""
        y_true, y_pred = self._true_pred(mod, rows, h, no_skip)
        if y_true is None:
            raise ValueError(f"diagnose_h: no rows for horizon {h} in {rows}")
        q = diagnose(y_true, y_pred, n_features=mod.dfr)
        q["dfr"] = mod.dfr
        return q

    def qof_by_h(self, mod: NARX_SR_D, rows: range, no_skip: bool = False) -> List[Dict[str, float]]:
        """The QoF per horizon (a list of hh dicts) over the given rows."""
        return [self.diagnose_h(mod, rows, h, no_skip) for h in range(1, self.hh + 1)]

    def qof_train(self, mod: NARX_SR_D) -> List[Dict[str, float]]:
        return self.qof_by_h(mod, self.tr_rows)

    def qof_fit(self, mod: NARX_SR_D) -> List[Dict[str, float]]:
        return self.qof_by_h(mod, self.fit_rows)

    def qof_validation(self, mod: NARX_SR_D) -> List[Dict[str, float]]:
        """The QoF on the first validation block (the fit rows' when there is none)."""
        return self.qof_fit(mod) if self.n_val == 0 else self.qof_by_h(mod, self.blocks[0], no_skip=True)

    def qof_validation_all(self, mod: NARX_SR_D, fresh) -> List[Dict[str, float]]:
        """
        The mean QoF over every validation block: block 0 from `mod`, block i from a fresh
        model `fresh(size)` whose parameters and coefficients are fitted on the rows before
        the block (a rolling origin inside the training rows).
        """
        q0 = self.qof_validation(mod)
        if self.val_folds == 1 or self.n_val == 0:
            return q0
        qs = [q0]
        for i in range(1, self.val_folds):
            rows = range(0, self.blocks[i].start)
            m_i = self.train(fresh(rows.stop), rows)
            qs.append(self.qof_by_h(m_i, self.blocks[i], no_skip=True))
        return [{k: float(np.mean([q[h][k] for q in qs])) for k in QOF_KEYS} for h in range(self.hh)]

    @staticmethod
    def j_of(qof: Sequence[Dict[str, float]], qof_v: Sequence[Dict[str, float]], extra: int = 0) -> np.ndarray:
        """
        The selection score per horizon: J_h = sMAPE_h (validation) * (1 + 2 p / n), with
        p = dfr + 1 + extra (the parameters of horizon h, the intercept and the fitted nonlinear
        parameters included) and n the rows diagnosed on the fit rows.
        """
        return np.array([qv["SMAPE"] * (1.0 + 2.0 * (q["dfr"] + 1.0 + extra) / q["m"]) for q, qv in zip(qof, qof_v)])

    def score_of(self, qof: Sequence[Dict[str, float]], qof_v: Optional[Sequence[Dict[str, float]]] = None,
                 extra: int = 0) -> float:
        """The scalar score: the mean of J_h over the horizons, or J_1 when hSel = 1."""
        sj = self.j_of(qof, qof if qof_v is None else qof_v, extra)
        s = float(sj[0]) if self.h_sel == 1 else float(sj.mean())
        return s if math.isfinite(s) else math.inf

    # -- the final evaluation -----------------------------------------------------------------

    def tnt(self, f: Union[str, Formula]):
        """
        The final evaluation of a formula: the model fitted on all the training rows, then the
        base's rolling validation on the testing rows (retraining the coefficients every hp["rc"]
        rows on a sliding window); returns (model, QoF per horizon on the testing rows).
        """
        mod = self.fit_full(f)
        yf = mod.rollValidate(self.rc)
        return mod, self.qof_tnt(mod, yf)

    def qof_tnt(self, mod: NARX_SR_D, yf: np.ndarray) -> List[Dict[str, float]]:
        """The QoF per horizon of the rolling forecasts `yf` (te_size-by-hh) of the testing rows."""
        out = []
        te = self.te_size
        for h in range(1, self.hh + 1):
            n = te - h + 1
            idx = np.arange(self.tr_size, self.tr_size + n)
            y_true = mod.y[idx + h - 1]
            y_pred = yf[:n, h - 1]
            if mod._yForm is not None:
                y_true = mod._yForm.inverse_transform(y_true.reshape(-1, 1)).ravel()
                y_pred = mod._yForm.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            q = diagnose(y_true, y_pred, n_features=mod.dfr)
            q["dfr"] = mod.dfr
            out.append(q)
        return out

    def report(self, cand: Candidate, tnt: bool = True) -> str:
        """A text report of a candidate: formula, score, in-sample and validation sMAPE, and
        (optionally) the rolling TnT sMAPE on the testing rows."""
        lines = [f"formula:    {cand.formula_str}", f"key:        {cand.key}",
                 f"score J:    {cand.score:.4f}   (n_terms = {cand.n_terms}, n_free = {cand.n_free}){cand.why}"]
        if not cand.failed:
            lines.append("fit sMAPE:  " + " ".join(f"{v:7.3f}" for v in cand.smape_h))
            lines.append("val sMAPE:  " + " ".join(f"{q['SMAPE']:7.3f}" for q in cand.qof_v))
            lines.append("J per h:    " + " ".join(f"{v:7.3f}" for v in cand.score_h))
            if tnt:
                _, qt = self.tnt(cand.formula.fresh())
                lines.append("TnT sMAPE:  " + " ".join(f"{q['SMAPE']:7.3f}" for q in qt) +
                             f"   mean {np.mean([q['SMAPE'] for q in qt]):.3f}")
        return "\n".join(lines)
