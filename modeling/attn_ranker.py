"""
Attn_Ranker Module: Attention Scores as an Importance per Column

An attention-based regressor whose attention weights rank the columns of a design matrix -- the
ranking the `Ranked` selection technique consumes (`SubsetSearch.ranked`, `rankedElimAll`,
`TermLibrary.selectBy (Ranked, scorer, rank)`): the ranker decides which column leaves at each
step, the selection score only where to stop.  The class has the shape of `CEM_Selector`, the
other vector-producing algorithm of the platform: hyper-parameters in the constructor, `run ()`
does the work, and the result is a vector (`scores`, as `probs` is CEM's) that the selectors read.

The regressor (`AttnRegressor`) gives EVERY column its own embedding, e_j = x_j w_j + b_j (a
(n, d) weight, not one `Linear (1, d)` shared by all columns), so the attention a column receives
can depend on WHICH column it is: with a shared embedding and mean pooling a model is
permutation-invariant over the columns and its attention ranks value distributions, not relevance.
Then one block of self-attention over the n column tokens (residual + LayerNorm), a GELU
feed-forward, mean pooling over the tokens and a linear output.

Two scores per column, both measured on the rows the ranker was given (the fit rows of the
selection, never the testing rows), averaged over `seeds` trainings:

    attention        s_k = the mean attention column k receives as a KEY, over rows and queries
                     (the NARX_SR paper's eq. 4; the heads averaged)
    gradient x input g_k = mean over rows of |d yhat / d x_k * x_k|

`run ()` returns the attention score (`score = "attn"`) or the gradient score (`"grad"`); both are
kept (`scores`, `gradScores`).  Another regressor (the paper's `MHAttn_Regressor`) can be plugged in
through `regressor`, so the two can be compared on one footing.  A ranking is only as good as its training: standardize (done
here), average over seeds, and keep the rows the chooser fits on.

PyTorch is imported on first use (an optional dependency: `pip install torch`); without it the
constructor raises an ImportError that says so.  The module is the Python-first form of
ScalaTion's `autograd/Attn_Ranker.scala` (to follow the experiment that compares the attention
ranking with the coefficient ranking).

Author: Yousef Fekri Dabanloo
Date: 2026-10-15
Version: 1.0
License: MIT License
Note: AI assisted code
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

__all__ = ["Attn_Ranker"]


def _torch():
    """PyTorch and its nn module, imported on demand (an optional dependency)."""
    try:
        import torch
        from torch import nn
    except ImportError as e:                                             # pragma: no cover
        raise ImportError("Attn_Ranker needs PyTorch (pip install torch); the coefficient ranker needs nothing") from e
    return torch, nn


def _regressor(torch: Any, nn: Any):
    """The `AttnRegressor` class, built once torch is known to be present."""

    class AttnRegressor(nn.Module):
        """
        Self-attention over the n columns of a row, each column its own token: per-column
        embedding e_j = x_j w_j + b_j (d_model), multi-head self-attention with residual and
        LayerNorm, a GELU feed-forward with residual and LayerNorm, mean pooling over the tokens,
        a linear output.  `forward (x)` returns (yhat (B, 1), attention (B, n, n), heads averaged).
        """

        def __init__(self, n: int, dModel: int = 32, heads: int = 1, hidden: int = 64, dropout: float = 0.1) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.randn(n, dModel) * 0.5)         # per-column embedding weights
            self.b = nn.Parameter(torch.zeros(n, dModel))                # per-column embedding biases
            self.attn = nn.MultiheadAttention(dModel, heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(dModel)
            self.ff = nn.Sequential(nn.Linear(dModel, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dModel))
            self.norm2 = nn.LayerNorm(dModel)
            self.out = nn.Linear(dModel, 1)

        def forward(self, x):                                            # x: (B, n)
            e = x.unsqueeze(-1) * self.w + self.b                        # (B, n, d): column j's own embedding
            a, A = self.attn(e, e, e, need_weights=True, average_attn_weights=True)   # A: (B, n, n), rows = queries
            h = self.norm1(e + a)
            h = self.norm2(h + self.ff(h))
            return self.out(h.mean(dim=1)), A

    return AttnRegressor


class Attn_Ranker:
    """
    Trains attention regressors on (x, y) and ranks the columns of x (see the module docstring).

    Parameters
    ----------
    x : np.ndarray, shape (m, n)
        The design matrix (the fit rows of the selection; standardized here, column by column).
    y : np.ndarray, shape (m,)
        The response (standardized here).
    dModel : int, default=32
        The embedding size.
    heads : int, default=1
        The attention heads (the attention matrix is averaged over them).
    hidden : int, default=64
        The feed-forward width.
    dropout : float, default=0.1
        The dropout of the attention and the feed-forward.
    epochs : int, default=100
        The training epochs per seed.
    lr : float, default=1e-3
        Adam's learning rate.
    weightDecay : float, default=1e-4
        Adam's weight decay (with far more parameters than rows, some shrinkage is needed).
    batch : int, default=16
        The mini-batch size.
    seeds : sequence of int, default=(0, 1, 2, 3, 4)
        The trainings whose scores are averaged.
    stream : int, default=0
        Added to every seed (a run's stream).
    score : {"attn", "grad"}, default="attn"
        The score `run` returns: the attention received as a key, or gradient x input.
    regressor : callable, optional
        A factory n -> nn.Module whose `forward (x)` returns (yhat, attention), the attention
        (B, n, n) or already averaged over the batch (n, n); None => `AttnRegressor` with the
        per-column embedding.  The paper's `MHAttn_Regressor` (neuralforecasting) fits here as
        `lambda n: MHAttn_Regressor (input_dim = n, d_model = 72, num_heads = 4)`, for the comparison.

    Attributes
    ----------
    scores, gradScores : np.ndarray or None
        The two scores per column after `run` (None before).
    losses : list of float
        The final training loss of every seed.
    """

    DEFAULT = dict(dModel=32, heads=1, hidden=64, dropout=0.1, epochs=100, lr=1e-3, weightDecay=1e-4, batch=16)

    def __init__(self, x: np.ndarray, y: np.ndarray, dModel: int = 32, heads: int = 1, hidden: int = 64,
                 dropout: float = 0.1, epochs: int = 100, lr: float = 1e-3, weightDecay: float = 1e-4,
                 batch: int = 16, seeds: Sequence[int] = (0, 1, 2, 3, 4), stream: int = 0,
                 score: str = "attn", regressor: Optional[Any] = None) -> None:
        _torch()                                                         # fail here, not in run, when torch is missing
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        if x.ndim != 2 or len(y) != x.shape[0]:
            raise ValueError(f"Attn_Ranker: x must be (m, n) and y (m,), got {x.shape} and {y.shape}")
        if score not in ("attn", "grad"):
            raise ValueError(f"Attn_Ranker: score must be 'attn' or 'grad', got {score!r}")
        self.m, self.n = x.shape
        self.xs = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)        # standardized columns
        self.ys = (y - y.mean()) / (y.std() + 1e-12)
        self.dModel, self.heads, self.hidden, self.dropout = int(dModel), int(heads), int(hidden), float(dropout)
        self.epochs, self.lr, self.weightDecay, self.batch = int(epochs), float(lr), float(weightDecay), int(batch)
        self.seeds, self.stream, self.score = list(seeds), int(stream), score
        self.regressor = regressor
        self.scores: Optional[np.ndarray] = None
        self.gradScores: Optional[np.ndarray] = None
        self.losses: List[float] = []

    @property
    def name(self) -> str:
        return "attn"

    def _train(self, seed: int):
        """One regressor trained from one seed; returns it (eval mode)."""
        torch, nn = _torch()
        torch.manual_seed(seed + self.stream)
        X = torch.tensor(self.xs)
        Y = torch.tensor(self.ys).unsqueeze(-1)
        net = (_regressor(torch, nn)(self.n, self.dModel, self.heads, self.hidden, self.dropout)
               if self.regressor is None else self.regressor(self.n))
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weightDecay)
        net.train()
        loss_v = float("nan")
        for _ in range(self.epochs):
            perm = torch.randperm(self.m)
            for i in range(0, self.m, self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                yhat, _ = net(X[idx])
                loss = nn.functional.mse_loss(yhat, Y[idx])
                loss.backward()
                opt.step()
                loss_v = float(loss.item())
        self.losses.append(loss_v)
        net.eval()
        return net, X

    def run(self) -> np.ndarray:
        """
        Trains one regressor per seed on the standardized rows and averages the two scores over the
        seeds: `scores` (attention received as a key, eq. 4) and `gradScores` (gradient x input);
        returns the one chosen by `score`, a length-n vector (higher = more important).
        """
        torch, _ = _torch()
        attn = np.zeros(self.n)
        grad = np.zeros(self.n)
        self.losses = []
        for seed in self.seeds:
            net, X = self._train(seed)
            with torch.no_grad():
                _, A = net(X)                                            # (m, n, n): rows = queries, columns = keys
                A = A.mean(dim=0) if A.dim() == 3 else A                 # (n, n): over the rows (a heat map is one already)
                attn += A.mean(dim=0).numpy()                            # received as a key: over the queries
            Xg = X.clone().requires_grad_(True)
            yhat, _ = net(Xg)
            yhat.sum().backward()
            grad += (Xg.grad * Xg).abs().mean(dim=0).detach().numpy()   # gradient x input, over rows
        self.scores = attn / len(self.seeds)
        self.gradScores = grad / len(self.seeds)
        return self.scores if self.score == "attn" else self.gradScores

    def ranking(self) -> List[int]:
        """The columns from most to least important by the chosen score (after `run`)."""
        s = self.scores if self.score == "attn" else self.gradScores
        if s is None:
            raise ValueError("Attn_Ranker: call run first")
        return [int(j) for j in np.argsort(-s)]

    def __repr__(self) -> str:
        return (f"Attn_Ranker(m={self.m}, n={self.n}, dModel={self.dModel}, heads={self.heads}, epochs={self.epochs}, "
                f"seeds={len(self.seeds)}, score={self.score!r})")
