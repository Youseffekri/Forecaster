# Changelog

## 1.1.0 -- 2026-10-12: LLM-NARX-SR core layer, four datasets, optimizer options

New
- `modeling/forecasting/formula.py` -- the NARX formula terms (lags, functions, powers, products,
  waves, and the free-parameter terms shift / rate / pow / sat / hinge / trans / ema / dq / wave(?)),
  `Par` with roles and data-driven bounds, `Formula` (key, formula_str, proposal_str, columns, redundancy).
- `modeling/forecasting/formula_parser.py` -- `parse_formula` / `try_parse_formula` with reasons, `grammar_lines`.
- `modeling/forecasting/narx_sr_d.py` -- `NARX_SR_D(Forecaster_D)`: `from_formula` factory, ridge per
  horizon, free parameters by `optimizer="varpro" | "lbfgsb" | "joint"`, `n_free` degrees of freedom.
- `modeling/optimization/varpro.py` -- `VarPro` (reduced residual / objective, `fit(a0, optimizer)`), `ridge_solve`.
- `modeling/forecasting/llm_narx_sr/formula_scorer.py` -- `FormulaScorer` (hp, fit rows and validation blocks,
  `score` -> `Candidate`, rejection rules, `tnt`, `report`), `qof_table`.
- `modeling/forecasting/llm_narx_sr/datasets.py` -- `Example_Covid`, `Example_ILI`, `Example_ETTh1`,
  `Example_GEFCom`, `DatasetConfig`, `Datasets` (COVID2, COVID, ILI, ETTH1, ETTH1_1Y, GEF14).
- `modeling/forecasting/llm_narx_sr/eda.py` -- periodogram `top_freqs` for `wave(k)`.
- `data/ETTh1.csv` (original ETDataset file, md5 8381763947c85f4be6ac456c508460d6),
  `data/gefcom2014_load_daily.csv` (2,099 days) and `tools/build_gefcom_daily.py` that builds it.
- `tests/` -- 46 pytest tests (parser, model + optimizers, scorer, datasets); `Examples/Example_LLM_NARX_SR.py`.
- `pyproject.toml` (`pip install -e .`), `__init__.py` files, `requirements.txt` re-saved as UTF-8 (it was UTF-16).

Unchanged
- The existing models (`AR_YW`, `ARX`, `ARX_D`, `ARX_Symb`, `ARX_Symb_D`, `MHAttn_Regressor`) and `Util`.
  `NARX_SR_D` on a plain-lag formula with `lam=0` reproduces `ARX_D` (tested).

Known
- `Examples/Example_Covid19.py` imports `modeling.neuralnet.neuralnet_3l`, a module no longer in the tree.
- `data/national_illness_clip.csv` (ILI) is not shipped; `Example_ILI` says where it goes.
