# Where the ScalaTion sources used for the port live (this sandbox)

- Base (Dr. Miller's tree, the current ScalaTion): /home/claude/work/rev4/in_miller/scalation/
  - modeling/Fit.scala, FitM.scala, Model.scala, FeatureSelection.scala, SubsetSearch.scala, CEM_Selector.scala, Predictor.scala
  - modeling/forecasting/{Forecaster, Forecaster_D, Forecaster_Reg, ForecastMatrix, Diagnoser, MakeMatrix4TS, AR, RandomWalk,
    ARX, ARX_D, ARX_SR, ARX_SR_D, ARX_Formula, FormulaParser, ExoScreen, Example_Covid, Example_ILI}.scala
  - modeling/forecasting/nonlinear/{NARX_Formula, NARX_FormulaParser, NARX_SR_D}.scala
  - mathstat/Transform.scala, VectorD.scala
- rev4.0 package (the LLM-NARX-SR search layer): /home/claude/work/rev4/rev4.0/scalation/modeling/forecasting/nonlinear/llm_narx_sr/*.scala
  and /home/claude/work/rev4/rev4.0/scalation/modeling/LLMClient.scala, forecasting/{Example_ETTh1, Example_GEFCom}.scala
- ScalaTion run logs (UTF-8) showing the exact printed output: /home/claude/work/logs_utf8/*.log
