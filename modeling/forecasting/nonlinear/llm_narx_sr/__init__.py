"""
LLM-NARX-SR: LLM-guided symbolic regression over NARX models -- the Python port of ScalaTion's
`modeling/forecasting/nonlinear/llm_narx_sr` package (rev4.0), module by module:

    formula_scorer        FormulaScorer (and its `hp`), Candidate (the fit-and-score engine, L1)
    eda4ts                EDA4TS, topFreqs, ljungBox, ... (the data profile and the prompt block, L1)
    datasets              DatasetConfig, Datasets, ExoMode, ExoScreen (what an experiment runs on, L2)
    term_library          TermLibrary (term libraries and the CEM / backward / ... selection, L2)
    search_arm            Method, Source, ProposerKind, LibraryKind, ArmSpec, Trace, ExpRun, SearchArm (L3)
    search_loop           Proposer, Reject, SearchLoop (the propose -> trim -> score -> feed back loop, L4)
    genetic_proposer      RandomTerms, GeneticProposer (the zero-LLM proposers, L4)
    llm_proposer          LLMProposer (the prompt and the parsing of the reply, L4)
    explainer             TermEffect, Claim, Explanation, Explainer (L4)
    baseline_arm          BaselineArm (the anchors, L5)
    pipeline_arm          PipelineArm (every searching arm, L5)
    plausibility_checker  PlausibilityReport, PlausibilityChecker (L5)
    significance          SigResult, Significance, wilcoxon_test (L5)
    llm_narx_sr           LLMModel, LLM_NARX_SR (the experiment harness, L6)

and `modeling/llm_client.py` (LLMClient, OpenAIClient, AnthropicClient, LocalClient, CachedClient,
CountingClient, MockLLMClient).
"""
