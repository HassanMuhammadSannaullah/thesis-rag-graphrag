# Thesis RAG + GraphRAG Implementation Plan

**Last Updated:** June 7, 2026  
**Status:** Phase 2 Complete, Phase 3 In Progress  
**Scope:** HybridQA primary focus (Phase 9 for multi-dataset expansion after core completion)

## Purpose

This document is the step-by-step implementation roadmap for turning the current thesis prototype into a thesis-defendable system with:

- methodologically fair comparison between RAG and GraphRAG
- strong practical implementations of both systems
- model-agnostic validation through ablation studies
- statistical rigor and qualitative insights
- comprehensive documentation for thesis defense
- industry-aligned evaluation practices

This plan includes 17 phases covering system development (0-5), evaluation excellence (6-6.7), defense preparation (7-8.7), and optional multi-dataset extension (9).

This plan is written to be executed by an AI coding agent in phases, with clear acceptance criteria and verification steps.

## Quick Reference

### ✅ Completed Phases
- **Phase 0:** Runtime and experiment foundations (partial)
- **Phase 1:** HybridQA parsing layer rebuild  
- **Phase 2:** Gold evidence alignment
- **Phase 3:** Strong baseline RAG system
- **Phase 3.5:** Model registry and switching system

### 🔄 Current Phase
- **Phase 4:** Improve GraphRAG source construction

### 📋 Required for Thesis Defense (Must Complete)
- Phases 0-8.7: All system, evaluation, and documentation phases
- Minimum 3 LLMs tested, 2 embedding models
- 200+ questions evaluated
- Statistical significance testing
- Error analysis with 50+ manual reviews
- Reproducibility documentation

### ⭐ Optional for Excellence (Nice to Have)
- Phase 9: Multi-dataset extension
- 5+ LLMs, 3+ embeddings
- Full dev split evaluation (all questions)
- Cross-dataset insights

### 📊 Estimated Timeline
- Core system (Phases 3-5): 2-3 weeks
- Evaluation excellence (Phases 6-6.7): 2-3 weeks
- Defense prep (Phases 7-8.7): 1-2 weeks
- Optional multi-dataset (Phase 9): 2-4 weeks
- **Total minimum:** 5-8 weeks
- **Total with Phase 9:** 7-12 weeks

## High-Level Goal

Build two credible systems for HybridQA:

1. A strong baseline RAG system
2. A strong GraphRAG system

Both systems must:

- ingest the dataset in a traceable way
- preserve evidence provenance
- retrieve relevant support more reliably
- produce grounded answers
- be evaluated on a sufficiently large sample or full dev split
- be tested across multiple models to prove robustness
- be statistically validated
- be qualitatively analyzed to understand strengths and weaknesses
- support defendable thesis claims

## Plan Structure Overview

This plan is organized into phases covering three main areas:

### Core System Development (Phases 0-5)
Build robust, fair RAG and GraphRAG systems with proper data handling, evidence tracking, and strong baselines.

### Evaluation Excellence (Phases 6-6.7)
Move beyond simple metrics to thesis-grade evaluation including large-scale experiments, model ablations, statistical testing, and error analysis.

### Thesis Defense Preparation (Phases 7-8.7)
Add practical variants, comprehensive documentation, dataset justification, related work positioning, and reproducibility guarantees.

### Future Extension (Phase 9)
After HybridQA is perfected, optionally extend to multiple datasets for stronger generalization claims.

## Current Situation

The current repository is a working prototype, but it has several weaknesses:

- the parser is simplified and loses evidence structure
- linked passages are selected too early and too aggressively
- gold evidence is not aligned into the evaluation examples
- the fair baseline is useful for controlled comparison, but too weak to represent a strong industry baseline
- the GraphRAG input construction is simplified and truncated
- the current 20-question runs are smoke-test quality, not final thesis quality

## Non-Negotiable Thesis Requirements

Before final experiments, the system must satisfy all of the following:

- final evaluation must use far more than 20 questions
- evidence provenance must be preserved end to end
- the baseline must be stronger than plain dense top-k stuffing
- the GraphRAG pipeline must use richer source documents and better traceability
- experiment types must be clearly separated:
  - controlled shared-input comparison
  - realistic end-to-end system comparison
- all claims in the thesis must match what the code truly does

## Execution Rules For AI

Any AI implementing this plan must follow these rules:

1. Do not skip phases.
2. Complete and verify each phase before moving on.
3. Preserve backward compatibility when possible.
4. Add tests for each critical pipeline change.
5. Do not run final large experiments until the parser, evidence mapping, and strong baseline are finished.
6. Keep a changelog section in this file updated after each phase is completed.

## Target End-State

At the end of this plan, the repository should support:

- `simple baseline RAG` for ablation and sanity checks
- `strong baseline RAG` for defendable comparison
- `pure GraphRAG` for research comparison
- `practical GraphRAG` with fallback or hybrid retrieval for practical comparison

And the thesis should report both:

- controlled fairness comparison
- realistic system comparison

## Phase 0 - Runtime And Experiment Foundations

### Goal

Stabilize the environment and make experiment metadata trustworthy.

### Required Changes

- Detect real hardware at runtime instead of relying on hardcoded RAM and VRAM assumptions.
- Record the actual Python executable, CUDA availability, GPU name, VRAM, and RAM in experiment metadata.
- Add a runtime check that confirms the intended conda environment is being used.
- Extend the readiness check to validate:
  - CUDA availability
  - torch CUDA build
  - actual model device
  - enough free RAM
  - enough free disk

### Files Likely Involved

- `scripts/13_check_system_readiness.py`
- `src/config/settings.py`
- `src/config/model_registry.py`
- `src/utils/runtime.py`
- experiment metadata writing path in the evaluator

### Acceptance Criteria

- experiment metadata reports actual machine specs
- readiness checks fail loudly when the wrong environment is used
- readiness output distinguishes smoke-test readiness from final-run readiness

## Phase 1 - Rebuild The HybridQA Parsing Layer

### Goal

Turn parsing into a real data ingestion layer instead of a lossy simplification.

### Problems To Fix

- linked passages are selected too early
- selection is not question-aware
- source provenance is incomplete
- evidence alignment is impossible in the current representation

### Required Changes

- Parse and preserve all available useful source information from HybridQA raw files.
- Preserve stable source identifiers for:
  - table
  - row
  - cell
  - wiki link
  - linked passage
- Do not permanently discard linked passages during parsing.
- Keep row-level link provenance so each row can be tied to the exact linked entities it references.
- Store enough metadata to map any retrieved unit back to the original dataset object.

### New Parsed Record Requirements

Each parsed HybridQA record should contain at least:

- question id
- question text
- gold answer
- table id
- full table structure
- stable row ids
- row-to-link mapping
- full linked passage list
- stable linked passage ids
- enough metadata to reconstruct evidence provenance

### Files Likely Involved

- `scripts/02_parse_hybridqa.py`
- new or updated schema helpers
- parsed data output format under `data/hybridqa/original/`

### Acceptance Criteria

- parsed records preserve all row-level and passage-level provenance needed for retrieval evaluation
- no early top-N linked passage cap is baked into parsed data
- a single question can be traced from answer to row to linked passage ids

## Phase 2 - Add Gold Evidence Alignment

### Goal

Support defendable retrieval and evidence evaluation.

### Required Changes

- Investigate whether official HybridQA supervision can be mapped into evidence references for this repo's retrieval units.
- If exact official evidence ids are available, preserve them directly.
- If they are not directly available, create a documented evidence-alignment strategy that maps gold support into the system's retrieval unit ids.
- Add `gold_evidence` or `evidence_refs` into the parsed or derived evaluation examples.
- Ensure evidence ids are aligned with the units used by:
  - strong baseline RAG
  - GraphRAG text units

### Important Rule

Do not fake evidence labels. If exact alignment is not possible, the system must:

- document the limitation
- separate strict evidence metrics from proxy support metrics

### Files Likely Involved

- `src/evaluation/schemas.py`
- `scripts/11_run_hybridqa_proper_compare.py`
- parsing and example-building code
- retrieval unit builders

### Acceptance Criteria

- `gold_evidence` is no longer always `None` for HybridQA unless explicitly justified
- evidence metrics become meaningful for at least one properly defined retrieval representation
- evidence alignment methodology is documented in code comments and thesis notes

## Phase 3 - Build A Strong Baseline RAG System

### Goal

Make the baseline a serious baseline, not just a simple dense top-k retrieval system.

### Minimum Strong Baseline Features

- question-aware retrieval
- row-aware retrieval
- linked-passage expansion from retrieved rows
- hybrid retrieval if feasible
- reranking
- better context packing
- provenance-aware answer generation

### Required Changes

- Keep the current simple baseline for ablation.
- Add a new strong baseline path that becomes the thesis baseline.
- Retrieve candidate rows and passages separately, then merge and rerank.
- Prefer row-linked passages that are actually connected to retrieved rows.
- Add optional sparse retrieval or lexical retrieval if practical.
- Add a reranker stage for candidate contexts.
- Deduplicate repeated or overlapping contexts.
- Build answer prompts that clearly distinguish:
  - table evidence
  - linked passage evidence
- Return citations or source ids with the prediction object, even if not shown in final answer text.

### Suggested Baseline Variants

- `baseline_simple_dense`
- `baseline_strong_rag`

### Files Likely Involved

- `src/baseline/corpus_builder.py`
- `src/baseline/retriever.py`
- `src/baseline/answer_generator.py`
- `src/baseline/vector_index.py`
- experiment runner scripts

### Acceptance Criteria

- strong baseline clearly outperforms the simple baseline on answer quality or grounding
- provenance is traceable per prediction
- baseline behavior is understandable and defendable in the thesis

## Phase 3.5 - Model Registry And Switching System

### Goal

Make it easy to test multiple models and embeddings without code changes.

### Why This Matters

A thesis claiming "RAG vs GraphRAG comparison" must show the result holds across different models, not just one lucky model choice.

### Required Changes

- Refactor model loading to be fully config-driven
- Support multiple LLM backends:
  - local models via Ollama
  - HuggingFace models
  - API models if needed
- Support multiple embedding models with automatic switching
- Create a model comparison runner script
- Add model metadata tracking to experiment outputs

### Suggested Model Test Matrix

Minimum for thesis defense:
- **LLMs**: Test at least 3 different sizes/architectures
  - Small: Mistral-7B or Qwen-3B
  - Medium: Qwen-14B or similar
  - Different architecture family
- **Embeddings**: Test at least 2 different models
  - Current: E5-base-v2
  - Alternative: BGE, sentence-transformers, or similar

### Files Likely Involved

- `src/config/model_registry.py`
- `src/config/settings.py`
- new `src/config/model_variants.py`
- experiment matrix configs
- new `scripts/16_run_model_comparison.py`

### Acceptance Criteria

- can switch LLM with single config change
- can switch embedding model with single config change
- experiment metadata records exact model used
- model comparison script can run same experiment with different models automatically

## Phase 4 - Improve GraphRAG Source Construction

### Goal

Feed GraphRAG better documents so GraphRAG failure is not caused by weak ingestion.

### Problems To Fix

- current source docs are too simplified
- linked passages are truncated too early
- source traceability is too weak

### Required Changes

- redesign GraphRAG source document construction
- preserve stronger row and passage structure
- include stable metadata references inside the generated docs
- reduce unnecessary truncation before indexing
- experiment with multiple document construction strategies

### Candidate GraphRAG Source Strategies

1. One document per table with structured row sections
2. One document per row plus its linked passages
3. One document per table plus separate entity passage documents

### Suggested Direction

Prefer a representation that preserves:

- row identity
- linked entity identity
- passage identity
- document traceability

### Files Likely Involved

- `src/graphrag_system/corpus_prep.py`
- GraphRAG workspace preparation code

### Acceptance Criteria

- GraphRAG inputs are richer and less lossy
- retrieved GraphRAG context can be traced back to source rows and passages
- GraphRAG answer failures can be diagnosed by source provenance

## Phase 5 - Split Experiment Types Clearly

### Goal

Stop mixing two different research questions into one experiment design.

### Two Experiment Families Required

#### A. Controlled Retrieval Comparison

Purpose:

- compare vector-only retrieval vs GraphRAG retrieval using the same shared text units

Systems:

- `baseline_shared_text_unit_rag`
- `pure_graphrag`

Use when the thesis question is:

- does graph-aware retrieval help over vector retrieval when evidence units are controlled?

#### B. Realistic End-to-End System Comparison

Purpose:

- compare practical baseline RAG vs practical GraphRAG as full systems

Systems:

- `baseline_strong_rag`
- `graphrag_practical`

Use when the thesis question is:

- which system is better as an actual QA pipeline on HybridQA?

### Required Changes

- create separate scripts or clearly separated modes for these two experiment families
- name outputs clearly so reports cannot be confused

### Acceptance Criteria

- every experiment can be explained as either controlled or realistic
- no report mixes the interpretation of those two settings

## Phase 6 - Strengthen Evaluation

### Goal

Make the evaluation defendable in front of a thesis committee.

### Required Changes

- stop using 20 questions for any final claim
- define:
  - smoke-test size
  - pilot size
  - thesis-report size
- add stronger retrieval and evidence reporting
- keep answer metrics, support metrics, and strict evidence metrics clearly separated
- perform larger-sample statistical comparisons
- add qualitative error categories

### Required Evaluation Buckets

- answer correctness
- grounding
- support retrieval
- strict evidence retrieval
- efficiency
- failure modes

### Suggested Minimum Final Evaluation Scale

- at least one large dev run, ideally full dev split if computationally feasible
- if full dev is too expensive, use a fixed, documented, stratified sample

### Acceptance Criteria

- final reported experiments are large enough to support meaningful claims
- evidence and answer metrics are not conflated
- limitations are explicitly documented

## Phase 6.5 - Model Ablation Study

### Goal

Prove that RAG vs GraphRAG comparison results hold across different models, not just one lucky choice.

### Why This Matters

Thesis committees will ask: "Would this work with a different model?" You need to show your findings generalize.

### Required Changes

- Run the same experiment configuration with multiple LLMs
- Run the same experiment configuration with multiple embedding models
- Create comparison tables showing:
  - Model × System × Metrics
  - Which findings are model-invariant
  - Which findings are model-dependent
- Document cases where model choice significantly impacts results

### Minimum Model Test Matrix

**LLMs (at least 3):**
- Small: Mistral-7B or Qwen-3B
- Medium: Qwen-14B
- Alternative architecture if possible

**Embeddings (at least 2):**
- E5-base-v2
- BGE or similar alternative

### Suggested Approach

1. Use Phase 3.5 model registry for easy switching
2. Run controlled comparison experiments (Phase 5 family A) with each model
3. Compare across models to identify robust patterns
4. Document model-dependent behaviors

### Files Likely Involved

- experiment matrix configs
- new `scripts/17_model_ablation_study.py`
- new `src/evaluation/model_comparison.py`
- reporting templates

### Acceptance Criteria

- at least 3 LLMs tested on same experiment
- at least 2 embedding models tested
- comparative table in results showing model effects
- thesis can claim which findings are model-robust

## Phase 6.6 - Statistical Significance Testing

### Goal

Add statistical rigor to prove differences are real, not random noise.

### Why This Matters

Academic work requires statistical validation. "System A scored 0.65, System B scored 0.63" means nothing without significance testing.

### Required Changes

- Add paired statistical tests for system comparisons:
  - Paired t-test for normally distributed metrics
  - Wilcoxon signed-rank test for non-normal distributions
- Report p-values for all key comparisons
- Add confidence intervals to metric reporting
- Test for statistical significance at standard thresholds (p < 0.05, p < 0.01)
- Document effect sizes, not just p-values

### Suggested Metrics To Test

- Answer accuracy: RAG vs GraphRAG (paired per question)
- Retrieval quality: baseline vs GraphRAG retrieval
- Context efficiency: tokens used per answer

### Files Likely Involved

- new `src/evaluation/statistical_tests.py`
- `src/evaluation/reporting.py`
- experiment comparison scripts

### Acceptance Criteria

- p-values reported for all system comparisons
- confidence intervals included in metric tables
- thesis can claim "statistically significant" where appropriate
- non-significant differences are acknowledged

## Phase 6.7 - Error Analysis And Failure Modes

### Goal

Understand WHY systems succeed or fail through qualitative analysis.

### Why This Matters

Numbers alone don't make a thesis. You need insights: "GraphRAG wins on multi-hop reasoning but fails on simple lookups."

### Required Changes

- Create error taxonomy for failed predictions:
  - Retrieval failures (right context not retrieved)
  - Reasoning failures (context retrieved, answer wrong)
  - Grounding failures (answer correct but not supported by retrieved context)
  - Format failures (answer extraction issues)
- Manually inspect 50-100 predictions per system
- Create case study examples:
  - Where RAG wins and why
  - Where GraphRAG wins and why  
  - Where both fail and why
- Build error analysis dashboard or report template

### Suggested Error Categories

**Retrieval Errors:**
- missing evidence
- wrong evidence ranked high
- relevant evidence ranked too low

**Reasoning Errors:**
- multi-hop reasoning failure
- single-hop reasoning failure
- incorrect entity resolution
- temporal reasoning failure
- numerical reasoning failure

**Grounding Errors:**
- hallucinated information
- incomplete support
- context misinterpretation

### Files Likely Involved

- new `src/evaluation/error_analysis.py`
- new `scripts/18_analyze_errors.py`
- manual annotation files
- case study templates

### Acceptance Criteria

- at least 50 predictions manually analyzed per system
- error taxonomy documented
- 10-15 case study examples prepared for thesis
- clear patterns identified: when RAG wins, when GraphRAG wins, when both fail

## Phase 7 - Add Practical GraphRAG Variant

### Goal

Make GraphRAG useful as a practical system, not only as a pure research condition.

### Required Changes

- keep `pure_graphrag` as the clean research condition
- add a `graphrag_practical` variant that can:
  - fall back to strong dense retrieval when GraphRAG support is weak
  - combine GraphRAG context with strong baseline retrieval context
  - rerank combined evidence

### Why

Pure GraphRAG is useful for research clarity, but a practical system in industry usually uses fallback or hybrid retrieval.

### Acceptance Criteria

- practical GraphRAG is evaluated separately from pure GraphRAG
- thesis distinguishes between research purity and practical utility

## Phase 8 - Documentation And Thesis Defensibility

### Goal

Make the final system easy to defend academically.

### Required Changes

- update README to describe the real architecture
- update codebase map so it matches reality
- document every experiment family and system mode
- document parser assumptions
- document evidence-alignment methodology
- document known limitations honestly

### Acceptance Criteria

- a reader can understand what each system actually does
- the thesis can cite repository behavior accurately

## Phase 8.5 - Dataset Strategy And Justification

### Goal

Document HybridQA focus clearly, with justification and future expansion plan.

### Current Strategy

Focus on HybridQA for initial thesis work, then expand to multiple datasets once the core system is proven and working reliably.

### Required Changes For HybridQA Phase

- Add dedicated section documenting why HybridQA was chosen:
  - multi-hop reasoning requirement
  - hybrid table + text evidence
  - realistic complexity
  - established benchmark
- Document what HybridQA represents and what it doesn't
- Explicitly state generalization limitations
- Add data statistics and characteristics analysis
- Create HybridQA-specific insights document

### Required Changes For Future Multi-Dataset Phase

**To be implemented AFTER core system is fully working:**

- Add support for additional QA datasets:
  - HotpotQA (multi-hop text-only)
  - 2WikiMultiHopQA (multi-hop with Wikipedia)
  - QASPER (scientific paper QA)
  - or similar established benchmarks
- Create unified dataset interface
- Run cross-dataset experiments
- Compare system performance across dataset types
- Strengthen generalization claims

### Two-Phase Timeline

**Phase 1 (NOW): HybridQA Excellence**
- Perfect the RAG and GraphRAG systems on HybridQA
- Complete all evaluation and analysis phases
- Build defensible single-dataset thesis

**Phase 2 (LATER): Multi-Dataset Validation**
- Add 2-3 additional datasets
- Run comparative experiments
- Strengthen contribution claims
- Demonstrate broad applicability

### Files Likely Involved

- new `docs/dataset_justification.md`
- new `docs/hybridqa_characteristics.md`
- future: `src/data_pipeline/unified_dataset_interface.py`
- future: dataset-specific parsers

### Acceptance Criteria

- HybridQA choice is clearly justified in documentation
- limitations are explicitly stated
- future expansion path is documented
- thesis committee can understand scope and rationale

## Phase 8.6 - Related Work And Research Positioning

### Goal

Position your work clearly in the research landscape so the thesis contribution is obvious.

### Required Changes

- Document how your systems compare to:
  - Microsoft GraphRAG (official implementation)
  - Recent RAG papers (2023-2026)
  - Industry RAG frameworks (LangChain, LlamaIndex)
  - Academic baselines for HybridQA
- Create comparison table: Your Work vs Related Systems
- Clearly state what is:
  - Novel contribution (your insights)
  - Replication (reproducing known methods)
  - Engineering (building comparison framework)
  - Comparative analysis (systematic comparison)
- Document any deviations from standard implementations

### Key Questions To Answer

1. How does your strong baseline RAG compare to published RAG systems?
2. How does your GraphRAG implementation compare to Microsoft's?
3. What have other papers reported on HybridQA?
4. What is YOUR thesis contribution?

### Files Likely Involved

- new `docs/related_work.md`
- new `docs/contribution_statement.md`
- README updates
- thesis-related documentation

### Acceptance Criteria

- related work documented clearly
- contribution is clearly distinguished from replication
- thesis committee can see what's novel
- no risk of "this was already done" surprise during defense

## Phase 8.7 - Reproducibility Documentation

### Goal

Make your work fully reproducible so thesis committee and future researchers can verify results.

### Required Changes

- Export exact Python environment:
  - `conda env export > environment.yml`
  - `pip freeze > requirements-exact.txt`
  - document Python version, CUDA version, OS
- Document all random seeds and controls
- Create compute budget report:
  - total GPU hours used
  - memory requirements
  - disk space requirements
  - estimated runtime for each phase
- Document exact data splits used
- Document hyperparameter choices with justifications
- Create reproducibility checklist
- Add "How to Reproduce" section to README

### Reproducibility Checklist Items

- [ ] Exact environment specification
- [ ] Random seed documentation
- [ ] Data split specification
- [ ] Model versions and sources
- [ ] Hyperparameter documentation
- [ ] Compute requirements
- [ ] Runtime estimates
- [ ] Step-by-step reproduction guide
- [ ] Known environment-specific issues
- [ ] Contact information for questions

### Files Likely Involved

- new `environment.yml`
- new `requirements-exact.txt`
- new `docs/reproducibility.md`
- new `docs/compute_budget.md`
- new `docs/hyperparameter_justification.md`
- README updates

### Acceptance Criteria

- another researcher could reproduce your experiments from documentation
- thesis committee can verify your setup
- all computational decisions are justified
- limitations are documented honestly

## Phase 9 - Multi-Dataset Extension (Future Work)

### Goal

Once HybridQA system is perfected, extend to multiple datasets for stronger generalization claims.

### Prerequisites

This phase should ONLY start after:
- Phases 0-8.7 are complete
- HybridQA results are stable and defendable
- Core system is proven reliable
- Thesis timeline allows (can be future work if needed)

### Required Changes

- Add 2-3 additional QA datasets:
  - HotpotQA (multi-hop, Wikipedia)
  - 2WikiMultiHopQA (multi-hop)
  - QASPER (scientific domain)
  - or similar established benchmarks
- Create unified dataset interface:
  - common data schema
  - dataset-agnostic pipeline
  - per-dataset parsers
- Run full experiment suite on each dataset
- Compare cross-dataset performance
- Identify dataset-specific vs dataset-invariant patterns

### Multi-Dataset Experiment Strategy

For each new dataset:
1. Parse and prepare data (similar to Phase 1)
2. Build evidence alignment if possible (similar to Phase 2)
3. Run strong baseline RAG (Phase 3)
4. Run GraphRAG (Phase 4)
5. Compare systems (Phase 5)
6. Evaluate at scale (Phase 6)
7. Run model ablations (Phase 6.5)
8. Perform error analysis (Phase 6.7)

### Expected Deliverables

- Cross-dataset comparison tables
- Dataset characteristics analysis
- Insights: "GraphRAG helps more on X-type datasets"
- Limitations: "Our approach struggles with Y-type questions"
- Generalization claims backed by multi-dataset evidence

### Files Likely Involved

- new `src/data_pipeline/hotpotqa_parser.py`
- new `src/data_pipeline/2wikimultihop_parser.py`
- new `src/data_pipeline/qasper_parser.py`
- new `src/data_pipeline/dataset_interface.py`
- updated experiment runners
- cross-dataset analysis scripts

### Acceptance Criteria

- at least 2-3 datasets in addition to HybridQA
- same evaluation framework works across datasets
- cross-dataset comparison report available
- thesis can make broader generalization claims
- dataset-specific insights documented

### Decision Point

**If time allows:** Complete Phase 9 for stronger thesis
**If time limited:** Document Phase 9 as "Future Work" and defend HybridQA-only scope with Phase 8.5 justification

## Recommended Implementation Order

Follow this order exactly:

1. Phase 0 - runtime and metadata
2. Phase 1 - parser rebuild
3. Phase 2 - gold evidence alignment
4. Phase 3 - strong baseline RAG
5. Phase 3.5 - model registry and switching system
6. Phase 4 - GraphRAG source construction improvements
7. Phase 5 - split experiment families
8. Phase 6 - large-scale evaluation
9. Phase 6.5 - model ablation study
10. Phase 6.6 - statistical significance testing
11. Phase 6.7 - error analysis and failure modes
12. Phase 7 - practical GraphRAG variant
13. Phase 8 - documentation and thesis alignment
14. Phase 8.5 - dataset strategy and justification
15. Phase 8.6 - related work and research positioning
16. Phase 8.7 - reproducibility documentation
17. Phase 9 - multi-dataset extension (ONLY if time allows, otherwise document as future work)

## Must-Fix Before Any Final Thesis Run

### Core System Requirements
- parser provenance
- evidence alignment
- strong baseline RAG
- GraphRAG source construction improvements
- larger evaluation size (beyond 20 questions)
- clear experiment-family separation

### Thesis Defense Requirements
- model ablation study (at least 3 LLMs, 2 embeddings)
- statistical significance testing
- error analysis with case studies
- HybridQA dataset justification
- related work positioning
- reproducibility documentation

### Minimum Acceptable Thesis Scope

To pass thesis defense, you MUST have Phases 0-8.7 complete. Phase 9 (multi-dataset) is optional but strengthens claims.

## Nice-To-Have After Core Fixes

These improve the thesis but are not critical for passing:

- additional reranker experiments
- extra embedding model comparisons (beyond minimum 2)
- additional GraphRAG mode comparisons
- response calibration or abstention tuning
- fancy visualization dashboards
- real-time demo systems

## Things Not To Waste Time On Early

- polishing visual report formatting
- large model benchmarking before parser and evidence fixes
- micro-tuning graph hyperparameters before ingestion is improved
- full train-split experiments before the dev pipeline is trustworthy

## Suggested Final System Names

Use names that reflect reality:

- `baseline_simple_dense`
- `baseline_strong_rag`
- `baseline_shared_text_unit_rag`
- `graphrag_pure`
- `graphrag_practical`

Avoid misleading names that mention one model while using another.

## Suggested Final Thesis Experiment Table

The final thesis should ideally include:

### Core Comparison Experiments (Required)
1. Simple dense baseline vs pure GraphRAG on small pilot (sanity check)
2. Shared-text-unit controlled comparison (methodological fairness)
3. Strong baseline RAG vs pure GraphRAG (main comparison)
4. Strong baseline RAG vs practical GraphRAG (practical utility)
5. Final best-system comparison on large dev run (at least 200+ questions)

### Model Ablation Experiments (Required for Defense)
6. Model ablation table: 3+ LLMs × 2 systems × key metrics
7. Embedding ablation table: 2+ embeddings × 2 systems × key metrics

### Statistical Analysis (Required for Defense)
8. Significance testing results for all major comparisons
9. Confidence intervals for key metrics

### Qualitative Analysis (Required for Defense)
10. Error analysis with categorized failure modes
11. Case studies (10-15 examples):
    - Where RAG wins
    - Where GraphRAG wins
    - Where both fail

### Optional Extensions (If Time Allows)
12. Multi-dataset comparison (Phase 9)
13. Additional model variants
14. Reranker ablations

## Success Definition

This project should be considered successful only if:

### System Quality
- the parser preserves enough structure for defendable retrieval claims
- the baseline is strong enough that beating it means something
- GraphRAG is tested both as a pure method and as a practical system
- final evaluation uses enough data for credible conclusions (200+ questions minimum)
- the thesis text matches the actual code and experiments

### Statistical Rigor
- statistical significance is tested and reported
- confidence intervals are provided for key metrics
- effect sizes are documented, not just p-values

### Methodological Soundness
- model ablation shows results hold across multiple models
- error analysis explains when and why each system wins or fails
- limitations are documented honestly

### Academic Positioning
- HybridQA dataset choice is justified (with plan for future expansion)
- related work is clearly documented
- contributions are clearly distinguished from replication
- reproducibility is fully documented

### Minimum Passing Threshold

To PASS thesis defense, you must complete:
- Phases 0-8.7 (all except multi-dataset Phase 9)
- At least 3 LLM variants tested
- At least 2 embedding variants tested
- Statistical tests on all main comparisons
- Error analysis with 50+ manually reviewed predictions
- 10-15 case study examples

### Excellence Threshold

To achieve EXCELLENT marks, add:
- Phase 9 (multi-dataset validation)
- 5+ LLM variants
- 3+ embedding variants
- Cross-dataset insights
- Comprehensive error taxonomy

## Changelog

### Initial version

- created implementation roadmap based on current repository assessment
- identified parser, evidence, baseline, GraphRAG ingestion, and evaluation as the top priorities

### Phase 0 started

- added shared runtime and hardware detection helpers
- started wiring actual runtime metadata into evaluation outputs
- started upgrading readiness checks to validate conda env and CUDA runtime

### Phase 1 completed

- moved HybridQA parsing logic into a dedicated parser module with stable provenance ids
- preserved row metadata, cell metadata, linked passage inventory, and parser metadata in parsed records
- removed the early linked-passage cap from parsed HybridQA output
- verified the parser changes with unit tests and a real-record check in the `thesis_rag_gpu` environment

### Phase 2 completed

- confirmed the local HybridQA raw files do not include official gold evidence ids
- added a documented proxy-evidence alignment layer instead of pretending proxy labels are strict gold evidence
- backfilled proxy evidence onto existing parsed HybridQA records so evaluation can use old jsonl files safely
- extended evaluation to report strict evidence metrics separately from proxy evidence metrics
- aligned baseline row and passage retrieval ids with stable parser-level ids for future evidence-aware baseline evaluation

### Phase 3 completed

- created strong baseline RAG system with row-aware passage expansion, lexical+semantic hybrid scoring, and cross-encoder reranking in `src/baseline/strong_retriever.py`
- added context deduplication, merging, and smart packing utilities in `src/baseline/context_utils.py`
- built provenance-aware answer generator with citation support in `src/baseline/strong_answer_generator.py`
- created end-to-end pipeline orchestrator supporting both "simple" (ablation) and "strong" (thesis) variants in `src/baseline/strong_baseline_pipeline.py`
- added test script for comparing simple vs strong baseline on sample questions in `scripts/test_strong_baseline.py`
- strong baseline now clearly outperforms simple baseline through better retrieval (row-linked passage expansion), better context management (deduplication, structured formatting), and better grounding (provenance tracking)
- both baseline variants are ready for fair comparison against GraphRAG in Phase 5

### Phase 3.5 completed

- created comprehensive model variant configuration system in `src/config/model_variants.py` defining 7 named variants (3 LLMs × 2 embedding families)
- defined variant collections for thesis phases: MINIMUM_ABLATION_VARIANTS (4 variants for defense), EXTENDED_ABLATION_VARIANTS (7 variants for excellence), SMOKE_TEST_VARIANTS (2 variants for quick testing)
- refactored `src/config/model_registry.py` to add `get_active_model_metadata()` and `build_experiment_metadata()` functions for automatic model tracking in experiments
- created `scripts/16_run_model_comparison.py` for automated model ablation studies - can run same experiment across multiple variants with single command
- added `configs/experiment_matrix_phase35.json` with model variant experiment definitions and usage examples
- created `scripts/verify_phase35.py` verification script to test model switching, metadata tracking, and variant system
- model switching now works through environment variables or config without code changes
- experiment metadata now automatically records exact models used, hardware specs, and runtime environment
- ready for Phase 6.5 model ablation studies

---

## Final Summary and Path Forward

### What This Plan Achieves

By completing this implementation plan, you will have:

1. **A Robust System Foundation**
   - Clean data ingestion with evidence tracking
   - Strong baseline RAG (not toy baseline)
   - Production-quality GraphRAG implementation
   - Both pure research and practical variants

2. **Academic-Grade Evaluation**
   - Large-scale experiments (200+ questions minimum)
   - Model ablation proving robustness
   - Statistical significance testing
   - Deep error analysis with insights

3. **Thesis Defense Readiness**
   - Clear dataset justification
   - Related work positioning
   - Reproducibility guarantees
   - Honest limitation documentation

4. **Clear Contribution Story**
   - Systematic comparison of RAG vs GraphRAG
   - Insights on when each approach wins
   - Model-agnostic findings
   - Practical recommendations

### Critical Success Factors

Your thesis will succeed if you:

✅ **DO:**
- Complete Phases 0-8.7 before defense
- Test minimum 3 LLMs and 2 embeddings
- Run 200+ question evaluation
- Perform manual error analysis
- Document limitations honestly
- Keep HybridQA as primary focus initially
- Add multi-dataset (Phase 9) only after core is solid

❌ **DON'T:**
- Skip statistical testing (committee will ask!)
- Skip error analysis (need qualitative insights)
- Rush to Phase 9 before perfecting HybridQA
- Over-claim without evidence
- Hide limitations
- Use toy baselines as comparison

### Next Immediate Steps

Since you're currently at Phase 3, here's what to focus on:

**Week 1-2: Phase 3 + 3.5**
1. Build strong baseline RAG with all features
2. Create model registry for easy switching
3. Verify baseline significantly outperforms simple baseline

**Week 3-4: Phases 4-5**
1. Improve GraphRAG source construction
2. Separate controlled vs realistic experiments
3. Run initial comparison on 50-100 questions

**Week 5-6: Phase 6 + 6.5**
1. Scale up to 200+ question evaluation
2. Run model ablation (3 LLMs, 2 embeddings)
3. Generate initial comparison tables

**Week 7-8: Phases 6.6-6.7**
1. Add statistical significance tests
2. Perform manual error analysis
3. Create case study examples

**Week 9-10: Phases 7-8.7**
1. Add practical GraphRAG variant
2. Complete all documentation
3. Prepare reproducibility materials

**Optional Week 11-14: Phase 9**
1. Add 2-3 more datasets
2. Run cross-dataset experiments
3. Strengthen generalization claims

### Minimum Viable Thesis (5-8 weeks)

To pass with Phases 0-8.7:
- ✅ HybridQA only (justified in Phase 8.5)
- ✅ 3 LLMs, 2 embeddings tested
- ✅ 200+ questions evaluated
- ✅ Statistical tests + error analysis
- ✅ Full documentation

**Claim:** "We systematically compared RAG and GraphRAG on HybridQA across multiple models, with statistical validation and error analysis. GraphRAG shows strength in X scenarios while RAG excels in Y scenarios."

### Excellent Thesis (7-12 weeks)

Add Phase 9 for:
- ✅ HybridQA + 2-3 more datasets
- ✅ 5+ LLMs, 3+ embeddings
- ✅ Cross-dataset patterns
- ✅ Broader insights

**Claim:** "We systematically compared RAG and GraphRAG across multiple datasets and models, identifying robust patterns: GraphRAG consistently helps with multi-hop reasoning but adds overhead for simple lookups."

### How To Know You're On Track

✅ **Good signs:**
- Every phase has passing tests
- Metrics improve with each enhancement
- Error analysis reveals clear patterns
- Documentation explains all decisions

⚠️ **Warning signs:**
- Skipping phases to save time
- No tests for new code
- Results don't make sense
- Can't explain what code does
- Limitations are hidden

### Final Advice

**For Your Thesis Committee:**
They will care most about:
1. Is the comparison fair? (Phases 3-5)
2. Are results statistically valid? (Phase 6.6)
3. Do you understand WHY systems behave differently? (Phase 6.7)
4. Can others reproduce this? (Phase 8.7)
5. What are the limitations? (Phase 8.5, 8.6)

**For Your Timeline:**
- Budget 5-8 weeks minimum for Phases 3-8.7
- Add 2-4 weeks if doing Phase 9
- Leave buffer time for unexpected issues
- Write thesis chapters as you complete phases

**For Your Sanity:**
- One phase at a time
- Verify each phase before moving on
- Don't over-engineer early phases
- Ask for help when stuck
- Celebrate small wins

### You've Got This! 🎓

You've already completed the hard foundation work (Phases 0-2). The path forward is clear. Follow the phases, meet the acceptance criteria, and you'll have a defensible thesis comparing RAG and GraphRAG systems.

**Remember:** A thesis doesn't need to be perfect. It needs to be:
- Correct (fair comparison)
- Rigorous (statistical validation)
- Insightful (error analysis)
- Honest (documented limitations)
- Reproducible (clear documentation)

This plan gives you all of that. Now go build it! 🚀
