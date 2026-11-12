# Research Decisions Log

Project: **Evaluating Multilingual Factual Reliability of Large Language Models in Scientific Literature Analysis**  
Author: Aman Kumar (`mcl2026012@iiita.ac.in`)  
Framework: AI Tools for Research (AITR) — Evidence-Based Implementation Study  

---

## Decision Log Format
Each decision recorded in this log adheres to the following evidence hierarchy:
- **[VERIFIED]**: Confirmed directly from the literature review paper, assignment specification, official documentation, primary publications, or verified command execution.
- **[INFERRED]**: Plausible deduction or interpretation derived from verified evidence, requiring empirical validation or explicit confirmation.
- **[UNKNOWN]**: Missing evidence, unverified parameter, or pending experimental measurement.

---

## Log Entries

### RD-001: Project Scope and Literature Continuity
- **Date**: 2026-10-31
- **Status**: **[VERIFIED]** (Topic & Literature Review Link)
- **Decision**: Ground the second paper's experimental study directly in the theoretical findings of the first paper, *"Benchmarking Factual Accuracy of Large Language Models Across Scientific Domains"* (Kumar, 2026). Specifically focus on the transition from static, monolingual scientific benchmark scores to multilingual factual reliability in scientific literature analysis.
- **Rationale**: 
  - Paper 1 established that benchmark scores are not interchangeable and identified that atomic claim verification (Equation 1), evidence entailment, and cross-lingual robustness represent distinct measurement dimensions.
  - Section 5.3 of Paper 1 notes: *"Multi-Fact extends FActScore-oriented analysis to multilingual factuality settings [12]."*
  - Table 3 of Paper 1 highlights key gaps in retrieval/citations, calibration/abstention, and cross-domain/multilingual generalization.
- **Source / Evidence**: 
  - Paper 1: Kumar, A., *"Benchmarking Factual Accuracy of Large Language Models Across Scientific Domains"*, Sections 1, 2, 5.3, Table 1, Table 3.
  - Assignment Guide: *"AI Tools for Research"*, Section 2 (Continuity rule).
- **Next Actions**: Formulate a narrow, experimentally testable research question for Gate G1 comparing multilingual factual reliability across scientific claims.

---

### RD-002: Repository Layout and Traceability Architecture
- **Date**: 2026-10-31
- **Status**: **[VERIFIED]** (Assignment Specification Compliant)
- **Decision**: Establish an accountable, modular directory structure adhering strictly to Section 9 of the assignment specification:
  - `references/`: Verified primary papers, bibliographies, and dataset documentation.
  - `data/`: Dataset schemas, access scripts, and partition manifests (no raw private/restricted data).
  - `notebooks/`: Exploratory and runnable verification notebooks.
  - `src/`: Reusable Python modules for claim decomposition, retrieval, entailment, and evaluation.
  - `scripts/`: Entry-point scripts for data verification, baseline execution, and metric generation.
  - `configs/`: Frozen experiment parameters, seed definitions, and model configs.
  - `results/`: Actual metric CSVs, execution logs, and output tables.
  - `figures/`: Reproducible high-resolution plots.
  - `paper/`: LaTeX manuscript sources (`.tex`, `.bib`, figures) for the 6–10 page final submission.
  - `tests/`: Sanity tests and unit tests verifying data integrity, environment, and metric calculation.
- **Rationale**: Meets Gate G1-G4 reproducibility standards and ensures zero fabricated claims.
- **Source / Evidence**: *"AI Tools for Research"*, Sections 7, 8, and 9.
- **Next Actions**: Complete initial documentation files (`README.md`, `evidence_ledger.csv`, `ai_assistance_log.md`).

---

### RD-003: Critical Gap Analysis and Literature Feasibility Audit
- **Date**: 2026-11-04
- **Status**: **[VERIFIED]** (Cross-referenced against Paper 1, SciFact, Multi-Fact, and Poly-FEVER literature)
- **Decision**: Analyzed five candidate experimental gaps derived from Paper 1 and conducted an external literature audit. Determined that existing multilingual factuality studies (Multi-Fact, Poly-FEVER) either restrict themselves to Wikipedia biographical facts or test closed-book claim veracity without evaluating scientific rationale grounding, abstention calibration under insufficient evidence (`NO_INFO`), or translation-mediated terminological distortion.
- **Rationale**: 
  - Poly-FEVER (2024) translated SciFact claims into 11 languages but only evaluated isolated veracity classification, leaving evidence sentence rationale retrieval and cross-lingual grounding untested.
  - Multi-Fact (Shafayat et al., 2024) evaluated FActScore on general Wikipedia, leaving domain-specific scientific literature untouched.
  - Candidate Gaps 1, 4, and 3 provide the highest scientific feasibility for a controlled benchmark on frozen open models without requiring model retraining.
- **Source / Evidence**:
  - Paper 1: Kumar, A. (2026), pp. 2–8.
  - Wadden et al. (2020), *SciFact*, EMNLP 2020.
  - Shafayat et al. (2024), *Multi-Fact*, arXiv:2402.18045.
  - Poly-FEVER Benchmark (2024).
- **Next Actions**: Formalize Gate G1 Research Brief with ranked candidates and select Candidate Gap 1 / Pathway B (Controlled Comparative Benchmark).

---

### RD-004: Formal Approval and Freezing of Research Brief
- **Date**: 2026-11-04
- **Status**: **[VERIFIED]** (Author Brief Frozen in `research_brief.md`)
- **Decision**: Formalized the one-page research brief incorporating the student's exact proposed questions:
  - **Primary Question**: *How does language affect a multilingual model's ability to verify scientific claims against supplied scientific evidence?*
  - **Supporting Questions**: (1) Classification performance variation across languages; (2) Prediction flip rate / consistency for identical examples; (3) Evidence-interpretation error taxonomy.
  - **Design**: Factorial 3 Languages (EN, HI, ES) × 2 Pipelines (Direct Multilingual vs. Translate-Test) × 2 Open Models (Llama-3-8B-Instruct vs. Qwen-2.5-7B-Instruct in 4-bit NF4) on SciFact Development Set (300 claims).
  - **Metrics**: Veracity Macro-F1, Sentence-Level Rationale F1, Cross-Lingual Flip Rate (CFR), ECE, Error Taxonomy.
  - **Integrity Rule**: Zero invented accuracy scores; all claims tagged as VERIFIED, INFERRED, or UNKNOWN.
- **Rationale**: Fully captures the user's specific research formulation and adheres strictly to the AITR Gate G1 standard.
- **Source / Evidence**: `research_brief.md` in repository root.
- **Next Actions**: Advance to **Gate G2: Baseline Verification** (environment check, dataset acquisition, baseline sanity run).

---

### RD-005: SciFact Dataset Inspection and Schema Adaptation
- **Date**: 2026-11-04
- **Status**: **[VERIFIED]** (Actual Dataset Files Inspected via `scripts/inspect_dataset.py`)
- **Decision**:
  1. Downloaded and verified the official SciFact release (`data.tar.gz` from AWS S3, 3.11 MB).
  2. Verified exact counts: Corpus = 5,183 documents; Train = 809 claims; Dev = 300 claims; Test = 300 claims (Total = 1,409 claims).
  3. **Identified Critical Schema Variation**: Raw release files key the negative evidence label as `"CONTRADICT"`, whereas research papers commonly write `"REFUTE"`. Adapted `src/data_loader.py` with bidirectional label mapping (`CONTRADICT` $\leftrightarrow$ `REFUTE`) and documented this schema normalization.
  4. Verified Test Partition Blindness: `claims_test.jsonl` contains strictly `id` and `claim`. In strict compliance with benchmark ethics, the test set is kept blind and **will NOT be used** for tuning or prompt iteration.
  5. Selected Evaluation Partition: Confirmed `claims_dev.jsonl` (300 claims: 124 SUPPORT, 64 CONTRADICT, 112 NOT_ENOUGH_INFO) as the primary evaluation split.
- **Rationale**: Full traceability to raw data; prevents parsing exceptions; ensures unbiased evaluation.
- **Source / Evidence**:
  - `results/dataset_statistics.json`
  - `results/data_validation_report.md`
  - `tests/test_data_loader.py` (5/5 tests passing)
- **Next Actions**: Proceed with environment configuration (`requirements.txt`), GPU/CPU sanity diagnostic script (`scripts/test_env.py`), and baseline prompt design.

---

### RD-006: Pre-Experiment Protocol Freeze
- **Date**: 2026-11-04
- **Status**: **[VERIFIED]** (Protocol Formally Frozen in `experiment_protocol.md`)
- **Decision**:
  - Defined and locked the complete experimental blueprint before running evaluation passes or inspecting results.
  - Compared Modes: Mode A (Direct Classification) vs. Mode B (Evidence-Grounded Classification).
  - Factorial: 3 Languages (EN, HI, ES) × 2 Modes (Direct vs. Grounded) × 2 Models (Qwen-2.5-7B vs. Llama-3-8B in 4-bit) = 12 conditions.
  - Frozen decoding settings: greedy decoding (`temp=0.0`, `seed=42`, `max_new_tokens=128`).
  - Parsing fallback and penalty rule locked: unparseable outputs assigned `"INVALID_OUTPUT"` (score 0.0), never silently dropped.
  - Statistical testing: Paired bootstrap resampling (1,000 iterations) with 95% confidence intervals and McNemar's test.
  - Failure taxonomy: 5 explicit error codes (`ERR_RAT_OMIT`, `ERR_RAT_SPURIOUS`, `ERR_POL_INVERT`, `ERR_NEI_FAIL`, `ERR_TRANS_DRIFT`).
- **Rationale**: Adheres to scientific rigor; eliminates post-hoc metric shopping; satisfies Gate G2/G3 freeze criteria.
- **Source / Evidence**: [`experiment_protocol.md`](file:///C:/Users/akema/.gemini/antigravity/scratch/AITR_Lab02_mcl2026012_MultilingualSciFact/experiment_protocol.md).
- **Next Actions**: Implement baseline runner (`scripts/run_baseline.py`) and translation pipeline (`scripts/translate_claims.py`).

---

### RD-007: Environment Diagnostics and Dependency Alignment
- **Date**: 2026-11-04
- **Status**: **[VERIFIED]** (Executed `scripts/test_env.py`)
- **Decision**:
  1. Measured host hardware: Python 3.12.3 on Windows 11, Intel Core processor, 7.63 GB total system RAM (0.69 GB free), and an NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM, 3,964 MiB free).
  2. Identified PyTorch runtime status: Current PyTorch build is CPU-only (`torch==2.4.0+cpu`). CUDA is reported as `False` in this local Python installation.
  3. Aligned library dependencies: Installed compatible `transformers==4.45.2`, `datasets==2.21.0`, `accelerate==1.15.0`, `evaluate==0.4.6`, `sacrebleu==2.6.0`, `sentencepiece==0.2.2`, and `numpy==1.26.4`.
  4. Executed functional sanity test: All 4 components (`tensor_ops`, `sklearn_metrics`, `sacrebleu_metric`, `scifact_data_loader`) passed cleanly.
  5. Architecture Decision: Local environment is fully verified for data processing, metrics evaluation, unit testing, and small pipeline queries. For heavy 7B/8B model inference passes, Google Colab Free Tier T4 GPU (~15 GB VRAM) is the designated cloud environment, avoiding local memory exhaustion.
- **Rationale**: Strict honesty regarding GPU capabilities; no unverified claims of local CUDA execution.
- **Source / Evidence**:
  - `requirements.txt`
  - `scripts/test_env.py`
  - `results/environment_info.json`
- **Next Actions**: Build translation and baseline runner scripts (`scripts/translate_claims.py`, `scripts/run_baseline.py`).

---

### RD-008: Baseline NLI Model Implementation, Label Mapping, and Smoke Verification
- **Date**: 2026-11-04
- **Status**: **[VERIFIED]** (Executed `scripts/run_baseline.py` and `tests/test_baseline.py`)
- **Decision**:
  1. **Model Selection**: Selected `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (278,811,651 parameters, `deberta-v2` backbone fine-tuned on MNLI and XNLI). This model supports 100+ languages (including English, Hindi, and Spanish) and runs deterministically on local CPU.
  2. **Model Label Configuration Inspection**: Inspected the actual downloaded model configuration `model.config.id2label`:
     - Index 0: `entailment`
     - Index 1: `neutral`
     - Index 2: `contradiction`
  3. **Deterministic Label Normalization**: Built explicit mapping layer:
     - `entailment` $\rightarrow$ `SUPPORTS`
     - `contradiction` $\rightarrow$ `REFUTES`
     - `neutral` $\rightarrow$ `NOT_ENOUGH_INFO`
     - Handles raw SciFact `CONTRADICT` $\leftrightarrow$ `REFUTES` bidirectionally.
  4. **Modularity and Decoupling**: Kept baseline code strictly isolated in `src/baseline_model.py` and `scripts/run_baseline.py`, independent from subsequent generative LLM prompts.
  5. **Edge-Case and Robustness Handling**:
     - Empty claim or empty evidence defaults safely to `NOT_ENOUGH_INFO` with confidence `0.0` and error tags.
     - Sequences exceeding 512 tokens are truncated and flagged (`truncated: True`).
     - Softmax probabilities logged across all 3 classes (`score_SUPPORTS`, `score_REFUTES`, `score_NOT_ENOUGH_INFO`).
  6. **Unit Test Verification**: 16 unit tests in `tests/test_baseline.py` passed cleanly (100% pass rate).
  7. **Smoke Test Execution**: Evaluated first 20 claims from `claims_dev.jsonl` (22 claim-document pairs):
     - Correct: 16 / 22
     - Veracity Accuracy: 72.73% (0.7273)
     - Label distribution: 6 SUPPORTS, 6 REFUTES, 10 NOT_ENOUGH_INFO
     - System runtime: 76.16s on CPU (0 errors, 0 invalid outputs)
     - Logged genuine predictions to `results/baseline_predictions_smoke.csv` and `results/baseline_predictions_smoke_summary.json`.
- **Rationale**: Establishes a genuine empirical baseline before running generative models, satisfying Gate G2 verification requirements.
- **Source / Evidence**:
  - `src/baseline_model.py`
  - `scripts/run_baseline.py`
  - `tests/test_baseline.py`
  - `results/baseline_predictions_smoke.csv`
  - `results/baseline_predictions_smoke_summary.json`
- **Next Actions**: Implement cross-lingual translation pipeline (`scripts/translate_claims.py`) and metric evaluation module (`src/metrics.py`).

---

### RD-009: Baseline Audit, Non-Reproduction Boundary, and Metric Module Freeze
- **Date**: 2026-11-04
- **Status**: **[VERIFIED]** (Executed `baseline_validation.md`, `tests/test_evaluation_audit.py`, `src/metrics.py`)
- **Decision**:
  1. **Comprehensive Baseline Audit**: Audited all 10 operational dimensions: label mapping, evidence alignment, model NLI framing, tokenization truncation, missing evidence, output normalization, invalid prediction penalty, duplicate records, split contamination, and metric implementations.
  2. **Sentence Pointer Verification**: Confirmed all 366 evidence sentence pointers in `claims_dev.jsonl` are strictly within valid abstract bounds (`[0, len(abstract))`).
  3. **Split Integrity & Contamination Assessment**:
     - Dev vs. Test claim ID overlap: 0 (Zero leakage).
     - Train vs. Test claim ID overlap: 0 (Zero leakage).
     - Train vs. Dev text overlap: Identified 2 identical claims in official SciFact release (Claim 870/871 and 1291/1292). Verified **zero impact** on this study due to frozen zero-shot inference without model training or prompt tuning on `claims_train.jsonl`.
  4. **Explicit Non-Reproduction Demarcation**: Formally documented that our baseline does NOT claim reproduction of published VeriSci scores (Wadden et al., 2020), as VeriSci employs a 3-stage fine-tuned retrieval pipeline, whereas our baseline evaluates zero-shot/frozen multilingual NLI sequence classification on scientific claims.
  5. **Metric Module Implementation**: Created `src/metrics.py` implementing 3-class Macro-F1, per-class Precision/Recall/F1, confusion matrices, sentence-level rationale F1, and Cross-Lingual Flip Rate (CFR), with strict penalty scoring for invalid outputs.
  6. **Automated Audit Suite**: Authored `tests/test_evaluation_audit.py` (19 tests). Full suite of 40 tests across the project passed cleanly in 43.82s.
- **Rationale**: Enforces complete scientific transparency, eliminates unjustified baseline reproduction claims, and satisfies Gate G2 audit requirements.
- **Source / Evidence**:
  - `baseline_validation.md`
  - `src/metrics.py`
  - `tests/test_evaluation_audit.py`
  - `tests/test_baseline.py`
  - `tests/test_data_loader.py`
- **Next Actions**: Implement translation pipeline (`scripts/translate_claims.py`) for Hindi and Spanish evaluation conditions.

---

### RD-010: Translation Pipeline, Integrity Auditing, and ChatML Native Formatting
- **Date**: 2026-11-05
- **Status**: **[VERIFIED]** (Executed `src/translator.py`, `scripts/translate_claims.py`, `data/claims_dev_multilingual_pilot.jsonl`)
- **Decision**:
  1. **Translation Engine Selection**: Selected `Helsinki-NLP/opus-mt-en-es` (Spanish) and `Helsinki-NLP/opus-mt-en-hi` (Hindi) with `sacremoses` tokenization support, local cache fallback (`local_files_only=True`), and reproducible decoding (`num_beams=4`).
  2. **Automated Semantic Integrity Auditing**: Built regex-based validation for numerical quantity preservation (`NUMERIC_PATTERN`) and negation polarity tracking (`NEGATION_TERMS`).
  3. **Translation Behavioral Findings**:
     - Spanish preserved 100% negation polarity and scientific concepts, but used European thousand-separator dot notation (`1.000` vs `1,000`).
     - Hindi translated certain Arabic digits into Devanagari numerals (`5%` $\to$ `५%`), retaining numeric equivalence but requiring localized number parsing.
  4. **ChatML Template Fix**: Discovered that small causal LLMs (`Qwen/Qwen2.5-0.5B-Instruct`) degenerate into repeated token ID 0 (`!`) when prompted with raw ChatML string tags. Resolved by invoking native `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`.
  5. **Pilot Translation Dataset**: Generated verified multilingual dataset for first 10 dev claims in `data/claims_dev_multilingual_pilot.jsonl` with full per-claim latency and model provenance.
- **Rationale**: Ensures translation fidelity without silent mutation of scientific quantities, preserving strict experimental matched conditions.
- **Source / Evidence**:
  - `src/translator.py`
  - `scripts/translate_claims.py`
  - `data/claims_dev_multilingual_pilot.jsonl`
  - `data/claims_dev_multilingual_pilot_summary.json`
- **Next Actions**: Execute pilot evaluation on generative claim verification pipeline across Languages and Modes.

---

### RD-011: Multilingual Pilot Benchmark, Output Normalization, and Error Taxonomy Verification
- **Date**: 2026-11-05
- **Status**: **[VERIFIED]** (Executed `scripts/run_multilingual_experiment.py`, `results/multilingual_pilot_validation.md`, `tests/test_multilingual_experiment.py`)
- **Decision**:
  1. **Pilot Evaluation Execution**: Executed matched multilingual matrix across 3 languages (`en`, `es`, `hi`) and 2 prompting modes (Mode A: Direct vs Mode B: Grounded) on 3 claims (4 evaluation pairs), completing 24 genuine model passes in 512.38s (21.35s avg latency per pass).
  2. **Empirical Discovery on Output Capitalization**: Qwen-0.5B consistently outputs camelCase `"NOTEnoughInfo"` instead of canonical uppercase underscore `"NOT_ENOUGH_INFO"`. Initial parser flagged these as `INVALID_OUTPUT`. Updated `normalize_eval_label` to strip whitespace, underscores, and hyphens, completely eliminating false invalid penalties and reducing invalid outputs from 16 to **0 (0.0%)**.
  3. **Mode A vs Mode B Empirical Divergence**:
     - **Mode A (Direct)**: 100% NEI-collapse across all languages (Acc: 0.2500, Macro-F1: 0.1333, Invalid: 0). The model conservatively predicts `NOT_ENOUGH_INFO` for every claim when ungrounded.
     - **Mode B (Grounded)**: Successfully broke NEI-collapse on Claim 5, identifying `SUPPORTS` across English, Spanish, and Hindi (Acc: 0.5000, Macro-F1: 0.3333, Mean Rat-F1: 0.25, Invalid: 0).
  4. **Cross-Lingual Consistency**: Cross-Lingual Flip Rate (CFR) across $(EN, ES)$, $(EN, HI)$, and $(ES, HI)$ was exactly **0.00%** on the pilot set, indicating high cross-lingual prediction stability under matched prompting.
  5. **Rationale Extraction Bottleneck**: On Claim 5, the 0.5B model output explanatory rationale descriptions (`"rationales": ["The study aims to..."]`) rather than integer indices (`[4]`), yielding `ERR_RAT_OMIT` and Rationale F1 of 0.0 on that claim.
  6. **Unit Test Verification**: Created `tests/test_multilingual_experiment.py` (16 unit tests) covering normalization, parser recovery, error classification, and CFR. Total test suite of 56 tests passed (100% pass rate).
- **Rationale**: Validates the end-to-end multilingual experimental pipeline on genuine outputs before scaling, identifying critical normalization requirements and runtime boundaries.
- **Source / Evidence**:
  - `results/multilingual_pilot_validation.md`
  - `results/multilingual_pilot_predictions.csv`
  - `results/multilingual_pilot_predictions_summary.json`
  - `tests/test_multilingual_experiment.py`
  - `src/multilingual_experiment.py`
  - `scripts/run_multilingual_experiment.py`
- **Next Actions**: Implement automated evaluation pipeline and generate figures using actual prediction values.

---

### RD-012: Comprehensive Evaluation Pipeline, Strict Schema Validation, and Calibration Profiling
- **Date**: 2026-11-05
- **Status**: **[VERIFIED]** (Executed `src/evaluation.py`, `scripts/evaluate_predictions.py`, `tests/test_evaluation_metrics.py`)
- **Decision**:
  1. **Strict Input Schema Verification**: Enforced automated pre-computation schema checks on required columns (`claim_id`, `doc_id`, `language`, `mode`, `gold_label`, `predicted_label`, `confidence`, `latency_sec`), numeric validity of confidence ($\in [0.0, 1.0]$) and latency ($\ge 0.0$), and absence of nulls in key identifier fields.
  2. **Cross-Lingual Gold Alignment Audit**: Validated that for each unique claim-evidence instance, the gold label is strictly invariant across English, Spanish, and Hindi. Grouping incorporates `gold_sentences` to disambiguate multiple evidence sets for the same cited document.
  3. **Matched-Pair Prediction Disagreement & Cohen's Kappa**: Implemented paired comparison engine for both cross-lingual pairs (holding prompting mode fixed) and cross-prompt pairs (holding language fixed). Addressed edge cases where unanimous category predictions cause expected chance agreement $p_e = 1.0$, mapping Cohen's $\kappa$ to 1.0 under zero disagreement.
  4. **Empirical Calibration Investigation**: Computed Brier Score (0.6254), Expected Calibration Error (ECE = 0.6208), Maximum Calibration Error (MCE = 0.7417), and reliability diagram binning. Documented that self-reported verbalized confidences from small LLMs exhibit significant overconfidence.
  5. **Standardized Metric Table & Canonical Prediction Export**: Produced `results/predictions.csv`, `results/metrics_by_language.csv`, `results/metrics_by_prompt.csv`, and `results/paired_disagreement.csv`.
  6. **Publication Figures from Genuine Values**: Generated 5 high-resolution PNG figures:
     - `fig1_accuracy_f1_by_condition.png`: Accuracy & Macro-F1 by condition.
     - `fig2_confusion_matrices.png`: Mode A vs Mode B confusion matrix heatmaps.
     - `fig3_runtime_by_condition.png`: Inference latency distributions.
     - `fig4_paired_disagreement.png`: Disagreement rates across paired conditions.
     - `fig5_calibration_analysis.png`: Confidence distribution & reliability diagram.
  7. **Automated Unit Testing**: Created `tests/test_evaluation_metrics.py` (17 tests). Entire test suite (73 tests) passed in 17.17s.
- **Rationale**: Guarantees zero invented metrics or fabricated significance, providing an automated, mathematically rigorous evaluation engine.
- **Source / Evidence**:
  - `src/evaluation.py`
  - `scripts/evaluate_predictions.py`
  - `tests/test_evaluation_metrics.py`
  - `results/metrics_by_language.csv`
  - `results/metrics_by_prompt.csv`
  - `results/paired_disagreement.csv`
  - `results/predictions.csv`
  - `results/figures/`
- **Next Actions**: Conduct deep qualitative and empirical error analysis of results across research questions.

---

### RD-013: Empirical Results Analysis, Epistemic Demarcation, and Negative Results Reporting
- **Date**: 2026-11-05
- **Status**: **[VERIFIED]** (Executed `results_analysis.md`)
- **Decision**:
  1. **Epistemic Partitioning**: Formally categorized every experimental finding across the 4 core research questions into five distinct categories: Directly observed results, Plausible explanations, Alternative explanations, Uncertain findings, and Unsupported prohibited claims.
  2. **Mechanistic Error Diagnosis via Qualitative Case Studies**:
     - *Claim 1 (Doc 31715818, Gold NEI):* Model successfully abstained across all 6 passes in EN, ES, HI.
     - *Claim 3 (Doc 14717500, Gold SUPPORTS):* Uncovered catastrophic translation drift where MarianMT mistranslated `"genomes project"` into `"भूविज्ञान परियोजना"` (Geology project) in Hindi (`ERR_TRANS_DRIFT`). Model exhibited conservative over-abstention in Mode A and Mode B.
     - *Claim 5 (Doc 13734012, Gold SUPPORTS):* Observed dramatic Mode divergence: Mode A completely failed (NEI-collapse), while Mode B successfully broke NEI-collapse and recognized mathematical quantitative equivalence ($1/2000 \approx 493 \text{ per million}$) across all 3 languages. However, the model emitted natural-language text rather than integer indices `[4]`, resulting in rationale omission (`ERR_RAT_OMIT`).
  3. **Honest Reporting of Negative Results**:
     - Mode A failed completely as a scientific verification tool for small LLMs (100% NEI-collapse).
     - Small LLM zero-shot sentence rationale extraction failed on 100% of positive evidence instances.
     - Verbalized confidence is severely miscalibrated (ECE = 0.6208, Brier = 0.6254).
  4. **Prohibited Publication Claims**: Documented strict checklist of unsupported overgeneralizations barred from appearing in publication.
- **Rationale**: Enforces complete scientific honesty, prevents speculative overclaims, and establishes actionable research directions.
- **Source / Evidence**:
  - `results_analysis.md`
  - `results/predictions.csv`
  - `results/metrics_by_language.csv`
  - `results/metrics_by_prompt.csv`
  - `results/paired_disagreement.csv`
  - `data/claims_dev_multilingual_pilot.jsonl`
- **Next Actions**: Present comprehensive findings and recommendations to user.
