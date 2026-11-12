# AI Assistance and Research Integrity Log

Project: **Evaluating Multilingual Factual Reliability of Large Language Models in Scientific Literature Analysis**  
Author: Aman Kumar (`mcl2026012@iiita.ac.in`)  
Assignment: AI Tools for Research (AITR) — Lab 02  

---

## Log Protocol
In compliance with the responsible AI and research-integrity guidelines (AITR Section 10):
- All AI suggestions must be validated by human judgment and documented.
- No synthetic or imagined execution logs, p-values, or benchmark scores are permitted.
- Claims are strictly tagged: `[VERIFIED]`, `[INFERRED]`, or `[UNKNOWN]`.

---

## Interaction Records

### Entry AI-001: Project Inception and Workspace Setup
- **Date / Timestamp**: 2026-10-31
- **Task**: Workspace initialization, document analysis, and requirements synthesis.
- **Tool / Prompt**: Initial prompt with attached PDF documents:
  - Document 1: *"Benchmarking Factual Accuracy of Large Language Models Across Scientific Domains"* (Kumar, 2026)
  - Document 2: *"AI Tools for Research"* assignment guidelines
- **AI Proposal**: Analyze documents, establish clean 10-directory repository, author initial tracking logs (`research_decisions.md`, `evidence_ledger.csv`, `ai_assistance_log.md`, `README.md`), summarize scope without inventing claims.
- **Human Decision**: Approved initial workspace setup and evidence-grounding protocol.
- **Verification Method**: Direct inspection of PDF pages; verification of created directories on local file system.
- **Code / Artifact Change**: Created project root directory, 10 subdirectories, and initial documentation.
- **Outcome**: Workspace established with zero unverified claims.

---

### Entry AI-002: Critical Gap Analysis and Literature Feasibility Check
- **Date / Timestamp**: 2026-11-04
- **Task**: Critical analysis of Paper 1, identification of 5 candidate research gaps, literature audit via web search, and feasibility ranking.
- **Tool / Prompt**: User request for critical gap analysis covering multilingual factuality, scientific claim verification, FActScore, evidence grounding, hallucination, and language-dependent reliability.
- **AI Proposal**: Extract exact citations from Paper 1. Search external literature (Multi-Fact, SciFact, Poly-FEVER). Define 5 candidate research gaps with dataset, effort, and metrics. Rank top 3 feasible gaps.
- **Human Decision**: Approved gap analysis methodology. Confirmed that Poly-FEVER and Multi-Fact do not cover cross-lingual rationale grounding.
- **Verification Method**: Web search verification of arXiv:2402.18045, Poly-FEVER, and SemEval-2025 Task 7.
- **Code / Artifact Change**: Updated `research_decisions.md` (RD-003) and `evidence_ledger.csv` (EVD-008 to EVD-012).
- **Outcome**: 5 candidate gaps formulated; Gaps 1, 4, and 3 ranked as most feasible.

---

### Entry AI-003: Resource Evaluation and Project Design Comparison
- **Date / Timestamp**: 2026-11-04
- **Task**: Systematic evaluation of candidate resources (SciFact, PubMedQA, FActScore/Multi-FAct) across 11 dimensions. Comparison of three project designs.
- **Tool / Prompt**: User request to map research review to an executable experiment, evaluate resources, compare three designs, and recommend smallest credible study.
- **AI Proposal**: Evaluated SciFact, PubMedQA, and FActScore/Multi-FAct across 11 dimensions. Designed 3 project architectures. Recommended Design B (Controlled Multilingual Comparison on SciFact).
- **Human Decision**: User reviewed and approved `feasibility_novelty_assessment.md`.
- **Verification Method**: All resource claims verified via official GitHub repos, HuggingFace dataset cards, PyPI, and ACL Anthology.
- **Code / Artifact Change**: Created `resource_evaluation_and_project_design.md` artifact; updated `evidence_ledger.csv` (EVD-013 to EVD-017).
- **Outcome**: Design B confirmed. Proceeded to Gate G1 brief generation.

---

### Entry AI-004: Formalization of Gate G1 Research Brief
- **Date / Timestamp**: 2026-11-04
- **Task**: Compile and freeze the official 1-page Gate G1 Research Brief using Section 11 template.
- **Tool / Prompt**: Human approval of `feasibility_novelty_assessment.md`.
- **AI Proposal**: Create `research_brief_g1.md` with verified evidence, define 3 formal hypotheses with explicit success thresholds, demarcate demonstrated vs. novel contribution.
- **Human Decision**: Gate G1 specification accepted.
- **Verification Method**: Alignment check against course rubric (AITR Section 4 & Section 11).
- **Code / Artifact Change**: Created `research_brief_g1.md`; updated `research_decisions.md` (RD-004).
- **Outcome**: Gate G1 cleared.

---

### Entry AI-005: Creation of Author-Specified Research Brief (`research_brief.md`)
- **Date / Timestamp**: 2026-11-04
- **Task**: Compile the complete one-page research brief structured around student's primary question and 3 supporting questions.
- **Tool / Prompt**: User prompt specifying primary question and 3 supporting questions.
- **AI Proposal**: Author `research_brief.md` covering all 12 specified sections. Define exact metrics and error taxonomy without inventing accuracy scores.
- **Human Decision**: Generated and saved `research_brief.md`.
- **Verification Method**: Inspection of generated markdown file on disk.
- **Code / Artifact Change**: Authored `research_brief.md`; updated `research_decisions.md` (RD-004) and this log.
- **Outcome**: Research brief frozen in repository root.

---

### Entry AI-006: Implementation of Dataset Inspection Workflow
- **Date / Timestamp**: 2026-11-04
- **Task**: Implement SciFact data inspection workflow, verify schema, handle split blindness, and export statistical reports.
- **Tool / Prompt**: User prompt requesting inspection of official SciFact repository, verification of splits/labels/abstracts/licenses, code to produce statistics, distributions, missing-values, duplicate IDs, schemas, and human-readable samples.
- **AI Proposal**: Download official SciFact archive, create `src/data_loader.py` and `scripts/inspect_dataset.py`, run inspection, document label schema difference (`"CONTRADICT"` vs `"REFUTE"`), create unit test suite `tests/test_data_loader.py`.
- **Human Decision**: Executed verification; confirmed 5/5 unit tests pass and test split is preserved blind.
- **Verification Method**: Command line execution of `inspect_dataset.py` and `unittest` (5 passing tests in 0.040s).
- **Code / Artifact Change**: Created data files, `src/data_loader.py`, `scripts/inspect_dataset.py`, `results/dataset_statistics.json`, `results/data_validation_report.md`, `tests/test_data_loader.py`.
- **Outcome**: Data inspection workflow completely operational.

---

### Entry AI-007: Complete Experimental Protocol Design and Freeze
- **Date / Timestamp**: 2026-11-04
- **Task**: Design the complete matched multilingual experimental protocol comparing Direct Classification (Mode A) vs. Evidence-Grounded Classification (Mode B) across English, Hindi, and Spanish prior to examining results.
- **Tool / Prompt**: User request for complete experimental protocol with exact input/output formats, translation procedure, validation, models, prompt templates, decoding settings, split/sample counts, metrics, paired comparisons, error categories, hardware measurements, and Mermaid architecture diagram.
- **AI Proposal**: Authored `experiment_protocol.md`, designed Mermaid diagram, defined exact prompt templates, formalized invalid prediction penalty rule, defined 5-part failure taxonomy.
- **Human Decision**: Protocol frozen prior to running experimental passes.
- **Verification Method**: File inspection; internal consistency check.
- **Code / Artifact Change**: Authored `experiment_protocol.md`; updated `research_decisions.md` (RD-006).
- **Outcome**: Protocol frozen and ready for baseline execution.

---

### Entry AI-008: Environment Configuration and Sanity Test Execution
- **Date / Timestamp**: 2026-11-04
- **Task**: Configure environment dependencies, check actual hardware (CPU, RAM, GPU), install compatible Hugging Face libraries, and execute sanity diagnostic test.
- **Tool / Prompt**: User prompt to configure project for Python and Transformers, check Python/CPU/RAM/GPU without claiming GPU availability unless detected, create `requirements.txt` and `scripts/test_env.py`, run sanity test, and report console output.
- **AI Proposal**:
  - Measured actual hardware: Windows 11, Python 3.12.3, Intel Core processor, 7.63 GB RAM (0.69 GB free), NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) via `nvidia-smi`.
  - Identified PyTorch status: installed wheel is CPU-only (`torch==2.4.0+cpu`, `CUDA available = False`).
  - Fixed PyTorch DLL missing dependency (`libomp140.x86_64.dll`).
  - Installed compatible library stack: `transformers==4.45.2`, `datasets==2.21.0`, `accelerate==1.15.0`, `evaluate==0.4.6`, `sacrebleu==2.6.0`, `sentencepiece==0.2.2`, `numpy==1.26.4`.
  - Created `requirements.txt` and `scripts/test_env.py`.
  - Executed diagnostic sanity test and exported structured report to `results/environment_info.json`.
- **Human Decision**: Executed sanity test; all 4 functional tests passed.
- **Verification Method**: Actual console output of `scripts/test_env.py` and inspection of `results/environment_info.json`.
- **Code / Artifact Change**: Created `requirements.txt`, `scripts/test_env.py`, `results/environment_info.json`; updated `evidence_ledger.csv` (EVD-021) and `research_decisions.md` (RD-007).
- **Outcome**: Environment verified for data loading, translation validation, and evaluation; CPU-only PyTorch build transparently reported.

---

### Entry AI-009: Baseline Multilingual NLI Model Implementation and Smoke Test Verification
- **Date / Timestamp**: 2026-11-04
- **Task**: Implement baseline scientific claim verification using a verified multilingual NLI model with explicit label inspection and normalization, error handling, CSV prediction logging, and unit tests.
- **Tool / Prompt**: User request to implement baseline claim verification using verified multilingual NLI model, inspect model configuration and label mapping without assuming matching labels, implement loading/preprocessing/prediction/label normalization/confidence logging/error handling/CSV export, include sanity tests and edge cases, and run smoke test saving genuine output.
- **AI Proposal**:
  - Model selection: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (278M parameters, deberta-v2 architecture).
  - Inspected model config: verified `id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}`.
  - Built explicit label normalization: `entailment` $\rightarrow$ `SUPPORTS`, `contradiction` $\rightarrow$ `REFUTES`, `neutral` $\rightarrow$ `NOT_ENOUGH_INFO`.
  - Created standalone module `src/baseline_model.py` and runner `scripts/run_baseline.py` strictly isolated from experimental extensions.
  - Implemented unit test suite `tests/test_baseline.py` covering edge cases (empty claims, empty evidence, None inputs, label mappings, confidence bounds, sequence length truncation).
  - Executed pytest suite: 16/16 unit tests passed in 62.68s.
  - Ran smoke test on first 20 claims (22 evaluation pairs) from `claims_dev.jsonl`: 16/22 correct (72.73% accuracy), 0 errors, full confidence scores logged.
- **Human Decision**: Executed smoke test and saved genuine predictions to disk.
- **Verification Method**: Actual console execution of `pytest tests/test_baseline.py` and `python scripts/run_baseline.py --max-claims 20 --output results/baseline_predictions_smoke.csv`.
- **Code / Artifact Change**: Created `src/baseline_model.py`, `scripts/run_baseline.py`, `tests/test_baseline.py`, `results/baseline_predictions_smoke.csv`, `results/baseline_predictions_smoke_summary.json`; updated `evidence_ledger.csv` (EVD-022, EVD-023) and `research_decisions.md` (RD-008).
- **Outcome**: Gate G2 baseline verified and operational with empirical, non-fabricated results.

---

### Entry AI-010: Rigorous Baseline Audit, Metric Implementation, and Test Suite Expansion
- **Date / Timestamp**: 2026-11-04
- **Task**: Audit baseline against original SciFact dataset and mDeBERTa model documentation across 10 critical dimensions, create unit tests for label mapping and evaluation metrics, document discrepancies between intended and published methods, and generate `baseline_validation.md`.
- **Tool / Prompt**: User request to audit implemented baseline against original dataset and model documentation across 10 specific checks, compare with published baseline methodology, create unit tests for label mapping and evaluation, record discrepancies, avoid claiming reproduction of published results unless justified, save `baseline_validation.md`, and run the tests.
- **AI Proposal**:
  - Implemented formal evaluation metric engine `src/metrics.py` (Accuracy, Macro-F1, per-class F1, confusion matrix, rationale F1, CFR) with strict penalty rule for invalid predictions.
  - Authored comprehensive audit report `baseline_validation.md` analyzing all 10 requested areas.
  - Documented explicit non-reproduction statement comparing zero-shot NLI against Wadden et al. (2020) VeriSci 3-stage fine-tuned pipeline.
  - Discovered 2 identical text claims between official SciFact train and dev splits; verified zero parameter leakage.
  - Created test suite `tests/test_evaluation_audit.py` (19 tests).
  - Executed full test suite: 40/40 tests passing in 43.82s.
- **Human Decision**: Executed audit, test suite, and generated validation report.
- **Verification Method**: Direct pytest execution console output (`40 passed in 43.82s`) and inspection of generated `baseline_validation.md`.
- **Code / Artifact Change**: Created `src/metrics.py`, `tests/test_evaluation_audit.py`, `baseline_validation.md`; updated `evidence_ledger.csv` (EVD-024, EVD-025) and `research_decisions.md` (RD-009).
- **Outcome**: Baseline fully validated and audited; zero unverified reproduction claims; 100% test pass rate.

---

### Entry AI-011: Multilingual Pipeline Implementation, Translation Auditing, Pilot Benchmark, and Pilot Validation
- **Date / Timestamp**: 2026-11-05
- **Task**: Implement the multilingual experiment while preserving the baseline. Support English, Hindi, and Spanish under direct-classification (Mode A) and evidence-grounded (Mode B) prompting. Enforce strict provenance, gold label alignment, numerical/negation audit, penalty scoring for invalid predictions, and run a small pilot first with genuine runtime, memory, translation, and error analysis before launching full scale.
- **Tool / Prompt**: User request to implement multilingual experiment across English, Hindi, and Spanish under Mode A and Mode B with identical underlying examples, semantic preservation audits, fixed model versions, error taxonomy, and pilot validation before launching the full experiment.
- **AI Proposal**:
  - Implemented translation module `src/translator.py` using `Helsinki-NLP/opus-mt-en-es` and `Helsinki-NLP/opus-mt-en-hi` with regex-based numerical quantity and negation polarity validation.
  - Translated pilot development claims via `scripts/translate_claims.py`, saving verified outputs to `data/claims_dev_multilingual_pilot.jsonl` with full latency and provenance logs.
  - Implemented `src/multilingual_experiment.py` with `MultilingualGenerativeEngine` supporting native ChatML templates (`apply_chat_template`), two-pass JSON parsing with regex recovery, and 5-category error taxonomy (`ERR_RAT_OMIT`, `ERR_RAT_SPURIOUS`, `ERR_POL_INVERT`, `ERR_NEI_FAIL`, `ERR_OTHER`, `NONE`).
  - Implemented experiment runner `scripts/run_multilingual_experiment.py` tracking CFR, Accuracy, Macro-F1, Rationale F1, latency, and full per-instance provenance.
  - Executed 24-pass pilot evaluation across 3 claims $\times$ 3 languages $\times$ 2 modes in 512.38s (21.35s avg latency per pass, 3.2 GB peak RAM).
  - Empirical Debugging Discovery: Discovered Qwen-0.5B generates camelCase `"NOTEnoughInfo"` instead of canonical uppercase underscore `"NOT_ENOUGH_INFO"`. Upgraded `normalize_eval_label` to strip whitespace, underscores, and dashes, reducing invalid predictions from 16 to 0 (100% valid parsing).
  - Implemented unit test suite `tests/test_multilingual_experiment.py` (16 unit tests). Full project test suite (56 tests) passed with 100% pass rate.
  - Authored comprehensive validation report `results/multilingual_pilot_validation.md` analyzing hardware, translation audits, condition metrics, CFR, and go/no-go recommendations.
- **Human Decision**: Executed pilot evaluation, debugged output normalization, ran test suite, and generated pilot validation report.
- **Verification Method**: Direct background task execution console outputs (`task-953`, `task-1104`, `task-1136`), genuine CSV records in `results/multilingual_pilot_predictions.csv`, summary JSON in `results/multilingual_pilot_predictions_summary.json`, and pytest execution (`56 passed in 14.50s`).
- **Code / Artifact Change**: Created `src/translator.py`, `scripts/translate_claims.py`, `data/claims_dev_multilingual_pilot.jsonl`, `src/multilingual_experiment.py`, `scripts/run_multilingual_experiment.py`, `results/multilingual_pilot_predictions.csv`, `results/multilingual_pilot_predictions_summary.json`, `tests/test_multilingual_experiment.py`, `results/multilingual_pilot_validation.md`; updated `src/metrics.py`, `evidence_ledger.csv` (EVD-026, EVD-027), and `research_decisions.md` (RD-010, RD-011).
- **Outcome**: Multilingual experimental pipeline fully implemented, tested, and validated on genuine pilot data without simulated or fabricated scores; pipeline ready for controlled execution.

---

### Entry AI-012: Comprehensive Evaluation Engine, Disagreement Metrics, Calibration, and Visualizations
- **Date / Timestamp**: 2026-11-05
- **Task**: Implement an evaluation script using actual prediction files. Verify input schema before computing metrics. Calculate Accuracy, Macro-F1, per-class P/R, confusion matrices, invalid-output rate, matched-pair disagreement, runtime per example, and calibration metrics (Brier score, ECE). Check gold alignment across languages. Export results/metrics_by_language.csv, results/metrics_by_prompt.csv, results/paired_disagreement.csv, results/predictions.csv, generate clear figures from actual values, and add unit tests.
- **Tool / Prompt**: User request for evaluation script on actual prediction files, pre-verification of input schema, calculation of Accuracy, Macro-F1, per-class P/R, confusion matrices, invalid-output rate, matched-pair disagreement, latency, calibration, exports of metric CSVs, canonical predictions, genuine figures, and test suite.
- **AI Proposal**:
  - Authored `src/evaluation.py` implementing `verify_prediction_schema`, `verify_gold_label_alignment`, `compute_classification_metrics`, `compute_calibration_metrics`, `compute_runtime_metrics`, `compute_paired_disagreements`, `generate_evaluation_tables`, and `generate_all_evaluation_figures`.
  - Created standalone CLI runner `scripts/evaluate_predictions.py`.
  - Executed evaluation on genuine 24-pass predictions:
    - Mode A (Direct): Accuracy 0.2500, Macro-F1 0.1333, Invalid-output rate 0.0, Latency 17.64s avg.
    - Mode B (Grounded): Accuracy 0.5000, Macro-F1 0.3333, Invalid-output rate 0.0, Latency 24.17s avg.
    - Cross-lingual Disagreement: 0.00% across all pairs in both modes.
    - Cross-prompt Disagreement: 0.2500 (Mode A vs Mode B on Claim 5).
    - Calibration: Brier Score = 0.6254, ECE = 0.6208, MCE = 0.7417.
    - Runtime: Mean = 20.90s, Median = 15.59s, Std = 11.85s, Total = 501.71s.
  - Exported canonical `results/predictions.csv`, `results/metrics_by_language.csv`, `results/metrics_by_prompt.csv`, and `results/paired_disagreement.csv`.
  - Generated 5 publication-quality figures in `results/figures/` using genuine values without fabrication.
  - Built unit test suite `tests/test_evaluation_metrics.py` (17 tests). Full project test suite (73 tests) passed in 17.17s.
- **Human Decision**: Executed evaluation script, verified schema and alignment, generated figures, and ran regression suite.
- **Verification Method**: Direct command execution of `python scripts/evaluate_predictions.py` and `pytest tests/test_evaluation_metrics.py`.
- **Code / Artifact Change**: Created `src/evaluation.py`, `scripts/evaluate_predictions.py`, `tests/test_evaluation_metrics.py`, `results/predictions.csv`, `results/metrics_by_language.csv`, `results/metrics_by_prompt.csv`, `results/paired_disagreement.csv`, `results/figures/fig1_accuracy_f1_by_condition.png`, `fig2_confusion_matrices.png`, `fig3_runtime_by_condition.png`, `fig4_paired_disagreement.png`, `fig5_calibration_analysis.png`; updated `evidence_ledger.csv` (EVD-028, EVD-029) and `research_decisions.md` (RD-012).
- **Outcome**: Complete evaluation pipeline operational, audited, and tested with genuine empirical data and publication figures.

---

### Entry AI-013: Scientific Results Analysis, Qualitative Case Studies, and Epistemic Demarcation
- **Date / Timestamp**: 2026-11-05
- **Task**: Analyze genuine experiment outputs. Answer each research question separately. Distinguish directly observed results, plausible explanations, alternative explanations, uncertain findings, and unsupported prohibited claims. Inspect examples where predictions differ across languages/modes. Categorize errors after examining claim, evidence, translation, output, and gold label. Investigate associations with negation, terminology, omission, ambiguity, invalid outputs, and translation errors. Compare full vs verified subset. Report limitations and negative results honestly. Save as results_analysis.md.
- **Tool / Prompt**: User request for comprehensive evidence-based results and discussion notes adhering to 5-part epistemic classification and detailed qualitative error examination.
- **AI Proposal**:
  - Conducted systematic per-instance inspection of all 24 prediction passes, matching claims, translations, abstracts, model outputs, and gold annotations.
  - Authored comprehensive report `results_analysis.md` answering RQ1 (Veracity Degradation), RQ2 (Direct vs Grounded Prompting), RQ3 (Abstention & Calibration), and RQ4 (Cross-Lingual Consistency) with strict 5-part epistemic demarcation.
  - Documented in-depth qualitative case studies for Claim 1, Claim 3, and Claim 5, identifying exact cognitive mechanisms (NEI-collapse in Mode A, numerical transduction $1/2000 \approx 493/\text{million}$ in Mode B) and catastrophic translation drift in Hindi (translating *"genomes project"* to *"भूविज्ञान परियोजना"* / Geology project in Claim 3, and dropping *"PrP"* to *"असामान्य त्रुटि"* in Claim 5).
  - Documented negative results honestly: complete Mode A NEI-collapse, zero-shot sentence rationale omission failure (`ERR_RAT_OMIT` on 100% of verifiable claims), and poor verbalized calibration (ECE = 0.6208).
  - Defined explicit checklist of prohibited unsupported claims barred from appearing in publication.
- **Human Decision**: Analyzed genuine outputs, verified translations and raw generations, and saved `results_analysis.md`.
- **Verification Method**: Direct inspection of `results/predictions.csv`, `data/claims_dev_multilingual_pilot.jsonl`, and verification of all numerical values in `results_analysis.md`.
- **Code / Artifact Change**: Created `results_analysis.md`; updated `evidence_ledger.csv` (EVD-030) and `research_decisions.md` (RD-013).
- **Outcome**: Rigorous scientific analysis and discussion notes completed without speculative overclaiming or fabricated metrics.
