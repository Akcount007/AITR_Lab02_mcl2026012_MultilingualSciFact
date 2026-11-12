# Baseline Validation and Audit Report: Multilingual Scientific Claim Verification

**Project**: Evaluating Multilingual Factual Reliability of Large Language Models in Scientific Literature Analysis  
**Author**: Aman Kumar (`mcl2026012@iiita.ac.in`)  
**Course**: AI Tools for Research (AITR) — Lab 02  
**Baseline Model**: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (278,811,651 parameters, `deberta-v2`)  
**Audit Date**: 2026-11-04  
**Audit Status**: **[VERIFIED]** (All 10 Audit Dimensions Examined Against Primary Datasets and Checkpoint Documentation)

---

## 1. Executive Summary

This document reports the technical audit of the implemented scientific claim verification baseline against the official SciFact dataset ([Wadden et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.278/)) and the Hugging Face model documentation for [`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli).

A total of **40 automated tests** (21 core tests + 19 audit tests) were executed. All 40 passed without error.

```text
============================== Test Suite Summary ==============================
tests/test_data_loader.py ........  5 PASSED in  0.04s
tests/test_baseline.py ........... 16 PASSED in 62.68s
tests/test_evaluation_audit.py ... 19 PASSED in 37.33s
--------------------------------------------------------------------------------
Total Automated Tests:             40 PASSED (100% Pass Rate, 0 Failures)
================================================================================
```

---

## 2. Dataset Label Mapping Audit

### 2.1 Raw Dataset vs. Task Taxonomy
The raw SciFact release files (`claims_train.jsonl`, `claims_dev.jsonl`) store veracity labels with a schema discrepancy relative to published papers:

| Schema Level | Positive Class | Negative Class | Neutral / Ungrounded Class |
| :--- | :--- | :--- | :--- |
| **SciFact Raw JSON** | `"SUPPORT"` | `"CONTRADICT"` | Empty evidence `{}` / absent |
| **SciFact Paper Prose** | `SUPPORT` | `REFUTE` | `NOT_ENOUGH_INFO` |
| **FEVER / Standard Factuality** | `SUPPORTS` | `REFUTES` | `NOT_ENOUGH_INFO` |
| **Pre-Registered Protocol** | `SUPPORTS` | `REFUTES` | `NOT_ENOUGH_INFO` |

### 2.2 Bidirectional Normalization in `src/data_loader.py`
The data loader implements bidirectional mapping via `normalize_label()`:
```python
LABEL_MAPPING = {
    "SUPPORT": "SUPPORT", "SUPPORTS": "SUPPORT",
    "CONTRADICT": "CONTRADICT", "CONTRADICTS": "CONTRADICT",
    "REFUTE": "CONTRADICT", "REFUTES": "CONTRADICT",
    "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO", "NEI": "NOT_ENOUGH_INFO", "NO_INFO": "NOT_ENOUGH_INFO"
}
```
In `src/metrics.py`, `normalize_eval_label()` standardizes all variants to the pre-registered 3-class canonical set: `{"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}`. Any unmapped or malformed string maps deterministically to `"INVALID_OUTPUT"`.

---

## 3. Evidence and Claim Alignment Audit

### 3.1 Sentence Pointer Indexing
- **Audit Question**: Are sentence indices in SciFact `evidence[doc_id][...]["sentences"]` 0-indexed or 1-indexed?
- **Verification Result**: Inspected all 366 evidence sentence pointers across the 300 claims in `claims_dev.jsonl`.
  - Minimum pointer: index `0` (points to the first sentence of `corpus[doc_id]["abstract"]`).
  - Maximum pointer: within `len(abstract) - 1`.
  - **Out-of-bounds count**: **0 out of 366 pointers** (`0.0%`).
  - Indexing is strictly **0-indexed**, matching Python array slicing directly.

### 3.2 Evidence Extraction Logic
In `scripts/run_baseline.py` (`build_evidence_text()`):
- When `use_gold_rationale=True`: Sentences indicated by gold `rationale_sentences` are concatenated in original order.
- When `use_gold_rationale=False`: The entire abstract is concatenated.
- When a cited document has no rationale sentences (or evidence is `{}`): The entire abstract is provided as premise, and the gold label is assigned `NOT_ENOUGH_INFO`.

---

## 4. Correct Use of the Model

### 4.1 NLI Framing (Premise vs. Hypothesis)
- **Standard NLI Definition**: A Premise ($P$) entails, contradicts, or is neutral towards a Hypothesis ($H$).
- **Scientific Claim Verification Mapping**:
  $$\text{Premise } (P) = \text{Scientific Evidence Text (PubMed Abstract Sentences)}$$
  $$\text{Hypothesis } (H) = \text{Scientific Claim to be Verified}$$
- **Implementation Check**: In `src/baseline_model.py`:
  ```python
  inputs = self.tokenizer(evidence, claim, truncation=True, max_length=512, return_tensors="pt")
  ```
  The first argument (`evidence`) maps to Sequence A ($P$), and the second argument (`claim`) maps to Sequence B ($H$). This adheres strictly to the formulation established in Wadden et al. (2020).

### 4.2 Verified Model Configuration
Inspected directly from `model.config`:
- `model_type`: `deberta-v2`
- `num_labels`: 3
- `id2label`: `{0: 'entailment', 1: 'neutral', 2: 'contradiction'}`
- `label2id`: `{'contradiction': 2, 'entailment': 0, 'neutral': 1}`

The normalization function maps:
- `entailment` (0) $\longrightarrow$ `SUPPORTS`
- `neutral` (1) $\longrightarrow$ `NOT_ENOUGH_INFO`
- `contradiction` (2) $\longrightarrow$ `REFUTES`

---

## 5. Tokenization and Truncation Audit

### 5.1 Maximum Sequence Length
The model backbone (`mDeBERTa-v3-base`) supports a maximum position embedding length of 512 tokens.

### 5.2 Truncation Strategy Verification
- **Audit Check**: When $\text{length}(P) + \text{length}(H) > 512$, does truncation discard the claim or the evidence?
- **Empirical Test**: Executed 600-word premise paired with a short hypothesis claim.
- **Result**:
  ```text
  Total tokens: 512
  Decoded ends with: "... rd Word Word Word Word[SEP] Hypothesis claim.[SEP]"
  Is hypothesis in decoded output: True
  ```
  Hugging Face `AutoTokenizer` employs `truncation="longest_first"` by default for pair inputs. When the evidence abstract is exceptionally long, tokens are trimmed from the evidence while preserving the hypothesis claim intact.
- In `scripts/run_baseline.py`, whenever an input reaches the 512-token ceiling, the `truncated` column in the prediction CSV is flagged as `True`.

---

## 6. Missing and Empty Evidence Handling

SciFact contains claims where no supporting or refuting evidence exists in cited abstracts:
- In `claims_dev.jsonl`, **112 out of 300 claims** (37.33%) have empty evidence `{}`.
- Among these, some cite PubMed documents where the claim cannot be verified (`cited_doc_ids`), while others cite no documents.
- **Handling in Baseline**:
  1. Claims with cited documents but empty rationale: Abstract text is fed to the model; gold label is evaluated as `NOT_ENOUGH_INFO`.
  2. Claims with no cited documents: Handled via `_empty_prediction()`, assigning `predicted_label = "NOT_ENOUGH_INFO"`, `confidence = 0.0`, and reason code `"no_evidence_docs"`.
  3. Empty claim string or empty evidence string: Safely caught before model forward pass, preventing crashes.

---

## 7. Output Normalization and Invalid Prediction Protocol

### 7.1 Protocol Enforcement
In compliance with the pre-registered protocol ([`experiment_protocol.md`](file:///C:/Users/akema/.gemini/antigravity/scratch/AITR_Lab02_mcl2026012_MultilingualSciFact/experiment_protocol.md)):
- Any prediction that does not match one of `{"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}` is marked as `INVALID_OUTPUT`.
- **Zero-Tolerance Penalty Rule**: `INVALID_OUTPUT` records are **never silently dropped** from the evaluation sample. They are scored as incorrect ($0.0$), penalizing both Accuracy and Macro-F1.

---

## 8. Duplicate Examples and Dataset Integrity Audit

Executed direct scans across all three official partitions:

```text
Split Name      Total Claims   Unique IDs   Duplicate Claims
Train Split     809            809          0
Dev Split       300            300          0
Test Split      300            300          0
Corpus Split    5,183 docs     5,183 docs   0
```

- **Duplicate IDs**: 0 across all partitions.
- **Corpus Integrity**: All document IDs referenced in `claims_dev.jsonl` exist in `corpus.jsonl`.

---

## 9. Train / Test Contamination Risk Assessment

### 9.1 Partition Leakage Audit

| Comparison | Overlapping Claim IDs | Overlapping Claim Text | Leakage Risk Assessment |
| :--- | :---: | :---: | :--- |
| **Dev vs. Test** | **0** | **0** | **Zero leakage**. Complete partition separation. |
| **Train vs. Test** | **0** | **0** | **Zero leakage**. Test partition strictly preserved. |
| **Train vs. Dev** | **0** | **2** | **Identified minor text overlap in official release**. |

### 9.2 Investigation of Train-Dev Text Overlap in Official SciFact Release
Two claims in `claims_dev.jsonl` share identical text with claims in `claims_train.jsonl`:

1. **Claim Text**: *"Obesity decreases life quality."*
   - Dev ID: `870` (cited: `[195689316]`, evidence: `{}`)
   - Train ID: `871` (cited: `[195689316]`, evidence: `{}`)
2. **Claim Text**: *"There is no association between HNF4A mutations and diabetes risks."*
   - Dev ID: `1292` (cited: `[56893404]`, evidence: `{}`)
   - Train ID: `1291` (cited: `[56893404]`, evidence: `{}`)

> [!IMPORTANT]
> **Impact on this project**: **Zero contamination**. Because our baseline and experimental models are evaluated under **frozen zero-shot inference** (no model training, fine-tuning, or prompt optimization on `claims_train.jsonl`), this official dataset artifact introduces no parameter leakage into our evaluation.

---

## 10. Metric Implementation and Comparison with Published Baselines

### 10.1 Published Baseline Methodology (Wadden et al., EMNLP 2020)
In the original SciFact publication, the authors propose **VeriSci**, a three-stage pipeline:
1. **Abstract Retrieval**: BM25 / TF-IDF retrieval of candidate abstracts from the 5,183-document corpus.
2. **Rationale Selection**: Fine-tuned SciBERT / RoBERTa-large binary classifier predicting which abstract sentences constitute evidence.
3. **Stance Prediction**: Fine-tuned 2-way classifier (`SUPPORT` vs. `CONTRADICT`) applied to selected rationale sentences. If no sentences are selected, the claim is classified as `NOT_ENOUGH_INFO`.

**Published VeriSci Results** (Wadden et al., 2020, Table 2):
- *Oracle Rationale Selection* (gold rationale provided): Abstract-level Stance F1 $\approx 70.0 - 75.0\%$.
- *End-to-End Pipeline* (retrieval + selection + stance): Abstract-level Selection+Stance F1 $\approx 45.0 - 50.0\%$.

### 10.2 Implemented Baseline Methodology
Our baseline evaluates **Zero-Shot / Frozen Multilingual NLI**:
- Model is **not fine-tuned on SciFact** training data.
- Directly performs 3-class sequence classification ($P, H \to \{\text{SUPPORTS}, \text{REFUTES}, \text{NOT\_ENOUGH\_INFO}\}$).
- Evaluates stance classification given gold evidence rationales.

### 10.3 Explicit Non-Reproduction Statement
> [!WARNING]
> **Do not claim reproduction of Wadden et al. (2020)**:
> 1. Our baseline does **not** fine-tune RoBERTa/SciBERT on SciFact train split.
> 2. Our baseline does **not** execute the 3-stage retrieval pipeline.
> 3. Our baseline measures out-of-the-box multilingual NLI transfer capability on scientific claim-evidence pairs, serving as a lower-bound reference point for multilingual generative evaluations.

---

## 11. Empirical Verification on Smoke Test Partition

The baseline was executed on the first 20 claims of `claims_dev.jsonl` (22 evaluation pairs):

- **Command**: `python scripts/run_baseline.py --max-claims 20 --output results/baseline_predictions_smoke.csv`
- **Output CSV**: [`results/baseline_predictions_smoke.csv`](file:///C:/Users/akema/.gemini/antigravity/scratch/AITR_Lab02_mcl2026012_MultilingualSciFact/results/baseline_predictions_smoke.csv)
- **Summary JSON**: [`results/baseline_predictions_smoke_summary.json`](file:///C:/Users/akema/.gemini/antigravity/scratch/AITR_Lab02_mcl2026012_MultilingualSciFact/results/baseline_predictions_smoke_summary.json)

### Formal Metrics (via `src/metrics.py`)

$$\text{Accuracy} = \mathbf{0.7273} \quad (16 / 22)$$
$$\text{Macro-F1} = \mathbf{0.7313} \quad (\text{Macro-P: } 0.7556, \text{Macro-R: } 0.7389)$$

#### Detailed Class Breakdown
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **SUPPORTS** | 1.0000 | 0.6667 | 0.8000 | 9 |
| **REFUTES** | 0.6667 | 0.8000 | 0.7273 | 5 |
| **NOT_ENOUGH_INFO** | 0.6000 | 0.7500 | 0.6667 | 8 |

#### Confusion Matrix
$$\begin{pmatrix}
6 & 0 & 3 \\
0 & 4 & 1 \\
0 & 2 & 6
\end{pmatrix}$$
*(Rows: Gold [SUPPORTS, REFUTES, NEI]; Columns: Predicted [SUPPORTS, REFUTES, NEI])*

---

## 12. Discrepancy Ledger (Intended vs. Implemented)

| Dimension | Intended Specification | Implemented Method | Resolution & Justification |
| :--- | :--- | :--- | :--- |
| **Negative Label Name** | `REFUTES` | `CONTRADICT` in raw data, `REFUTES` in evaluation | Handled via bidirectional normalization in `src/data_loader.py` and `src/metrics.py`. |
| **Model Type** | Multilingual NLI | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | Selected and verified. 278M parameters, CPU-runnable, supports English, Spanish, and Hindi. |
| **SciFact Split** | Dev Split (300 claims) | Dev Split (300 claims) | Primary evaluation partition confirmed. Test split preserved blind. |
| **Baseline Role** | Lower-bound anchor | Zero-shot frozen NLI | Strictly decoupled from generative prompt experiments; not claimed as VeriSci reproduction. |
| **Invalid Output Handling** | Never drop invalid outputs | Strict penalty scoring | Handled in `src/metrics.py`: unmapped outputs assigned `INVALID_OUTPUT` (score 0.0). |

---

## 13. Audit Conclusion & Gate G2 Status

The baseline claim-verification pipeline meets all Gate G2 integrity standards:
- Verified data loaders with zero-leakage split isolation.
- Verified deterministic label mapping matching checkpoint configuration.
- Real execution outputs exported with softmax confidences and rationale linkage.
- Comprehensive unit test suite with 100% pass rate.

**Gate G2 Status**: **VERIFIED & OPERATIONAL**. Ready to proceed with cross-lingual translation pipeline (`scripts/translate_claims.py`).
