# One-Page Research Brief

**Title:** Cross-Lingual Evidence Grounding and Prediction Stability in Scientific Claim Verification  
**Author:** Aman Kumar (`mcl2026012@iiita.ac.in`)  
**Institution:** Indian Institute of Information Technology, Allahabad  
**Context:** M.Tech Implementation Study — *AI Tools for Research (AITR)*  
**Date:** 2026-11-04  
**Integrity Protocol:** All claims tagged as `[VERIFIED]`, `[INFERRED]`, or `[UNKNOWN]`. No invented accuracy numbers.  

---

## 1. Research Questions

### Primary Question
**How does language affect a multilingual model's ability to verify scientific claims against supplied scientific evidence?**

### Supporting Questions
1. **SQ1 (Performance Variation):** How does claim verification performance (veracity classification Macro-F1 and sentence-level evidence rationale F1) vary when identical scientific claims are evaluated in English, Hindi, and Spanish against English literature abstracts?
2. **SQ2 (Cross-Lingual Prediction Stability):** How often do a model's predicted veracity labels and selected rationale sentences change for the exact same scientific claim when presented across different languages (cross-lingual flip rate and pairwise agreement)?
3. **SQ3 (Evidence-Interpretation Error Taxonomy):** What specific categories of evidence-interpretation errors occur across languages (e.g., rationale omission, spurious evidence attribution, translation-mediated terminological drift, or failure to abstain on unanswerable claims)?

---

## 2. Research Objectives
1. **Quantify cross-lingual performance disparities:** Benchmark open-weights multilingual LLMs on scientific claim verification, isolating claim language while holding scientific evidence abstracts constant in English `[VERIFIED design]`.
2. **Measure instance-level prediction consistency:** Compute pairwise consistency and prediction flip rates across languages to evaluate whether models maintain semantic invariance or exhibit surface-language sensitivity `[INFERRED]`.
3. **Establish an empirical error taxonomy:** Qualitatively and quantitatively categorize failure modes across direct prompting and translation-mediated pipelines to identify where factual breakdown originates `[INFERRED]`.

---

## 3. Testable Hypotheses

- **H1 (Grounding Degradation Asymmetry):** When claims are evaluated in non-English languages (Hindi, Spanish) against English abstracts, sentence-level rationale selection F1 degrades more substantially relative to the English baseline than coarse 3-way veracity Macro-F1 does.
- **H2 (Cross-Lingual Instability):** Presenting semantically equivalent scientific claims in different languages produces a non-trivial prediction flip rate ($\text{CFR} > 0$), where the model changes its verdict despite unchanged scientific evidence.
- **H3 (Pipeline-Specific Failure Modes):** Direct Multilingual Prompting exhibits higher rationale omission rates due to cross-lingual attention dispersion, whereas the Translate-Test pipeline exhibits higher false-refutation rates caused by machine translation distorting domain-specific biochemical terminology.
- **H4 (Abstention Vulnerability):** In non-English conditions, models exhibit lower accuracy on unanswerable (`NOT_ENOUGH_INFO`) claims, hallucinating `SUPPORT` or `REFUTE` labels rather than appropriately abstaining.

---

## 4. Experimental Variables

| Variable Type | Variable Name | Operational Definition & Levels |
| :--- | :--- | :--- |
| **Independent Variable 1** | **Claim Language** | 3 levels: English (`en`, baseline), Hindi (`hi`, Indic, non-Latin script), Spanish (`es`, Romance, Latin script). |
| **Independent Variable 2** | **Inference Pipeline** | 2 levels: (1) *Direct Multilingual Prompting* (non-English claim + English abstract); (2) *Translate-Test Pipeline* (non-English claim machine-translated to English $\to$ English claim + English abstract). |
| **Independent Variable 3** | **Model Architecture** | 2 open-weights model families: Meta Llama-3-8B-Instruct vs. Alibaba Qwen-2.5-7B-Instruct (both 4-bit NF4 quantized). |
| **Controlled Variables** | **Evidence Context & Decoding** | Evidence abstracts remain strictly in English (PubMed source text); decoding parameters frozen (greedy decoding, $\text{temperature} = 0.0$, $\text{max\_new\_tokens} = 128$); identical system prompt schemas. |
| **Dependent Variable 1** | **Veracity Performance** | 3-way Macro-F1, Precision, and Recall across `SUPPORT`, `REFUTE`, and `NOT_ENOUGH_INFO`. |
| **Dependent Variable 2** | **Evidence Grounding** | Sentence-level Rationale Precision, Recall, and Strict/Loose F1 against gold sentence indices. |
| **Dependent Variable 3** | **Prediction Stability** | Cross-Lingual Flip Rate ($\text{CFR}$) and Cohen’s Kappa ($\kappa$) between language pairs $(L_1, L_2)$ on identical claim IDs. |
| **Dependent Variable 4** | **Abstention & Calibration** | Abstention Accuracy on the `NOT_ENOUGH_INFO` partition; Expected Calibration Error (ECE). |
| **Dependent Variable 5** | **Error Distribution** | Proportions of errors mapped to 4 taxonomy classes: Rationale Omission, Spurious Attribution, Terminological Drift, and Polarity Inversion. |

---

## 5. Dataset & Language Conditions

- **Source Dataset:** SciFact Development Set `[VERIFIED: github.com/allenai/scifact, Wadden et al., 2020]`.
  - Size: 300 expert-annotated biomedical claims mapped to 5,183 candidate PubMed abstracts.
  - Labels: `SUPPORT`, `REFUTE`, `NOT_ENOUGH_INFO` with gold sentence rationale indices.
  - License: CC BY 4.0 (claims & annotations), ODC-By 1.0 (abstracts).
- **Language Conditions:**
  - **English (`en`):** Original expert-written claims (gold standard reference).
  - **Hindi (`hi`):** Machine-translated claims using NLLB-200-3.3B (`hin_Deva`), representing an Indic, morphologically rich language with lower scientific NLP representation.
  - **Spanish (`es`):** Machine-translated claims using NLLB-200-3.3B (`spa_Latn`), representing a high-resource Romance language with extensive Latinate biomedical cognates.
- **Quality Control:** 15% random sample of translated claims subjected to manual spot-checking for terminological preservation and semantic equivalence before running evaluation.

---

## 6. Baselines & Prompting Strategies

1. **English Gold Baseline:** English claim + English abstract $\to$ frozen LLM. Serves as the upper-bound performance ceiling for each model.
2. **Strategy A (Direct Multilingual Prompting):** Non-English claim (Hindi/Spanish) + English abstract $\to$ frozen multilingual LLM. Prompts instruct the model to reason across the language gap and output standardized JSON:
   ```json
   {"veracity": "SUPPORT|REFUTE|NOT_ENOUGH_INFO", "rationale_sentences": [1, 3], "confidence": 0.85}
   ```
3. **Strategy B (Translate-Test Pipeline):** Non-English claim $\to$ NLLB-200-3.3B translates to English $\to$ English prompt $\to$ frozen LLM.
4. **Deterministic Controls:** All prompts use explicit zero-shot instructions with identical schema definitions, zero temperature, and fixed random seeds.

---

## 7. Metrics & Mathematical Definitions

- **Veracity Macro-F1:**
  $$\text{Macro-F1} = \frac{1}{3} \sum_{c \in \{\text{SUP}, \text{REF}, \text{NEI}\}} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
- **Sentence-Level Rationale F1:**
  $$\text{Rationale-F1} = \frac{2 \cdot |S_{\text{pred}} \cap S_{\text{gold}}|}{|S_{\text{pred}}| + |S_{\text{gold}}|}$$
  Evaluated under strict matching (label must also match) and loose matching (sentence overlap regardless of label).
- **Cross-Lingual Flip Rate (CFR):**
  $$\text{CFR}(L_1, L_2) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(\hat{y}_i^{(L_1)} \ne \hat{y}_i^{(L_2)}\right)$$
- **Expected Calibration Error (ECE):** Partition predictions into $M$ confidence bins $B_m$:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

---

## 8. Success Criteria & Decision Rules

- **Criterion 1 (Cross-Lingual Effect Significance):** Observed drops in Veracity Macro-F1 and Rationale F1 between English and Hindi/Spanish must be statistically significant at $p < 0.05$ via paired bootstrap resampling (1,000 iterations).
- **Criterion 2 (Grounding Sensitivity Demonstration):** Support for H1 requires the relative drop in Rationale F1 ($\Delta R / R_{\text{en}}$) to exceed the relative drop in Veracity Macro-F1 ($\Delta V / V_{\text{en}}$) by at least 1.5×.
- **Criterion 3 (Flip Rate Detection):** Detection of a non-trivial prediction flip rate ($\text{CFR} \ge 10\%$) between English and non-English claims.
- **Criterion 4 (Traceable Error Categorization):** Successful manual and automated categorization of 100% of divergent predictions into the 4-part error taxonomy.

---

## 9. Possible Contradictory Findings

- **Contradiction to H1:** Rationale F1 drops less than or equal to Veracity Macro-F1, indicating that sentence-level evidence grounding is more robust to translation than high-level veracity inference.
- **Contradiction to H2:** Cross-lingual flip rate is negligible ($\text{CFR} < 3\%$), demonstrating that multilingual LLMs possess strong semantic invariance across scripts and languages in biomedical contexts.
- **Contradiction to H3:** Direct Multilingual Prompting outperforms Translate-Test across all metrics for Hindi, showing that joint representation spaces handle scientific semantics better than explicit machine translation.
- **Contradiction to H4:** Abstention accuracy on `NOT_ENOUGH_INFO` claims improves in non-English prompts, indicating that multilingual prompting induces conservative hedging rather than overconfidence.

---

## 10. Compute Requirements & Feasibility

- **Hardware:** Google Colab Free Tier (Single T4 GPU, ~15GB VRAM) or equivalent local machine `[VERIFIED]`.
- **Quantization:** 4-bit NormalFloat (NF4) via `bitsandbytes` loads 7B/8B models in ~4.5–5.0 GB VRAM `[VERIFIED]`.
- **Execution Budget:** 12 experimental conditions (3 languages × 2 pipelines × 2 models) × 300 claims = 3,600 inference passes.
- **Throughput:** At ~2.0 seconds per pass, total inference time is ~2.0–2.5 hours wall-clock.
- **Execution Protocol:** Sequential execution (translate claims once $\to$ run Model 1 $\to$ clear cache $\to$ run Model 2) to eliminate OOM risks.

---

## 11. Scope Limitations

1. **Dataset Boundary:** Evaluated on SciFact dev set (300 claims). While standard for controlled LLM probing, statistical power is bounded; findings must be reported with 95% bootstrap confidence intervals.
2. **Domain Focus:** Restricted to biomedical literature abstracts. Results may not generalize to formal mathematics, physics proofs, or chemistry retrosynthesis.
3. **Model Scale:** Evaluates open-weights 7B–8B models. Frontier proprietary models (e.g., GPT-4o, Claude 3.5 Sonnet) may exhibit higher multilingual alignment.
4. **Translation Engine:** Claims translated using NLLB-200-3.3B. While open and reproducible, machine translation quality may introduce domain artifacts that are measured as part of the error taxonomy.

---

## 12. Demarcation: Prior Published Work vs. This Study

| Prior Published Benchmark / Work | What Was Demonstrated `[VERIFIED]` | What Remains Unproven | Novel Experimental Contribution of This Study |
| :--- | :--- | :--- | :--- |
| **SciFact**<br>*(Wadden et al., 2020)* | Benchmarked evidence retrieval, rationale selection, and veracity prediction on English biomedical claims. | Cross-lingual claim behavior; non-English queries against English abstracts. | First evaluation of sentence-level rationale selection and evidence grounding under cross-lingual claim input. |
| **Poly-FEVER**<br>*(Zhang et al., 2024)* | Translated SciFact/FEVER claims into 11 languages for closed-book binary veracity classification. | Open-book evidence grounding; sentence rationale extraction; abstention on unanswerable claims. | Evaluates open-book grounding with PubMed abstracts, sentence-level rationale F1, and `NOT_ENOUGH_INFO` abstention. |
| **Multi-FAct**<br>*(Shafayat et al., 2024)* | Evaluated multilingual LLM factuality via atomic claims (FActScore) on Wikipedia biographies. | Domain-specific scientific literature; peer-reviewed abstract grounding. | Evaluates structured scientific evidence verification rather than general biographical Wikipedia generation. |
| **Cross-Lingual NLI Studies** | Evaluated Translate-Test vs. Direct Multilingual models on generic XNLI datasets. | Impact on fine-grained scientific rationales and domain terminological drift. | Establishes a domain-specific error taxonomy quantifying terminological drift vs. rationale omission in scientific claim verification. |
