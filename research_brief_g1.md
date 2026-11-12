# Gate G1: One-Page Research Brief

**Project Title:** Cross-Lingual Evidence Grounding in Scientific Claim Verification: Do LLMs Lose the Rationale When the Query Changes Language?  
**Author:** Aman Kumar (`mcl2026012@iiita.ac.in`)  
**Institution:** Indian Institute of Information Technology, Allahabad  
**Course / Milestone:** AI Tools for Research (AITR) — Gate G1 Submission  
**Date:** 2026-11-04  
**Integrity Status:** All claims tagged as `[VERIFIED]`, `[INFERRED]`, or `[UNKNOWN]`  

---

## 1. Specification Matrix (Assignment Section 11 Template)

| Field | Project Information & Verified Evidence |
| :--- | :--- |
| **Review topic and key papers** | **Topic:** Evaluating Multilingual Factual Reliability of Large Language Models in Scientific Literature Analysis.<br>**Key References Verified from Paper 1:**<br>1. *SciFact*: Wadden et al. (2020), EMNLP 2020 `[VERIFIED]` (Ref [16])<br>2. *Multi-Fact*: Shafayat et al. (2024), arXiv:2402.18045 / COLM 2024 `[VERIFIED]` (Ref [12])<br>3. *FActScore*: Min et al. (2023), EMNLP 2023 `[VERIFIED]` (Ref [9])<br>4. *PubMedQA*: Jin et al. (2019), EMNLP-IJCNLP 2019 `[VERIFIED]` (Ref [3])<br>5. *Poly-FEVER*: Zhang et al. (2024), ACL / OpenReview `[VERIFIED]` |
| **Research gap** | Existing scientific claim verification benchmarks (SciFact, PubMedQA) are strictly **English-only**, whereas multilingual factuality benchmarks (Multi-Fact on Wikipedia biographies, Poly-FEVER on closed-book veracity) **completely omit sentence-level evidence grounding (rationale extraction)** and abstention calibration under insufficient evidence. It is unknown whether cross-lingual claim input degrades rationale grounding faster than veracity labeling, or triggers rationale hallucination. `[VERIFIED from literature audit]` |
| **Research question** | **Primary Question:** When large language models verify biomedical claims against English PubMed abstracts, does presenting claims in Hindi or Spanish degrade sentence-level evidence grounding more severely than coarse veracity classification, and does cross-lingual input impair the model's ability to abstain when evidence is insufficient?<br>**Sub-Question 1 (Pipeline Trade-off):** Does a Translate-Test pipeline mitigate cross-lingual grounding collapse compared to Direct Multilingual Prompting, or does it introduce terminological drift?<br>**Sub-Question 2 (Calibration / Abstention):** Does cross-lingual claim formulation cause models to become uncalibrated and over-commit on unanswerable (`NOT_ENOUGH_INFO`) claims? |
| **Study pathway** | **Pathway B: Controlled Comparative Study** `[VERIFIED]`<br>A factorial 3 (Languages: EN, HI, ES) × 2 (Pipelines: Direct Multilingual vs. Translate-Test) × 2 (Open Models: Llama-3-8B-Instruct vs. Qwen-2.5-7B-Instruct) controlled evaluation. |
| **Dataset & measurement setup** | **Primary Dataset:** SciFact Development Set (300 expert-annotated claims paired with PubMed abstracts, sentence-level gold rationales, and 3-way labels: SUPPORT / REFUTE / NOT_ENOUGH_INFO) `[VERIFIED]`.<br>**License:** CC BY 4.0 (claims/annotations), ODC-By 1.0 (abstracts) `[VERIFIED]`.<br>**Translation Protocol:** Claims translated to Hindi and Spanish using NLLB-200-3.3B (open-weights, CC-BY-NC 4.0), with 15% random subset audited for terminological fidelity. |
| **Baseline & comparators** | **Internal Baseline:** English Zero-Shot Prompting on SciFact dev set (same models, same prompts, gold English claims) `[VERIFIED]`.<br>**Comparators:**<br>1. *Direct Multilingual Prompting* (Non-English claim + English PubMed abstract).<br>2. *Translate-Test Pipeline* (Non-English claim translated to English $\to$ English verification against abstract).<br>**Evaluated Models:** Llama-3-8B-Instruct and Qwen-2.5-7B-Instruct (4-bit NF4 quantized, deterministic temperature=0.0). |
| **Primary metric & control** | **Primary Grounding Metric:** Sentence-level Rationale Precision, Recall, and F1 (strict and loose matching) `[VERIFIED]`.<br>**Primary Veracity Metric:** 3-way Macro-F1 across SUPPORT, REFUTE, NOT_ENOUGH_INFO `[VERIFIED]`.<br>**Calibration / Abstention Metric (Deep Analysis):** Expected Calibration Error (ECE) and Abstention Accuracy on `NOT_ENOUGH_INFO` instances `[VERIFIED]`.<br>**Controls Held Constant:** Abstract corpus (English PubMed text), prompt structure, decoding parameters (greedy `temp=0`), and evaluation splits. |
| **Computing and risks** | **Hardware:** Google Colab T4 GPU (~15GB VRAM) or equivalent local GPU `[VERIFIED]`.<br>**Compute Budget:** ~3,600 inference calls total (12 conditions × 300 claims), runtime ~2.5–3 hours wall-clock `[INFERRED]`.<br>**Risk & Fallback:** If NLLB-200 translation exhibits severe terminological drift on Hindi biomedical entities, fall back to Google Translate API for the 300 claims and document the divergence as a systematic error category. |
| **First executable milestone** | **Gate G2 Baseline Verification:** Set up environment test script (`scripts/test_env.py`), download SciFact dev set, load Qwen-2.5-7B in 4-bit, and execute deterministic English zero-shot baseline on 20 sanity claims, verifying pipeline input/output parsing and metric calculation. |

---

## 2. Testable Hypotheses, Variables, Decision Rules & Contradictions

### Hypothesis 1 (H1: Grounding Asymmetry)
- **Hypothesis:** Sentence-level rationale selection F1 drops significantly more sharply than 3-way veracity Macro-F1 when moving from English to Hindi/Spanish claims against English PubMed abstracts.
- **Independent Variable:** Claim language (English, Hindi, Spanish).
- **Dependent Variables:** Veracity Macro-F1, Sentence Rationale F1.
- **Decision Rule / Success Threshold:** $\Delta \text{Rationale F1} > 1.5 \times \Delta \text{Veracity F1}$ relative to English baseline ($p < 0.05$ via bootstrap resampling).
- **Contradicting Observation:** Rationale F1 decreases by the same or lesser proportion as Veracity F1, indicating that language shifts affect high-level reasoning and fine-grained evidence localization equally.

### Hypothesis 2 (H2: Pipeline Trade-off)
- **Hypothesis:** The Translate-Test pipeline achieves higher rationale grounding F1 than Direct Multilingual Prompting for Hindi, but exhibits higher false-refutation rates on claims containing complex drug/gene abbreviations due to translation drift.
- **Independent Variable:** Pipeline mode (Direct Multilingual vs. Translate-Test).
- **Dependent Variables:** Rationale F1, Terminological Drift Error Rate.
- **Decision Rule / Success Threshold:** Translate-Test yields higher Rationale F1 by at least $+5\%$ points on Hindi, with $\ge 20\%$ of its remaining errors classified as translation entity distortion.
- **Contradicting Observation:** Direct Multilingual Prompting matches or outperforms Translate-Test in both rationale selection and terminological fidelity.

### Hypothesis 3 (H3: Multilingual Abstention Failure)
- **Hypothesis:** In non-English claim settings, models exhibit higher Expected Calibration Error (ECE) and an increased tendency to hallucinate SUPPORT/REFUTE verdicts on unanswerable (`NOT_ENOUGH_INFO`) instances rather than abstaining.
- **Independent Variable:** Claim language on `NOT_ENOUGH_INFO` subset.
- **Dependent Variables:** Abstention Accuracy on NEI claims, ECE.
- **Decision Rule / Success Threshold:** Abstention accuracy on NEI claims drops by $\ge 10\%$ points in Hindi/Spanish compared to English, accompanied by elevated confidence scores ($>0.70$).
- **Contradicting Observation:** Models maintain equal or higher abstention rates on non-English NEI claims without overconfident hallucination.

---

## 3. Demarcation of Demonstrated vs. Novel Contribution

| Demonstrated Prior Work (Do Not Claim Novelty) | Novel Empirical Contribution of This Study |
| :--- | :--- |
| English scientific claim verification on SciFact using fine-tuned BERT/RoBERTa (Wadden et al., 2020) `[VERIFIED]` | First controlled benchmark evaluating cross-lingual scientific evidence grounding (sentence rationale F1) against English PubMed abstracts |
| Closed-book multilingual claim veracity classification without evidence retrieval or rationales (Poly-FEVER, 2024) `[VERIFIED]` | First open-book evaluation testing whether cross-lingual input induces rationale hallucination (citing irrelevant sentences as evidence) |
| Multilingual atomic factuality on Wikipedia biographies (Multi-Fact / FActScore, 2023–2024) `[VERIFIED]` | Direct empirical trade-off analysis between Direct Multilingual Prompting and Translate-Test pipelines on scientific biomedical literature |
| General-domain translation evaluation on standard benchmarks `[VERIFIED]` | Systematic evaluation of language-dependent calibration and selective abstention under insufficient scientific evidence (`NOT_ENOUGH_INFO`) |
