# Results Analysis and Scientific Discussion

**Document:** `results_analysis.md`  
**Date:** 2026-11-05  
**Author:** Aman Kumar (`mcl2026012@iiita.ac.in`)  
**Project:** Cross-Lingual Evidence Grounding in Scientific Claim Verification  
**Evaluation Scope:** Genuine empirical outputs from `Qwen/Qwen2.5-0.5B-Instruct` across matched English (`en`), Spanish (`es`), and Hindi (`hi`) conditions under Direct Classification (Mode A) and Evidence-Grounded Classification (Mode B).  
**Underlying Data Artifacts:**  
- Canonical Predictions: [`results/predictions.csv`](results/predictions.csv) ($N = 24$ genuine inference passes)  
- Metrics by Language: [`results/metrics_by_language.csv`](results/metrics_by_language.csv)  
- Metrics by Prompt: [`results/metrics_by_prompt.csv`](results/metrics_by_prompt.csv)  
- Paired Disagreement: [`results/paired_disagreement.csv`](results/paired_disagreement.csv)  
- Multilingual Claims: [`data/claims_dev_multilingual_pilot.jsonl`](data/claims_dev_multilingual_pilot.jsonl)  
- Pilot Summary: [`results/multilingual_pilot_predictions_summary.json`](results/multilingual_pilot_predictions_summary.json)  
- Figures: [`results/figures/`](results/figures/)  

---

## 1. Executive Summary & Epistemic Framework

This report analyzes the genuine empirical outputs of the multilingual scientific claim verification experiment. In strict compliance with scientific integrity principles:
1. **Zero fabricated metrics:** All numerical values, rates, latencies, and confusion matrices originate from genuine command executions on local hardware.
2. **Explicit epistemic demarcation:** Every research finding is partitioned into five distinct categories:
   - **[Observed] Directly observed results:** Facts mathematically calculated from genuine prediction logs.
   - **[Plausible] Plausible explanations:** High-probability hypotheses consistent with literature and empirical data.
   - **[Alternative] Alternative explanations:** Competing mechanisms that could explain the same observation.
   - **[Uncertain] Findings that remain uncertain:** Inconclusive findings requiring larger statistical power or ablation.
   - **[Prohibited] Unsupported claims:** Overgeneralizations that are unsupported by the evidence and must not appear in the paper.

---

## 2. Separate Analysis of Core Research Questions

### 2.1 Research Question 1 (RQ1: Cross-Lingual Veracity Degradation)
> *Does presenting scientific claims in non-English languages (Spanish and Hindi) degrade coarse veracity classification performance compared to English when evidence abstracts remain in English?*

#### 1. Directly Observed Results:
- In **Mode A (Direct Classification)**: Accuracy was identically **0.2500** across English, Spanish, and Hindi ($1/4$ correct per language). Macro-F1 was identically **0.1333** across all three languages.
- In **Mode B (Evidence-Grounded Classification)**: Accuracy was identically **0.5000** across English, Spanish, and Hindi ($2/4$ correct per language). Macro-F1 was identically **0.3333** across all three languages.
- On the pilot dataset, **no degradation** in coarse veracity classification accuracy or Macro-F1 was observed between English, Spanish, and Hindi within either prompting mode.

#### 2. Plausible Explanations:
- The small model (`Qwen2.5-0.5B-Instruct`) relies on coarse lexical overlap and heavy conservative hedging rather than nuanced syntactic parsing. Because the evidence abstract remained in English in all conditions, the model's decision boundary was dominated by abstract-level keywords rather than claim-language syntax.

#### 3. Alternative Explanations:
- The 3 pilot claims evaluated (Claims 1, 3, 5) contain prominent international scientific terminology (*biomaterials*, *genomes project*, *PrP*, *UK*, *1/2000*) that survived translation into Spanish and Hindi with sufficient phonetic or lexical signal for the model to treat them equivalently.
- The sample size of 4 instances per condition ($N=24$ total) has insufficient statistical power to reveal subtle cross-lingual performance deltas that might emerge at scale ($N=469$).

#### 4. Findings that Remain Uncertain:
- Whether the apparent cross-lingual performance parity persists across complex biochemical claims with lower cognate overlap or more intricate clausal dependencies.
- Whether a larger LLM (e.g., 7B or 8B) with richer multilingual representations would exhibit cross-lingual divergence due to deeper semantic processing.

#### 5. Unsupported Claims That Must NOT Appear in the Paper:
- ❌ *"Qwen-2.5-0.5B demonstrates complete cross-lingual invariance across scientific claim verification."* (Overgeneralization from $N=4$ pilot instances).
- ❌ *"Machine translation of scientific claims introduces zero performance penalty for Indic languages."* (Refuted by qualitative translation analysis showing severe terminological drift in Hindi).

---

### 2.2 Research Question 2 (RQ2: Direct vs. Evidence-Grounded Prompting)
> *Does evidence-grounded classification (Mode B: joint sentence rationale extraction + veracity) improve veracity reliability compared to direct classification (Mode A), and does sentence rationale extraction degrade more sharply than veracity?*

#### 1. Directly Observed Results:
- Veracity Accuracy doubled from **0.2500** in Mode A to **0.5000** in Mode B across all three languages ($12/24$ total correct in Mode B vs. $6/24$ in Mode A).
- Veracity Macro-F1 increased from **0.1333** in Mode A to **0.3333** in Mode B across all languages.
- In Mode A, the model exhibited complete **NEI-collapse** (100% of predictions across all 12 passes were `NOT_ENOUGH_INFO`).
- In Mode B, the model broke NEI-collapse on Claim 5, correctly predicting `SUPPORTS` across English, Spanish, and Hindi.
- Sentence-level Rationale F1 was **1.0000** on Claim 1 (where gold rationales were empty `[]`), but **0.0000** on Claim 3 and Claim 5 (where gold rationales were `[2, 5]`, `[7]`, and `[4]`), yielding a mean Rationale F1 of **0.2500** in Mode B.
- Rationale extraction degraded more severely than veracity: on Claim 5, veracity was 100% correct across all languages, while strict sentence rationale F1 was 0.00% across all languages (`ERR_RAT_OMIT`).

#### 2. Plausible Explanations:
- Requiring the model to inspect numbered sentences in Mode B forces attention over specific context segments, overcoming the default conservative prior to abstain (`NOT_ENOUGH_INFO`).
- A 494M parameter model lacks the instruction-following fidelity to output integer arrays formatted as `[4]` without few-shot exemplars, defaulting instead to natural-language descriptions (`"The study aims to investigate..."`).

#### 3. Alternative Explanations:
- The prompt length and ChatML structure for Mode B provide more contextual cues than Mode A, inadvertently biasing the generation length and triggering different output tokens.
- Truncation at `max_new_tokens=35` prevented the model from completing its JSON rationale array after emitting explanatory text.

#### 4. Findings that Remain Uncertain:
- Whether increasing `max_new_tokens` to 128 or adding 1-shot in-context demonstrations would enable the model to output valid integer sentence indices.
- Whether larger models (7B/8B) experience the same rationale omission failure or successfully output sentence indices.

#### 5. Unsupported Claims That Must NOT Appear in the Paper:
- ❌ *"Evidence grounding guarantees superior scientific reasoning in small language models."* (The model correctly predicted the label on Claim 5 but completely failed sentence pointer extraction).
- ❌ *"Small LLMs can reliably generate structured JSON rationale indices zero-shot."* (Zero-shot integer extraction failed on 100% of positive evidence pairs).

---

### 2.3 Research Question 3 (RQ3: Abstention, False Certainty, and Calibration)
> *Does cross-lingual claim presentation cause models to over-commit or fail to abstain on claims where evidence is actually insufficient (`NOT_ENOUGH_INFO`), and how does evidence grounding affect calibration?*

#### 1. Directly Observed Results:
- On Claim 1 (Gold `NOT_ENOUGH_INFO`), the model **abstained correctly** in 100% of passes (6/6 passes across English, Spanish, and Hindi in both Mode A and Mode B).
- The model produced **zero false certainty errors (`ERR_NEI_FAIL`)** on unanswerable claims: it never hallucinated `SUPPORTS` or `REFUTES` when evidence was genuinely absent.
- However, the model exhibited severe **over-abstention** on verifiable claims: on Claim 3 (Gold `SUPPORTS`), it predicted `NOT_ENOUGH_INFO` across all 6 passes in Mode A and all 6 passes in Mode B.
- Mode A showed severe overconfidence on incorrect predictions: the model emitted self-reported confidence of **1.00** on all incorrect `NOT_ENOUGH_INFO` predictions, resulting in a Brier Score of **0.7508** and an Expected Calibration Error (ECE) of **0.7417**.
- Mode B exhibited improved calibration: Brier Score dropped to **0.5000** and ECE dropped to **0.5000**.

#### 2. Plausible Explanations:
- Small models exhibit an asymmetric failure mode: rather than hallucinating false facts, they default to conservative hedging (`NOT_ENOUGH_INFO`) whenever semantic complexity exceeds their reasoning capacity.
- Verbalized self-reported confidences in small causal LLMs do not represent true Bayesian posterior probabilities; they reflect token generation heuristics where `"confidence": 1.0` is frequently memorized from training corpora.

#### 3. Alternative Explanations:
- The absence of explicit probability calibration (e.g., Platt scaling or temperature scaling on output logits) causes raw verbalized confidences to cluster unnaturally at extreme values ($0.0$ or $1.0$).
- The evidence abstract for Claim 3 (Doc 14717500, *Rare Variants Create Synthetic Genome-Wide Associations*) contains hedging language (*"theoretical possibility"*, *"stochastically"*), which small models interpret as insufficient evidence.

#### 4. Findings that Remain Uncertain:
- Whether calibration error is driven by verbalized prompting or by intrinsic model uncertainty.
- How logit-based calibration (e.g., softmax probabilities over label tokens) compares against self-reported verbalized confidence.

#### 5. Unsupported Claims That Must NOT Appear in the Paper:
- ❌ *"The model is well-calibrated when predicting scientific veracity."* (Refuted by empirical ECE of 0.6208 and Brier score of 0.6254).
- ❌ *"Cross-lingual input causes models to hallucinate false scientific evidence."* (The model never hallucinated evidence on NEI claims; its error was conservative over-abstention).

---

### 2.4 Research Question 4 (RQ4: Cross-Lingual Flip Rates & Consistency)
> *How consistent are the model's predictions across languages on the exact same underlying scientific claims (measured by Cross-Lingual Flip Rate CFR and Cohen's Kappa)?*

#### 1. Directly Observed Results:
- In Mode A: $\text{CFR}(EN, ES) = \mathbf{0.00\%}$, $\text{CFR}(EN, HI) = \mathbf{0.00\%}$, $\text{CFR}(ES, HI) = \mathbf{0.00\%}$ ($0/4$ flips per pair; agreement = 100%, Cohen's $\kappa = 1.0000$).
- In Mode B: $\text{CFR}(EN, ES) = \mathbf{0.00\%}$, $\text{CFR}(EN, HI) = \mathbf{0.00\%}$, $\text{CFR}(ES, HI) = \mathbf{0.00\%}$ ($0/4$ flips per pair; agreement = 100%, Cohen's $\kappa = 1.0000$).
- Cross-prompt disagreement holding language fixed: **25.00%** ($1/4$ disagreements, agreement = 75.0%, Cohen's $\kappa = 0.0000$) across English, Spanish, and Hindi.

#### 2. Plausible Explanations:
- The identical predictions across languages stem from the fact that the English evidence abstract was provided in full in all conditions. The model's internal attention over the English abstract dominated the generation logits, producing identical verdicts regardless of claim language.

#### 3. Alternative Explanations:
- The pilot subset of 3 claims happened to contain extreme cases (one unambiguous NEI claim, one difficult GWAS claim, and one numerical claim) where the model's decision threshold was far from the boundary, masking borderline cross-lingual instability.

#### 4. Findings that Remain Uncertain:
- Whether cross-lingual flips emerge on claims with subtle directional predicates (*inhibit* vs *activate*, *upregulate* vs *downregulate*).
- Whether a fully translated pipeline (where both claim and abstract are translated) would exhibit high flip rates.

#### 5. Unsupported Claims That Must NOT Appear in the Paper:
- ❌ *"Cross-lingual scientific fact-checking exhibits zero prediction volatility across languages."* (The pilot subset has $N=4$ pairs; claiming zero volatility universally is mathematically unjustified).

---

## 3. Qualitative Per-Instance Case Studies

To understand the exact mechanism of model behavior, every evaluation instance was examined at the raw text level:

```
                                Case Study Summary Matrix
┌─────────┬──────────────┬────────────┬──────────────────┬──────────────────┬─────────────────┐
│ Instance│ Doc ID       │ Gold Label │ Mode A (EN/ES/HI)│ Mode B (EN/ES/HI)│ Primary Error   │
├─────────┼──────────────┼────────────┼──────────────────┼──────────────────┼─────────────────┤
│ Claim 1 │ Doc 31715818 │ NEI        │ NEI / NEI / NEI  │ NEI / NEI / NEI  │ NONE (Correct)  │
│ Claim 3 │ Doc 14717500 │ SUPPORTS   │ NEI / NEI / NEI  │ NEI / NEI / NEI  │ ERR_OTHER (Hedg)│
│ Claim 5 │ Doc 13734012 │ SUPPORTS   │ NEI / NEI / NEI  │ SUP / SUP / SUP  │ ERR_RAT_OMIT    │
└─────────┴──────────────┴────────────┴──────────────────┴──────────────────┴─────────────────┘
```

---

### Case Study 1: Claim 1 (Doc 31715818) — Correct Abstention Across All Conditions

- **Claim (EN):** `"0-dimensional biomaterials show inductive properties."`
- **Claim (ES):** `"Los biomateriales de dimensión 0 muestran propiedades inductivas."`
- **Claim (HI):** `"0- विभिन्न जीव जीव- जीव-विज्ञानात्मक गुणों को दिखाता है."`
- **Abstract Evidence (Doc 31715818):**
  > Title: *New opportunities: the use of nanotechnologies to manipulate and track stem cells.*  
  > `[0]` Nanotechnologies are emerging platforms that could be useful in measuring, understanding, and manipulating stem cells.  
  > `[1]` Examples include magnetic nanoparticles and quantum dots for stem cell labeling and in vivo tracking...  
  > `[2]` This review examines the use of nanotechnologies for stem cell tracking, differentiation, and transplantation.  
  > `[3]` We further discuss their utility and the potential concerns regarding their cytotoxicity.
- **Gold Annotation:** Label = `NOT_ENOUGH_INFO`, Rationales = `[]`.
- **Model Output (Mode A, English):**
  ```json
  {"veracity": "NOT_ENOUGH_INFO", "confidence": 1}
  ```
- **Model Output (Mode B, Spanish):**
  ```text
  {"veracity": "NOTEnoughInfo", "rationales": ["The claim does not provide enough information to verify its accuracy.", "The given sentences do not directly address the"
  ```
- **Evaluation & Diagnosis:**
  - **Verdict:** **CORRECT** (`NOT_ENOUGH_INFO`).
  - **Mechanistic Finding:** The abstract discusses nanotechnologies and stem cells generally, but never mentions "0-dimensional biomaterials" or "inductive properties". The model successfully recognized the absence of evidence and abstained across all three languages in both modes.
  - **Translation Note:** The Hindi translation (`0- विभिन्न जीव जीव- जीव-विज्ञानात्मक गुणों को दिखाता है`) suffered morphological degradation ("0- various organism organism- biological properties"), yet the model still correctly abstained because the English abstract contained zero supporting overlap.

---

### Case Study 2: Claim 3 (Doc 14717500) — Reasoning Failure and Severe Translation Drift

- **Claim (EN):** `"1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants."`
- **Claim (ES):** `"El proyecto de 1.000 genomas permite mapear la variación de secuencia genética consistente en variantes raras con efectos de penetrancia más grandes que las variantes comunes."`
- **Claim (HI):** `"1,000 भूविज्ञान परियोजना आनुवंशिक अनुक्रम परिवर्तनों का निर्माण करने में समर्थ करती है जिसमें सामान्य अंतरों से बड़ा अंतर होता है।"` (*Transliteration:* `"1,000 bhu-vigyan pariyojana aanuvanshik anukram parivartanon ka nirmaan karne mein samarth karti hai..."`)*
- **Abstract Evidence (Doc 14717500):**
  > Title: *Rare Variants Create Synthetic Genome-Wide Associations*  
  > `[0]` Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases...  
  > `[2]` We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associations"...  
  > `[5]` We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals...  
  > `[7]` In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants...
- **Gold Annotation:** Label = `SUPPORTS`, Rationales = `[2, 5]` (Ev 1) and `[7]` (Ev 2).
- **Model Output (Mode A, English):**
  ```json
  {"veracity": "NOTEnoughInfo", "confidence": 1}
  ```
- **Model Output (Mode B, English):**
  ```text
  {"veracity": "NOTEnoughInfo", "rationales": ["The claim does not provide sufficient information about the nature of synthetic associations and their implications for GWAS results."
  ```
- **Evaluation & Diagnosis:**
  - **Verdict:** **INCORRECT** (`NOT_ENOUGH_INFO` vs. Gold `SUPPORTS`).
  - **Mechanistic Finding:** The model failed to perform multi-hop deductive entailment. Sentences `[2]`, `[5]`, and `[7]` establish that rare variants create synthetic associations that explain common variant signals, which supports the claim. However, the model hedged and stated in its explanation that the claim does not provide sufficient information about synthetic associations, demonstrating cognitive hesitation.
  - **Catastrophic Hindi Translation Error:** MarianMT mistranslated `"1,000 genomes project"` into `"1,000 bhu-vigyan pariyojana"` (Devanagari: `"1,000 भूविज्ञान परियोजना"`). In Hindi, **`bhu-vigyan` (भूविज्ञान) means Geology**, not Genetics (`jeenom`)! The claim literally read: *"1,000 Geology Project enables mapping of genetic sequence variation..."* This represents a severe domain-specific translation failure (`ERR_TRANS_DRIFT`).

---

### Case Study 3: Claim 5 (Doc 13734012) — Mode Divergence & Quantitative Equivalence

- **Claim (EN):** `"1/2000 in UK have abnormal PrP positivity."`
- **Claim (ES):** `"1/2000 en el Reino Unido tienen positividad PrP anormal."`
- **Claim (HI):** `"UK mein 1/2000 mein asaamanya truti hai"` (Devanagari: `"UK में 1/2000 में असामान्य त्रुटि है"`; *Literal gloss:* *"In UK 1/2000 have abnormal error/defect"*).
- **Abstract Evidence (Doc 13734012):**
  > Title: *Prevalent abnormal prion protein in human appendixes after bovine spongiform encephalopathy epizootic: large scale survey*  
  > `[4]` RESULTS Of the 32,441 appendix samples 16 were positive for abnormal PrP, indicating an overall prevalence of 493 per million population (95% confidence interval 282 to 801 per million).  
  > `[7]` CONCLUSIONS This study corroborates previous studies and suggests a high prevalence of infection with abnormal PrP...
- **Gold Annotation:** Label = `SUPPORTS`, Rationales = `[4]`.
- **Mathematical Grounding:**  
  $$\text{Prevalence} = \frac{16}{32,441} \approx 4.93 \times 10^{-4} = 493 \text{ per million} \approx \frac{1}{2,028} \approx \frac{1}{2,000}$$
- **Model Output (Mode A, English):**
  ```json
  {"veracity": "NOTEnoughInfo", "confidence": 1}
  ```
- **Model Output (Mode B, English):**
  ```text
  {"veracity": " SUPPORT ", "rationales": ["The study's objectives align closely with the claims about the prevalence of abnormal PrP positivity among archived appendix samples.", "
  ```
- **Model Output (Mode B, Spanish):**
  ```text
  {"veracity": "SUPPORT", "rationales": ["The study aims to investigate the prevalence of prion infections among individuals who died of BSE in England and Wales"
  ```
- **Evaluation & Diagnosis:**
  - **Mode Divergence:** Mode A failed completely (`NOT_ENOUGH_INFO`), while Mode B **succeeded in identifying `SUPPORTS`** across English, Spanish, and Hindi!
  - **Mechanistic Finding:** In Mode A, the model was overwhelmed by the raw unsegmented abstract and failed to connect $1/2000$ with $493 \text{ per million}$, collapsing to NEI. In Mode B, the numbered sentence prompt focused the model's attention on the results sentence `[4]`, enabling it to recognize the numerical entailment.
  - **Rationale Omission (`ERR_RAT_OMIT`):** Although veracity was 100% correct in Mode B, the model output natural-language explanations (`"The study aims to investigate..."`) instead of the required integer list `[4]`, resulting in a Rationale F1 score of **0.0000**.
  - **Hindi Translation Defect:** In Hindi, MarianMT translated `"abnormal PrP positivity"` into `"asaamanya truti"` (Devanagari: `"असामान्य त्रुटि"`, *abnormal error/defect*). `PrP` (prion protein) was completely erased from the Hindi claim, yet the model still predicted `SUPPORTS` because it anchored on the shared tokens `"UK"` and `"1/2000"`.

---

## 4. Systematic Error Categorization & Factor Association

Every error observed across the 24 pilot passes was categorized against potential failure mechanisms:

| Potential Association Factor | Empirical Association Level | Concrete Observed Evidence |
| :--- | :---: | :--- |
| **Negation Polarity Alteration** | **LOW** | Negation was preserved in 100% of Spanish claims. In Hindi, MarianMT added spurious negative particles (`nahin` / `नहीं`), but downstream model predictions were identical between Hindi and English (0.00% CFR). |
| **Scientific Terminology Drift** | **CRITICAL** | MarianMT mistranslated `"genomes"` to `"bhu-vigyan"` (भूविज्ञान = Geology) in Claim 3, and dropped `"PrP"` to `"truti"` (त्रुटि = error) in Claim 5. This is a severe threat to cross-lingual validity for lower-resource languages. |
| **Evidence Omission (`ERR_RAT_OMIT`)** | **SEVERE** | On 100% of positive evidence instances in Mode B, the model failed to output integer sentence indices, emitting free-form explanatory text instead. |
| **Unsupported Conclusions (False Certainty)** | **ABSENT** | The model never generated false support or refutation on unanswerable claims (`ERR_NEI_FAIL` = 0). |
| **Conservative Over-Abstention (`ERR_OTHER`)** | **SEVERE** | Mode A exhibited 100% NEI-collapse. The model conservatively predicted `NOT_ENOUGH_INFO` on claims that were fully supported by the abstract. |
| **Scientific Ambiguity & Transduction** | **MODERATE** | Converting fractions ($1/2000$) to frequency rates ($493 \text{ per million}$) required numerical reasoning that failed in Mode A but succeeded in Mode B. |
| **Invalid Syntax Outputs** | **MITIGATED** | Qwen-0.5B generated camelCase `"NOTEnoughInfo"`. Initially causing 16/24 invalid errors, robust string normalization reduced invalid outputs to 0/24 (0.0%). |

---

## 5. Comparative Audit: Full Translated Pilot (10 Claims) vs. Evaluated Subset (3 Claims)

To verify whether the evaluated claims are representative of the larger translated pool, we compared the automated quality metrics across all 10 translated pilot claims:

```
                            Pilot Translation Audit Summary (10 Claims)
┌───────────┬──────────────┬────────────────────────┬────────────────────────┬──────────────────────┐
│ Language  │ Total Claims │ Numerical Preserved    │ Negation Preserved     │ Mean MT Latency (s)  │
├───────────┼──────────────┼────────────────────────┼────────────────────────┼──────────────────────┤
│ Spanish   │ 10 claims    │ 9/10 (90.0%)*          │ 10/10 (100.0%)         │ 4.88 s               │
│ Hindi     │ 10 claims    │ 8/10 (80.0%)**         │ 6/10 (60.0%)***        │ 4.44 s               │
└───────────┴──────────────┴────────────────────────┴────────────────────────┴──────────────────────┘
* 1 apparent mismatch due to European thousand-dot typography ("1.000" vs "1,000").
** 2 apparent mismatches due to Devanagari digit conversion ("5%" -> "५%").
*** 4 apparent mismatches due to Hindi verbal negation particles ("नहीं").
```

### Key Comparative Findings:
1. **Typographical vs Semantic Shift:** Both Spanish and Hindi exhibit systematic typographical transformations (European thousand-dot separators in Spanish, Devanagari script numerals in Hindi). These do not alter semantic truth values, but require localized regex normalization.
2. **Biomedical Named Entity Recognition:** MarianMT preserves Romance-language biomedical cognates with high fidelity in Spanish (*positividad*, *mutación*, *biomateriales*), but frequently mistranslates domain-specific English terms into generic Hindi words (*genomes* $\to$ *geology*, *PrP* $\to$ *error*).
3. **Generalizability Limitation:** Because MarianMT exhibits noticeable terminological drift on Indic biomedical text, direct multilingual prompting in Hindi is partially confounded by translation quality. A human-verified or dictionary-guided translation pipeline is essential for large-scale Indic evaluation.

---

## 6. Honest Limitations & Negative Results

### 6.1 Critical Limitations
1. **Model Parameter Scale:** `Qwen2.5-0.5B-Instruct` is a compact 494M parameter model. Its capacity for multi-hop scientific deduction and strict structured output adherence is severely constrained compared to 7B or 8B parameter models.
2. **CPU Compute Latency:** Running local CPU inference at **21.35 seconds per pass** means that scaling to the full 300-claim development set ($2,400$ passes) would require **~14.2 hours** of uninterrupted CPU execution.
3. **Pilot Sample Size:** The pilot evaluated 3 claims across 4 evaluation pairs ($N=24$ total inferences). While sufficient to audit pipeline mechanics, schema validity, and runtime bounds, it lacks statistical power for inferential hypothesis testing (e.g., McNemar's test $p$-values are uninformative on $N=4$).
4. **Token Truncation in Mode B:** At `max_new_tokens=35`, Mode B generations were frequently cut off before closing the JSON string brace when the model generated lengthy explanations.

### 6.2 Explicit Negative Results
- **Negative Result 1:** Mode A (Direct Classification) completely failed as a scientific verification tool for small LLMs, collapsing to 100% `NOT_ENOUGH_INFO` predictions regardless of language.
- **Negative Result 2:** Zero-shot sentence rationale extraction completely failed on positive evidence claims; the model was incapable of outputting integer sentence arrays `[s_0, s_1]`, resulting in a Rationale F1 score of 0.00% on verifiable claims.
- **Negative Result 3:** Self-reported verbalized confidence from small causal LLMs is extremely poorly calibrated (ECE = 0.6208, Brier = 0.6254), exhibiting high overconfidence on incorrect predictions.

---

## 7. Prohibited Claims Checklist for Future Publication

To maintain scientific integrity and prevent reviewer rejection, the following claims **MUST NOT appear** in any paper or report based on this study:

- [ ] **DO NOT CLAIM:** *"Large language models achieve multilingual parity in scientific claim verification."* (Untrue; we evaluated a 0.5B model on a pilot subset where coarse accuracy was identical due to conservative NEI-collapse, not genuine multilingual mastery).
- [ ] **DO NOT CLAIM:** *"Indic languages suffer no performance degradation in scientific literature analysis."* (Untrue; qualitative inspection revealed severe translation drift in Hindi, including translating 'genomes' to 'geology').
- [ ] **DO NOT CLAIM:** *"Evidence grounding eliminates hallucinations in small LLMs."* (Untrue; while Mode B broke NEI-collapse on Claim 5, it failed to extract valid sentence indices on 100% of verifiable claims).
- [ ] **DO NOT CLAIM:** *"Statistical significance was established for cross-lingual equivalence."* (Untrue; sample size of $N=4$ pairs has zero statistical power to prove the null hypothesis of equivalence).
- [ ] **DO NOT CLAIM:** *"The model was calibrated."* (Untrue; empirical calibration error was 0.6208).
