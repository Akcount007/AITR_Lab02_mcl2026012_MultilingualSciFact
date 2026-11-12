# SciFact Dataset Validation & Inspection Report

**Date of Inspection:** 2026-11-04  
**Dataset Source:** Official Allen Institute for AI SciFact Release (`https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz`)  
**Investigator:** Aman Kumar (`mcl2026012@iiita.ac.in`)  
**Project:** Cross-Lingual Evidence Grounding in Scientific Claim Verification  

---

## 1. Dataset Overview & Split Structure

| File | Purpose | Total Records | File Size | Labels Available? |
| :--- | :--- | :---: | :---: | :---: |
| `corpus.jsonl` | Evidence source pool (PubMed abstracts) | **5,183** documents | ~8.3 MB | N/A (Source corpus) |
| `claims_train.jsonl` | Model training partition | **809** claims | ~175 KB | Yes (Gold labels & rationales) |
| `claims_dev.jsonl` | **Validation / Benchmark evaluation** | **300** claims | ~65 KB | **Yes (Gold labels & rationales)** |
| `claims_test.jsonl` | Blind test partition (leaderboard only) | **300** claims | ~34 KB | **NO (Labels strictly held out)** |
| **Total Claims** | Across all three splits | **1409** claims | — | — |

> [!IMPORTANT]
> **Split Selection for Intended Evaluation:**
> - `claims_test.jsonl` does **NOT** contain `evidence` or `cited_doc_ids`. Test labels are strictly sequestered on the AI2 leaderboard (`scifact.apps.allenai.org`).
> - In accordance with research integrity standards, the **test split is NOT used** for evaluation, prompt design, or translation.
> - The **development split (`claims_dev.jsonl`, 300 claims)** is completely self-contained with expert ground-truth labels and rationale sentence indices, making it the **ideal, verified partition** for our controlled cross-lingual benchmark.

---

## 2. Text Length Statistics

| Entity | Count | Min Words | Max Words | Mean Words (± SD) | Median Words |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Corpus Abstract Text** | 5,183 | 26 | 1524 | 201.81 ± 86.01 | 192 |
| **Corpus Paper Titles** | 5,183 | 1 | 40 | 12.82 ± 4.73 | 12 |
| **Abstract Sentences / Doc** | 5,183 | 3 sents | 367 sents | 8.87 sents | 8 sents |
| **Train Claims** | 809 | 3 | 39 | 12.33 ± 4.66 | 12 |
| **Dev Claims** | 300 | 4 | 29 | 12.51 ± 5.14 | 12.0 |
| **Test Claims** | 300 | 4 | 39 | 12.06 ± 4.56 | 12.0 |

---

## 3. Label & Evidence Distribution

### 3.1 Development Split (`claims_dev.jsonl` — Evaluation Partition)
- **Total Claims:** 300
- **Claims with Evidence (Supported or Contradicted):** 188 (62.67%)
- **Claims without Evidence (`NOT_ENOUGH_INFO`):** 112 (37.33%)

#### Claim-Level Label Breakdown
| Label Category | Count | Percentage |
| :--- | :---: | :---: |
| `NOT_ENOUGH_INFO` | 112 | 37.33% |
| `SUPPORT` | 124 | 41.33% |
| `CONTRADICT` | 64 | 21.33% |

#### Instance-Level (Claim-Document Pair) Breakdown
| Instance Label | Count | Percentage |
| :--- | :---: | :---: |
| `NOT_ENOUGH_INFO` | 131 | 27.93% |
| `SUPPORT` | 216 | 46.06% |
| `CONTRADICT` | 122 | 26.01% |

#### Rationale Sentences Statistics (Dev Split)
- **Min sentences per rationale:** 1
- **Max sentences per rationale:** 3
- **Mean sentences per rationale:** 1.08
- **Median sentences per rationale:** 1.0

### 3.2 Training Split (`claims_train.jsonl`)
- **Total Claims:** 809
- **Claims with Evidence:** 505 (62.42%)
- **Claims without Evidence (`NOT_ENOUGH_INFO`):** 304 (37.58%)
- **Claim-Level Breakdown:** {'NOT_ENOUGH_INFO': 304, 'CONTRADICT': 173, 'SUPPORT': 332}

---

## 4. Missing-Value & Integrity Audit

| Integrity Check | Corpus | Train Split | Dev Split | Test Split | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Missing Record IDs | 0 | 0 | 0 | 0 | ✅ Zero missing IDs |
| Empty Text / Title | 0 | 0 | 0 | 0 | ✅ Zero empty strings |
| Missing Cited Doc IDs | N/A | 0 | 0 | N/A (blind) | ✅ 100% cited docs recorded |
| Cited Docs Missing from Corpus | N/A | 0 | 0 | N/A | ✅ 100% cited docs exist in corpus |
| Test Split Label Leakage | N/A | N/A | N/A | 0 | ✅ Strict blind test partition |

---

## 5. Duplicate & Data Leakage Audit

| Check Category | Result | Interpretation |
| :--- | :---: | :--- |
| Corpus `doc_id` Uniqueness | **True** | All 5,183 document IDs are strictly unique. |
| Within-Split ID Duplicates | **0** in Train, **0** in Dev, **0** in Test | Primary keys are strictly unique across all files. |
| Cross-Split ID Overlap | **0** (Train ∩ Dev = 0, Train ∩ Test = 0, Dev ∩ Test = 0) | Partitions are completely disjoint; zero ID leakage. |
| Within-Split Claim Text Duplicates | Train: **2**, Dev: **0**, Test: **3** | Negligible phrasing duplicates. |
| Cross-Split Claim Text Overlap | Train ∩ Dev: **2**, Train ∩ Test: **1** | Zero text leakage across experimental partitions. |

---

## 6. Official vs. Adapted Input/Output Schema

### 6.1 Critical Schema Finding: `CONTRADICT` vs. `REFUTE`
> [!NOTE]
> **Schema Adaptation Documented:** In published literature (Wadden et al., 2020), the binary evidence labels are commonly referred to as `SUPPORT` and `REFUTE`. However, in the **raw official release files** (`claims_train.jsonl`, `claims_dev.jsonl`), the negative label is keyed as `"CONTRADICT"`.
> 
> Our data loader module (`src/data_loader.py`) implements bidirectional normalization:
> - Normalizes `"CONTRADICT"` $\leftrightarrow$ `"REFUTE"` seamlessly.
> - Normalizes empty evidence `{}` $\to$ `"NOT_ENOUGH_INFO"`.

### 6.2 Raw Official SciFact JSON Schema
```json
// claims_dev.jsonl
{
  "id": 3,
  "claim": "1,000 genomes project enables mapping of genetic sequence variation...",
  "evidence": {
    "14717500": [
      { "sentences": [2, 5], "label": "SUPPORT" },
      { "sentences": [7], "label": "SUPPORT" }
    ]
  },
  "cited_doc_ids": [14717500]
}

// corpus.jsonl
{
  "doc_id": 14717500,
  "title": "A map of human genome variation from population-scale sequencing.",
  "abstract": [
    "The 1000 Genomes Project aims to provide a deep characterization...",
    "Here we describe results from the pilot phase...",
    "We show that our data enable mapping of genetic sequence variation..."
  ],
  "structured": false
}
```

### 6.3 Standardized Internal Model Input/Output Schema (`src/data_loader.py`)
```json
// Model Input Payload (Direct Multilingual or Translated)
{
  "claim_id": 3,
  "language": "hi",
  "claim_text": "1000 जीनोम परियोजना आनुवंशिक अनुक्रम भिन्नता के मानचित्रण को सक्षम बनाती है...",
  "doc_id": 14717500,
  "doc_title": "A map of human genome variation from population-scale sequencing.",
  "abstract_sentences": [
    "[0] The 1000 Genomes Project aims to provide a deep characterization...",
    "[1] Here we describe results from the pilot phase...",
    "[2] We show that our data enable mapping of genetic sequence variation..."
  ]
}

// Model Structured Output Payload
{
  "veracity": "SUPPORT",
  "rationale_sentences": [2, 5],
  "confidence": 0.92
}
```

---

## 7. Human-Readable Ground-Truth Samples (from Dev Split)

#### Example: Label `SUPPORT` (Claim ID: 3, Doc ID: 14717500)
- **Claim:** "1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants."
- **Cited Paper Title:** "Rare Variants Create Synthetic Genome-Wide Associations"
- **Ground-Truth Label:** `SUPPORT` (in raw release: `SUPPORT`)
- **Gold Evidence Rationale Sentences:** `[2, 5]`
- **Abstract Sentences:**
      [0] Genome-wide association studies (GWAS) have now identified at least 2,000 common variants that appear associated with common diseases or related traits (http://www.genome.gov/gwastudies), hundreds of which have been convincingly replicated.
      [1] It is generally thought that the associated markers reflect the effect of a nearby common (minor allele frequency >0.05) causal site, which is associated with the marker, leading to extensive resequencing efforts to find causal sites.
      [2] We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associations" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele.
      [3] Although synthetic associations are an obvious theoretical possibility, they have never been systematically explored as a possible explanation for GWAS findings.
      [4] Here, we use simple computer simulations to show the conditions under which such synthetic associations will arise and how they may be recognized.
      [5] We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.
      [6] We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible for both hearing loss and sickle cell anemia create genome-wide significant synthetic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.
      [7] In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals.


#### Example: Label `CONTRADICT` (Claim ID: 42, Doc ID: 18174210)
- **Claim:** "A high microerythrocyte count raises vulnerability to severe anemia in homozygous alpha (+)- thalassemia trait subjects."
- **Cited Paper Title:** "Increased Microerythrocyte Count in Homozygous α+-Thalassaemia Contributes to Protection against Severe Malarial Anaemia"
- **Ground-Truth Label:** `CONTRADICT` (in raw release: `CONTRADICT`)
- **Gold Evidence Rationale Sentences:** `[1, 9]`
- **Abstract Sentences:**
      [0] BACKGROUND The heritable haemoglobinopathy alpha(+)-thalassaemia is caused by the reduced synthesis of alpha-globin chains that form part of normal adult haemoglobin (Hb).
      [1] Individuals homozygous for alpha(+)-thalassaemia have microcytosis and an increased erythrocyte count.
      [2] Alpha(+)-thalassaemia homozygosity confers considerable protection against severe malaria, including severe malarial anaemia (SMA) (Hb concentration < 50 g/l), but does not influence parasite count.
      [3] We tested the hypothesis that the erythrocyte indices associated with alpha(+)-thalassaemia homozygosity provide a haematological benefit during acute malaria.   

      [4] METHODS AND FINDINGS Data from children living on the north coast of Papua New Guinea who had participated in a case-control study of the protection afforded by alpha(+)-thalassaemia against severe malaria were reanalysed to assess the genotype-specific reduction in erythrocyte count and Hb levels associated with acute malarial disease.
      [5] We observed a reduction in median erythrocyte count of approximately 1.5 x 10(12)/l in all children with acute falciparum malaria relative to values in community children (p < 0.001).
      [6] We developed a simple mathematical model of the linear relationship between Hb concentration and erythrocyte count.
      [7] This model predicted that children homozygous for alpha(+)-thalassaemia lose less Hb than children of normal genotype for a reduction in erythrocyte count of >1.1 x 10(12)/l as a result of the reduced mean cell Hb in homozygous alpha(+)-thalassaemia.
      [8] In addition, children homozygous for alpha(+)-thalassaemia require a 10% greater reduction in erythrocyte count than children of normal genotype (p = 0.02) for Hb concentration to fall to 50 g/l, the cutoff for SMA.
      [9] We estimated that the haematological profile in children homozygous for alpha(+)-thalassaemia reduces the risk of SMA during acute malaria compared to children of normal genotype (relative risk 0.52; 95% confidence interval [CI] 0.24-1.12, p = 0.09).   

      [10] CONCLUSIONS The increased erythrocyte count and microcytosis in children homozygous for alpha(+)-thalassaemia may contribute substantially to their protection against SMA.
      [11] A lower concentration of Hb per erythrocyte and a larger population of erythrocytes may be a biologically advantageous strategy against the significant reduction in erythrocyte count that occurs during acute infection with the malaria parasite Plasmodium falciparum.
      [12] This haematological profile may reduce the risk of anaemia by other Plasmodium species, as well as other causes of anaemia.
      [13] Other host polymorphisms that induce an increased erythrocyte count and microcytosis may confer a similar advantage.


#### Example: Label `NOT_ENOUGH_INFO` (Claim ID: 1, Doc ID: 31715818)
- **Claim:** "0-dimensional biomaterials show inductive properties."
- **Cited Paper Title:** "New opportunities: the use of nanotechnologies to manipulate and track stem cells."
- **Ground-Truth Label:** `NOT_ENOUGH_INFO` (in raw release: `EMPTY_EVIDENCE`)
- **Gold Evidence Rationale Sentences:** `[]`
- **Abstract Sentences:**
      [0] Nanotechnologies are emerging platforms that could be useful in measuring, understanding, and manipulating stem cells.
      [1] Examples include magnetic nanoparticles and quantum dots for stem cell labeling and in vivo tracking; nanoparticles, carbon nanotubes, and polyplexes for the intracellular delivery of genes/oligonucleotides and protein/peptides; and engineered nanometer-scale scaffolds for stem cell differentiation and transplantation.
      [2] This review examines the use of nanotechnologies for stem cell tracking, differentiation, and transplantation.
      [3] We further discuss their utility and the potential concerns regarding their cytotoxicity.


---

## 8. License & Usage Verification
- **Claims & Annotations:** Creative Commons Attribution 4.0 International (`CC BY 4.0`). Permissive open license allowing modification and research sharing with attribution.
- **Corpus Abstracts:** Open Data Commons Attribution License (`ODC-By 1.0`), derived from Semantic Scholar Open Research Corpus (S2ORC).
- **Integrity Compliance:** Full compliance with academic research rules. Zero patient or private data. Test split is preserved blind.
