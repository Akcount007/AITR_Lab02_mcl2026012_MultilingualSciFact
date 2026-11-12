# Data Access, Schema, and Partition Protocol

## 1. Benchmark Origin and Licensing
This repository evaluates the **SciFact** scientific claim verification benchmark (Wadden et al., EMNLP 2020).
- **Official Source:** [allenai/scifact](https://github.com/allenai/scifact)
- **Primary Publication:** Wadden, D., Lin, S., Lo, K., Wang, L. L., Guan, M., van Zuylen, M., Cohan, A., & Hajishirzi, H. (2020). *Fact or Fiction: Verifying Scientific Claims*. EMNLP 2020.
- **License:** Apache License 2.0 (open for academic research and evaluation).
- **Domain:** Peer-reviewed biomedical literature indexed in PubMed / Semantic Scholar Open Research Corpus (S2ORC).

---

## 2. Partition Structure and Sequestration Protocol

To preserve strict scientific integrity and prevent data leakage:

| File | Purpose | Total Claims | Total Pairs | Status in Study |
| :--- | :--- | :--- | :--- | :--- |
| `claims_dev.jsonl` | Official Development Split | 300 claims | 469 pairs | Active evaluation anchor |
| `corpus.jsonl` | PubMed Abstract Corpus | 5,183 abstracts | N/A | Evidence source (Titles & Sentences) |
| `claims_train.jsonl` | Official Training Split | 809 claims | 919 pairs | Inspected for contamination auditing only |
| `claims_test.jsonl` | Official Test Split | 300 claims | Unlabelled | **Strictly Sequestered / Blind** |

> **Scientific Integrity Rule:** The test split (`claims_test.jsonl`) has unlabelled gold veracity and was strictly unexamined and sequestered throughout all stages of model development, prompt formulation, and evaluation to prevent test contamination.

---

## 3. Schema Definitions

### 3.1 Claims Schema (`claims_dev.jsonl`)
```json
{
  "id": 1,
  "claim": "0-dimensional biomaterials show inductive properties.",
  "evidence": {
    "31715818": [
      {
        "sentences": [],
        "label": "NOT_ENOUGH_INFO"
      }
    ]
  },
  "cited_doc_ids": [31715818]
}
```

### 3.2 Evidence Corpus Schema (`corpus.jsonl`)
```json
{
  "doc_id": 31715818,
  "title": "Recent advances in stem cell tracking and differentiation using nanotechnology",
  "abstract": [
    "Stem cell-based regenerative medicine holds great promise...",
    "Nanotechnology has emerged as a powerful tool..."
  ],
  "structured": false
}
```

### 3.3 Multilingual Pilot Dataset (`claims_dev_multilingual_pilot.jsonl`)
Produced via Helsinki-NLP neural machine translation (`opus-mt-en-es` and `opus-mt-en-hi`) under beam search (`num_beams=4`):
```json
{
  "id": 5,
  "claim_en": "1/2000 in UK have abnormal PrP positivity.",
  "claim_es": "1/2000 en el Reino Unido tienen positividad anormal de PrP.",
  "claim_hi": "ब्रिटेन में 1/2000 में असामान्य त्रुटि सकारात्मकता है।",
  "evidence": {
    "13734012": [
      {
        "sentences": [4],
        "label": "SUPPORTS"
      }
    ]
  },
  "translation_provenance": {
    "es": {"model": "Helsinki-NLP/opus-mt-en-es", "beam_size": 4},
    "hi": {"model": "Helsinki-NLP/opus-mt-en-hi", "beam_size": 4}
  }
}
```

---

## 4. Replication and Download Instructions
Raw data files can be synchronized or verified using:
```bash
# Verify data directory integrity
python scripts/inspect_dataset.py --action summary
```
All raw data files in this repository conform strictly to the official open SciFact release.
