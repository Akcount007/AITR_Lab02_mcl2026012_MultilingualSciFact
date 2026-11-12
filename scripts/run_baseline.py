#!/usr/bin/env python3
"""
run_baseline.py — Baseline scientific claim verification on SciFact dev set.

Runs MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (multilingual NLI) on
SciFact claim-evidence pairs and exports predictions to CSV.

This script is the BASELINE reference point and must remain separate
from experimental extensions (multilingual conditions, prompt variants).

Usage:
    python scripts/run_baseline.py                         # Full dev set
    python scripts/run_baseline.py --max-claims 20         # Smoke test
    python scripts/run_baseline.py --output results/my.csv # Custom output
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---- Project imports ----
# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_corpus, load_scifact_split, normalize_label
from src.baseline_model import BaselineNLIModel, VALID_SCIFACT_LABELS

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_baseline")


# ---------------------------------------------------------------------------
# Evidence construction helpers
# ---------------------------------------------------------------------------

def build_evidence_text(
    claim_record: dict,
    corpus: dict,
    use_gold_rationale: bool = True,
) -> list:
    """
    Build (doc_id, gold_label, evidence_text, rationale_indices) tuples
    for a single claim record.

    If use_gold_rationale is True and gold rationale sentences are available,
    only those sentences are concatenated as evidence. Otherwise the full
    abstract is concatenated.

    Returns a list of tuples, one per cited document.
    """
    entries = []
    parsed = claim_record.get("parsed_evidence", [])
    cited_docs = claim_record.get("cited_doc_ids", [])

    if parsed:
        # Group by doc_id to avoid duplicates
        seen_docs = set()
        for ev in parsed:
            doc_id = ev["doc_id"]
            if doc_id is None or doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)

            gold_label = normalize_label(ev["label"])
            rationale_idx = ev.get("rationale_sentences", [])

            if doc_id not in corpus:
                logger.warning("Claim %s: doc_id %s not in corpus — skipping.",
                               claim_record["id"], doc_id)
                continue

            abstract_sents = corpus[doc_id].get("abstract", [])

            if use_gold_rationale and rationale_idx:
                # Use only gold rationale sentences
                evidence_sents = []
                for si in sorted(rationale_idx):
                    if 0 <= si < len(abstract_sents):
                        evidence_sents.append(abstract_sents[si])
                    else:
                        logger.warning(
                            "Claim %s, doc %s: rationale index %d out of range "
                            "(abstract has %d sentences).",
                            claim_record["id"], doc_id, si, len(abstract_sents),
                        )
                evidence_text = " ".join(evidence_sents)
            else:
                # Use full abstract
                evidence_text = " ".join(abstract_sents)

            entries.append((doc_id, gold_label, evidence_text, rationale_idx))
    elif cited_docs:
        # No parsed evidence but cited docs exist -> NOT_ENOUGH_INFO
        for doc_id in cited_docs:
            if doc_id not in corpus:
                continue
            abstract_sents = corpus[doc_id].get("abstract", [])
            evidence_text = " ".join(abstract_sents)
            entries.append((doc_id, "NOT_ENOUGH_INFO", evidence_text, []))

    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_baseline(
    claims_path: str,
    corpus_path: str,
    output_path: str,
    max_claims: int = 0,
    use_gold_rationale: bool = True,
):
    """Run baseline NLI inference and export predictions."""

    start_time = time.time()

    # ---- Load data ----
    logger.info("Loading corpus from %s", corpus_path)
    corpus = load_corpus(corpus_path)
    logger.info("Corpus loaded: %d documents", len(corpus))

    logger.info("Loading claims from %s", claims_path)
    claims = load_scifact_split(claims_path, corpus)
    logger.info("Claims loaded: %d", len(claims))

    if max_claims > 0:
        claims = claims[:max_claims]
        logger.info("Limiting to first %d claims (smoke test mode)", max_claims)

    # ---- Load model ----
    model = BaselineNLIModel()
    label_map = model.load()
    model_info = model.get_model_info()

    logger.info("=" * 60)
    logger.info("MODEL CONFIGURATION REPORT")
    logger.info("=" * 60)
    logger.info("Model: %s", model_info["model_name"])
    logger.info("Type: %s", model_info["model_type"])
    logger.info("Parameters: %s", f"{model_info['total_parameters']:,}")
    logger.info("Device: %s", model_info["device"])
    logger.info("id2label: %s", model_info["id2label"])
    logger.info("NLI -> SciFact mapping: %s", model_info["nli_to_scifact_mapping"])
    logger.info("Max sequence length: %d", model_info["max_seq_length"])
    logger.info("=" * 60)

    # ---- Prepare output ----
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    csv_fields = [
        "claim_id",
        "doc_id",
        "claim_text",
        "evidence_text_preview",
        "gold_label",
        "predicted_label",
        "raw_nli_label",
        "confidence",
        "score_SUPPORTS",
        "score_REFUTES",
        "score_NOT_ENOUGH_INFO",
        "correct",
        "truncated",
        "claim_len",
        "evidence_len",
        "rationale_indices",
    ]

    predictions = []
    total = 0
    correct = 0
    errors = 0
    skipped_no_evidence = 0
    label_counts = {"SUPPORTS": 0, "REFUTES": 0, "NOT_ENOUGH_INFO": 0, "UNMAPPED": 0}

    # ---- Run inference ----
    logger.info("Starting inference on %d claims...", len(claims))

    for ci, claim_rec in enumerate(claims):
        claim_text = claim_rec["claim"]
        claim_id = claim_rec["id"]

        evidence_entries = build_evidence_text(claim_rec, corpus, use_gold_rationale)

        if not evidence_entries:
            # No evidence at all — predict NOT_ENOUGH_INFO by default
            skipped_no_evidence += 1
            result = model._empty_prediction(claim_text, "", reason="no_evidence_docs")
            row = {
                "claim_id": claim_id,
                "doc_id": "N/A",
                "claim_text": claim_text[:200],
                "evidence_text_preview": "",
                "gold_label": "NOT_ENOUGH_INFO",
                "predicted_label": result["predicted_label"],
                "raw_nli_label": result["raw_nli_label"],
                "confidence": result["confidence"],
                "score_SUPPORTS": 0.0,
                "score_REFUTES": 0.0,
                "score_NOT_ENOUGH_INFO": 0.0,
                "correct": result["predicted_label"] == "NOT_ENOUGH_INFO",
                "truncated": False,
                "claim_len": len(claim_text),
                "evidence_len": 0,
                "rationale_indices": "[]",
            }
            predictions.append(row)
            total += 1
            if row["correct"]:
                correct += 1
            label_counts[result["predicted_label"]] = label_counts.get(result["predicted_label"], 0) + 1
            continue

        for doc_id, gold_label, evidence_text, rationale_idx in evidence_entries:
            total += 1
            try:
                result = model.predict(claim=claim_text, evidence=evidence_text)
            except Exception as e:
                logger.error(
                    "Prediction failed for claim %d, doc %s: %s",
                    claim_id, doc_id, e,
                )
                errors += 1
                result = {
                    "predicted_label": "NOT_ENOUGH_INFO",
                    "raw_nli_label": "ERROR",
                    "confidence": 0.0,
                    "all_scores": {lbl: 0.0 for lbl in VALID_SCIFACT_LABELS},
                    "truncated": False,
                    "claim_len": len(claim_text),
                    "evidence_len": len(evidence_text),
                    "error": str(e),
                }

            # Normalize gold label for comparison
            # Gold uses CONTRADICT in raw SciFact; our predictions use REFUTES
            gold_normalized = gold_label
            if gold_normalized == "CONTRADICT":
                gold_normalized = "REFUTES"
            elif gold_normalized == "SUPPORT":
                gold_normalized = "SUPPORTS"

            is_correct = result["predicted_label"] == gold_normalized
            if is_correct:
                correct += 1

            label_counts[result["predicted_label"]] = label_counts.get(
                result["predicted_label"], 0
            ) + 1

            row = {
                "claim_id": claim_id,
                "doc_id": doc_id,
                "claim_text": claim_text[:200],
                "evidence_text_preview": evidence_text[:200],
                "gold_label": gold_normalized,
                "predicted_label": result["predicted_label"],
                "raw_nli_label": result["raw_nli_label"],
                "confidence": result["confidence"],
                "score_SUPPORTS": result["all_scores"].get("SUPPORTS", 0.0),
                "score_REFUTES": result["all_scores"].get("REFUTES", 0.0),
                "score_NOT_ENOUGH_INFO": result["all_scores"].get("NOT_ENOUGH_INFO", 0.0),
                "correct": is_correct,
                "truncated": result.get("truncated", False),
                "claim_len": result.get("claim_len", 0),
                "evidence_len": result.get("evidence_len", 0),
                "rationale_indices": str(rationale_idx),
            }
            predictions.append(row)

        if (ci + 1) % 50 == 0:
            logger.info("Processed %d / %d claims (%d pairs so far)...",
                        ci + 1, len(claims), total)

    elapsed = time.time() - start_time

    # ---- Write CSV ----
    logger.info("Writing %d predictions to %s", len(predictions), output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(predictions)

    # ---- Summary ----
    accuracy = correct / total if total > 0 else 0.0

    summary = {
        "model": model_info["model_name"],
        "model_type": model_info["model_type"],
        "parameters": model_info["total_parameters"],
        "device": model_info["device"],
        "id2label": model_info["id2label"],
        "nli_to_scifact_mapping": model_info["nli_to_scifact_mapping"],
        "claims_file": claims_path,
        "total_claims_input": len(claims),
        "total_prediction_pairs": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "errors": errors,
        "skipped_no_evidence": skipped_no_evidence,
        "predicted_label_distribution": label_counts,
        "use_gold_rationale": use_gold_rationale,
        "elapsed_seconds": round(elapsed, 2),
        "output_file": output_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    logger.info("=" * 60)
    logger.info("BASELINE RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("Total prediction pairs: %d", total)
    logger.info("Correct: %d", correct)
    logger.info("Accuracy: %.4f", accuracy)
    logger.info("Errors: %d", errors)
    logger.info("Skipped (no evidence): %d", skipped_no_evidence)
    logger.info("Predicted label distribution: %s", label_counts)
    logger.info("Elapsed: %.2f seconds", elapsed)
    logger.info("Output: %s", output_path)
    logger.info("=" * 60)

    # Save summary JSON alongside CSV
    summary_path = output_path.replace(".csv", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run baseline NLI claim verification on SciFact.",
    )
    parser.add_argument(
        "--claims", type=str,
        default=str(PROJECT_ROOT / "data" / "claims_dev.jsonl"),
        help="Path to claims JSONL file (default: data/claims_dev.jsonl).",
    )
    parser.add_argument(
        "--corpus", type=str,
        default=str(PROJECT_ROOT / "data" / "corpus.jsonl"),
        help="Path to corpus JSONL file (default: data/corpus.jsonl).",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "results" / "baseline_predictions.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--max-claims", type=int, default=0,
        help="Limit to first N claims (0 = all). Use for smoke testing.",
    )
    parser.add_argument(
        "--no-gold-rationale", action="store_true",
        help="Use full abstracts instead of gold rationale sentences.",
    )
    args = parser.parse_args()

    run_baseline(
        claims_path=args.claims,
        corpus_path=args.corpus,
        output_path=args.output,
        max_claims=args.max_claims,
        use_gold_rationale=not args.no_gold_rationale,
    )


if __name__ == "__main__":
    main()
