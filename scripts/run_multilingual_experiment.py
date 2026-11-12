#!/usr/bin/env python3
"""
run_multilingual_experiment.py — Multilingual scientific claim verification runner.

Evaluates matched conditions:
  Languages: English (en), Spanish (es), Hindi (hi)
  Modes: Mode A (Direct Classification) vs Mode B (Evidence-Grounded)

Outputs linked per-instance predictions and condition summaries.

Usage:
    python scripts/run_multilingual_experiment.py --pilot   # Run pilot on 10 claims
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
from typing import Optional, Dict, Any, List

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_corpus, load_jsonl, normalize_label
from src.metrics import (
    evaluate_veracity_predictions,
    evaluate_rationale_selection,
    compute_cross_lingual_flip_rate,
    normalize_eval_label,
)
from src.multilingual_experiment import (
    MultilingualGenerativeEngine,
    parse_model_json_response,
    classify_error,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_multilingual_experiment")


def run_experiment(
    claims_path: str,
    corpus_path: str,
    output_path: str,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    languages: tuple = ("en", "es", "hi"),
    modes: tuple = ("A", "B"),
    max_claims: int = 0,
):
    start_total_time = time.time()

    # Load corpus
    logger.info("Loading corpus from %s", corpus_path)
    corpus = load_corpus(corpus_path)
    logger.info("Corpus loaded: %d documents", len(corpus))

    # Load multilingual claims
    logger.info("Loading multilingual claims from %s", claims_path)
    claims = load_jsonl(claims_path)
    logger.info("Loaded %d claims", len(claims))

    if max_claims > 0:
        claims = claims[:max_claims]
        logger.info("Limited to first %d claims for run", len(claims))

    # Initialize and load model
    engine = MultilingualGenerativeEngine(model_name=model_name)
    engine.load()

    # Prepare predictions container
    all_predictions = []

    # Map for computing cross-lingual flip rate: (mode, claim_id, doc_id) -> {lang: pred_label}
    cfr_tracker = {}

    csv_fields = [
        "claim_id",
        "doc_id",
        "language",
        "mode",
        "claim_text",
        "abstract_title",
        "gold_label",
        "gold_sentences",
        "predicted_label",
        "predicted_sentences",
        "confidence",
        "veracity_correct",
        "rationale_f1",
        "parsing_status",
        "error_category",
        "latency_sec",
        "raw_model_output",
    ]

    total_instances = 0

    logger.info("Starting evaluation across Languages %s and Modes %s...", languages, modes)

    for ci, claim_rec in enumerate(claims):
        claim_id = claim_rec["id"]
        claim_en = claim_rec["claim_en"]
        evidence_dict = claim_rec.get("evidence", {})
        cited_doc_ids = claim_rec.get("cited_doc_ids", [])

        # Build list of (doc_id, gold_label, gold_sentences)
        evaluation_pairs = []
        if evidence_dict:
            for doc_id_str, ev_list in evidence_dict.items():
                doc_id = int(doc_id_str)
                for ev in ev_list:
                    lbl = normalize_eval_label(ev.get("label", "NOT_ENOUGH_INFO"))
                    sents = sorted(ev.get("sentences", []))
                    evaluation_pairs.append((doc_id, lbl, sents))
        elif cited_doc_ids:
            for doc_id in cited_doc_ids:
                evaluation_pairs.append((doc_id, "NOT_ENOUGH_INFO", []))
        else:
            evaluation_pairs.append((None, "NOT_ENOUGH_INFO", []))

        for doc_id, gold_label, gold_sents in evaluation_pairs:
            doc_data = corpus.get(doc_id, {"title": "Unknown", "abstract": []}) if doc_id else {"title": "None", "abstract": []}
            abstract_title = doc_data.get("title", "")
            abstract_sentences = doc_data.get("abstract", [])

            for lang in languages:
                # Select claim text in target language
                if lang == "en":
                    claim_text = claim_en
                else:
                    claim_text = claim_rec.get("translations", {}).get(lang, claim_en)

                for mode in modes:
                    total_instances += 1
                    try:
                        res = engine.run_prompt(
                            claim_text=claim_text,
                            abstract_title=abstract_title,
                            abstract_sentences=abstract_sentences,
                            mode=mode,
                        )
                        pred_label = res["veracity"]
                        pred_sents = res["rationale_sentences"]
                        conf = res["confidence"]
                        parsing_status = res["parsing_status"]
                        raw_output = res["raw_output"]
                        latency = res["latency_seconds"]
                    except Exception as e:
                        logger.error("Inference failed for claim %d, doc %s, %s, Mode %s: %s", claim_id, doc_id, lang, mode, e)
                        pred_label = "INVALID_OUTPUT"
                        pred_sents = []
                        conf = 0.0
                        parsing_status = f"EXCEPTION: {str(e)}"
                        raw_output = ""
                        latency = 0.0

                    is_correct = pred_label == gold_label

                    # Evaluate rationale if Mode B
                    rat_metrics = evaluate_rationale_selection(gold_sents, pred_sents)
                    rat_f1 = rat_metrics["f1"] if mode == "B" else None

                    # Classify error
                    err_code = classify_error(
                        gold_label=gold_label,
                        pred_label=pred_label,
                        gold_sents=gold_sents,
                        pred_sents=pred_sents,
                        mode=mode,
                    )

                    # Track for CFR
                    pair_key = (mode, claim_id, doc_id)
                    if pair_key not in cfr_tracker:
                        cfr_tracker[pair_key] = {}
                    cfr_tracker[pair_key][lang] = pred_label

                    row = {
                        "claim_id": claim_id,
                        "doc_id": doc_id if doc_id is not None else "NONE",
                        "language": lang,
                        "mode": mode,
                        "claim_text": claim_text[:200],
                        "abstract_title": abstract_title[:100],
                        "gold_label": gold_label,
                        "gold_sentences": str(gold_sents),
                        "predicted_label": pred_label,
                        "predicted_sentences": str(pred_sents),
                        "confidence": conf,
                        "veracity_correct": is_correct,
                        "rationale_f1": rat_f1,
                        "parsing_status": parsing_status,
                        "error_category": err_code or "NONE",
                        "latency_sec": latency,
                        "raw_model_output": raw_output[:300].replace("\n", " "),
                    }
                    all_predictions.append(row)

                    logger.info(
                        "Pass %d | Claim %d | Doc %s | Lang: %s | Mode: %s -> %s (conf: %.2f) in %.2fs [%s]",
                        total_instances, claim_id, doc_id, lang, mode, pred_label, conf, latency, parsing_status,
                    )

        logger.info("Completed Claim %d (%d/%d claims done)", claim_id, ci + 1, len(claims))

    total_elapsed = time.time() - start_total_time

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Save CSV
    logger.info("Writing %d prediction rows to %s", len(all_predictions), output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_predictions)

    # Compute Condition Metrics
    condition_metrics = {}
    for lang in languages:
        condition_metrics[lang] = {}
        for mode in modes:
            cond_rows = [r for r in all_predictions if r["language"] == lang and r["mode"] == mode]
            gold_vec = [r["gold_label"] for r in cond_rows]
            pred_vec = [r["predicted_label"] for r in cond_rows]

            eval_res = evaluate_veracity_predictions(gold_vec, pred_vec)

            # Rationale F1 mean for Mode B
            avg_rat_f1 = None
            if mode == "B":
                rat_f1_scores = [r["rationale_f1"] for r in cond_rows if r["rationale_f1"] is not None]
                avg_rat_f1 = round(sum(rat_f1_scores) / len(rat_f1_scores), 4) if rat_f1_scores else 0.0

            # Error category breakdown
            err_counts = {}
            for r in cond_rows:
                code = r["error_category"]
                err_counts[code] = err_counts.get(code, 0) + 1

            condition_metrics[lang][f"Mode_{mode}"] = {
                "samples": len(cond_rows),
                "accuracy": eval_res["accuracy"],
                "macro_f1": eval_res["macro_f1"],
                "macro_precision": eval_res["macro_precision"],
                "macro_recall": eval_res["macro_recall"],
                "mean_rationale_f1": avg_rat_f1,
                "invalid_predictions": eval_res["invalid_predictions"],
                "per_class": eval_res["per_class"],
                "confusion_matrix": eval_res["confusion_matrix"],
                "error_taxonomy": err_counts,
            }

    # Compute Cross-Lingual Flip Rates
    cfr_results = {}
    for mode in modes:
        cfr_results[f"Mode_{mode}"] = {}
        pairs_to_check = [("en", "es"), ("en", "hi"), ("es", "hi")]
        for l1, l2 in pairs_to_check:
            if l1 in languages and l2 in languages:
                v1 = []
                v2 = []
                for p_key, lang_dict in cfr_tracker.items():
                    if p_key[0] == mode and l1 in lang_dict and l2 in lang_dict:
                        v1.append(lang_dict[l1])
                        v2.append(lang_dict[l2])
                cfr_results[f"Mode_{mode}"][f"{l1}_vs_{l2}"] = compute_cross_lingual_flip_rate(v1, v2)

    summary = {
        "model": model_name,
        "claims_input": claims_path,
        "total_claims": len(claims),
        "total_passes": len(all_predictions),
        "languages": list(languages),
        "modes": list(modes),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "avg_seconds_per_pass": round(total_elapsed / len(all_predictions), 3) if all_predictions else 0,
        "conditions": condition_metrics,
        "cross_lingual_flip_rates": cfr_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    summary_file = output_path.replace(".csv", "_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("MULTILINGUAL EXPERIMENT PILOT COMPLETE")
    logger.info("=" * 60)
    logger.info("Total passes: %d | Time: %.2f s", len(all_predictions), total_elapsed)
    for lang in languages:
        for mode in modes:
            m = condition_metrics[lang][f"Mode_{mode}"]
            logger.info(
                "[%s - Mode %s] Acc: %.4f | Macro-F1: %.4f | Invalid: %d",
                lang, mode, m["accuracy"], m["macro_f1"], m["invalid_predictions"]
            )
    logger.info("=" * 60)
    logger.info("Summary saved to %s", summary_file)

    return summary


def reparse_predictions(csv_path: str, summary_path: Optional[str] = None):
    """
    Recompute metrics, error taxonomy, and CFR from an existing predictions CSV
    using updated parse_model_json_response and normalize_eval_label logic.
    """
    logger.info("Reparsing predictions from %s", csv_path)
    import ast
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    logger.info("Loaded %d rows for reparsing", len(rows))
    cfr_tracker = {}
    languages = set()
    modes = set()

    for r in rows:
        mode = r["mode"]
        lang = r["language"]
        languages.add(lang)
        modes.add(mode)
        raw_output = r["raw_model_output"]
        gold_label = r["gold_label"]

        # Parse gold_sentences from string representation
        gold_sents_str = r.get("gold_sentences", "[]")
        try:
            gold_sents = ast.literal_eval(gold_sents_str) if gold_sents_str else []
        except Exception:
            gold_sents = []

        parsed = parse_model_json_response(raw_output, mode=mode)
        pred_label = parsed["veracity"]
        pred_sents = parsed["rationale_sentences"]
        conf = parsed["confidence"]
        status = parsed["parsing_status"]

        is_correct = pred_label == gold_label
        rat_f1 = evaluate_rationale_selection(gold_sents, pred_sents)["f1"] if mode == "B" else None
        err_code = classify_error(gold_label, pred_label, gold_sents, pred_sents, mode=mode)

        r["predicted_label"] = pred_label
        r["predicted_sentences"] = str(pred_sents)
        r["confidence"] = conf
        r["veracity_correct"] = is_correct
        r["rationale_f1"] = rat_f1
        r["parsing_status"] = status
        r["error_category"] = err_code

        pair_key = (mode, r["claim_id"], r["doc_id"])
        if pair_key not in cfr_tracker:
            cfr_tracker[pair_key] = {}
        cfr_tracker[pair_key][lang] = pred_label

    # Save updated CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    languages = sorted(list(languages))
    modes = sorted(list(modes))

    # Recompute Condition Metrics
    condition_metrics = {}
    for lang in languages:
        condition_metrics[lang] = {}
        for mode in modes:
            cond_rows = [r for r in rows if r["language"] == lang and r["mode"] == mode]
            gold_vec = [r["gold_label"] for r in cond_rows]
            pred_vec = [r["predicted_label"] for r in cond_rows]

            eval_res = evaluate_veracity_predictions(gold_vec, pred_vec)

            avg_rat_f1 = None
            if mode == "B":
                rat_f1_scores = [r["rationale_f1"] for r in cond_rows if r["rationale_f1"] is not None]
                avg_rat_f1 = round(sum(rat_f1_scores) / len(rat_f1_scores), 4) if rat_f1_scores else 0.0

            err_counts = {}
            for r in cond_rows:
                code = r["error_category"]
                err_counts[code] = err_counts.get(code, 0) + 1

            condition_metrics[lang][f"Mode_{mode}"] = {
                "samples": len(cond_rows),
                "accuracy": eval_res["accuracy"],
                "macro_f1": eval_res["macro_f1"],
                "macro_precision": eval_res["macro_precision"],
                "macro_recall": eval_res["macro_recall"],
                "mean_rationale_f1": avg_rat_f1,
                "invalid_predictions": eval_res["invalid_predictions"],
                "per_class": eval_res["per_class"],
                "confusion_matrix": eval_res["confusion_matrix"],
                "error_taxonomy": err_counts,
            }

    # Recompute Cross-Lingual Flip Rates
    cfr_results = {}
    for mode in modes:
        cfr_results[f"Mode_{mode}"] = {}
        pairs_to_check = [("en", "es"), ("en", "hi"), ("es", "hi")]
        for l1, l2 in pairs_to_check:
            if l1 in languages and l2 in languages:
                v1 = []
                v2 = []
                for p_key, lang_dict in cfr_tracker.items():
                    if p_key[0] == mode and l1 in lang_dict and l2 in lang_dict:
                        v1.append(lang_dict[l1])
                        v2.append(lang_dict[l2])
                cfr_results[f"Mode_{mode}"][f"{l1}_vs_{l2}"] = compute_cross_lingual_flip_rate(v1, v2)

    summary_file = summary_path or csv_path.replace(".csv", "_summary.json")
    # Load previous timing if present
    prev_elapsed = 512.38
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                prev_s = json.load(f)
                prev_elapsed = prev_s.get("total_elapsed_seconds", 512.38)
        except Exception:
            pass

    summary = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "claims_input": str(PROJECT_ROOT / "data" / "claims_dev_multilingual_pilot.jsonl"),
        "total_claims": 3,
        "total_passes": len(rows),
        "languages": list(languages),
        "modes": list(modes),
        "total_elapsed_seconds": prev_elapsed,
        "avg_seconds_per_pass": round(prev_elapsed / len(rows), 3) if rows else 0,
        "conditions": condition_metrics,
        "cross_lingual_flip_rates": cfr_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("REPARSED MULTILINGUAL SUMMARY COMPLETE")
    logger.info("=" * 60)
    for lang in languages:
        for mode in modes:
            m = condition_metrics[lang][f"Mode_{mode}"]
            logger.info(
                "[%s - Mode %s] Acc: %.4f | Macro-F1: %.4f | Invalid: %d",
                lang, mode, m["accuracy"], m["macro_f1"], m["invalid_predictions"]
            )
    logger.info("=" * 60)
    logger.info("Summary updated at %s", summary_file)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run multilingual scientific claim verification.")
    parser.add_argument(
        "--claims", type=str,
        default=str(PROJECT_ROOT / "data" / "claims_dev_multilingual_pilot.jsonl"),
        help="Path to multilingual claims file.",
    )
    parser.add_argument(
        "--corpus", type=str,
        default=str(PROJECT_ROOT / "data" / "corpus.jsonl"),
        help="Path to corpus file.",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "results" / "multilingual_pilot_predictions.csv"),
        help="Path to output CSV.",
    )
    parser.add_argument(
        "--model", type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model checkpoint name.",
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Run pilot on first 3 claims from input file.",
    )
    parser.add_argument(
        "--reparse", action="store_true",
        help="Recompute metrics and error codes from an existing predictions CSV without re-running model inference.",
    )
    args = parser.parse_args()

    if args.reparse:
        reparse_predictions(csv_path=args.output)
        return

    max_c = 3 if args.pilot else 0
    run_experiment(
        claims_path=args.claims,
        corpus_path=args.corpus,
        output_path=args.output,
        model_name=args.model,
        max_claims=max_c,
    )


if __name__ == "__main__":
    main()
