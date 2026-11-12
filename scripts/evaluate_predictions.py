#!/usr/bin/env python3
"""
evaluate_predictions.py — Master evaluation runner for scientific claim verification.

Performs:
  1. Input schema verification.
  2. Cross-lingual gold-label alignment audit.
  3. Classification metric computation (Accuracy, Macro-F1, per-class P/R/F1, confusion matrices).
  4. Invalid-output rate tracking (strict penalty rules).
  5. Matched-pair disagreement computation (Cross-Lingual CFR and Cohen's Kappa).
  6. Runtime and latency profiling per example.
  7. Confidence calibration analysis (Brier Score, ECE, Reliability Diagrams).
  8. Canonical exports:
     - results/metrics_by_language.csv
     - results/metrics_by_prompt.csv
     - results/paired_disagreement.csv
     - results/predictions.csv
  9. Publication figure generation using actual values:
     - results/figures/fig1_accuracy_f1_by_condition.png
     - results/figures/fig2_confusion_matrices.png
     - results/figures/fig3_runtime_by_condition.png
     - results/figures/fig4_paired_disagreement.png
     - results/figures/fig5_calibration_analysis.png

Usage:
    python scripts/evaluate_predictions.py
    python scripts/evaluate_predictions.py --input results/multilingual_pilot_predictions.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    verify_prediction_schema,
    verify_gold_label_alignment,
    generate_evaluation_tables,
    generate_all_evaluation_figures,
    compute_calibration_metrics,
    compute_runtime_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_predictions")


def run_evaluation(
    input_path: str,
    output_predictions_path: str,
    results_dir: str,
    figures_dir: str,
) -> int:
    logger.info("Loading predictions from %s", input_path)
    if not os.path.exists(input_path):
        logger.error("Input file does not exist: %s", input_path)
        return 1

    df = pd.read_csv(input_path)
    logger.info("Loaded %d prediction rows.", len(df))

    # 1. Verify schema
    logger.info("Verifying prediction schema...")
    schema_valid, schema_issues = verify_prediction_schema(df)
    if not schema_valid:
        logger.error("Schema validation FAILED with issues:")
        for iss in schema_issues:
            logger.error("  - %s", iss)
        return 1
    logger.info("Schema verification: PASS (All required columns, types, and bounds valid).")

    # 2. Verify gold-label alignment across languages
    logger.info("Verifying gold label alignment across languages...")
    align_valid, align_issues = verify_gold_label_alignment(df)
    if not align_valid:
        logger.error("Gold label alignment FAILED with issues:")
        for iss in align_issues:
            logger.error("  - %s", iss)
        return 1
    logger.info("Gold label alignment: PASS (Zero gold label divergence across languages).")

    # 3. Export canonical predictions.csv
    os.makedirs(os.path.dirname(output_predictions_path) if os.path.dirname(output_predictions_path) else ".", exist_ok=True)
    df.to_csv(output_predictions_path, index=False, encoding="utf-8")
    logger.info("Exported canonical predictions to %s", output_predictions_path)

    # 4. Generate structured metrics tables
    logger.info("Computing evaluation metrics and paired disagreements...")
    tables = generate_evaluation_tables(df)

    metrics_lang_path = os.path.join(results_dir, "metrics_by_language.csv")
    metrics_prompt_path = os.path.join(results_dir, "metrics_by_prompt.csv")
    paired_disagree_path = os.path.join(results_dir, "paired_disagreement.csv")

    tables["metrics_by_language"].to_csv(metrics_lang_path, index=False, encoding="utf-8")
    tables["metrics_by_prompt"].to_csv(metrics_prompt_path, index=False, encoding="utf-8")
    tables["paired_disagreement"].to_csv(paired_disagree_path, index=False, encoding="utf-8")

    logger.info("Exported %s (%d rows)", metrics_lang_path, len(tables["metrics_by_language"]))
    logger.info("Exported %s (%d rows)", metrics_prompt_path, len(tables["metrics_by_prompt"]))
    logger.info("Exported %s (%d rows)", paired_disagree_path, len(tables["paired_disagreement"]))

    # 5. Calibration investigation
    cal_metrics = compute_calibration_metrics(df)
    logger.info(
        "Calibration Analysis: Brier Score = %.4f | ECE = %.4f | MCE = %.4f",
        cal_metrics["brier_score"], cal_metrics["ece"], cal_metrics["mce"]
    )

    # 6. Runtime investigation
    rt_metrics = compute_runtime_metrics(df)
    logger.info(
        "Runtime Analysis: Mean = %.2fs | Median = %.2fs | Std = %.2fs | Min = %.2fs | Max = %.2fs | Total = %.2fs",
        rt_metrics["mean_sec"], rt_metrics["median_sec"], rt_metrics["std_sec"],
        rt_metrics["min_sec"], rt_metrics["max_sec"], rt_metrics["total_sec"]
    )

    # 7. Generate publication-quality figures
    logger.info("Generating publication-quality figures in %s...", figures_dir)
    fig_paths = generate_all_evaluation_figures(df, figures_dir=figures_dir)
    for fig_id, p in fig_paths.items():
        logger.info("Generated figure [%s]: %s", fig_id, p)

    # 8. Print formatted console summary
    print("\n" + "=" * 78)
    print("                 SCIENTIFIC FACTUALITY EVALUATION REPORT")
    print("=" * 78)
    print(f"Predictions File:      {output_predictions_path} (N = {len(df)})")
    print(f"Schema Status:         PASS (8/8 core fields valid)")
    print(f"Gold Alignment:        PASS (100% matched across languages)")
    print(f"Brier Score:           {cal_metrics['brier_score']:.4f}")
    print(f"Expected Calib Error:  {cal_metrics['ece']:.4f}")
    print(f"Average Latency:       {rt_metrics['mean_sec']:.2f} s / instance (Total: {rt_metrics['total_sec']:.1f} s)")
    print("-" * 78)

    print("\n[METRICS BY PROMPT MODE]")
    print(tables["metrics_by_prompt"][["mode", "sample_count", "accuracy", "macro_f1", "macro_precision", "macro_recall", "invalid_output_rate", "mean_latency_sec"]].to_string(index=False))

    print("\n[METRICS BY LANGUAGE AND MODE]")
    print(tables["metrics_by_language"][["language", "mode", "sample_count", "accuracy", "macro_f1", "invalid_output_rate", "mean_latency_sec"]].to_string(index=False))

    print("\n[MATCHED-PAIR PREDICTION DISAGREEMENT]")
    print(tables["paired_disagreement"][["comparison_type", "pair_name", "condition_fixed", "total_pairs", "disagreement_count", "disagreement_rate", "cohens_kappa"]].to_string(index=False))

    print("-" * 78)
    print("Generated Artifacts:")
    print(f"  - Predictions:          {output_predictions_path}")
    print(f"  - Metrics by Language:  {metrics_lang_path}")
    print(f"  - Metrics by Prompt:    {metrics_prompt_path}")
    print(f"  - Paired Disagreement:  {paired_disagree_path}")
    print(f"  - Figure 1 (Acc & F1):  {fig_paths['fig1']}")
    print(f"  - Figure 2 (Confusion): {fig_paths['fig2']}")
    print(f"  - Figure 3 (Latency):   {fig_paths['fig3']}")
    print(f"  - Figure 4 (Paired):    {fig_paths['fig4']}")
    print(f"  - Figure 5 (Calib):     {fig_paths['fig5']}")
    print("=" * 78 + "\n")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate scientific claim verification predictions.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "results" / "multilingual_pilot_predictions.csv"),
        help="Path to input predictions CSV.",
    )
    parser.add_argument(
        "--output-predictions",
        type=str,
        default=str(PROJECT_ROOT / "results" / "predictions.csv"),
        help="Path to export canonical predictions CSV.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(PROJECT_ROOT / "results"),
        help="Directory to save metric tables.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "figures"),
        help="Directory to save figures.",
    )
    args = parser.parse_args()

    ret = run_evaluation(
        input_path=args.input,
        output_predictions_path=args.output_predictions,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
