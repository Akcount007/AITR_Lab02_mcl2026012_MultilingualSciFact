"""
evaluation.py — Comprehensive evaluation module for scientific claim verification.

Provides:
  1. Input schema verification and cross-lingual gold-label alignment audit.
  2. Classification metric computation (Accuracy, Macro-F1, per-class P/R/F1, confusion matrices).
  3. Invalid-output rate calculation (strict penalty tracking).
  4. Matched-pair prediction disagreement (Cross-Lingual Flip Rates and Cohen's Kappa).
  5. Runtime and latency profiling per example.
  6. Calibration analysis (Expected Calibration Error, Brier Score, Reliability Diagram data).
  7. Export utilities for metric tables and visualization figures using actual values.
"""

from typing import Dict, List, Tuple, Any, Optional, Set
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    brier_score_loss,
)

from src.metrics import CANONICAL_LABELS, normalize_eval_label

REQUIRED_COLUMNS = {
    "claim_id",
    "doc_id",
    "language",
    "mode",
    "gold_label",
    "predicted_label",
    "confidence",
    "latency_sec",
}


def verify_prediction_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Verify input DataFrame schema, column types, value bounds, and nulls.

    Returns:
        (is_valid: bool, issues: List[str])
    """
    issues = []

    # 1. Required column presence
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        issues.append(f"Missing required columns: {sorted(list(missing_cols))}")
        return False, issues

    # 2. Row count check
    if len(df) == 0:
        issues.append("Prediction DataFrame is empty (0 rows).")
        return False, issues

    # 3. Null values check in essential columns
    for col in ["claim_id", "doc_id", "language", "mode", "gold_label", "predicted_label"]:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            issues.append(f"Column '{col}' contains {null_count} null/NaN values.")

    # 4. Confidence validity
    if not pd.api.types.is_numeric_dtype(df["confidence"]):
        issues.append("Column 'confidence' must be numeric.")
    else:
        out_of_bounds = df[(df["confidence"] < 0.0) | (df["confidence"] > 1.0)]
        if len(out_of_bounds) > 0:
            issues.append(f"Column 'confidence' has {len(out_of_bounds)} values outside [0.0, 1.0].")

    # 5. Latency validity
    if not pd.api.types.is_numeric_dtype(df["latency_sec"]):
        issues.append("Column 'latency_sec' must be numeric.")
    else:
        neg_latency = df[df["latency_sec"] < 0.0]
        if len(neg_latency) > 0:
            issues.append(f"Column 'latency_sec' has {len(neg_latency)} negative values.")

    # 6. Valid gold labels
    invalid_gold = []
    for g in df["gold_label"].unique():
        norm = normalize_eval_label(str(g))
        if norm not in CANONICAL_LABELS:
            invalid_gold.append(g)
    if invalid_gold:
        issues.append(f"Invalid gold label values detected: {invalid_gold}")

    return len(issues) == 0, issues


def verify_gold_label_alignment(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Verify that gold labels remain strictly aligned with identical example IDs across languages.

    Returns:
        (is_aligned: bool, issues: List[str])
    """
    issues = []
    key_cols = ["claim_id", "doc_id"]
    if "gold_sentences" in df.columns:
        key_cols.append("gold_sentences")

    grouped = df.groupby(key_cols)
    for key_vals, group in grouped:
        unique_golds = set(group["gold_label"].unique())
        if len(unique_golds) > 1:
            issues.append(
                f"Gold label mismatch for instance {key_vals}: "
                f"observed divergent gold labels {unique_golds} across languages/modes."
            )

    return len(issues) == 0, issues


def compute_classification_metrics(
    gold_labels: List[str],
    pred_labels: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute Accuracy, Macro-F1, per-class Precision/Recall/F1, confusion matrix,
    and invalid-output rate with strict penalty rules.
    """
    if labels is None:
        labels = CANONICAL_LABELS

    n_total = len(gold_labels)
    if n_total == 0:
        return {
            "samples": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "invalid_output_rate": 0.0,
            "invalid_predictions": 0,
            "per_class": {},
            "confusion_matrix": [[0] * len(labels) for _ in range(len(labels))],
        }

    norm_gold = [normalize_eval_label(g) for g in gold_labels]
    norm_pred = [normalize_eval_label(p) for p in pred_labels]

    invalid_count = sum(1 for p in norm_pred if p == "INVALID_OUTPUT")
    invalid_rate = round(invalid_count / n_total, 4)

    # Accuracy strictly penalized: INVALID_OUTPUT never matches valid gold
    acc = float(accuracy_score(norm_gold, norm_pred))

    # Per-class and Macro metrics on CANONICAL_LABELS
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        norm_gold,
        norm_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        norm_gold,
        norm_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    per_class = {}
    for i, lbl in enumerate(labels):
        per_class[lbl] = {
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1": round(float(f1_per[i]), 4),
            "support": int(sup_per[i]),
        }

    cm = confusion_matrix(norm_gold, norm_pred, labels=labels)

    return {
        "samples": n_total,
        "accuracy": round(acc, 4),
        "macro_f1": round(float(f1_macro), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "invalid_output_rate": invalid_rate,
        "invalid_predictions": invalid_count,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def compute_calibration_metrics(
    df: pd.DataFrame,
    n_bins: int = 5,
) -> Dict[str, Any]:
    """
    Compute calibration metrics: Brier score, Expected Calibration Error (ECE),
    Maximum Calibration Error (MCE), and binned reliability diagram data.
    """
    if len(df) == 0:
        return {
            "brier_score": None,
            "ece": None,
            "mce": None,
            "bins": [],
        }

    confidences = df["confidence"].values
    norm_gold = [normalize_eval_label(g) for g in df["gold_label"]]
    norm_pred = [normalize_eval_label(p) for p in df["predicted_label"]]

    correctness = np.array([1.0 if g == p else 0.0 for g, p in zip(norm_gold, norm_pred)])

    # Brier score on correctness indicator: (confidence - is_correct)^2
    brier = float(brier_score_loss(correctness, confidences))

    # Equal-width binning for ECE
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_data = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        # Include upper edge on final bin
        if i == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)

        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_acc = float(np.mean(correctness[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            gap = abs(bin_acc - bin_conf)
            ece += (bin_count / len(df)) * gap
            mce = max(mce, gap)
        else:
            bin_acc = None
            bin_conf = None
            gap = None

        bin_data.append({
            "bin_index": i,
            "bin_range": f"[{low:.2f}, {high:.2f}]",
            "count": bin_count,
            "accuracy": round(bin_acc, 4) if bin_acc is not None else None,
            "confidence": round(bin_conf, 4) if bin_conf is not None else None,
            "calibration_gap": round(gap, 4) if gap is not None else None,
        })

    return {
        "brier_score": round(brier, 4),
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "bins": bin_data,
    }


def compute_runtime_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate descriptive statistics for inference latency per example."""
    latencies = df["latency_sec"].dropna().values
    if len(latencies) == 0:
        return {
            "mean_sec": 0.0,
            "std_sec": 0.0,
            "median_sec": 0.0,
            "min_sec": 0.0,
            "max_sec": 0.0,
            "total_sec": 0.0,
        }

    return {
        "mean_sec": round(float(np.mean(latencies)), 4),
        "std_sec": round(float(np.std(latencies)), 4),
        "median_sec": round(float(np.median(latencies)), 4),
        "min_sec": round(float(np.min(latencies)), 4),
        "max_sec": round(float(np.max(latencies)), 4),
        "total_sec": round(float(np.sum(latencies)), 4),
    }


def compute_paired_disagreements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute matched-pair prediction disagreement and Cohen's Kappa across:
      1. Cross-Lingual pairs (en vs es, en vs hi, es vs hi) holding mode fixed.
      2. Cross-Prompt pairs (Mode A vs Mode B) holding language fixed.
    """
    rows = []

    # 1. Cross-Lingual Comparisons (holding Mode fixed)
    modes = sorted(df["mode"].unique())
    lang_pairs = [("en", "es"), ("en", "hi"), ("es", "hi")]

    df_copy = df.copy()
    if "gold_sentences" in df_copy.columns:
        df_copy["instance_key"] = (
            df_copy["claim_id"].astype(str)
            + "_"
            + df_copy["doc_id"].astype(str)
            + "_"
            + df_copy["gold_sentences"].astype(str)
        )
    else:
        df_copy["instance_key"] = df_copy["claim_id"].astype(str) + "_" + df_copy["doc_id"].astype(str)

    for mode in modes:
        sub_mode = df_copy[df_copy["mode"] == mode]
        pivot = sub_mode.pivot(index="instance_key", columns="language", values="predicted_label")

        for l1, l2 in lang_pairs:
            if l1 in pivot.columns and l2 in pivot.columns:
                valid_pairs = pivot[[l1, l2]].dropna()
                n_pairs = len(valid_pairs)
                if n_pairs > 0:
                    v1 = valid_pairs[l1].tolist()
                    v2 = valid_pairs[l2].tolist()
                    disagreements = sum(1 for a, b in zip(v1, v2) if a != b)
                    disagree_rate = round(disagreements / n_pairs, 4)
                    agree_rate = round(1.0 - disagree_rate, 4)
                    try:
                        k_val = float(cohen_kappa_score(v1, v2))
                        kappa = 1.0 if (np.isnan(k_val) and disagree_rate == 0.0) else (round(k_val, 4) if not np.isnan(k_val) else 0.0)
                    except Exception:
                        kappa = 1.0 if disagree_rate == 0.0 else 0.0

                    rows.append({
                        "comparison_type": "cross_lingual",
                        "pair_name": f"{l1}_vs_{l2}",
                        "condition_fixed": f"Mode_{mode}",
                        "total_pairs": n_pairs,
                        "disagreement_count": disagreements,
                        "disagreement_rate": disagree_rate,
                        "agreement_rate": agree_rate,
                        "cohens_kappa": kappa,
                    })

    # 2. Cross-Prompt Comparisons (holding Language fixed)
    languages = sorted(df["language"].unique())
    for lang in languages:
        sub_lang = df_copy[df_copy["language"] == lang]
        pivot = sub_lang.pivot(index="instance_key", columns="mode", values="predicted_label")
        if "A" in pivot.columns and "B" in pivot.columns:
            valid_pairs = pivot[["A", "B"]].dropna()
            n_pairs = len(valid_pairs)
            if n_pairs > 0:
                v1 = valid_pairs["A"].tolist()
                v2 = valid_pairs["B"].tolist()
                disagreements = sum(1 for a, b in zip(v1, v2) if a != b)
                disagree_rate = round(disagreements / n_pairs, 4)
                agree_rate = round(1.0 - disagree_rate, 4)
                try:
                    k_val = float(cohen_kappa_score(v1, v2))
                    kappa = 1.0 if (np.isnan(k_val) and disagree_rate == 0.0) else (round(k_val, 4) if not np.isnan(k_val) else 0.0)
                except Exception:
                    kappa = 1.0 if disagree_rate == 0.0 else 0.0

                rows.append({
                    "comparison_type": "cross_prompt",
                    "pair_name": "Mode_A_vs_Mode_B",
                    "condition_fixed": f"lang_{lang}",
                    "total_pairs": n_pairs,
                    "disagreement_count": disagreements,
                    "disagreement_rate": disagree_rate,
                    "agreement_rate": agree_rate,
                    "cohens_kappa": kappa,
                })

    return pd.DataFrame(rows)


def generate_evaluation_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Generate structured metrics tables:
      1. metrics_by_language
      2. metrics_by_prompt
      3. paired_disagreement
    """
    # Table 1: Metrics by Language (Language overall & Language x Mode)
    lang_rows = []
    languages = sorted(df["language"].unique())
    modes = sorted(df["mode"].unique())

    # Overall per language
    for lang in languages:
        sub = df[df["language"] == lang]
        cls_m = compute_classification_metrics(sub["gold_label"].tolist(), sub["predicted_label"].tolist())
        cal_m = compute_calibration_metrics(sub)
        rt_m = compute_runtime_metrics(sub)

        # Mean rationale F1 if Mode B present
        b_sub = sub[sub["mode"] == "B"]
        rat_f1 = None
        if len(b_sub) > 0 and "rationale_f1" in b_sub.columns:
            valid_rats = b_sub["rationale_f1"].dropna().tolist()
            rat_f1 = round(float(np.mean(valid_rats)), 4) if valid_rats else None

        lang_rows.append({
            "language": lang,
            "mode": "OVERALL",
            "sample_count": cls_m["samples"],
            "accuracy": cls_m["accuracy"],
            "macro_f1": cls_m["macro_f1"],
            "macro_precision": cls_m["macro_precision"],
            "macro_recall": cls_m["macro_recall"],
            "precision_SUPPORTS": cls_m["per_class"].get("SUPPORTS", {}).get("precision", 0.0),
            "recall_SUPPORTS": cls_m["per_class"].get("SUPPORTS", {}).get("recall", 0.0),
            "f1_SUPPORTS": cls_m["per_class"].get("SUPPORTS", {}).get("f1", 0.0),
            "precision_REFUTES": cls_m["per_class"].get("REFUTES", {}).get("precision", 0.0),
            "recall_REFUTES": cls_m["per_class"].get("REFUTES", {}).get("recall", 0.0),
            "f1_REFUTES": cls_m["per_class"].get("REFUTES", {}).get("f1", 0.0),
            "precision_NOT_ENOUGH_INFO": cls_m["per_class"].get("NOT_ENOUGH_INFO", {}).get("precision", 0.0),
            "recall_NOT_ENOUGH_INFO": cls_m["per_class"].get("NOT_ENOUGH_INFO", {}).get("recall", 0.0),
            "f1_NOT_ENOUGH_INFO": cls_m["per_class"].get("NOT_ENOUGH_INFO", {}).get("f1", 0.0),
            "invalid_output_rate": cls_m["invalid_output_rate"],
            "mean_latency_sec": rt_m["mean_sec"],
            "median_latency_sec": rt_m["median_sec"],
            "std_latency_sec": rt_m["std_sec"],
            "brier_score": cal_m["brier_score"],
            "ece": cal_m["ece"],
            "mean_rationale_f1": rat_f1,
        })

        # By Language x Mode
        for mode in modes:
            sub_lm = sub[sub["mode"] == mode]
            if len(sub_lm) > 0:
                cls_lm = compute_classification_metrics(sub_lm["gold_label"].tolist(), sub_lm["predicted_label"].tolist())
                cal_lm = compute_calibration_metrics(sub_lm)
                rt_lm = compute_runtime_metrics(sub_lm)

                lm_rat_f1 = None
                if mode == "B" and "rationale_f1" in sub_lm.columns:
                    valid_rats = sub_lm["rationale_f1"].dropna().tolist()
                    lm_rat_f1 = round(float(np.mean(valid_rats)), 4) if valid_rats else None

                lang_rows.append({
                    "language": lang,
                    "mode": mode,
                    "sample_count": cls_lm["samples"],
                    "accuracy": cls_lm["accuracy"],
                    "macro_f1": cls_lm["macro_f1"],
                    "macro_precision": cls_lm["macro_precision"],
                    "macro_recall": cls_lm["macro_recall"],
                    "precision_SUPPORTS": cls_lm["per_class"].get("SUPPORTS", {}).get("precision", 0.0),
                    "recall_SUPPORTS": cls_lm["per_class"].get("SUPPORTS", {}).get("recall", 0.0),
                    "f1_SUPPORTS": cls_lm["per_class"].get("SUPPORTS", {}).get("f1", 0.0),
                    "precision_REFUTES": cls_lm["per_class"].get("REFUTES", {}).get("precision", 0.0),
                    "recall_REFUTES": cls_lm["per_class"].get("REFUTES", {}).get("recall", 0.0),
                    "f1_REFUTES": cls_lm["per_class"].get("REFUTES", {}).get("f1", 0.0),
                    "precision_NOT_ENOUGH_INFO": cls_lm["per_class"].get("NOT_ENOUGH_INFO", {}).get("precision", 0.0),
                    "recall_NOT_ENOUGH_INFO": cls_lm["per_class"].get("NOT_ENOUGH_INFO", {}).get("recall", 0.0),
                    "f1_NOT_ENOUGH_INFO": cls_lm["per_class"].get("NOT_ENOUGH_INFO", {}).get("f1", 0.0),
                    "invalid_output_rate": cls_lm["invalid_output_rate"],
                    "mean_latency_sec": rt_lm["mean_sec"],
                    "median_latency_sec": rt_lm["median_sec"],
                    "std_latency_sec": rt_lm["std_sec"],
                    "brier_score": cal_lm["brier_score"],
                    "ece": cal_lm["ece"],
                    "mean_rationale_f1": lm_rat_f1,
                })

    metrics_by_language = pd.DataFrame(lang_rows)

    # Table 2: Metrics by Prompt Mode
    prompt_rows = []
    for mode in modes:
        sub_p = df[df["mode"] == mode]
        cls_p = compute_classification_metrics(sub_p["gold_label"].tolist(), sub_p["predicted_label"].tolist())
        cal_p = compute_calibration_metrics(sub_p)
        rt_p = compute_runtime_metrics(sub_p)

        p_rat_f1 = None
        if mode == "B" and "rationale_f1" in sub_p.columns:
            valid_rats = sub_p["rationale_f1"].dropna().tolist()
            p_rat_f1 = round(float(np.mean(valid_rats)), 4) if valid_rats else None

        prompt_rows.append({
            "mode": mode,
            "sample_count": cls_p["samples"],
            "accuracy": cls_p["accuracy"],
            "macro_f1": cls_p["macro_f1"],
            "macro_precision": cls_p["macro_precision"],
            "macro_recall": cls_p["macro_recall"],
            "precision_SUPPORTS": cls_p["per_class"].get("SUPPORTS", {}).get("precision", 0.0),
            "recall_SUPPORTS": cls_p["per_class"].get("SUPPORTS", {}).get("recall", 0.0),
            "f1_SUPPORTS": cls_p["per_class"].get("SUPPORTS", {}).get("f1", 0.0),
            "precision_REFUTES": cls_p["per_class"].get("REFUTES", {}).get("precision", 0.0),
            "recall_REFUTES": cls_p["per_class"].get("REFUTES", {}).get("recall", 0.0),
            "f1_REFUTES": cls_p["per_class"].get("REFUTES", {}).get("f1", 0.0),
            "precision_NOT_ENOUGH_INFO": cls_p["per_class"].get("NOT_ENOUGH_INFO", {}).get("precision", 0.0),
            "recall_NOT_ENOUGH_INFO": cls_p["per_class"].get("NOT_ENOUGH_INFO", {}).get("recall", 0.0),
            "f1_NOT_ENOUGH_INFO": cls_p["per_class"].get("NOT_ENOUGH_INFO", {}).get("f1", 0.0),
            "invalid_output_rate": cls_p["invalid_output_rate"],
            "mean_latency_sec": rt_p["mean_sec"],
            "median_latency_sec": rt_p["median_sec"],
            "std_latency_sec": rt_p["std_sec"],
            "brier_score": cal_p["brier_score"],
            "ece": cal_p["ece"],
            "mean_rationale_f1": p_rat_f1,
        })

    metrics_by_prompt = pd.DataFrame(prompt_rows)

    # Table 3: Paired Disagreement Table
    paired_disagreement = compute_paired_disagreements(df)

    return {
        "metrics_by_language": metrics_by_language,
        "metrics_by_prompt": metrics_by_prompt,
        "paired_disagreement": paired_disagreement,
    }


def generate_all_evaluation_figures(df: pd.DataFrame, figures_dir: str):
    """
    Generate and save five clear publication-quality figures using actual values:
      1. fig1_accuracy_f1_by_condition.png
      2. fig2_confusion_matrices.png
      3. fig3_runtime_by_condition.png
      4. fig4_paired_disagreement.png
      5. fig5_calibration_analysis.png
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = sns.color_palette("deep")

    # -------------------------------------------------------------
    # Figure 1: Accuracy and Macro-F1 by Condition
    # -------------------------------------------------------------
    tables = generate_evaluation_tables(df)
    m_lang = tables["metrics_by_language"]
    # Filter to specific language x mode rows (exclude OVERALL)
    plot_df = m_lang[m_lang["mode"].isin(["A", "B"])].copy()
    plot_df["Condition"] = plot_df["language"].str.upper() + " (Mode " + plot_df["mode"] + ")"

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(plot_df))
    width = 0.35

    rects1 = ax.bar(x - width / 2, plot_df["accuracy"], width, label="Accuracy", color="#1f77b4")
    rects2 = ax.bar(x + width / 2, plot_df["macro_f1"], width, label="Macro-F1", color="#2ca02c")

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Scientific Claim Verification: Accuracy & Macro-F1 across Conditions", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Condition"], rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=True)

    # Label heights
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "fig1_accuracy_f1_by_condition.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 2: Confusion Matrices
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, mode in enumerate(["A", "B"]):
        sub_mode = df[df["mode"] == mode]
        cm_dict = compute_classification_metrics(sub_mode["gold_label"].tolist(), sub_mode["predicted_label"].tolist())
        cm = np.array(cm_dict["confusion_matrix"])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues" if mode == "A" else "Greens",
            xticklabels=["SUPP", "REF", "NEI"],
            yticklabels=["SUPP", "REF", "NEI"],
            cbar=False,
            ax=axes[i],
            annot_kws={"size": 13, "weight": "bold"},
        )
        mode_label = "Direct Classification" if mode == "A" else "Evidence-Grounded"
        axes[i].set_title(f"Mode {mode}: {mode_label} (N={len(sub_mode)})", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Predicted Label", fontsize=10)
        axes[i].set_ylabel("Gold Label", fontsize=10)

    fig.suptitle("Confusion Matrices across Experimental Modes (All Languages)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "fig2_confusion_matrices.png")
    fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 3: Runtime per Example by Condition
    # -------------------------------------------------------------
    languages = sorted(df["language"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(
        data=df,
        x="language",
        y="latency_sec",
        hue="mode",
        palette={"A": "#4c72b0", "B": "#55a868"},
        ax=ax,
    )
    ax.set_title("Runtime Latency per Example across Language and Prompt Mode", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Language Condition", fontsize=11)
    ax.set_ylabel("Inference Latency (seconds)", fontsize=11)
    ax.set_xticks(range(len(languages)))
    ax.set_xticklabels([f"English (en)", "Spanish (es)", "Hindi (hi)"][:len(languages)], fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Mode A: Direct", "Mode B: Grounded"], title="Prompting Mode", loc="upper right")

    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "fig3_runtime_by_condition.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 4: Paired Disagreement (Cross-Lingual & Cross-Prompt)
    # -------------------------------------------------------------
    paired_df = tables["paired_disagreement"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    paired_df["label"] = paired_df["pair_name"] + "\n(" + paired_df["condition_fixed"] + ")"
    sns.barplot(
        data=paired_df,
        x="label",
        y="disagreement_rate",
        hue="comparison_type",
        palette={"cross_lingual": "#3470a3", "cross_prompt": "#e24a33"},
        ax=ax,
    )
    ax.set_title("Matched-Pair Prediction Disagreement Rates", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Disagreement Rate (0.0 to 1.0)", fontsize=11)
    ax.set_xlabel("Paired Comparison Condition", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(title="Comparison Type", loc="upper left")

    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h >= 0:
            ax.annotate(f"{h:.2f}", xy=(p.get_x() + p.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "fig4_paired_disagreement.png")
    fig.savefig(fig4_path, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 5: Confidence & Calibration Analysis
    # -------------------------------------------------------------
    fig, (ax_hist, ax_cal) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Histogram of confidence values
    sns.histplot(
        df["confidence"],
        bins=5,
        kde=False,
        color="#8172b3",
        ax=ax_hist,
    )
    ax_hist.set_title("Confidence Distribution (N=24)", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("Predicted Confidence", fontsize=10)
    ax_hist.set_ylabel("Frequency", fontsize=10)
    ax_hist.set_xlim(-0.05, 1.05)

    # Reliability diagram for overall data
    cal_res = compute_calibration_metrics(df, n_bins=5)
    valid_bins = [b for b in cal_res["bins"] if b["count"] > 0]
    b_conf = [b["confidence"] for b in valid_bins]
    b_acc = [b["accuracy"] for b in valid_bins]

    ax_cal.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax_cal.scatter(b_conf, b_acc, color="#d95f02", s=80, zorder=5, label="Observed Bins")
    ax_cal.plot(b_conf, b_acc, color="#d95f02", alpha=0.7)
    ax_cal.set_title(f"Reliability Diagram (ECE: {cal_res['ece']:.3f}, Brier: {cal_res['brier_score']:.3f})", fontsize=11, fontweight="bold")
    ax_cal.set_xlabel("Mean Predicted Confidence", fontsize=10)
    ax_cal.set_ylabel("Observed Accuracy", fontsize=10)
    ax_cal.set_xlim(-0.05, 1.05)
    ax_cal.set_ylim(-0.05, 1.05)
    ax_cal.legend(loc="upper left")

    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, "fig5_calibration_analysis.png")
    fig.savefig(fig5_path, dpi=300)
    plt.close(fig)

    return {
        "fig1": fig1_path,
        "fig2": fig2_path,
        "fig3": fig3_path,
        "fig4": fig4_path,
        "fig5": fig5_path,
    }
