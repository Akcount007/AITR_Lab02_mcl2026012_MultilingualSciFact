"""
Evaluation metrics module for scientific claim verification.

Provides:
  1. Veracity classification evaluation (Macro-F1, Precision, Recall, Accuracy, Per-class stats)
  2. Strict handling of invalid / unparseable outputs (penalized, never dropped)
  3. Sentence-level rationale selection metrics (Precision, Recall, F1)
  4. Cross-Lingual Flip Rate (CFR) across paired language conditions
"""

from typing import Dict, List, Set, Any, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

CANONICAL_LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
LABEL_TO_INT = {lbl: idx for idx, lbl in enumerate(CANONICAL_LABELS)}


def normalize_eval_label(label: str) -> str:
    """Standardize label string to canonical 3-class taxonomy."""
    if not isinstance(label, str):
        return "INVALID_OUTPUT"
    cleaned = label.strip().upper().replace("_", "").replace(" ", "").replace("-", "")
    if cleaned in ("SUPPORT", "SUPPORTS", "ENTAILMENT", "SUPPORTED"):
        return "SUPPORTS"
    if cleaned in ("CONTRADICT", "CONTRADICTS", "REFUTE", "REFUTES", "CONTRADICTION", "REFUTED"):
        return "REFUTES"
    if cleaned in ("NOTENOUGHINFO", "NEI", "NOINFO", "NEUTRAL", "NOTENOUGHINFORMATION", "INSUFFICIENT", "INSUFFICIENTINFO"):
        return "NOT_ENOUGH_INFO"
    return "INVALID_OUTPUT"


def evaluate_veracity_predictions(
    gold_labels: List[str],
    pred_labels: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate 3-class veracity classification predictions.

    Args:
        gold_labels: List of gold standard labels.
        pred_labels: List of predicted labels.
        labels: Optional label order (defaults to CANONICAL_LABELS).

    Returns:
        Dict with keys:
          - accuracy: float
          - macro_f1: float
          - macro_precision: float
          - macro_recall: float
          - total_samples: int
          - invalid_predictions: int
          - per_class: Dict[str, Dict[str, float]]
          - confusion_matrix: List[List[int]]
    """
    if labels is None:
        labels = CANONICAL_LABELS

    if len(gold_labels) != len(pred_labels):
        raise ValueError(
            f"Length mismatch: {len(gold_labels)} gold vs {len(pred_labels)} predicted."
        )

    if len(gold_labels) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "total_samples": 0,
            "invalid_predictions": 0,
            "per_class": {lbl: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for lbl in labels},
            "confusion_matrix": [[0] * len(labels) for _ in labels],
        }

    norm_gold = [normalize_eval_label(g) for g in gold_labels]
    norm_pred = [normalize_eval_label(p) for p in pred_labels]

    invalid_count = sum(1 for p in norm_pred if p == "INVALID_OUTPUT")

    # Scikit-learn calculation over canonical labels
    acc = accuracy_score(norm_gold, norm_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        norm_gold,
        norm_pred,
        labels=labels,
        average=None,
        zero_division=0.0,
    )

    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        norm_gold,
        norm_pred,
        labels=labels,
        average="macro",
        zero_division=0.0,
    )

    cm = confusion_matrix(norm_gold, norm_pred, labels=labels)

    per_class = {}
    for idx, lbl in enumerate(labels):
        per_class[lbl] = {
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx]),
        }

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "total_samples": len(gold_labels),
        "invalid_predictions": invalid_count,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def evaluate_rationale_selection(
    gold_sentences: List[int],
    pred_sentences: List[int],
) -> Dict[str, float]:
    """
    Compute sentence-level Precision, Recall, and F1 for rationale selection.

    Args:
        gold_sentences: List of 0-indexed gold sentence indices.
        pred_sentences: List of 0-indexed predicted sentence indices.

    Returns:
        Dict with precision, recall, f1.
    """
    gold_set = set(gold_sentences)
    pred_set = set(pred_sentences)

    if not gold_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gold_set or not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    intersection = gold_set.intersection(pred_set)
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(gold_set) if gold_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_cross_lingual_flip_rate(
    preds_lang_a: List[str],
    preds_lang_b: List[str],
) -> Dict[str, Any]:
    """
    Compute Cross-Lingual Flip Rate (CFR) between two matched prediction vectors.

    CFR is the proportion of identical examples whose predicted veracity
    flips between Language A and Language B.
    """
    if len(preds_lang_a) != len(preds_lang_b):
        raise ValueError(
            f"Mismatched predictions: {len(preds_lang_a)} vs {len(preds_lang_b)}"
        )

    if len(preds_lang_a) == 0:
        return {"flip_rate": 0.0, "flips": 0, "total": 0}

    flips = 0
    norm_a = [normalize_eval_label(p) for p in preds_lang_a]
    norm_b = [normalize_eval_label(p) for p in preds_lang_b]

    for pa, pb in zip(norm_a, norm_b):
        if pa != pb:
            flips += 1

    flip_rate = flips / len(norm_a)
    return {
        "flip_rate": round(flip_rate, 4),
        "flips": flips,
        "total": len(norm_a),
    }
