"""
test_evaluation_metrics.py — Unit tests for evaluation module, schema verification,
classification metrics, calibration analysis, runtime profiling, and paired disagreement.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    verify_prediction_schema,
    verify_gold_label_alignment,
    compute_classification_metrics,
    compute_calibration_metrics,
    compute_runtime_metrics,
    compute_paired_disagreements,
    generate_evaluation_tables,
)


@pytest.fixture
def sample_valid_df():
    """Create a minimal valid evaluation DataFrame with 2 claims x 2 languages x 2 modes = 8 rows."""
    data = [
        # Claim 1 (Doc 101, Gold: SUPPORTS)
        {"claim_id": 1, "doc_id": 101, "language": "en", "mode": "A", "gold_label": "SUPPORTS", "predicted_label": "SUPPORTS", "confidence": 0.9, "latency_sec": 12.5, "gold_sentences": "[1]"},
        {"claim_id": 1, "doc_id": 101, "language": "en", "mode": "B", "gold_label": "SUPPORTS", "predicted_label": "SUPPORTS", "confidence": 0.85, "latency_sec": 15.0, "gold_sentences": "[1]"},
        {"claim_id": 1, "doc_id": 101, "language": "es", "mode": "A", "gold_label": "SUPPORTS", "predicted_label": "SUPPORTS", "confidence": 0.88, "latency_sec": 11.0, "gold_sentences": "[1]"},
        {"claim_id": 1, "doc_id": 101, "language": "es", "mode": "B", "gold_label": "SUPPORTS", "predicted_label": "NOT_ENOUGH_INFO", "confidence": 0.4, "latency_sec": 14.2, "gold_sentences": "[1]"},
        # Claim 2 (Doc 102, Gold: NOT_ENOUGH_INFO)
        {"claim_id": 2, "doc_id": 102, "language": "en", "mode": "A", "gold_label": "NOT_ENOUGH_INFO", "predicted_label": "NOT_ENOUGH_INFO", "confidence": 0.95, "latency_sec": 8.0, "gold_sentences": "[]"},
        {"claim_id": 2, "doc_id": 102, "language": "en", "mode": "B", "gold_label": "NOT_ENOUGH_INFO", "predicted_label": "NOT_ENOUGH_INFO", "confidence": 0.92, "latency_sec": 9.5, "gold_sentences": "[]"},
        {"claim_id": 2, "doc_id": 102, "language": "es", "mode": "A", "gold_label": "NOT_ENOUGH_INFO", "predicted_label": "NOT_ENOUGH_INFO", "confidence": 0.94, "latency_sec": 7.8, "gold_sentences": "[]"},
        {"claim_id": 2, "doc_id": 102, "language": "es", "mode": "B", "gold_label": "NOT_ENOUGH_INFO", "predicted_label": "NOT_ENOUGH_INFO", "confidence": 0.91, "latency_sec": 9.2, "gold_sentences": "[]"},
    ]
    return pd.DataFrame(data)


class TestSchemaVerification:
    def test_valid_schema_passes(self, sample_valid_df):
        is_valid, issues = verify_prediction_schema(sample_valid_df)
        assert is_valid is True
        assert len(issues) == 0

    def test_missing_column_fails(self, sample_valid_df):
        bad_df = sample_valid_df.drop(columns=["confidence"])
        is_valid, issues = verify_prediction_schema(bad_df)
        assert is_valid is False
        assert any("confidence" in s for s in issues)

    def test_empty_dataframe_fails(self):
        empty_df = pd.DataFrame(columns=["claim_id", "doc_id", "language", "mode", "gold_label", "predicted_label", "confidence", "latency_sec"])
        is_valid, issues = verify_prediction_schema(empty_df)
        assert is_valid is False
        assert any("empty" in s.lower() for s in issues)

    def test_confidence_out_of_bounds_fails(self, sample_valid_df):
        bad_df = sample_valid_df.copy()
        bad_df.loc[0, "confidence"] = 1.5
        is_valid, issues = verify_prediction_schema(bad_df)
        assert is_valid is False
        assert any("outside [0.0, 1.0]" in s for s in issues)

    def test_negative_latency_fails(self, sample_valid_df):
        bad_df = sample_valid_df.copy()
        bad_df.loc[0, "latency_sec"] = -5.0
        is_valid, issues = verify_prediction_schema(bad_df)
        assert is_valid is False
        assert any("negative" in s for s in issues)

    def test_null_values_fail(self, sample_valid_df):
        bad_df = sample_valid_df.copy()
        bad_df.loc[0, "predicted_label"] = np.nan
        is_valid, issues = verify_prediction_schema(bad_df)
        assert is_valid is False
        assert any("null/NaN" in s for s in issues)


class TestGoldLabelAlignment:
    def test_aligned_labels_pass(self, sample_valid_df):
        is_aligned, issues = verify_gold_label_alignment(sample_valid_df)
        assert is_aligned is True
        assert len(issues) == 0

    def test_misaligned_labels_fail(self, sample_valid_df):
        bad_df = sample_valid_df.copy()
        # Change gold label for es Mode A on Claim 1
        bad_df.loc[2, "gold_label"] = "REFUTES"
        is_aligned, issues = verify_gold_label_alignment(bad_df)
        assert is_aligned is False
        assert any("Gold label mismatch" in s for s in issues)


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        gold = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        pred = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        res = compute_classification_metrics(gold, pred)
        assert res["accuracy"] == 1.0
        assert res["macro_f1"] == 1.0
        assert res["invalid_output_rate"] == 0.0
        assert res["invalid_predictions"] == 0

    def test_all_wrong_predictions(self):
        gold = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        pred = ["REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS"]
        res = compute_classification_metrics(gold, pred)
        assert res["accuracy"] == 0.0
        assert res["macro_f1"] == 0.0

    def test_invalid_output_penalty(self):
        gold = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        pred = ["SUPPORTS", "INVALID_OUTPUT", "UNRECOGNIZED_LABEL"]
        res = compute_classification_metrics(gold, pred)
        # 1 correct out of 3 -> accuracy = 1/3
        assert res["accuracy"] == pytest.approx(1 / 3, 0.001)
        # 2 invalid out of 3
        assert res["invalid_predictions"] == 2
        assert res["invalid_output_rate"] == pytest.approx(2 / 3, 0.001)

    def test_per_class_metrics(self):
        gold = ["SUPPORTS", "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        pred = ["SUPPORTS", "NOT_ENOUGH_INFO", "REFUTES", "NOT_ENOUGH_INFO"]
        res = compute_classification_metrics(gold, pred)
        # SUPPORTS: TP=1, FP=0, FN=1 -> P=1.0, R=0.5, F1=0.6667
        supp = res["per_class"]["SUPPORTS"]
        assert supp["precision"] == 1.0
        assert supp["recall"] == 0.5
        assert supp["f1"] == pytest.approx(2 / 3, 0.01)
        # REFUTES: TP=1, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
        assert res["per_class"]["REFUTES"]["f1"] == 1.0


class TestCalibrationMetrics:
    def test_perfect_calibration(self):
        df = pd.DataFrame({
            "gold_label": ["SUPPORTS", "REFUTES"],
            "predicted_label": ["SUPPORTS", "REFUTES"],
            "confidence": [1.0, 1.0],
        })
        res = compute_calibration_metrics(df, n_bins=2)
        assert res["brier_score"] == 0.0
        assert res["ece"] == 0.0

    def test_imperfect_calibration(self):
        df = pd.DataFrame({
            "gold_label": ["SUPPORTS", "SUPPORTS"],
            "predicted_label": ["SUPPORTS", "REFUTES"],  # 1 correct, 1 wrong
            "confidence": [1.0, 1.0],                   # Overconfident on both
        })
        res = compute_calibration_metrics(df, n_bins=2)
        # Correctness: [1, 0], Conf: [1, 1] -> Brier = (0^2 + 1^2)/2 = 0.5
        assert res["brier_score"] == 0.5
        # Bin [0.5, 1.0]: count=2, acc=0.5, conf=1.0, gap=0.5 -> ECE = 0.5
        assert res["ece"] == 0.5


class TestRuntimeMetrics:
    def test_runtime_statistics(self):
        df = pd.DataFrame({"latency_sec": [10.0, 20.0, 30.0]})
        res = compute_runtime_metrics(df)
        assert res["mean_sec"] == 20.0
        assert res["median_sec"] == 20.0
        assert res["min_sec"] == 10.0
        assert res["max_sec"] == 30.0
        assert res["total_sec"] == 60.0


class TestPairedDisagreement:
    def test_disagreement_cross_lingual_and_prompt(self, sample_valid_df):
        paired_df = compute_paired_disagreements(sample_valid_df)
        assert len(paired_df) > 0
        assert "comparison_type" in paired_df.columns
        assert "disagreement_rate" in paired_df.columns
        assert "cohens_kappa" in paired_df.columns

        # In sample_valid_df, for Mode A: en vs es has identical predictions (0 disagreement)
        sub_mode_a = paired_df[(paired_df["condition_fixed"] == "Mode_A") & (paired_df["pair_name"] == "en_vs_es")]
        assert len(sub_mode_a) == 1
        assert sub_mode_a.iloc[0]["disagreement_rate"] == 0.0
        assert sub_mode_a.iloc[0]["disagreement_count"] == 0

        # For Mode B: on Claim 1, en predicted SUPPORTS, es predicted NOT_ENOUGH_INFO (1 disagreement out of 2)
        sub_mode_b = paired_df[(paired_df["condition_fixed"] == "Mode_B") & (paired_df["pair_name"] == "en_vs_es")]
        assert len(sub_mode_b) == 1
        assert sub_mode_b.iloc[0]["disagreement_count"] == 1
        assert sub_mode_b.iloc[0]["disagreement_rate"] == 0.5


class TestTableGeneration:
    def test_generate_evaluation_tables_columns(self, sample_valid_df):
        tables = generate_evaluation_tables(sample_valid_df)
        assert "metrics_by_language" in tables
        assert "metrics_by_prompt" in tables
        assert "paired_disagreement" in tables

        m_lang = tables["metrics_by_language"]
        for expected_col in ["language", "mode", "accuracy", "macro_f1", "brier_score", "ece"]:
            assert expected_col in m_lang.columns

        m_prompt = tables["metrics_by_prompt"]
        for expected_col in ["mode", "accuracy", "macro_f1", "mean_latency_sec"]:
            assert expected_col in m_prompt.columns
