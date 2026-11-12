"""
test_evaluation_audit.py — Rigorous unit tests for label mapping,
prediction evaluation, data integrity, and split contamination.
"""

import json
import sys
from pathlib import Path
import pytest

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import normalize_label, parse_claim_evidence, load_corpus, load_scifact_split
from src.baseline_model import NLI_TO_SCIFACT, VALID_SCIFACT_LABELS
from src.metrics import (
    normalize_eval_label,
    evaluate_veracity_predictions,
    evaluate_rationale_selection,
    compute_cross_lingual_flip_rate,
    CANONICAL_LABELS,
)


# ===========================================================================
# 1. Label Mapping Tests
# ===========================================================================

class TestLabelMappingAudit:
    """Audit label normalization and alignment across schemas."""

    def test_nli_to_scifact_canonical(self):
        """Verify NLI standard labels map strictly to 3 canonical SciFact classes."""
        assert NLI_TO_SCIFACT["entailment"] == "SUPPORTS"
        assert NLI_TO_SCIFACT["neutral"] == "NOT_ENOUGH_INFO"
        assert NLI_TO_SCIFACT["contradiction"] == "REFUTES"

    def test_normalize_eval_label_variants(self):
        """All known variants map to the canonical 3-class set."""
        assert normalize_eval_label("SUPPORT") == "SUPPORTS"
        assert normalize_eval_label("SUPPORTS") == "SUPPORTS"
        assert normalize_eval_label("entailment") == "SUPPORTS"

        assert normalize_eval_label("CONTRADICT") == "REFUTES"
        assert normalize_eval_label("CONTRADICTS") == "REFUTES"
        assert normalize_eval_label("REFUTE") == "REFUTES"
        assert normalize_eval_label("REFUTES") == "REFUTES"
        assert normalize_eval_label("contradiction") == "REFUTES"

        assert normalize_eval_label("NOT_ENOUGH_INFO") == "NOT_ENOUGH_INFO"
        assert normalize_eval_label("NEI") == "NOT_ENOUGH_INFO"
        assert normalize_eval_label("NO_INFO") == "NOT_ENOUGH_INFO"
        assert normalize_eval_label("neutral") == "NOT_ENOUGH_INFO"

    def test_invalid_label_produces_invalid_output(self):
        """Garbage or unparseable labels return INVALID_OUTPUT."""
        assert normalize_eval_label("UNKNOWN") == "INVALID_OUTPUT"
        assert normalize_eval_label("MAYBE") == "INVALID_OUTPUT"
        assert normalize_eval_label("") == "INVALID_OUTPUT"
        assert normalize_eval_label("Random text from LLM hallucination") == "INVALID_OUTPUT"


# ===========================================================================
# 2. Prediction Evaluation Metric Tests
# ===========================================================================

class TestPredictionEvaluationAudit:
    """Audit metric calculation accuracy, edge cases, and penalty behavior."""

    def test_perfect_predictions(self):
        """When predictions equal gold, all metrics must be 1.0."""
        gold = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        pred = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        res = evaluate_veracity_predictions(gold, pred)

        assert res["accuracy"] == 1.0
        assert res["macro_f1"] == 1.0
        assert res["macro_precision"] == 1.0
        assert res["macro_recall"] == 1.0
        assert res["invalid_predictions"] == 0

    def test_completely_wrong_predictions(self):
        """When predictions are systematically inverted."""
        gold = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        pred = ["REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS"]
        res = evaluate_veracity_predictions(gold, pred)

        assert res["accuracy"] == 0.0
        assert res["macro_f1"] == 0.0
        assert res["macro_precision"] == 0.0
        assert res["macro_recall"] == 0.0

    def test_invalid_predictions_penalized_never_dropped(self):
        """
        Critical Protocol Check:
        Invalid predictions must NOT be dropped; they must be penalized.
        """
        gold = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS"]
        pred = ["SUPPORTS", "REFUTES", "MALFORMED_OUTPUT", "ANOTHER_ERROR"]
        res = evaluate_veracity_predictions(gold, pred)

        assert res["total_samples"] == 4
        assert res["invalid_predictions"] == 2
        # Correct = 2 out of 4 = 0.50
        assert res["accuracy"] == 0.50
        assert res["per_class"]["NOT_ENOUGH_INFO"]["recall"] == 0.0

    def test_empty_input_handling(self):
        """Gracefully handle empty gold/pred lists without crashing."""
        res = evaluate_veracity_predictions([], [])
        assert res["total_samples"] == 0
        assert res["accuracy"] == 0.0
        assert res["macro_f1"] == 0.0

    def test_length_mismatch_raises_value_error(self):
        """Mismatched vector lengths raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate_veracity_predictions(["SUPPORTS"], ["SUPPORTS", "REFUTES"])

    def test_confusion_matrix_dimensions(self):
        """Confusion matrix must be 3x3 for the 3 canonical classes."""
        gold = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        pred = ["SUPPORTS", "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO"]
        res = evaluate_veracity_predictions(gold, pred)

        cm = res["confusion_matrix"]
        assert len(cm) == 3
        assert len(cm[0]) == 3
        assert len(cm[1]) == 3
        assert len(cm[2]) == 3


# ===========================================================================
# 3. Rationale Selection & Flip Rate Tests
# ===========================================================================

class TestRationaleAndFlipRateAudit:
    """Audit rationale sentence overlap and cross-lingual flip rate."""

    def test_rationale_exact_match(self):
        """Exact sentence index match yields F1 = 1.0."""
        res = evaluate_rationale_selection([1, 3], [1, 3])
        assert res["precision"] == 1.0
        assert res["recall"] == 1.0
        assert res["f1"] == 1.0

    def test_rationale_partial_overlap(self):
        """Gold [1, 2], Pred [2, 3] -> P=0.5, R=0.5, F1=0.5."""
        res = evaluate_rationale_selection([1, 2], [2, 3])
        assert res["precision"] == 0.5
        assert res["recall"] == 0.5
        assert res["f1"] == 0.5

    def test_rationale_empty_both(self):
        """Both empty is a valid match (e.g. for NOT_ENOUGH_INFO)."""
        res = evaluate_rationale_selection([], [])
        assert res["precision"] == 1.0
        assert res["recall"] == 1.0
        assert res["f1"] == 1.0

    def test_cross_lingual_flip_rate_zero(self):
        """Identical predictions between languages yield CFR = 0.0."""
        en = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        hi = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        res = compute_cross_lingual_flip_rate(en, hi)
        assert res["flip_rate"] == 0.0
        assert res["flips"] == 0

    def test_cross_lingual_flip_rate_partial(self):
        """1 flip out of 4 predictions yields CFR = 0.25."""
        en = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "SUPPORTS"]
        hi = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "REFUTES"]
        res = compute_cross_lingual_flip_rate(en, hi)
        assert res["flip_rate"] == 0.25
        assert res["flips"] == 1


# ===========================================================================
# 4. Data Integrity and Split Contamination Audit
# ===========================================================================

class TestDataIntegrityAndContaminationAudit:
    """Verify official dataset split integrity and absence of contamination."""

    @pytest.fixture(scope="class")
    def dataset_splits(self):
        data_dir = PROJECT_ROOT / "data"
        train = [json.loads(line) for line in open(data_dir / "claims_train.jsonl", encoding="utf-8")]
        dev = [json.loads(line) for line in open(data_dir / "claims_dev.jsonl", encoding="utf-8")]
        test = [json.loads(line) for line in open(data_dir / "claims_test.jsonl", encoding="utf-8")]
        corpus = {d["doc_id"]: d for d in [json.loads(line) for line in open(data_dir / "corpus.jsonl", encoding="utf-8")]}
        return {"train": train, "dev": dev, "test": test, "corpus": corpus}

    def test_zero_dev_test_id_leakage(self, dataset_splits):
        """Zero claim ID overlap between dev and test sets."""
        dev_ids = {c["id"] for c in dataset_splits["dev"]}
        test_ids = {c["id"] for c in dataset_splits["test"]}
        assert len(dev_ids.intersection(test_ids)) == 0

    def test_zero_train_test_id_leakage(self, dataset_splits):
        """Zero claim ID overlap between train and test sets."""
        train_ids = {c["id"] for c in dataset_splits["train"]}
        test_ids = {c["id"] for c in dataset_splits["test"]}
        assert len(train_ids.intersection(test_ids)) == 0

    def test_zero_train_dev_id_leakage(self, dataset_splits):
        """Zero claim ID overlap between train and dev sets."""
        train_ids = {c["id"] for c in dataset_splits["train"]}
        dev_ids = {c["id"] for c in dataset_splits["dev"]}
        assert len(train_ids.intersection(dev_ids)) == 0

    def test_dev_no_duplicate_ids(self, dataset_splits):
        """Every claim ID in dev set must be unique."""
        dev_ids = [c["id"] for c in dataset_splits["dev"]]
        assert len(dev_ids) == len(set(dev_ids))

    def test_evidence_sentence_bounds(self, dataset_splits):
        """All evidence sentence pointers must be strictly within bounds."""
        corpus = dataset_splits["corpus"]
        for c in dataset_splits["dev"]:
            for doc_id_str, ev_list in c.get("evidence", {}).items():
                doc_id = int(doc_id_str)
                assert doc_id in corpus, f"Doc {doc_id} cited in claim {c['id']} missing from corpus"
                num_sents = len(corpus[doc_id]["abstract"])
                for ev in ev_list:
                    for s_idx in ev.get("sentences", []):
                        assert 0 <= s_idx < num_sents, (
                            f"Claim {c['id']}, doc {doc_id}: sentence index {s_idx} "
                            f"out of bounds [0, {num_sents})"
                        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
