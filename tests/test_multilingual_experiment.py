"""
test_multilingual_experiment.py — Unit tests for multilingual experimental pipeline,
label normalization, JSON parsing, error classification, and cross-lingual metrics.
"""

import sys
from pathlib import Path
import pytest

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import (
    normalize_eval_label,
    evaluate_veracity_predictions,
    evaluate_rationale_selection,
    compute_cross_lingual_flip_rate,
)
from src.multilingual_experiment import (
    parse_model_json_response,
    classify_error,
)


class TestLabelNormalization:
    """Verify that normalize_eval_label accurately maps all standard and model-specific variants."""

    def test_canonical_labels(self):
        assert normalize_eval_label("SUPPORTS") == "SUPPORTS"
        assert normalize_eval_label("REFUTES") == "REFUTES"
        assert normalize_eval_label("NOT_ENOUGH_INFO") == "NOT_ENOUGH_INFO"

    def test_synonyms_and_nli_labels(self):
        assert normalize_eval_label("SUPPORT") == "SUPPORTS"
        assert normalize_eval_label("ENTAILMENT") == "SUPPORTS"
        assert normalize_eval_label("CONTRADICT") == "REFUTES"
        assert normalize_eval_label("CONTRADICTION") == "REFUTES"
        assert normalize_eval_label("NEUTRAL") == "NOT_ENOUGH_INFO"
        assert normalize_eval_label("NEI") == "NOT_ENOUGH_INFO"
        assert normalize_eval_label("NO_INFO") == "NOT_ENOUGH_INFO"

    def test_case_and_formatting_variants(self):
        assert normalize_eval_label("  supports  ") == "SUPPORTS"
        assert normalize_eval_label("Refute") == "REFUTES"
        assert normalize_eval_label("not_enough_info") == "NOT_ENOUGH_INFO"
        assert normalize_eval_label("NOT ENOUGH INFO") == "NOT_ENOUGH_INFO"
        # Crucial empirical fix: Qwen camelCase outputs
        assert normalize_eval_label("NOTEnoughInfo") == "NOT_ENOUGH_INFO"
        assert normalize_eval_label("notenoughinfo") == "NOT_ENOUGH_INFO"

    def test_invalid_and_edge_cases(self):
        assert normalize_eval_label("UNKNOWN") == "INVALID_OUTPUT"
        assert normalize_eval_label("MAYBE") == "INVALID_OUTPUT"
        assert normalize_eval_label("") == "INVALID_OUTPUT"
        assert normalize_eval_label(None) == "INVALID_OUTPUT"
        assert normalize_eval_label(123) == "INVALID_OUTPUT"


class TestModelOutputParser:
    """Verify two-pass parser robustness against genuine LLM outputs."""

    def test_clean_mode_a_json(self):
        raw = '{"veracity": "SUPPORT", "confidence": 0.85}'
        res = parse_model_json_response(raw, mode="A")
        assert res["veracity"] == "SUPPORTS"
        assert res["confidence"] == 0.85
        assert res["parsing_status"] == "JSON_SUCCESS"

    def test_qwen_camelcase_json(self):
        raw = '{"veracity": "NOTEnoughInfo", "confidence": 1.0}'
        res = parse_model_json_response(raw, mode="A")
        assert res["veracity"] == "NOT_ENOUGH_INFO"
        assert res["confidence"] == 1.0
        assert res["parsing_status"] == "JSON_SUCCESS"

    def test_clean_mode_b_json(self):
        raw = '{"veracity": "CONTRADICT", "rationale_sentences": [1, 3], "confidence": 0.92}'
        res = parse_model_json_response(raw, mode="B")
        assert res["veracity"] == "REFUTES"
        assert res["rationale_sentences"] == [1, 3]
        assert res["confidence"] == 0.92
        assert res["parsing_status"] == "JSON_SUCCESS"

    def test_markdown_codeblock_wrapped_json(self):
        raw = '```json\n{"veracity": "SUPPORT", "rationale_sentences": [0], "confidence": 0.75}\n```'
        res = parse_model_json_response(raw, mode="B")
        assert res["veracity"] == "SUPPORTS"
        assert res["rationale_sentences"] == [0]
        assert res["confidence"] == 0.75
        assert res["parsing_status"] == "JSON_SUCCESS"

    def test_truncated_json_regex_recovery(self):
        # Model truncated midway through rationale list (observed in pilot)
        raw = '{"veracity": "NOTEnoughInfo", "rationales": ["The claim does not provide sufficient info'
        res = parse_model_json_response(raw, mode="B")
        assert res["veracity"] == "NOT_ENOUGH_INFO"
        assert res["parsing_status"] == "REGEX_FALLBACK"

    def test_malformed_text_penalized(self):
        raw = "I think the claim is probably true but not 100% sure without reading more."
        res = parse_model_json_response(raw, mode="B")
        assert res["veracity"] == "INVALID_OUTPUT"
        assert res["parsing_status"] == "PARSING_FAILED"
        assert res["confidence"] == 0.0


class TestErrorTaxonomy:
    """Verify error categorization according to experiment_protocol.md."""

    def test_exact_match(self):
        err = classify_error("SUPPORTS", "SUPPORTS", [2, 5], [2, 5], mode="B")
        assert err == "NONE"

    def test_polarity_inversion(self):
        err = classify_error("SUPPORTS", "REFUTES", [1], [1], mode="B")
        assert err == "ERR_POL_INVERT"
        err2 = classify_error("REFUTES", "SUPPORTS", [1], [1], mode="A")
        assert err2 == "ERR_POL_INVERT"

    def test_nei_failure(self):
        err = classify_error("NOT_ENOUGH_INFO", "SUPPORTS", [], [], mode="A")
        assert err == "ERR_NEI_FAIL"

    def test_rationale_omission(self):
        err = classify_error("SUPPORTS", "SUPPORTS", [2, 5], [], mode="B")
        assert err == "ERR_RAT_OMIT"

    def test_rationale_spurious(self):
        err = classify_error("NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO", [], [1, 2], mode="B")
        assert err == "ERR_RAT_SPURIOUS"


class TestCrossLingualMetrics:
    """Verify cross-lingual flip rate and veracity evaluation."""

    def test_cross_lingual_flip_rate(self):
        preds_en = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        preds_es = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        # Perfectly identical
        cfr = compute_cross_lingual_flip_rate(preds_en, preds_es)
        assert cfr["flip_rate"] == 0.0
        assert cfr["flips"] == 0

        # Divergent on 1 item
        preds_hi = ["SUPPORTS", "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO"]
        cfr_hi = compute_cross_lingual_flip_rate(preds_en, preds_hi)
        assert cfr_hi["flip_rate"] == pytest.approx(1 / 3, 0.01)
        assert cfr_hi["flips"] == 1
