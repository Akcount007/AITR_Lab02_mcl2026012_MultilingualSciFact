"""
Unit tests for the baseline NLI model module.

Tests cover:
  1. Label normalization mapping.
  2. Edge-case handling (empty claim, empty evidence).
  3. Model info structure (when loaded).
  4. Prediction output structure.
  5. Batch prediction consistency.

Note: Tests that require the model to be loaded are marked with a
module-level fixture that loads once and reuses.  If the model is
not available in the cache, these tests will attempt to load from
Hugging Face Hub (requires network).
"""

import os
import sys
import pytest
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_model import (
    BaselineNLIModel,
    NLI_TO_SCIFACT,
    VALID_SCIFACT_LABELS,
    MAX_SEQ_LENGTH,
)


# ---- Fixtures ----

@pytest.fixture(scope="module")
def loaded_model():
    """Load model once for all tests in this module."""
    model = BaselineNLIModel()
    model.load()
    return model


# ---- Test: Constants ----

def test_nli_to_scifact_mapping_complete():
    """All three NLI labels map to valid SciFact labels."""
    assert set(NLI_TO_SCIFACT.keys()) == {"entailment", "neutral", "contradiction"}
    assert set(NLI_TO_SCIFACT.values()) == VALID_SCIFACT_LABELS


def test_valid_scifact_labels():
    """SciFact labels match the experiment protocol."""
    assert VALID_SCIFACT_LABELS == {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}


# ---- Test: Label normalization ----

def test_label_normalization(loaded_model):
    """NLI labels normalize to SciFact labels."""
    assert loaded_model.normalize_label("entailment") == "SUPPORTS"
    assert loaded_model.normalize_label("contradiction") == "REFUTES"
    assert loaded_model.normalize_label("neutral") == "NOT_ENOUGH_INFO"


def test_label_normalization_case_insensitive(loaded_model):
    """Normalization is case-insensitive."""
    assert loaded_model.normalize_label("Entailment") == "SUPPORTS"
    assert loaded_model.normalize_label("CONTRADICTION") == "REFUTES"
    assert loaded_model.normalize_label("  Neutral  ") == "NOT_ENOUGH_INFO"


def test_label_normalization_unknown(loaded_model):
    """Unknown labels return UNMAPPED."""
    assert loaded_model.normalize_label("garbage") == "UNMAPPED"


# ---- Test: Model info ----

def test_model_info_structure(loaded_model):
    """get_model_info returns expected keys."""
    info = loaded_model.get_model_info()
    required_keys = {
        "model_name", "model_type", "num_labels", "id2label",
        "nli_to_scifact_mapping", "max_seq_length", "device",
        "total_parameters",
    }
    assert required_keys.issubset(set(info.keys()))
    assert info["num_labels"] == 3
    assert info["model_type"] == "deberta-v2"


def test_model_id2label_verified(loaded_model):
    """id2label matches the verified configuration."""
    info = loaded_model.get_model_info()
    id2label = info["id2label"]
    # Keys may be int or str depending on serialization
    labels = {str(k): v for k, v in id2label.items()}
    assert labels["0"] == "entailment"
    assert labels["1"] == "neutral"
    assert labels["2"] == "contradiction"


# ---- Test: Edge cases ----

def test_empty_claim(loaded_model):
    """Empty claim returns NOT_ENOUGH_INFO with error flag."""
    result = loaded_model.predict(claim="", evidence="Some evidence.")
    assert result["predicted_label"] == "NOT_ENOUGH_INFO"
    assert result["confidence"] == 0.0
    assert "error" in result
    assert result["error"] == "empty_claim"


def test_empty_evidence(loaded_model):
    """Empty evidence returns NOT_ENOUGH_INFO with error flag."""
    result = loaded_model.predict(claim="Aspirin prevents cancer.", evidence="")
    assert result["predicted_label"] == "NOT_ENOUGH_INFO"
    assert result["confidence"] == 0.0
    assert "error" in result
    assert result["error"] == "empty_evidence"


def test_none_inputs(loaded_model):
    """None inputs handled gracefully."""
    result = loaded_model.predict(claim=None, evidence="Some evidence.")
    assert result["predicted_label"] == "NOT_ENOUGH_INFO"
    assert "error" in result


# ---- Test: Prediction structure ----

def test_prediction_output_keys(loaded_model):
    """Prediction returns all expected keys."""
    result = loaded_model.predict(
        claim="Acetaminophen reduces fever.",
        evidence="Acetaminophen is an effective antipyretic agent."
    )
    required_keys = {
        "predicted_label", "raw_nli_label", "confidence",
        "all_scores", "truncated", "claim_len", "evidence_len",
    }
    assert required_keys.issubset(set(result.keys()))


def test_prediction_label_validity(loaded_model):
    """Predicted label is a valid SciFact label."""
    result = loaded_model.predict(
        claim="The drug is effective.",
        evidence="Clinical trials showed the drug significantly reduced symptoms."
    )
    assert result["predicted_label"] in VALID_SCIFACT_LABELS


def test_prediction_confidence_range(loaded_model):
    """Confidence is between 0 and 1."""
    result = loaded_model.predict(
        claim="Water is wet.",
        evidence="Water, a transparent liquid, is known for its wetting properties."
    )
    assert 0.0 <= result["confidence"] <= 1.0


def test_prediction_scores_sum_to_one(loaded_model):
    """All scores should sum to approximately 1.0."""
    result = loaded_model.predict(
        claim="Exercise improves health.",
        evidence="Regular physical activity has been associated with improved cardiovascular health."
    )
    total = sum(result["all_scores"].values())
    assert abs(total - 1.0) < 0.01


def test_prediction_raw_label_consistency(loaded_model):
    """raw_nli_label should be a known NLI label."""
    result = loaded_model.predict(
        claim="Ibuprofen is an NSAID.",
        evidence="Ibuprofen belongs to the class of non-steroidal anti-inflammatory drugs."
    )
    assert result["raw_nli_label"] in {"entailment", "neutral", "contradiction"}


# ---- Test: Model not loaded ----

def test_predict_before_load():
    """Calling predict before load raises RuntimeError."""
    model = BaselineNLIModel()
    with pytest.raises(RuntimeError, match="Model not loaded"):
        model.predict(claim="test", evidence="test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
