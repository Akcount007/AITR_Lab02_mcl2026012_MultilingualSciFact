"""
Baseline NLI model module for scientific claim verification.

Uses MoritzLaurer/mDeBERTa-v3-base-mnli-xnli as the backbone.
This is a multilingual DeBERTa-v3 model fine-tuned on MNLI + XNLI,
supporting 100+ languages including English, Hindi, and Spanish.

MODEL CONFIGURATION (VERIFIED 2026-11-04):
  id2label: {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
  label2id: {'contradiction': 2, 'entailment': 0, 'neutral': 1}
  model_type: deberta-v2
  num_labels: 3

LABEL MAPPING RATIONALE:
  NLI standard      ->  SciFact canonical
  -----------------------------------------
  entailment         ->  SUPPORTS
  contradiction      ->  REFUTES
  neutral            ->  NOT_ENOUGH_INFO

  SciFact raw uses "CONTRADICT"; we output "REFUTES" for consistency
  with the experimental protocol's 3-label taxonomy. The data_loader
  module handles the CONTRADICT <-> REFUTE bidirectional mapping
  when comparing to gold labels.

ASSUMPTIONS:
  1. The NLI premise-hypothesis framing maps scientific evidence
     (premise) against the claim (hypothesis). This is the standard
     approach in fact-verification literature (Wadden et al., 2020).
  2. The model operates on concatenated evidence sentences, not
     the full abstract. When gold rationale sentences are available,
     only those are used as evidence.
  3. Max sequence length is 512 tokens (mDeBERTa default). Inputs
     exceeding this are truncated from the right.
  4. The model runs in eval mode with no gradient computation.
  5. Confidence is the softmax probability of the predicted class.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
MAX_SEQ_LENGTH = 512

# Verified mapping from model output labels to SciFact task labels.
# Built by inspecting model.config.id2label (see module docstring).
NLI_TO_SCIFACT = {
    "entailment": "SUPPORTS",
    "neutral": "NOT_ENOUGH_INFO",
    "contradiction": "REFUTES",
}

VALID_SCIFACT_LABELS = {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class BaselineNLIModel:
    """
    Wraps a Hugging Face NLI model for scientific claim verification.

    Usage:
        model = BaselineNLIModel()
        model.load()
        result = model.predict(claim="...", evidence="...")
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._nli_to_scifact: Dict[str, str] = {}
        self._id2label: Dict[int, str] = {}

    def load(self, local_files_only: bool = False) -> Dict[str, str]:
        """
        Load model and tokenizer.  Inspect and log the actual label mapping.
        Tries local_files_only first to avoid network timeout delays when cached.
        Returns the verified NLI -> SciFact label mapping.
        """
        logger.info("Loading tokenizer: %s", self.model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=local_files_only)

        logger.info("Loading model: %s", self.model_name)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=True)
        except Exception:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=local_files_only)
        self.model.to(self.device)
        self.model.eval()

        # ---- Inspect actual label config ----
        self._id2label = dict(self.model.config.id2label)
        logger.info("Model id2label (from config): %s", self._id2label)

        # Build verified mapping
        self._nli_to_scifact = {}
        for idx, nli_label in self._id2label.items():
            nli_label_lower = nli_label.lower().strip()
            if nli_label_lower in NLI_TO_SCIFACT:
                self._nli_to_scifact[nli_label_lower] = NLI_TO_SCIFACT[nli_label_lower]
            else:
                logger.warning(
                    "Unexpected model label '%s' (id=%s) — not in NLI_TO_SCIFACT map. "
                    "Predictions with this label will be marked UNMAPPED.",
                    nli_label, idx,
                )

        logger.info("NLI -> SciFact mapping: %s", self._nli_to_scifact)

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Model loaded: %s | Parameters: %s | Device: %s",
            self.model_name, f"{param_count:,}", self.device,
        )

        return dict(self._nli_to_scifact)

    def _check_loaded(self):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

    def normalize_label(self, nli_label: str) -> str:
        """Map an NLI label string to a SciFact label."""
        mapped = self._nli_to_scifact.get(nli_label.lower().strip())
        if mapped is None:
            logger.warning("Label '%s' has no SciFact mapping — returning UNMAPPED", nli_label)
            return "UNMAPPED"
        return mapped

    def predict(
        self,
        claim: str,
        evidence: str,
    ) -> Dict[str, Any]:
        """
        Predict veracity of a scientific claim given evidence text.

        Args:
            claim: The scientific claim (hypothesis in NLI terms).
            evidence: The evidence text (premise in NLI terms).
                      Should be one or more concatenated evidence sentences.

        Returns:
            Dict with keys:
                predicted_label: str  (SUPPORTS | REFUTES | NOT_ENOUGH_INFO | UNMAPPED)
                raw_nli_label: str    (entailment | neutral | contradiction)
                confidence: float     (softmax probability of predicted class)
                all_scores: dict      (SciFact label -> probability)
                truncated: bool       (whether input was truncated)
                claim_len: int        (character length of claim)
                evidence_len: int     (character length of evidence)
        """
        self._check_loaded()

        # --- Edge-case handling ---
        if not claim or not claim.strip():
            logger.warning("Empty claim received.")
            return self._empty_prediction(claim, evidence, reason="empty_claim")

        if not evidence or not evidence.strip():
            logger.warning("Empty evidence received — defaulting to NOT_ENOUGH_INFO.")
            return self._empty_prediction(claim, evidence, reason="empty_evidence")

        # --- Tokenize ---
        # NLI convention: premise = evidence, hypothesis = claim
        inputs = self.tokenizer(
            evidence,
            claim,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
            padding=True,
        )
        truncated = inputs["input_ids"].shape[1] >= MAX_SEQ_LENGTH
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # --- Inference ---
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]  # shape: (num_labels,)
            probs = F.softmax(logits, dim=-1)

        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

        raw_nli_label = self._id2label[pred_idx].lower().strip()
        predicted_label = self.normalize_label(raw_nli_label)

        # Build per-label score dict
        all_scores = {}
        for idx, nli_lbl in self._id2label.items():
            scifact_lbl = self._nli_to_scifact.get(nli_lbl.lower().strip(), nli_lbl)
            all_scores[scifact_lbl] = round(probs[int(idx)].item(), 6)

        return {
            "predicted_label": predicted_label,
            "raw_nli_label": raw_nli_label,
            "confidence": round(confidence, 6),
            "all_scores": all_scores,
            "truncated": truncated,
            "claim_len": len(claim),
            "evidence_len": len(evidence),
        }

    def predict_batch(
        self,
        claims: List[str],
        evidences: List[str],
        batch_size: int = 16,
    ) -> List[Dict[str, Any]]:
        """Predict for a batch of claim-evidence pairs."""
        self._check_loaded()
        results = []
        for i in range(0, len(claims), batch_size):
            batch_claims = claims[i : i + batch_size]
            batch_evidences = evidences[i : i + batch_size]
            for c, e in zip(batch_claims, batch_evidences):
                results.append(self.predict(c, e))
        return results

    def _empty_prediction(
        self, claim: str, evidence: str, reason: str
    ) -> Dict[str, Any]:
        """Return a default prediction for degenerate inputs."""
        return {
            "predicted_label": "NOT_ENOUGH_INFO",
            "raw_nli_label": "N/A",
            "confidence": 0.0,
            "all_scores": {lbl: 0.0 for lbl in VALID_SCIFACT_LABELS},
            "truncated": False,
            "claim_len": len(claim) if claim else 0,
            "evidence_len": len(evidence) if evidence else 0,
            "error": reason,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Return a summary dict of model configuration."""
        self._check_loaded()
        return {
            "model_name": self.model_name,
            "model_type": self.model.config.model_type,
            "num_labels": self.model.config.num_labels,
            "id2label": dict(self._id2label),
            "nli_to_scifact_mapping": dict(self._nli_to_scifact),
            "max_seq_length": MAX_SEQ_LENGTH,
            "device": str(self.device),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
        }
