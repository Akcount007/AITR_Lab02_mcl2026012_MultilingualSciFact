"""
SciFact Multilingual Scientific Claim Verification Evaluation Package.
"""

from .data_loader import load_jsonl, load_corpus, load_scifact_split, parse_claim_evidence

__all__ = ["load_jsonl", "load_corpus", "load_scifact_split", "parse_claim_evidence"]
