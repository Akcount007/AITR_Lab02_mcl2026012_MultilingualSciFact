"""
Data loader module for SciFact scientific claim verification dataset.
Provides functions to load claims (train, dev, test) and corpus abstracts,
and to construct clean, evaluatable claim-abstract pairs.

SCHEMA ADAPTATION NOTE:
In the raw SciFact release files (claims_train.jsonl, claims_dev.jsonl),
the negative evidence label is represented as "CONTRADICT" in the JSON entries,
whereas published papers frequently refer to it as "REFUTE".
This module standardizes the internal representation to both formats:
canonical label "CONTRADICT" (raw) with alias support for "REFUTE".
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple, Set


LABEL_MAPPING = {
    "SUPPORT": "SUPPORT",
    "SUPPORTS": "SUPPORT",
    "CONTRADICT": "CONTRADICT",
    "CONTRADICTS": "CONTRADICT",
    "REFUTE": "CONTRADICT",
    "REFUTES": "CONTRADICT",
    "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
    "NEI": "NOT_ENOUGH_INFO",
    "NO_INFO": "NOT_ENOUGH_INFO"
}


def normalize_label(label: str) -> str:
    """Normalize label strings to canonical form ('SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO')."""
    cleaned = label.strip().upper()
    return LABEL_MAPPING.get(cleaned, cleaned)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Error parsing JSON on line {line_num} in {filepath}: {e}")
    return records


def load_corpus(corpus_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Load the SciFact corpus JSONL file.
    Returns a dictionary mapping doc_id (int) -> document dict with keys:
      - doc_id: int
      - title: str
      - abstract: List[str] (sentences)
      - structured: bool
    """
    corpus_list = load_jsonl(corpus_path)
    corpus = {}
    for doc in corpus_list:
        corpus[doc['doc_id']] = doc
    return corpus


def parse_claim_evidence(claim_record: Dict[str, Any], canonical_negative_label: str = "CONTRADICT") -> List[Dict[str, Any]]:
    """
    Parse the evidence field of a claim record into a standardized list of evidence entries.
    
    In SciFact schema:
      - "evidence": { "<doc_id>": [ {"sentences": [1, 2], "label": "SUPPORT" | "CONTRADICT"}, ... ] }
      - If "evidence" is empty {}:
          - If cited_doc_ids exists, each cited doc represents a NOT_ENOUGH_INFO instance.
          - If no cited_doc_ids, it is ungrounded / NOT_ENOUGH_INFO.
    
    Returns a list of standardized dicts:
      {
        "claim_id": int,
        "doc_id": int,
        "label": str ("SUPPORT" | "CONTRADICT" | "NOT_ENOUGH_INFO"),
        "rationale_sentences": List[int] (0-indexed sentence indices)
      }
    """
    claim_id = claim_record["id"]
    evidence_dict = claim_record.get("evidence", {})
    cited_doc_ids = claim_record.get("cited_doc_ids", [])
    
    parsed_entries = []
    
    if evidence_dict:
        for doc_id_str, ev_list in evidence_dict.items():
            doc_id = int(doc_id_str)
            for ev in ev_list:
                raw_label = ev.get("label", "NOT_ENOUGH_INFO")
                norm_label = normalize_label(raw_label)
                if canonical_negative_label == "REFUTE" and norm_label == "CONTRADICT":
                    norm_label = "REFUTE"
                elif canonical_negative_label == "CONTRADICT" and norm_label == "REFUTE":
                    norm_label = "CONTRADICT"
                    
                parsed_entries.append({
                    "claim_id": claim_id,
                    "doc_id": doc_id,
                    "label": norm_label,
                    "rationale_sentences": sorted(ev.get("sentences", []))
                })
    
    # Also capture cited documents that have no evidence (NOT_ENOUGH_INFO)
    evidence_doc_ids = {int(k) for k in evidence_dict.keys()}
    for cited_doc in cited_doc_ids:
        if cited_doc not in evidence_doc_ids:
            parsed_entries.append({
                "claim_id": claim_id,
                "doc_id": cited_doc,
                "label": "NOT_ENOUGH_INFO",
                "rationale_sentences": []
            })
            
    # If both evidence and cited_doc_ids are empty, record as global NOT_ENOUGH_INFO
    if not parsed_entries:
        parsed_entries.append({
            "claim_id": claim_id,
            "doc_id": None,
            "label": "NOT_ENOUGH_INFO",
            "rationale_sentences": []
        })
        
    return parsed_entries


def load_scifact_split(claims_path: str, corpus: Optional[Dict[int, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Load a SciFact claims split and optionally join with corpus abstracts.
    Each returned item contains:
      - id: claim ID
      - claim: English claim text
      - cited_doc_ids: list of cited document IDs (if present)
      - evidence_raw: raw evidence dictionary (if present)
      - parsed_evidence: standardized evidence entries (if present)
      - matched_docs: list of matched corpus documents (if corpus provided)
    """
    claims = load_jsonl(claims_path)
    enriched = []
    for c in claims:
        item = {
            "id": c["id"],
            "claim": c["claim"]
        }
        if "cited_doc_ids" in c:
            item["cited_doc_ids"] = c["cited_doc_ids"]
        if "evidence" in c:
            item["evidence_raw"] = c["evidence"]
            item["parsed_evidence"] = parse_claim_evidence(c)
            
        if corpus is not None and "cited_doc_ids" in c:
            item["matched_docs"] = [corpus[doc_id] for doc_id in c["cited_doc_ids"] if doc_id in corpus]
            
        enriched.append(item)
    return enriched
