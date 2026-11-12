#!/usr/bin/env python3
"""
translate_claims.py — Multilingual claim translation pipeline for SciFact.

Translates SciFact dev claims from English into Spanish (es) and Hindi (hi)
using verified translation models with automated quality verification.

Features:
  - Exact preservation of claim ID, cited documents, and gold evidence.
  - Automated numerical value preservation audit.
  - Automated negation polarity preservation audit.
  - Full translation provenance logged per record.
  - Pilot mode via --max-claims.

Usage:
    python scripts/translate_claims.py --max-claims 10  # Pilot run
    python scripts/translate_claims.py                  # Full dev split (300 claims)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_jsonl
from src.translator import ScientificClaimTranslator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("translate_claims")


def run_translation_pipeline(
    input_path: str,
    output_path: str,
    max_claims: int = 0,
    languages: list = ("es", "hi"),
):
    start_time = time.time()
    logger.info("Loading input claims from %s", input_path)
    claims = load_jsonl(input_path)
    logger.info("Loaded %d total claims", len(claims))

    if max_claims > 0:
        claims = claims[:max_claims]
        logger.info("Limiting to first %d claims (pilot mode)", max_claims)

    translator = ScientificClaimTranslator()

    results = []
    audit_stats = {
        lang: {
            "total": 0,
            "numeric_mismatches": 0,
            "negation_mismatches": 0,
            "failures": 0,
        }
        for lang in languages
    }

    logger.info("Starting translation into %s...", languages)

    for i, item in enumerate(claims):
        claim_id = item["id"]
        claim_en = item["claim"]

        rec = {
            "id": claim_id,
            "claim_en": claim_en,
            "cited_doc_ids": item.get("cited_doc_ids", []),
            "evidence": item.get("evidence", {}),
            "translations": {},
            "provenance": {},
            "audit": {},
        }

        for lang in languages:
            audit_stats[lang]["total"] += 1
            t0 = time.time()
            try:
                trans_text = translator.translate(claim_en, target_lang=lang)
                dt = time.time() - t0

                # Run automated consistency checks
                num_audit = translator.audit_numerical_consistency(claim_en, trans_text)
                neg_audit = translator.audit_negation_consistency(claim_en, trans_text, target_lang=lang)

                if not num_audit["numerics_preserved"]:
                    audit_stats[lang]["numeric_mismatches"] += 1
                    logger.warning(
                        "Claim %d [%s]: Missing numerics %s. EN: '%s' -> %s: '%s'",
                        claim_id, lang, num_audit["missing_numerics"], claim_en, lang, trans_text,
                    )

                if not neg_audit["polarity_preserved"]:
                    audit_stats[lang]["negation_mismatches"] += 1
                    logger.warning(
                        "Claim %d [%s]: Polarity drift. EN negation: %s, %s negation: %s. EN: '%s' -> %s: '%s'",
                        claim_id, lang, neg_audit["source_has_negation"], lang, neg_audit["target_has_negation"],
                        claim_en, lang, trans_text,
                    )

                rec["translations"][lang] = trans_text
                rec["provenance"][lang] = {
                    "model": f"Helsinki-NLP/opus-mt-en-{lang}",
                    "latency_sec": round(dt, 3),
                    "status": "SUCCESS",
                }
                rec["audit"][lang] = {
                    "numeric": num_audit,
                    "negation": neg_audit,
                }

            except Exception as e:
                audit_stats[lang]["failures"] += 1
                logger.error("Claim %d [%s] translation failed: %s", claim_id, lang, e)
                rec["translations"][lang] = ""
                rec["provenance"][lang] = {
                    "model": f"Helsinki-NLP/opus-mt-en-{lang}",
                    "latency_sec": 0.0,
                    "status": f"ERROR: {str(e)}",
                }
                rec["audit"][lang] = {
                    "numeric": {"numerics_preserved": False, "missing_numerics": []},
                    "negation": {"polarity_preserved": False},
                }

        results.append(rec)

        if (i + 1) % 25 == 0 or (i + 1) == len(claims):
            logger.info("Processed %d / %d claims", i + 1, len(claims))

    total_time = time.time() - start_time

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Save to JSONL
    logger.info("Writing %d translated claims to %s", len(results), output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    summary = {
        "input_file": input_path,
        "output_file": output_path,
        "total_claims": len(results),
        "languages": list(languages),
        "total_time_seconds": round(total_time, 2),
        "avg_seconds_per_claim": round(total_time / len(results), 3) if results else 0,
        "audit_stats": audit_stats,
    }

    summary_file = output_path.replace(".jsonl", "_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("TRANSLATION PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info("Total claims: %d", len(results))
    logger.info("Total time: %.2f seconds", total_time)
    for lang in languages:
        st = audit_stats[lang]
        logger.info(
            "[%s] Total: %d | Numeric Mismatches: %d | Polarity Mismatches: %d | Failures: %d",
            lang, st["total"], st["numeric_mismatches"], st["negation_mismatches"], st["failures"]
        )
    logger.info("Summary saved to %s", summary_file)
    logger.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Translate SciFact claims to Spanish and Hindi.")
    parser.add_argument(
        "--input", type=str,
        default=str(PROJECT_ROOT / "data" / "claims_dev.jsonl"),
        help="Input claims JSONL file.",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "data" / "claims_dev_multilingual.jsonl"),
        help="Output multilingual claims JSONL file.",
    )
    parser.add_argument(
        "--max-claims", type=int, default=0,
        help="Limit to first N claims (for pilot testing).",
    )
    args = parser.parse_args()

    run_translation_pipeline(
        input_path=args.input,
        output_path=args.output,
        max_claims=args.max_claims,
    )


if __name__ == "__main__":
    main()
