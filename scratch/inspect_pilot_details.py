import sys
import os
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")

import json
from src.data_loader import load_corpus, load_jsonl

corpus = load_corpus("data/corpus.jsonl")
claims = load_jsonl("data/claims_dev_multilingual_pilot.jsonl")

for c in claims[:3]:
    cid = c["id"]
    print(f"=== CLAIM {cid} ===")
    print("EN:", c["claim_en"])
    print("ES:", c["translations"]["es"])
    print("HI:", c["translations"]["hi"])
    print("Cited docs:", c.get("cited_doc_ids"))
    print("Evidence dict:", c.get("evidence"))
    for doc_id in c.get("cited_doc_ids", []):
        doc = corpus.get(doc_id)
        if doc:
            print(f"Doc {doc_id} Title: {doc['title']}")
            print("Abstract sents:")
            for idx, s in enumerate(doc["abstract"]):
                print(f"  [{idx}] {s}")
    print()
