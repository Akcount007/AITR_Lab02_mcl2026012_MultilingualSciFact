"""
Unit tests for data_loader module and SciFact dataset integrity.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_jsonl, load_corpus, load_scifact_split, parse_claim_evidence, normalize_label


class TestSciFactDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(PROJECT_ROOT, "data")
        cls.corpus_path = os.path.join(cls.data_dir, "corpus.jsonl")
        cls.dev_path = os.path.join(cls.data_dir, "claims_dev.jsonl")
        cls.test_path = os.path.join(cls.data_dir, "claims_test.jsonl")

    def test_corpus_exists_and_loads(self):
        self.assertTrue(os.path.exists(self.corpus_path), "corpus.jsonl does not exist")
        corpus = load_corpus(self.corpus_path)
        self.assertEqual(len(corpus), 5183, f"Expected 5,183 docs, got {len(corpus)}")
        first_doc = next(iter(corpus.values()))
        self.assertIn("doc_id", first_doc)
        self.assertIn("title", first_doc)
        self.assertIn("abstract", first_doc)
        self.assertIsInstance(first_doc["abstract"], list)

    def test_dev_split_loading_and_labels(self):
        self.assertTrue(os.path.exists(self.dev_path), "claims_dev.jsonl does not exist")
        dev_claims = load_jsonl(self.dev_path)
        self.assertEqual(len(dev_claims), 300, f"Expected 300 dev claims, got {len(dev_claims)}")

        # Check parsing of evidence
        sample_support = [c for c in dev_claims if c["id"] == 3][0]
        parsed = parse_claim_evidence(sample_support)
        self.assertTrue(len(parsed) >= 1)
        self.assertEqual(parsed[0]["label"], "SUPPORT")
        self.assertEqual(parsed[0]["doc_id"], 14717500)
        self.assertIn(2, parsed[0]["rationale_sentences"])

    def test_nei_parsing(self):
        dev_claims = load_jsonl(self.dev_path)
        sample_nei = [c for c in dev_claims if c["id"] == 1][0]
        parsed = parse_claim_evidence(sample_nei)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["label"], "NOT_ENOUGH_INFO")
        self.assertEqual(parsed[0]["rationale_sentences"], [])

    def test_label_normalization(self):
        self.assertEqual(normalize_label("SUPPORT"), "SUPPORT")
        self.assertEqual(normalize_label("CONTRADICT"), "CONTRADICT")
        self.assertEqual(normalize_label("refute"), "CONTRADICT")
        self.assertEqual(normalize_label("NEI"), "NOT_ENOUGH_INFO")

    def test_test_split_blindness(self):
        self.assertTrue(os.path.exists(self.test_path), "claims_test.jsonl does not exist")
        test_claims = load_jsonl(self.test_path)
        self.assertEqual(len(test_claims), 300, f"Expected 300 test claims, got {len(test_claims)}")
        for c in test_claims:
            self.assertNotIn("evidence", c, "Test split contains leaked evidence field!")
            self.assertNotIn("cited_doc_ids", c, "Test split contains leaked cited_doc_ids field!")


if __name__ == "__main__":
    unittest.main()
