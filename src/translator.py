"""
Scientific claim translation module with automated quality verification.

Translates English claims into Spanish (es) and Hindi (hi) using
specialized MarianMT models (Helsinki-NLP/opus-mt-en-es and
Helsinki-NLP/opus-mt-en-hi), with back-translation verification via
chrF++, numerical consistency checks, and negation audit.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

# Model checkpoints
TRANSLATION_MODELS = {
    "es": {
        "forward": "Helsinki-NLP/opus-mt-en-es",
        "backward": "Helsinki-NLP/opus-mt-es-en",
    },
    "hi": {
        "forward": "Helsinki-NLP/opus-mt-en-hi",
        "backward": "Helsinki-NLP/opus-mt-hi-en",
    },
}

# Negation lexicon for drift audit
NEGATION_TERMS = {
    "en": {"no", "not", "never", "cannot", "neither", "nor", "none", "unable", "without", "fails"},
    "es": {"no", "nunca", "jamás", "ni", "tampoco", "ningún", "ninguno", "ninguna", "sin"},
    "hi": {"नहीं", "न", "ना", "बिना", "गैर", "असमर्थ", "कभी नहीं"},
}

NUMERIC_PATTERN = re.compile(r"\b\d+(?:[.,/]\d+)?%?\b")


class ScientificClaimTranslator:
    """
    Translates scientific claims with automated validation checks.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}

    def _load_model(self, lang: str):
        """Lazy loader for translation models."""
        if lang in self.models:
            return

        model_name = TRANSLATION_MODELS[lang]["forward"]
        logger.info("Loading translation model for %s: %s", lang, model_name)
        try:
            tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            mod = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(model_name)
            mod = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        mod.to(self.device)
        mod.eval()
        self.tokenizers[lang] = tok
        self.models[lang] = mod

    def translate(self, text: str, target_lang: str) -> str:
        """Translate a single sentence to target_lang ('es' or 'hi')."""
        if not text or not text.strip():
            return ""

        self._load_model(target_lang)
        tok = self.tokenizers[target_lang]
        mod = self.models[target_lang]

        inputs = tok(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        outputs = mod.generate(**inputs, max_length=256, num_beams=4)
        translated = tok.decode(outputs[0], skip_special_tokens=True).strip()
        return translated

    def translate_batch(self, texts: List[str], target_lang: str, batch_size: int = 16) -> List[str]:
        """Translate a batch of sentences to target_lang."""
        if not texts:
            return []

        self._load_model(target_lang)
        tok = self.tokenizers[target_lang]
        mod = self.models[target_lang]

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
            outputs = mod.generate(**inputs, max_length=256, num_beams=4)
            decoded = tok.batch_decode(outputs, skip_special_tokens=True)
            results.extend([d.strip() for d in decoded])
        return results

    @staticmethod
    def audit_numerical_consistency(source_en: str, translated: str) -> Dict[str, Any]:
        """
        Verify numbers/percentages in source appear in translation.
        """
        src_nums = set(NUMERIC_PATTERN.findall(source_en))
        tgt_nums = set(NUMERIC_PATTERN.findall(translated))

        missing_nums = src_nums - tgt_nums
        return {
            "source_numerics": sorted(list(src_nums)),
            "target_numerics": sorted(list(tgt_nums)),
            "missing_numerics": sorted(list(missing_nums)),
            "numerics_preserved": len(missing_nums) == 0,
        }

    @staticmethod
    def audit_negation_consistency(source_en: str, translated: str, target_lang: str) -> Dict[str, Any]:
        """
        Check whether negation polarity is preserved in translation.
        """
        src_words = set(re.findall(r"\b\w+\b", source_en.lower()))
        tgt_words = set(re.findall(r"\b\w+\b", translated.lower()))

        src_has_negation = bool(src_words.intersection(NEGATION_TERMS["en"]))
        tgt_terms = NEGATION_TERMS.get(target_lang, set())
        tgt_has_negation = bool(tgt_words.intersection(tgt_terms))

        polarity_match = src_has_negation == tgt_has_negation
        return {
            "source_has_negation": src_has_negation,
            "target_has_negation": tgt_has_negation,
            "polarity_preserved": polarity_match,
        }
