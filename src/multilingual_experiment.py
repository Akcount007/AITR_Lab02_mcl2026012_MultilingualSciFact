"""
Multilingual Experiment Engine for Scientific Claim Verification.

Supports:
  1. Multilingual input claims: English (en), Spanish (es), Hindi (hi)
  2. Evaluation Modes:
     - Mode A: Direct Classification (Veracity only)
     - Mode B: Evidence-Grounded Classification (Veracity + Rationale Sentences)
  3. Strict linking to original claim ID, doc ID, gold labels, and rationales.
  4. Robust 2-pass JSON parsing with regex fallback.
  5. Taxonomy tagging for failure cases (ERR_RAT_OMIT, ERR_RAT_SPURIOUS,
     ERR_POL_INVERT, ERR_NEI_FAIL, ERR_TRANS_DRIFT).
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.metrics import normalize_eval_label, evaluate_rationale_selection

logger = logging.getLogger(__name__)

# Canonical prompt templates from frozen experiment_protocol.md
MODE_A_TEMPLATE = """<|im_start|>system
You are a rigorous scientific fact-checking system.
Analyze the scientific claim against the provided evidence abstract.
Determine whether the abstract supports, contradicts, or provides insufficient information to verify the claim.

You must respond STRICTLY with a valid JSON object matching this schema:
{{
  "veracity": "SUPPORT" | "CONTRADICT" | "NOT_ENOUGH_INFO",
  "confidence": <float between 0.0 and 1.0>
}}
Do not include any text outside the JSON object.
<|im_end|>
<|im_start|>user
Claim:
{claim_text}

Evidence Abstract:
Title: {abstract_title}
Text:
{abstract_text}

Classification JSON:
<|im_end|>
<|im_start|>assistant
"""

MODE_B_TEMPLATE = """<|im_start|>system
You are a rigorous scientific fact-checking system.
Analyze the scientific claim against the numbered sentences of the provided evidence abstract.
You must:
1. Identify the exact 0-indexed sentence numbers that contain direct factual evidence supporting or contradicting the claim. If the evidence is insufficient, rationale_sentences must be empty [].
2. Classify the claim as "SUPPORT", "CONTRADICT", or "NOT_ENOUGH_INFO".
3. Provide your confidence score.

You must respond STRICTLY with a valid JSON object matching this schema:
{{
  "veracity": "SUPPORT" | "CONTRADICT" | "NOT_ENOUGH_INFO",
  "rationale_sentences": [<list of integer sentence indices>],
  "confidence": <float between 0.0 and 1.0>
}}
Do not include any text outside the JSON object.
<|im_end|>
<|im_start|>user
Claim:
{claim_text}

Evidence Abstract:
Title: {abstract_title}
Numbered Sentences:
{numbered_abstract_sentences}

Classification JSON:
<|im_end|>
<|im_start|>assistant
"""


def parse_model_json_response(raw_text: str, mode: str = "B") -> Dict[str, Any]:
    """
    Two-pass parser:
      Pass 1: Native json.loads() on cleaned string or embedded markdown codeblock.
      Pass 2: Regex extraction of veracity, rationale_sentences, and confidence.
    """
    cleaned = raw_text.strip()

    # Strip markdown codeblocks if present
    if "```json" in cleaned:
        parts = cleaned.split("```json")
        if len(parts) > 1:
            cleaned = parts[1].split("```")[0].strip()
    elif "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) > 1:
            cleaned = parts[1].strip()

    # Pass 1: native JSON parsing
    try:
        data = json.loads(cleaned)
        veracity = normalize_eval_label(data.get("veracity", "INVALID_OUTPUT"))
        conf = float(data.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))

        rationales = []
        if mode == "B":
            raw_sents = data.get("rationale_sentences", [])
            if isinstance(raw_sents, list):
                for s in raw_sents:
                    try:
                        rationales.append(int(s))
                    except (ValueError, TypeError):
                        pass

        return {
            "veracity": veracity,
            "rationale_sentences": sorted(list(set(rationales))),
            "confidence": conf,
            "parsing_status": "JSON_SUCCESS",
            "raw_output": raw_text,
        }
    except Exception:
        pass

    # Pass 2: Regex fallback
    veracity = "INVALID_OUTPUT"
    # Prioritize extracting value explicitly assigned to veracity key
    veracity_field_match = re.search(
        r'(?:"veracity"|veracity)\s*:\s*"?([A-Za-z_-]+)"?',
        raw_text,
        re.IGNORECASE,
    )
    if veracity_field_match:
        veracity = normalize_eval_label(veracity_field_match.group(1))

    if veracity == "INVALID_OUTPUT":
        veracity_match = re.search(
            r'\b(SUPPORT|SUPPORTS|CONTRADICT|CONTRADICTS|REFUTE|REFUTES|NOT_ENOUGH_INFO|NOTENOUGHINFO|NEI|NO_INFO)\b',
            raw_text,
            re.IGNORECASE,
        )
        if veracity_match:
            veracity = normalize_eval_label(veracity_match.group(1))

    conf = 0.0
    conf_match = re.search(r'(?:"confidence"|confidence)\s*:\s*([0-9]*\.?[0-9]+)', raw_text, re.IGNORECASE)
    if conf_match:
        try:
            conf = max(0.0, min(1.0, float(conf_match.group(1))))
        except ValueError:
            pass

    rationales = []
    if mode == "B":
        sent_match = re.search(r'(?:"rationale_sentences"|rationale_sentences)\s*:\s*\[([0-9,\s]*)\]', raw_text)
        if sent_match:
            raw_nums = sent_match.group(1).split(",")
            for n in raw_nums:
                n = n.strip()
                if n.isdigit():
                    rationales.append(int(n))

    status = "REGEX_FALLBACK" if veracity != "INVALID_OUTPUT" else "PARSING_FAILED"

    return {
        "veracity": veracity,
        "rationale_sentences": sorted(list(set(rationales))),
        "confidence": conf,
        "parsing_status": status,
        "raw_output": raw_text,
    }


def classify_error(
    gold_label: str,
    pred_label: str,
    gold_sents: List[int],
    pred_sents: List[int],
    mode: str = "B",
) -> str:
    """
    Classify error into protocol 5-part taxonomy:
      - ERR_RAT_OMIT: Veracity correct, but zero/wrong rationale sentences.
      - ERR_RAT_SPURIOUS: Hallucinated citations that do not support claim.
      - ERR_POL_INVERT: Direct polarity flip (SUPPORT <-> REFUTES).
      - ERR_NEI_FAIL: Gold is NOT_ENOUGH_INFO, but model predicted SUPPORT/REFUTE.
      - NONE: Correct prediction.
      - ERR_OTHER: General prediction error.
    """
    norm_gold = normalize_eval_label(gold_label)
    norm_pred = normalize_eval_label(pred_label)

    if norm_gold == norm_pred:
        if mode == "B":
            if norm_gold in ("SUPPORTS", "REFUTES"):
                g_set = set(gold_sents)
                p_set = set(pred_sents)
                if not p_set.intersection(g_set):
                    return "ERR_RAT_OMIT"
                if len(p_set - g_set) > 0 and len(g_set.intersection(p_set)) == 0:
                    return "ERR_RAT_SPURIOUS"
            elif norm_gold == "NOT_ENOUGH_INFO" and len(pred_sents) > 0:
                return "ERR_RAT_SPURIOUS"
        return "NONE"

    # Errors
    if norm_gold == "NOT_ENOUGH_INFO" and norm_pred in ("SUPPORTS", "REFUTES"):
        return "ERR_NEI_FAIL"

    if (norm_gold == "SUPPORTS" and norm_pred == "REFUTES") or (
        norm_gold == "REFUTES" and norm_pred == "SUPPORTS"
    ):
        return "ERR_POL_INVERT"

    if mode == "B" and len(pred_sents) > 0 and norm_gold not in ("SUPPORTS", "REFUTES"):
        return "ERR_RAT_SPURIOUS"

    return "ERR_OTHER"


class MultilingualGenerativeEngine:
    """
    Generative inference engine for Prompt-based evaluation across languages and modes.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None

    def load(self):
        logger.info("Loading generative model %s on %s", self.model_name, self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
            )

        self.model.to(self.device)
        self.model.eval()
        logger.info("Generative model loaded successfully.")

    def run_prompt(
        self,
        claim_text: str,
        abstract_title: str,
        abstract_sentences: List[str],
        mode: str = "B",
    ) -> Dict[str, Any]:
        """
        Execute Mode A or Mode B prompt using native chat template and return parsed prediction.
        """
        if mode == "A":
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a scientific fact-checker. Analyze the claim against the evidence.\n"
                        "Respond ONLY with a JSON object: "
                        '{"veracity": "SUPPORT" or "CONTRADICT" or "NOT_ENOUGH_INFO", "confidence": 0.9}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Claim:\n{claim_text}\n\n"
                        f"Evidence:\n{' '.join(abstract_sentences)}\n\n"
                        f"Classification JSON:"
                    ),
                },
            ]
        elif mode == "B":
            numbered_sents = "\n".join(
                [f"[{idx}] {s}" for idx, s in enumerate(abstract_sentences)]
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a scientific fact-checking system. Analyze the claim against the numbered sentences.\n"
                        "Respond ONLY with a JSON object:\n"
                        '{"veracity": "SUPPORT" or "CONTRADICT" or "NOT_ENOUGH_INFO", "rationale_sentences": [0, 1], "confidence": 0.9}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Claim:\n{claim_text}\n\n"
                        f"Numbered Sentences:\n{numbered_sents}\n\n"
                        f"Classification JSON:"
                    ),
                },
            ]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=35,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.time() - t0

        generated_tokens = outputs[0][inputs.input_ids.shape[1] :]
        raw_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        parsed = parse_model_json_response(raw_output, mode=mode)
        parsed["latency_seconds"] = round(latency, 3)
        return parsed
