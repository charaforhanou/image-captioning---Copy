"""Post-processing for generated/reference token sequences.

R2Gen-aligned: the report text is plain (no FINDINGS/IMPRESSION section tags),
so we simply decode everything between [REPORT_START] and [EOS], skipping any
special tokens. If legacy section tags are present, they are still handled
(backward compatible), but their absence no longer yields an empty string.
"""

import re
from typing import Dict, Sequence, Union

import torch

from src.data.tokenizer import MedicalTokenizer


def clean_report_text(text: str) -> str:
    """Collapse whitespace and fix spacing before punctuation."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text


def _special_ids(tokenizer: MedicalTokenizer) -> set:
    """Collect the structural special-token ids to strip from plain text."""
    ids = set()
    for name in ("report_start_id", "finding_id", "end_finding_id",
                 "impression_id", "end_impression_id", "eos_id", "pad_id"):
        try:
            tid = getattr(tokenizer, name)
            if tid is not None:
                ids.add(tid)
        except Exception:
            pass
    return ids


def postprocess_generation(
    ids: Union[Sequence[int], torch.Tensor], tokenizer: MedicalTokenizer
) -> Dict[str, str]:
    """Decode a token-id sequence to cleaned report text.

    For R2Gen-style plain reports there are no section tags, so we decode all
    content tokens up to [EOS]. The findings/impression keys are kept for
    backward compatibility: if section tags ARE present they are split out,
    otherwise everything goes into ``findings`` (and ``full_text``).

    Returns:
        Dict with keys ``findings``, ``impression``, ``full_text``.
    """
    if torch.is_tensor(ids):
        ids = ids.tolist()

    has_sections = (tokenizer.finding_id in ids) or (tokenizer.impression_id in ids)
    specials = _special_ids(tokenizer)

    if has_sections:
        # --- Legacy path: split by section tags ---
        findings_ids, impression_ids = [], []
        section = None
        for token_id in ids:
            if token_id == tokenizer.finding_id:
                section = "findings"; continue
            if token_id == tokenizer.end_finding_id:
                section = None; continue
            if token_id == tokenizer.impression_id:
                section = "impression"; continue
            if token_id == tokenizer.end_impression_id:
                section = None; continue
            if token_id == tokenizer.eos_id:
                break
            if section == "findings":
                findings_ids.append(token_id)
            elif section == "impression":
                impression_ids.append(token_id)
        findings = clean_report_text(tokenizer.decode(findings_ids, skip_special_tokens=True))
        impression = clean_report_text(tokenizer.decode(impression_ids, skip_special_tokens=True))
        full_text = clean_report_text(f"{findings} {impression}")
        return {"findings": findings, "impression": impression, "full_text": full_text}

    # --- R2Gen path: plain text, no sections ---
    content_ids = []
    for token_id in ids:
        if token_id == tokenizer.eos_id:
            break
        if token_id in specials:
            continue
        content_ids.append(token_id)
    text = clean_report_text(tokenizer.decode(content_ids, skip_special_tokens=True))
    return {"findings": text, "impression": "", "full_text": text}