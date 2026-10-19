"""IU X-Ray report parsing, text cleaning, and target-sequence construction.

Parses the XML radiology reports in ``dataset/NLMCXR_reports/ecgen-radiology``,
cross-references them with the image/caption pairs in ``dataset/iu_xray.csv``,
and builds the structured target sequences used to train the decoder.
"""

import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Special tokens wrapping the FINDINGS / IMPRESSION sections of a report.
REPORT_START = "[REPORT_START]"
FINDING_START = "[FINDING]"
FINDING_END = "[END_FINDING]"
IMPRESSION_START = "[IMPRESSION]"
IMPRESSION_END = "[END_IMPRESSION]"
EOS = "[EOS]"

_ALLOWED_CHARS_RE = re.compile(r"[^a-z0-9\s.,;:\-()/%']")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class Report:
    """A single parsed XML radiology report."""

    uid: str
    comparison: str = ""
    indication: str = ""
    findings: str = ""
    impression: str = ""
    image_ids: List[str] = field(default_factory=list)


def clean_text(text: Optional[str]) -> str:
    """Lowercase and strip non-medical special characters from report text.

    Keeps letters, digits, whitespace, and punctuation commonly found in
    medical text (``. , ; : - ( ) / % '``). Collapses repeated whitespace.

    Args:
        text: Raw report text (may be ``None`` or empty).

    Returns:
        The cleaned, lowercased text. Empty string if ``text`` is falsy.
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = _ALLOWED_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def build_target_text(findings: str, impression: str) -> str:
    """Build the structured decoder target sequence for a report.

    Args:
        findings: Cleaned FINDINGS section text (may be empty).
        impression: Cleaned IMPRESSION section text (may be empty).

    Returns:
        A string of the form
        ``"[REPORT_START] [FINDING] <findings> [END_FINDING] [IMPRESSION] <impression> [END_IMPRESSION] [EOS]"``.
    """
    parts = [
        REPORT_START,
        FINDING_START,
        findings,
        FINDING_END,
        IMPRESSION_START,
        impression,
        IMPRESSION_END,
        EOS,
    ]
    return _WHITESPACE_RE.sub(" ", " ".join(p for p in parts if p)).strip()


def parse_report_xml(xml_path: Path) -> Optional[Report]:
    """Parse a single IU X-Ray XML report file.

    Args:
        xml_path: Path to an ``ecgen-radiology/<N>.xml`` file.

    Returns:
        A :class:`Report`, or ``None`` if the file has no ``uId``.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        logger.warning(f"Failed to parse {xml_path}: {exc}")
        return None

    uid_el = root.find("uId")
    if uid_el is None or not uid_el.get("id"):
        return None
    uid = uid_el.get("id")

    sections: Dict[str, str] = {}
    for abstract_text in root.findall(".//AbstractText"):
        label = (abstract_text.get("Label") or "").upper()
        sections[label] = (abstract_text.text or "").strip()

    image_ids = [
        img.get("id") for img in root.findall(".//parentImage") if img.get("id")
    ]

    return Report(
        uid=uid,
        comparison=sections.get("COMPARISON", ""),
        indication=sections.get("INDICATION", ""),
        findings=sections.get("FINDINGS", ""),
        impression=sections.get("IMPRESSION", ""),
        image_ids=image_ids,
    )


def parse_all_reports(reports_dir: Path) -> Dict[str, Report]:
    """Parse every XML report in ``reports_dir``.

    Args:
        reports_dir: Directory containing ``<N>.xml`` report files.

    Returns:
        Mapping from ``uId`` (e.g. ``"CXR1000"``) to :class:`Report`.
    """
    reports: Dict[str, Report] = {}
    xml_files = sorted(Path(reports_dir).glob("*.xml"))
    for xml_path in xml_files:
        report = parse_report_xml(xml_path)
        if report is not None:
            reports[report.uid] = report
    logger.info(f"Parsed {len(reports)} reports from {len(xml_files)} XML files")
    return reports


def build_metadata(
    images_dir: Path,
    reports_dir: Path,
    captions_csv: Path,
) -> pd.DataFrame:
    """Build the per-image metadata table used for training.

    Cross-references the XML reports (for structured FINDINGS/IMPRESSION
    text) with ``iu_xray.csv`` (for the authoritative image/caption pairs and
    as a fallback when an image is not referenced by any XML's
    ``parentImage`` list).

    Args:
        images_dir: Directory of PNG images (``dataset/NLMCXR_png``).
        reports_dir: Directory of XML reports
            (``dataset/NLMCXR_reports/ecgen-radiology``).
        captions_csv: Path to ``dataset/iu_xray.csv``.

    Returns:
        DataFrame with columns ``image, uid, findings, impression,
        target_text``, one row per image that exists on disk.
    """
    reports = parse_all_reports(reports_dir)

    # image_id (without extension) -> uid, built from each report's parentImage list.
    image_to_uid: Dict[str, str] = {}
    for uid, report in reports.items():
        for image_id in report.image_ids:
            image_to_uid[image_id] = uid

    captions_df = pd.read_csv(captions_csv)
    existing_images = {p.name for p in Path(images_dir).glob("*.png")}

    records = []
    for _, row in captions_df.iterrows():
        image_name = str(row["image"])
        if image_name not in existing_images:
            continue
        image_id = image_name[: -len(".png")] if image_name.endswith(".png") else image_name

        uid = image_to_uid.get(image_id)
        if uid is None:
            # Fallback: derive uid from the filename prefix before "_IM-".
            match = re.match(r"^(.*)_IM-", image_id)
            candidate = match.group(1) if match else None
            if candidate in reports:
                uid = candidate

        if uid is not None and uid in reports:
            report = reports[uid]
            findings = clean_text(report.findings)
            impression = clean_text(report.impression)
            if not findings and not impression:
                # XML had no usable sections; fall back to the CSV caption.
                findings = clean_text(str(row["caption"]))
                impression = ""
        else:
            # No matching XML report at all: use the CSV caption verbatim.
            uid = image_id.split("_IM-")[0]
            findings = clean_text(str(row["caption"]))
            impression = ""

        target_text = build_target_text(findings, impression)
        records.append(
            {
                "image": image_name,
                "uid": uid,
                "findings": findings,
                "impression": impression,
                "target_text": target_text,
            }
        )

    df = pd.DataFrame.from_records(records)
    logger.info(f"Built metadata for {len(df)} images across {df['uid'].nunique()} studies")
    return df


def make_splits(
    uids: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, str]:
    """Assign a train/val/test split to each unique study (uid).

    Splitting is done at the study level (not per-image) so that the
    frontal/lateral views of the same report never end up in different
    splits.

    Args:
        uids: List of unique study identifiers (e.g. ``["CXR1", "CXR2", ...]``).
        train_ratio: Fraction of studies assigned to the training split.
        val_ratio: Fraction of studies assigned to the validation split.
        test_ratio: Fraction of studies assigned to the test split.
        seed: Random seed for the shuffle.

    Returns:
        Mapping from uid to one of ``"train"``, ``"val"``, ``"test"``.

    Raises:
        ValueError: If the ratios do not sum to (approximately) 1.0.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    unique_uids = sorted(set(uids))
    rng = random.Random(seed)
    rng.shuffle(unique_uids)

    n = len(unique_uids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    split_map: Dict[str, str] = {}
    for i, uid in enumerate(unique_uids):
        if i < n_train:
            split_map[uid] = "train"
        elif i < n_train + n_val:
            split_map[uid] = "val"
        else:
            split_map[uid] = "test"
    return split_map
