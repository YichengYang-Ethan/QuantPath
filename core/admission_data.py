# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""Real admission data loader and normalizer.

Loads CSV files of real applicant outcomes and normalizes fields
(GPA scales, background types, etc.) into a uniform format for
calibration and statistical analysis.

CSV schema
----------
id, gender, bg_type, nationality, gpa, gpa_scale, gre, toefl, major,
intern_desc, has_paper, has_research, courses_note, program, result,
season, source
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_ADMISSIONS_DIR = _PACKAGE_ROOT / "data" / "admissions"

# ---------------------------------------------------------------------------
# GPA normalization
# ---------------------------------------------------------------------------

# Supported scales: 4, 4.3, 5, 100
_GPA_SCALE_TO_4: dict[float, list[tuple[float, float, float]]] = {
    # (threshold, mapped_start, mapped_end) — piecewise linear
    # Scale 100 -> 4.0
    100: [
        (95, 3.9, 4.0),
        (90, 3.7, 3.9),
        (85, 3.3, 3.7),
        (80, 3.0, 3.3),
        (75, 2.7, 3.0),
        (70, 2.3, 2.7),
        (60, 1.7, 2.3),
        (0, 0.0, 1.7),
    ],
    # Scale 5 -> 4.0
    5: [
        (4.8, 3.9, 4.0),
        (4.5, 3.7, 3.9),
        (4.0, 3.3, 3.7),
        (3.5, 3.0, 3.3),
        (3.0, 2.5, 3.0),
        (0, 0.0, 2.5),
    ],
    # Scale 4.3 -> 4.0 (cap at 4.0)
    4.3: [
        (4.0, 3.9, 4.0),
        (3.7, 3.7, 3.9),
        (3.3, 3.3, 3.7),
        (3.0, 3.0, 3.3),
        (0, 0.0, 3.0),
    ],
}


def normalize_gpa(gpa: float, scale: float) -> float:
    """Normalize a GPA value to the 4.0 scale.

    Parameters
    ----------
    gpa:
        The raw GPA value.
    scale:
        The GPA scale (4, 4.3, 5, or 100).

    Returns
    -------
    float
        GPA normalized to 0.0-4.0 range.
    """
    if scale == 4:
        return min(4.0, gpa)

    breakpoints = _GPA_SCALE_TO_4.get(scale)
    if breakpoints is None:
        # Unknown scale — attempt linear conversion
        return min(4.0, gpa * 4.0 / scale)

    for idx, (threshold, mapped_lo, mapped_hi) in enumerate(breakpoints):
        if gpa >= threshold:
            seg_top = scale if idx == 0 else breakpoints[idx - 1][0]
            if seg_top == threshold:
                return mapped_hi
            frac = (gpa - threshold) / (seg_top - threshold)
            return mapped_lo + frac * (mapped_hi - mapped_lo)

    return 0.0


# ---------------------------------------------------------------------------
# Background type classification
# ---------------------------------------------------------------------------

# Tier mapping for Chinese university classification
BG_TIER_MAP: dict[str, int] = {
    # Tier 1: Top overseas / C9 / Peking/Tsinghua
    "海本(Top10)": 1,
    "海本(Top15)": 1,
    "海本(Top20)": 1,
    "C9": 1,
    # Tier 2: Strong overseas / top 985
    "海本(Top30)": 2,
    "海本(Top50)": 2,
    "985": 2,
    # Tier 3: 211 / strong finance schools
    "两财一贸(211)": 3,
    "两财一贸": 3,
    "211": 3,
    # Tier 4: Other
    "双非一本": 4,
    "双非": 5,
    "海本(Top100)": 3,
    "海本": 3,
}


def classify_background(bg_type: str) -> int:
    """Map a background type string to a tier (1=strongest, 5=weakest).

    Performs fuzzy matching against known background categories.
    """
    bg_clean = bg_type.strip().replace(" ", "")

    # Exact match first
    if bg_clean in BG_TIER_MAP:
        return BG_TIER_MAP[bg_clean]

    # Partial match
    for key, tier in BG_TIER_MAP.items():
        if key in bg_clean or bg_clean in key:
            return tier

    # Keywords
    lower = bg_clean.lower()
    if "top10" in lower or "top15" in lower:
        return 1
    if "top20" in lower or "top30" in lower or "985" in lower or "c9" in lower:
        return 2
    if "211" in lower or "top50" in lower or "财" in lower or "贸" in lower:
        return 3
    if "海本" in lower:
        return 3
    if "双非" in lower:
        return 4

    return 4  # default


# ---------------------------------------------------------------------------
# Nationality classification
# ---------------------------------------------------------------------------

# Canonical nationality values
NATIONALITY_DOMESTIC = "domestic"  # US citizen / permanent resident
NATIONALITY_CHINA = "china"  # Chinese mainland
NATIONALITY_HK_TW = "hk_tw"  # Hong Kong, Macau, Taiwan
NATIONALITY_OTHER_INTL = "other_intl"  # Other international

_NATIONALITY_MAP: dict[str, str] = {
    "美籍": NATIONALITY_DOMESTIC,
    "美国": NATIONALITY_DOMESTIC,
    "us": NATIONALITY_DOMESTIC,
    "domestic": NATIONALITY_DOMESTIC,
    "greencard": NATIONALITY_DOMESTIC,
    "绿卡": NATIONALITY_DOMESTIC,
    "pr": NATIONALITY_DOMESTIC,
    "中国大陆": NATIONALITY_CHINA,
    "中国": NATIONALITY_CHINA,
    "大陆": NATIONALITY_CHINA,
    "china": NATIONALITY_CHINA,
    "mainland": NATIONALITY_CHINA,
    "港澳台": NATIONALITY_HK_TW,
    "香港": NATIONALITY_HK_TW,
    "台湾": NATIONALITY_HK_TW,
    "澳门": NATIONALITY_HK_TW,
    "hk": NATIONALITY_HK_TW,
    "taiwan": NATIONALITY_HK_TW,
}


def classify_nationality(nationality: str) -> str:
    """Map a nationality string to a canonical value.

    Returns one of: 'domestic', 'china', 'hk_tw', 'other_intl'.
    Empty/unknown values return 'china' (most common in MFE applicant pool).
    """
    val = nationality.strip().lower().replace(" ", "")
    if not val or val in ("不明", "n/a", "unknown"):
        return NATIONALITY_CHINA  # default for MFE applicant pool

    # Exact match
    if val in _NATIONALITY_MAP:
        return _NATIONALITY_MAP[val]

    # Partial match
    for key, canonical in _NATIONALITY_MAP.items():
        if key in val or val in key:
            return canonical

    return NATIONALITY_OTHER_INTL


# ---------------------------------------------------------------------------
# Intern strength scoring
# ---------------------------------------------------------------------------


def score_internships(intern_desc: str) -> float:
    """Score internship description on a 0-10 scale.

    Heuristic scoring based on keywords:
    - Number of internships
    - Quality indicators (顶级, top, 百亿, 头部)
    - Type indicators (量化, quant, 投行, IB, 对冲, hedge fund)
    """
    if not intern_desc or intern_desc.strip() in ("", "无", "N/A"):
        return 0.0

    desc = intern_desc.lower()
    score = 0.0

    # Count internships (Chinese: N段)
    for i, c in enumerate(desc):
        if c == "段" and i > 0 and desc[i - 1].isdigit():
            n = int(desc[i - 1])
            score += min(n * 1.5, 5.0)
            break

    # Quality keywords (Chinese + English)
    quality_keywords = {
        "顶级": 2.0, "top": 1.5, "百亿": 1.5, "头部": 1.5,
        "一线": 1.0, "知名": 0.8, "大型": 0.5,
    }
    for kw, pts in quality_keywords.items():
        if kw in desc:
            score += pts

    # Type keywords
    type_keywords = {
        "量化": 1.5, "quant": 1.5, "投行": 1.5, "ib": 1.0,
        "对冲": 1.5, "hedge": 1.5, "私募": 1.0, "qr": 1.0,
        "trading": 1.0, "研究": 0.8, "金工": 0.8,
        "三中一华": 2.0, "高盛": 2.0, "goldman": 2.0,
        "摩根": 2.0, "morgan": 1.5, "kaggle": 1.5,
    }
    for kw, pts in type_keywords.items():
        if kw in desc:
            score += pts

    return min(10.0, score)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class AdmissionRecord:
    """A single real applicant data point with normalized fields."""

    id: str = ""
    gender: str = ""  # M / F / empty
    bg_type: str = ""
    bg_tier: int = 4  # 1-5, computed from bg_type
    nationality: str = ""  # raw value
    nationality_canonical: str = ""  # domestic / china / hk_tw / other_intl
    gpa_raw: float = 0.0
    gpa_scale: float = 4.0
    gpa_normalized: float = 0.0  # on 4.0 scale
    gre: Optional[int] = None
    toefl: Optional[int] = None
    major: str = ""
    intern_desc: str = ""
    intern_score: float = 0.0  # 0-10 computed score
    has_paper: Optional[bool] = None
    has_research: Optional[bool] = None
    courses_note: str = ""
    program: str = ""
    result: str = ""  # accepted / rejected / waitlisted
    season: str = ""
    source: str = ""


@dataclass
class ProgramStats:
    """Aggregated statistics for a program from real data."""

    program_id: str = ""
    total_records: int = 0
    accepted: int = 0
    rejected: int = 0
    waitlisted: int = 0

    # Accepted applicant stats
    avg_gpa_accepted: float = 0.0
    avg_gre_accepted: float = 0.0
    avg_bg_tier_accepted: float = 0.0
    avg_intern_score_accepted: float = 0.0
    paper_rate_accepted: float = 0.0
    research_rate_accepted: float = 0.0
    female_rate_accepted: float = 0.0  # fraction of female among accepted
    nationality_dist_accepted: dict[str, int] = field(default_factory=dict)

    # Rejected applicant stats
    avg_gpa_rejected: float = 0.0
    avg_gre_rejected: float = 0.0

    # Computed metrics
    observed_acceptance_rate: float = 0.0

    # Feature importance (correlation with acceptance)
    feature_importance: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def _parse_bool(val: str) -> Optional[bool]:
    """Parse a boolean field that may be '是'/'否'/'不明'/etc."""
    val = val.strip().lower()
    if val in ("是", "yes", "true", "1", "有"):
        return True
    if val in ("否", "no", "false", "0", "无"):
        return False
    return None  # unknown


def _parse_int(val: str) -> Optional[int]:
    """Parse an integer, stripping non-numeric suffixes like '+'."""
    val = val.strip().rstrip("+").rstrip("分")
    if not val or val.lower() in ("", "n/a", "无", "不明"):
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _parse_float(val: str) -> float:
    """Parse a float value, defaulting to 0.0."""
    val = val.strip()
    if not val or val.lower() in ("n/a", "无", "不明"):
        return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0


def load_admission_csv(path: str | Path) -> list[AdmissionRecord]:
    """Load admission records from a CSV file.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    list[AdmissionRecord]
        Parsed and normalized records. Only includes records with
        result in ('accepted', 'rejected', 'waitlisted').
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Admission data file not found: {filepath}")

    records: list[AdmissionRecord] = []

    with open(filepath, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            result = row.get("result", "").strip().lower()
            if result not in ("accepted", "rejected", "waitlisted"):
                continue

            gpa_raw = _parse_float(row.get("gpa", "0"))
            gpa_scale = _parse_float(row.get("gpa_scale", "4"))
            if gpa_scale == 0:
                gpa_scale = 4.0

            bg_type = row.get("bg_type", "").strip()
            nationality_raw = row.get("nationality", "").strip()

            rec = AdmissionRecord(
                id=row.get("id", "").strip(),
                gender=row.get("gender", "").strip().upper(),
                bg_type=bg_type,
                bg_tier=classify_background(bg_type),
                nationality=nationality_raw,
                nationality_canonical=classify_nationality(nationality_raw),
                gpa_raw=gpa_raw,
                gpa_scale=gpa_scale,
                gpa_normalized=normalize_gpa(gpa_raw, gpa_scale),
                gre=_parse_int(row.get("gre", "")),
                toefl=_parse_int(row.get("toefl", "")),
                major=row.get("major", "").strip(),
                intern_desc=row.get("intern_desc", "").strip(),
                intern_score=score_internships(row.get("intern_desc", "")),
                has_paper=_parse_bool(row.get("has_paper", "")),
                has_research=_parse_bool(row.get("has_research", "")),
                courses_note=row.get("courses_note", "").strip(),
                program=row.get("program", "").strip(),
                result=result,
                season=row.get("season", "").strip(),
                source=row.get("source", "").strip(),
            )
            records.append(rec)

    return records


def load_all_admission_data() -> list[AdmissionRecord]:
    """Load all CSV files from the ``data/admissions/`` directory.

    Skips the template file. Returns combined records from all CSVs.
    """
    if not _ADMISSIONS_DIR.is_dir():
        return []

    all_records: list[AdmissionRecord] = []
    for csv_path in sorted(_ADMISSIONS_DIR.glob("*.csv")):
        if csv_path.stem == "template":
            continue
        all_records.extend(load_admission_csv(csv_path))
    return all_records


def validate_records(
    records: list[AdmissionRecord],
) -> list[str]:
    """Check loaded records for consistency issues.

    Returns a list of warning strings (empty = all OK).
    """
    warnings: list[str] = []

    # Check that same ID has consistent personal attributes.
    by_id: dict[str, AdmissionRecord] = {}
    for r in records:
        if not r.id:
            continue
        if r.id in by_id:
            ref = by_id[r.id]
            if abs(r.gpa_raw - ref.gpa_raw) > 0.01:
                warnings.append(
                    f"ID {r.id}: inconsistent GPA "
                    f"({r.gpa_raw} vs {ref.gpa_raw})"
                )
            if r.bg_type and ref.bg_type and r.bg_type != ref.bg_type:
                warnings.append(
                    f"ID {r.id}: inconsistent bg_type "
                    f"({r.bg_type} vs {ref.bg_type})"
                )
        else:
            by_id[r.id] = r

    # Range checks.
    for r in records:
        if r.gpa_normalized > 4.0:
            warnings.append(
                f"ID {r.id}: normalized GPA {r.gpa_normalized:.2f} > 4.0"
            )
        if r.gre is not None and (r.gre < 260 or r.gre > 340):
            warnings.append(
                f"ID {r.id}: GRE {r.gre} outside valid range 260-340"
            )

    return warnings


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------


def _safe_avg(values: list[float]) -> float:
    """Average of a list, returning 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


def compute_program_stats(
    records: list[AdmissionRecord],
    program_id: str,
) -> ProgramStats:
    """Compute aggregate statistics for a single program.

    Parameters
    ----------
    records:
        All admission records (will be filtered to program_id).
    program_id:
        The program ID to compute stats for.

    Returns
    -------
    ProgramStats
        Aggregated statistics including acceptance rates, average
        GPA/GRE for accepted vs rejected, and feature importance.
    """
    prog_records = [r for r in records if r.program == program_id]
    if not prog_records:
        return ProgramStats(program_id=program_id)

    accepted = [r for r in prog_records if r.result == "accepted"]
    rejected = [r for r in prog_records if r.result == "rejected"]
    waitlisted = [r for r in prog_records if r.result == "waitlisted"]

    stats = ProgramStats(
        program_id=program_id,
        total_records=len(prog_records),
        accepted=len(accepted),
        rejected=len(rejected),
        waitlisted=len(waitlisted),
    )

    # Accepted stats
    if accepted:
        stats.avg_gpa_accepted = _safe_avg([r.gpa_normalized for r in accepted])
        gre_vals = [r.gre for r in accepted if r.gre is not None]
        stats.avg_gre_accepted = _safe_avg(gre_vals) if gre_vals else 0.0
        stats.avg_bg_tier_accepted = _safe_avg([float(r.bg_tier) for r in accepted])
        stats.avg_intern_score_accepted = _safe_avg([r.intern_score for r in accepted])
        paper_known = [r for r in accepted if r.has_paper is not None]
        stats.paper_rate_accepted = (
            sum(1 for r in paper_known if r.has_paper) / len(paper_known)
            if paper_known
            else 0.0
        )
        research_known = [r for r in accepted if r.has_research is not None]
        stats.research_rate_accepted = (
            sum(1 for r in research_known if r.has_research) / len(research_known)
            if research_known
            else 0.0
        )
        # Gender stats
        gendered = [r for r in accepted if r.gender in ("M", "F")]
        stats.female_rate_accepted = (
            sum(1 for r in gendered if r.gender == "F") / len(gendered)
            if gendered
            else 0.0
        )
        # Nationality distribution
        nat_dist: dict[str, int] = {}
        for r in accepted:
            nat = r.nationality_canonical or "unknown"
            nat_dist[nat] = nat_dist.get(nat, 0) + 1
        stats.nationality_dist_accepted = nat_dist

    # Rejected stats
    if rejected:
        stats.avg_gpa_rejected = _safe_avg([r.gpa_normalized for r in rejected])
        gre_vals_rej = [r.gre for r in rejected if r.gre is not None]
        stats.avg_gre_rejected = _safe_avg(gre_vals_rej) if gre_vals_rej else 0.0

    # Observed acceptance rate
    decided = len(accepted) + len(rejected)
    stats.observed_acceptance_rate = len(accepted) / decided if decided > 0 else 0.0

    # Feature importance (simple correlation: avg_accepted vs avg_rejected)
    if accepted and rejected:
        stats.feature_importance = _compute_feature_importance(accepted, rejected)

    return stats


def _compute_feature_importance(
    accepted: list[AdmissionRecord],
    rejected: list[AdmissionRecord],
) -> dict[str, float]:
    """Compute simple feature importance as effect size between groups.

    Uses the difference of means normalized by pooled std as a proxy
    for feature discriminative power. Higher absolute value = more important.
    """
    import math

    features: dict[str, float] = {}

    def _effect_size(acc_vals: list[float], rej_vals: list[float]) -> float:
        if not acc_vals or not rej_vals:
            return 0.0
        mean_a = sum(acc_vals) / len(acc_vals)
        mean_r = sum(rej_vals) / len(rej_vals)
        var_a = sum((v - mean_a) ** 2 for v in acc_vals) / max(len(acc_vals), 1)
        var_r = sum((v - mean_r) ** 2 for v in rej_vals) / max(len(rej_vals), 1)
        pooled_std = math.sqrt((var_a + var_r) / 2) or 1.0
        return (mean_a - mean_r) / pooled_std

    # GPA (normalized)
    features["gpa"] = _effect_size(
        [r.gpa_normalized for r in accepted],
        [r.gpa_normalized for r in rejected],
    )

    # GRE
    acc_gre = [float(r.gre) for r in accepted if r.gre is not None]
    rej_gre = [float(r.gre) for r in rejected if r.gre is not None]
    features["gre"] = _effect_size(acc_gre, rej_gre)

    # Background tier (inverted: lower tier = better)
    features["bg_tier"] = -_effect_size(
        [float(r.bg_tier) for r in accepted],
        [float(r.bg_tier) for r in rejected],
    )

    # Intern score
    features["intern"] = _effect_size(
        [r.intern_score for r in accepted],
        [r.intern_score for r in rejected],
    )

    # Paper
    features["paper"] = _effect_size(
        [1.0 if r.has_paper else 0.0 for r in accepted if r.has_paper is not None],
        [1.0 if r.has_paper else 0.0 for r in rejected if r.has_paper is not None],
    )

    # Research
    features["research"] = _effect_size(
        [1.0 if r.has_research else 0.0 for r in accepted if r.has_research is not None],
        [1.0 if r.has_research else 0.0 for r in rejected if r.has_research is not None],
    )

    # Gender (female = 1, male = 0)
    acc_gender = [1.0 if r.gender == "F" else 0.0 for r in accepted if r.gender in ("M", "F")]
    rej_gender = [1.0 if r.gender == "F" else 0.0 for r in rejected if r.gender in ("M", "F")]
    features["gender_f"] = _effect_size(acc_gender, rej_gender)

    # Nationality (domestic = 1, international = 0)
    acc_nat = [
        1.0 if r.nationality_canonical == "domestic" else 0.0
        for r in accepted if r.nationality_canonical
    ]
    rej_nat = [
        1.0 if r.nationality_canonical == "domestic" else 0.0
        for r in rejected if r.nationality_canonical
    ]
    features["domestic"] = _effect_size(acc_nat, rej_nat)

    return features


def compute_all_program_stats(
    records: list[AdmissionRecord],
) -> dict[str, ProgramStats]:
    """Compute stats for every program found in the records.

    Returns
    -------
    dict[str, ProgramStats]
        Mapping of program_id to ProgramStats.
    """
    program_ids = sorted({r.program for r in records if r.program})
    return {pid: compute_program_stats(records, pid) for pid in program_ids}


def summarize_records(records: list[AdmissionRecord]) -> dict[str, Any]:
    """Generate a high-level summary of the admission dataset.

    Returns
    -------
    dict
        Summary with total counts, program breakdown, season info, etc.
    """
    programs = {}
    for r in records:
        if r.program not in programs:
            programs[r.program] = {"accepted": 0, "rejected": 0, "waitlisted": 0}
        programs[r.program][r.result] = programs[r.program].get(r.result, 0) + 1

    seasons = sorted({r.season for r in records if r.season})
    sources = sorted({r.source for r in records if r.source})

    # Gender breakdown
    gendered = [r for r in records if r.gender in ("M", "F")]
    gender_dist = {"M": 0, "F": 0}
    for r in gendered:
        gender_dist[r.gender] += 1

    # Nationality breakdown
    nat_dist: dict[str, int] = {}
    for r in records:
        nat = r.nationality_canonical or "unknown"
        nat_dist[nat] = nat_dist.get(nat, 0) + 1

    return {
        "total_records": len(records),
        "unique_applicants": len({r.id for r in records}),
        "programs": programs,
        "seasons": seasons,
        "sources": sources,
        "avg_gpa_normalized": _safe_avg([r.gpa_normalized for r in records]),
        "gre_available": sum(1 for r in records if r.gre is not None),
        "gender_dist": gender_dist,
        "nationality_dist": nat_dist,
    }
