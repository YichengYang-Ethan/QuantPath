#!/usr/bin/env python3
"""
QuantPath Training Data Preparation
Deduplication, standardization, tier recomputation, feature matrix generation.
"""

import csv
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("/Users/ethanyang/QuantPath/data/admissions")
INPUT_CSV = BASE / "collected_multidim.csv"
OUTPUT_FULL = BASE / "training_data_full.csv"
OUTPUT_MODEL = BASE / "training_data_model.csv"
OUTPUT_RICH = BASE / "training_data_rich.csv"
FEATURE_MATRIX = BASE / "feature_matrix.csv"
REPORT_MD = BASE / "data_quality_report.md"


# ── Canonical values ───────────────────────────────────────────────────────
VALID_RESULTS = {"accepted", "rejected", "waitlisted"}

PROGRAM_CANONICAL = {
    "baruch-mfe": "baruch-mfe",
    "cmu-mscf": "cmu-mscf",
    "columbia-msfe": "columbia-msfe",
    "princeton-mfin": "princeton-mfin",
    "mit-mfin": "mit-mfin",
    "berkeley-mfe": "berkeley-mfe",
    "uchicago-msfm": "uchicago-msfm",
    "gatech-qcf": "gatech-qcf",
    "cornell-mfe": "cornell-mfe",
    "nyu-mfe": "nyu-mfe",
    "nyu-tandon-mfe": "nyu-tandon-mfe",
    "stanford-mcf": "stanford-mcf",
    "ucla-mfe": "ucla-mfe",
    "uiuc-msfe": "uiuc-msfe",
    "rutgers-mqf": "rutgers-mqf",
    "bu-msmf": "bu-msmf",
    "toronto-mmf": "toronto-mmf",
    "utoronto-mmf": "toronto-mmf",
    "waterloo-mqf": "waterloo-mqf",
    "michigan-qfr": "michigan-qfr",
    "stonybrook-qf": "stonybrook-qf",
    "fordham-msqf": "fordham-msqf",
    "ncstate-mfm": "ncstate-mfm",
    "jhu-mfm": "jhu-mfm",
    "usc-msmf": "usc-msmf",
    "uwash-cfrm": "uwash-cfrm",
    "stevens-mfe": "stevens-mfe",
    "oxford-mcf": "oxford-mcf",
    "lse-mfe": "lse-mfe",
    "imperial-mfe": "imperial-mfe",
    "eth-qf": "eth-qf",
    "hkust-mfe": "hkust-mfe",
    "hec-mfe": "hec-mfe",
    "claremont-mfe": "claremont-mfe",
    "mfe-unknown": "mfe-unknown",
    "msfm-unknown": "msfm-unknown",
    "finance-unknown": "finance-unknown",
    "or-unknown": "or-unknown",
}

PROGRAM_ID_MAP = {prog: i for i, prog in enumerate(sorted(PROGRAM_CANONICAL.values()))}

UNDERGRAD_TIER_CANONICAL = {
    "t10": "T10", "t20": "T20", "t30": "T30", "t50": "T50",
    "c9": "C9", "985": "985", "211": "211",
    "iit": "IIT",
    "top_intl": "top_intl", "intl": "intl",
    "other_cn": "other_cn", "other": "other",
}

UNDERGRAD_TIER_ENCODING = {
    "C9": 1, "T10": 1, "IIT": 1,
    "T20": 2, "985": 2,
    "T30": 3, "211": 3,
    "T50": 4, "top_intl": 4,
    "intl": 5, "other_cn": 5, "other": 5,
}

INTERN_LEVEL_CANONICAL = {
    "us_top_quant": "us_top_quant",
    "us_quant": "us_quant",
    "us_bb": "us_bb",
    "us_finance": "us_finance",
    "us_tech": "us_tech",
    "cn_top": "cn_top",
    "china_top": "cn_top",
    "cn_finance": "cn_finance",
    "china_normal": "cn_finance",
    "other": "other",
    "none": "none",
}

INTERN_SCORE = {
    "us_top_quant": 10, "us_quant": 8, "us_bb": 7,
    "us_finance": 6, "cn_top": 6, "us_tech": 5,
    "cn_finance": 4, "other": 2, "none": 0,
}

INTERN_RELEVANCE_CANONICAL = {
    "quant_direct": "quant_direct",
    "quant_adjacent": "quant_adjacent",
    "somewhat_related": "somewhat_related",
    "unrelated": "unrelated",
}

RESEARCH_LEVEL_CANONICAL = {
    "published": "published",
    "significant": "significant",
    "relevant_experience": "minor",
    "minor": "minor",
    "none": "none",
}

RESEARCH_SCORE = {"published": 3, "significant": 2, "minor": 1, "none": 0}

NATIONALITY_CANONICAL = {
    "chinese": "chinese", "cn": "chinese", "china": "chinese",
    "us": "us", "american": "us", "domestic": "us",
    "indian": "indian", "india": "indian",
    "korean": "korean", "korea": "korean",
    "taiwanese": "other_asian", "japanese": "other_asian",
    "european": "european", "canadian": "european",
    "international": "international",
    "other": "other",
}

GENDER_CANONICAL = {"m": "M", "f": "F", "male": "M", "female": "F"}

MAJOR_RELEVANCE_SCORE = {
    "quant_direct": 1.0,
    "quant_adjacent": 0.7,
    "somewhat_related": 0.4,
    "not_related": 0.1,
    "unrelated": 0.1,
}


def content_hash(text):
    norm = re.sub(r'\s+', ' ', str(text).lower().strip())[:500]
    return hashlib.md5(norm.encode()).hexdigest()[:16]


def standardize_field(val, canonical_map, default=""):
    """Map a value to its canonical form."""
    if pd.isna(val) or str(val).strip() == "":
        return default
    key = str(val).strip().lower()
    return canonical_map.get(key, default)


def standardize_program(val):
    if pd.isna(val) or str(val).strip() == "":
        return ""
    key = str(val).strip().lower()
    return PROGRAM_CANONICAL.get(key, key)


def normalize_gpa(gpa, scale):
    """Normalize GPA to 4.0 scale."""
    if pd.isna(gpa) or str(gpa).strip() == "":
        return np.nan
    try:
        gpa_val = float(gpa)
    except (ValueError, TypeError):
        return np.nan

    if pd.isna(scale) or str(scale).strip() == "":
        if 0 < gpa_val <= 4.0:
            return gpa_val
        elif 50 <= gpa_val <= 100:
            return (gpa_val / 100) * 4.0
        return np.nan

    scale_str = str(scale).strip()
    if scale_str == "4.0":
        return gpa_val if 0 < gpa_val <= 4.0 else np.nan
    elif scale_str == "100":
        if 50 <= gpa_val <= 100:
            return (gpa_val / 100) * 4.0
        return np.nan
    elif scale_str == "5.0":
        if 0 < gpa_val <= 5.0:
            return (gpa_val / 5.0) * 4.0
        return np.nan
    elif scale_str == "10" or scale_str == "10.0":
        if 0 < gpa_val <= 10:
            return (gpa_val / 10.0) * 4.0
        return np.nan
    else:
        if 0 < gpa_val <= 4.0:
            return gpa_val
        return np.nan


def compute_tier(row):
    """Recompute quality tier based on actual field coverage."""
    has_result = str(row.get("result", "")).strip() in VALID_RESULTS
    rich_fields = 0

    if str(row.get("gpa", "")).strip():
        rich_fields += 1
    if str(row.get("undergrad_tier", "")).strip():
        rich_fields += 1
    if str(row.get("intern_level", "")).strip() and str(row.get("intern_level", "")).strip() != "none":
        rich_fields += 1
    if str(row.get("research_level", "")).strip() and str(row.get("research_level", "")).strip() != "none":
        rich_fields += 1
    if str(row.get("gender", "")).strip() and str(row.get("gender", "")).strip() != "unknown":
        rich_fields += 1
    if str(row.get("nationality", "")).strip() and str(row.get("nationality", "")).strip() not in ("", "unknown", "international"):
        rich_fields += 1
    if str(row.get("major", "")).strip():
        rich_fields += 1
    if str(row.get("gre_quant", "")).strip():
        rich_fields += 1

    if not has_result:
        return "D"
    if rich_fields >= 4:
        return "A"
    if rich_fields >= 2:
        return "B"
    if str(row.get("gpa", "")).strip() or str(row.get("gre_quant", "")).strip():
        return "C"
    return "D"


def deduplicate(df):
    """
    Two-phase deduplication:
    1. Exact: same source + same raw_text -> keep first
    2. Cross-source: same program + result + GPA (within 0.05) + season -> keep richest
    """
    print("  Phase 1: Exact deduplication (same source + raw_text)...")
    before = len(df)
    df["_raw_hash"] = df.apply(
        lambda r: hashlib.md5(
            f"{r.get('source', '')}:{str(r.get('raw_text', ''))[:500]}".lower().encode()
        ).hexdigest()[:16],
        axis=1
    )
    df = df.drop_duplicates(subset=["_raw_hash"], keep="first")
    print(f"    Removed {before - len(df)} exact duplicates")

    print("  Phase 2: Cross-source deduplication...")
    # Count filled fields per row for richness comparison
    important_fields = ["gpa", "undergrad_tier", "intern_level", "research_level",
                        "gender", "nationality", "major", "gre_quant", "toefl"]

    def count_filled(row):
        return sum(1 for f in important_fields if str(row.get(f, "")).strip() not in ("", "none", "unknown"))

    df["_richness"] = df.apply(count_filled, axis=1)

    # Normalize GPA for comparison
    df["_gpa_norm"] = df.apply(
        lambda r: normalize_gpa(r.get("gpa"), r.get("gpa_scale")), axis=1
    )

    # Group by (program, result, rounded GPA, season)
    df["_gpa_round"] = df["_gpa_norm"].round(1)
    df["_season_norm"] = df["season"].fillna("").str.strip().str.lower()
    df["_program_norm"] = df["program"].fillna("").str.strip().str.lower()
    df["_result_norm"] = df["result"].fillna("").str.strip().str.lower()

    # For each group with same program+result+gpa_round+season, keep the richest
    df = df.sort_values("_richness", ascending=False)

    # Only deduplicate where program, result, and gpa_round are ALL non-empty
    mask_valid = (
        (df["_program_norm"] != "") &
        (df["_result_norm"] != "") &
        (df["_gpa_round"].notna()) &
        (df["_season_norm"] != "")
    )

    # For valid rows: drop duplicates keeping richest
    valid = df[mask_valid].copy()
    invalid = df[~mask_valid].copy()

    before2 = len(valid)
    valid = valid.drop_duplicates(
        subset=["_program_norm", "_result_norm", "_gpa_round", "_season_norm"],
        keep="first"  # Already sorted by richness descending
    )
    print(f"    Removed {before2 - len(valid)} cross-source duplicates")

    df = pd.concat([valid, invalid], ignore_index=True)

    # Clean up temp columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])

    return df


def standardize(df):
    """Standardize all fields to canonical values."""
    print("  Standardizing fields...")

    # Program
    df["program"] = df["program"].apply(standardize_program)

    # Result - only keep valid results
    df["result"] = df["result"].apply(
        lambda x: str(x).strip().lower() if str(x).strip().lower() in VALID_RESULTS else ""
    )

    # Undergrad tier
    df["undergrad_tier"] = df["undergrad_tier"].apply(
        lambda x: standardize_field(x, UNDERGRAD_TIER_CANONICAL, "")
    )

    # Intern level
    df["intern_level"] = df["intern_level"].apply(
        lambda x: standardize_field(x, INTERN_LEVEL_CANONICAL, "")
    )

    # Intern relevance
    df["intern_relevance"] = df["intern_relevance"].apply(
        lambda x: standardize_field(x, INTERN_RELEVANCE_CANONICAL, "")
    )

    # Research level
    df["research_level"] = df["research_level"].apply(
        lambda x: standardize_field(x, RESEARCH_LEVEL_CANONICAL, "none")
    )

    # Nationality
    df["nationality"] = df["nationality"].apply(
        lambda x: standardize_field(x, NATIONALITY_CANONICAL, "")
    )

    # Gender
    df["gender"] = df["gender"].apply(
        lambda x: standardize_field(x, GENDER_CANONICAL, "")
    )

    # Season normalization
    def norm_season(s):
        if pd.isna(s) or str(s).strip() == "":
            return ""
        s = str(s).strip()
        # Already in format like "25Fall"
        m = re.match(r'^(\d{2})(Fall|Spring)$', s, re.I)
        if m:
            return f"{m.group(1)}{m.group(2).capitalize()}"
        # "2025 Fall" -> "25Fall"
        m2 = re.match(r'^(20)?(\d{2})\s*(Fall|Spring)', s, re.I)
        if m2:
            return f"{m2.group(2)}{m2.group(3).capitalize()}"
        return s

    df["season"] = df["season"].apply(norm_season)

    return df


def recompute_tiers(df):
    """Recompute quality tiers for all records."""
    print("  Recomputing quality tiers...")
    df["tier"] = df.apply(compute_tier, axis=1)
    return df


def generate_feature_matrix(df):
    """Generate numeric feature matrix for sklearn."""
    print("  Generating feature matrix...")

    # Only use records with accept/reject label
    model_df = df[df["result"].isin(["accepted", "rejected"])].copy()

    features = pd.DataFrame(index=model_df.index)

    # Program ID
    features["program_id"] = model_df["program"].map(PROGRAM_ID_MAP).fillna(-1).astype(int)

    # Result binary
    features["result_binary"] = (model_df["result"] == "accepted").astype(int)

    # GPA normalized to 4.0 scale
    features["gpa_normalized"] = model_df.apply(
        lambda r: normalize_gpa(r.get("gpa"), r.get("gpa_scale")), axis=1
    )

    # GRE quant
    features["gre_quant"] = pd.to_numeric(model_df["gre_quant"], errors="coerce")

    # GRE verbal
    features["gre_verbal"] = pd.to_numeric(model_df["gre_verbal"], errors="coerce")

    # TOEFL
    features["toefl"] = pd.to_numeric(model_df["toefl"], errors="coerce")

    # Undergrad tier encoded
    features["undergrad_tier_encoded"] = model_df["undergrad_tier"].map(UNDERGRAD_TIER_ENCODING)

    # Intern score
    features["intern_score"] = model_df["intern_level"].map(INTERN_SCORE).fillna(np.nan)
    # If intern_level is empty/none, leave as NaN
    features.loc[model_df["intern_level"].isin(["", "none"]) | model_df["intern_level"].isna(), "intern_score"] = np.nan

    # Intern count
    features["intern_count"] = pd.to_numeric(model_df["intern_count"], errors="coerce")

    # Research score
    features["research_score"] = model_df["research_level"].map(RESEARCH_SCORE)

    # Is international
    features["is_international"] = (~model_df["nationality"].isin(["us", ""])).astype(float)
    features.loc[model_df["nationality"] == "", "is_international"] = np.nan

    # Is female
    features["is_female"] = (model_df["gender"] == "F").astype(float)
    features.loc[model_df["gender"] == "", "is_female"] = np.nan

    # Major relevance
    features["major_relevance_score"] = model_df["major_relevance"].map(MAJOR_RELEVANCE_SCORE)

    # Has paper
    features["has_paper"] = (model_df["has_paper"] == "yes").astype(float)
    features.loc[model_df["has_paper"] == "", "has_paper"] = np.nan

    # Has research
    features["has_research"] = (model_df["has_research"] == "yes").astype(float)
    features.loc[model_df["has_research"] == "", "has_research"] = np.nan

    return features


def generate_report(df):
    """Generate comprehensive data quality report."""
    print("  Generating quality report...")

    total = len(df)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# MFE Admission Data Quality Report\n",
        f"**Generated**: {now}",
        f"**Total Records**: {total}\n",
    ]

    # By source
    lines.append("## Records by Source\n")
    lines.append("| Source | Count | % |")
    lines.append("|--------|------:|--:|")
    for src, cnt in df["source"].value_counts().items():
        lines.append(f"| {src} | {cnt} | {cnt*100/total:.1f}% |")

    # By tier
    lines.append("\n## Quality Tier Distribution\n")
    lines.append("| Tier | Description | Count | % |")
    lines.append("|------|-------------|------:|--:|")
    tier_info = {
        "A": ("A (Gold)", "4+ rich fields + result"),
        "B": ("B (Silver)", "2-3 rich fields + result"),
        "C": ("C (Bronze)", "GPA/GRE + result"),
        "D": ("D (Basic)", "Result only or no result"),
    }
    for t in ["A", "B", "C", "D"]:
        cnt = len(df[df["tier"] == t])
        name, desc = tier_info[t]
        lines.append(f"| {name} | {desc} | {cnt} | {cnt*100/total:.1f}% |")

    # Tier by source
    lines.append("\n## Tier Distribution by Source\n")
    lines.append("| Source | A | B | C | D | Total |")
    lines.append("|--------|--:|--:|--:|--:|------:|")
    for src in df["source"].value_counts().index:
        src_df = df[df["source"] == src]
        a = len(src_df[src_df["tier"] == "A"])
        b = len(src_df[src_df["tier"] == "B"])
        c = len(src_df[src_df["tier"] == "C"])
        d = len(src_df[src_df["tier"] == "D"])
        lines.append(f"| {src} | {a} | {b} | {c} | {d} | {len(src_df)} |")

    # Field coverage
    lines.append("\n## Field Coverage\n")
    lines.append("| Field | Records with data | Coverage % |")
    lines.append("|-------|------------------:|-----------:|")
    skip_fields = {"id", "source", "tier", "raw_text"}
    for field in df.columns:
        if field in skip_fields:
            continue
        cnt = len(df[df[field].astype(str).str.strip().ne("") & df[field].notna()])
        lines.append(f"| {field} | {cnt} | {cnt*100/total:.1f}% |")

    # Top programs
    lines.append("\n## Top 25 Programs by Record Count\n")
    lines.append("| Program | Count |")
    lines.append("|---------|------:|")
    prog_counts = df[df["program"].astype(str).str.strip().ne("")]["program"].value_counts().head(25)
    for prog, cnt in prog_counts.items():
        lines.append(f"| {prog} | {cnt} |")

    # Result distribution
    lines.append("\n## Result Distribution\n")
    lines.append("| Result | Count | % |")
    lines.append("|--------|------:|--:|")
    result_df = df[df["result"].astype(str).str.strip().ne("")]
    for r, cnt in result_df["result"].value_counts().items():
        lines.append(f"| {r} | {cnt} | {cnt*100/len(result_df):.1f}% |")

    # Top seasons
    lines.append("\n## Top Seasons\n")
    lines.append("| Season | Count |")
    lines.append("|--------|------:|")
    season_df = df[df["season"].astype(str).str.strip().ne("")]
    for s, cnt in season_df["season"].value_counts().head(10).items():
        lines.append(f"| {s} | {cnt} |")

    # Model readiness
    ab = len(df[df["tier"].isin(["A", "B"])])
    c = len(df[df["tier"] == "C"])
    d = len(df[df["tier"] == "D"])
    labeled = len(df[df["result"].isin(["accepted", "rejected", "waitlisted"])])

    lines.append("\n## Model-Readiness Summary\n")
    lines.append(f"- **Tier A+B (model-ready with rich features)**: {ab} records ({ab*100/total:.1f}%)")
    lines.append(f"- **Tier C (basic features)**: {c} records ({c*100/total:.1f}%)")
    lines.append(f"- **Tier D (needs enrichment)**: {d} records ({d*100/total:.1f}%)")
    lines.append(f"- **Records with accept/reject/waitlist label**: {labeled} ({labeled*100/total:.1f}%)")

    # Training data file stats
    lines.append("\n## Training Data Files\n")
    lines.append("| File | Records | Description |")
    lines.append("|------|--------:|-------------|")
    for fpath, desc in [
        (OUTPUT_FULL, "All cleaned records"),
        (OUTPUT_MODEL, "Tier A+B+C with accept/reject"),
        (OUTPUT_RICH, "Tier A+B only (multi-feature)"),
        (FEATURE_MATRIX, "Numeric feature matrix for sklearn"),
    ]:
        if fpath.exists():
            cnt = len(pd.read_csv(fpath))
            lines.append(f"| {fpath.name} | {cnt} | {desc} |")

    report = "\n".join(lines)
    REPORT_MD.write_text(report)
    print(f"  Report saved to {REPORT_MD}")


def main():
    print("=" * 70)
    print("QuantPath Training Data Preparation")
    print(f"Started: {datetime.now()}")
    print("=" * 70)

    # Load data
    print(f"\nLoading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    print(f"  Loaded {len(df)} records")

    # Step 1: Deduplicate
    print("\n--- STEP 1: Deduplication ---")
    df = deduplicate(df)
    print(f"  After dedup: {len(df)} records")

    # Step 2: Standardize
    print("\n--- STEP 2: Standardization ---")
    df = standardize(df)

    # Step 3: Recompute tiers
    print("\n--- STEP 3: Recompute tiers ---")
    df = recompute_tiers(df)
    for t in ["A", "B", "C", "D"]:
        cnt = len(df[df["tier"] == t])
        print(f"    Tier {t}: {cnt}")

    # Step 4: Output files
    print("\n--- STEP 4: Output files ---")

    # Full dataset
    df.to_csv(OUTPUT_FULL, index=False)
    print(f"  training_data_full.csv: {len(df)} records")

    # Model-ready (A+B+C with labels)
    model_df = df[
        df["tier"].isin(["A", "B", "C"]) &
        df["result"].isin(["accepted", "rejected"])
    ]
    model_df.to_csv(OUTPUT_MODEL, index=False)
    print(f"  training_data_model.csv: {len(model_df)} records")

    # Rich only (A+B)
    rich_df = df[
        df["tier"].isin(["A", "B"]) &
        df["result"].isin(["accepted", "rejected", "waitlisted"])
    ]
    rich_df.to_csv(OUTPUT_RICH, index=False)
    print(f"  training_data_rich.csv: {len(rich_df)} records")

    # Step 5: Feature matrix
    print("\n--- STEP 5: Feature matrix ---")
    features = generate_feature_matrix(df)
    features.to_csv(FEATURE_MATRIX, index=False)
    print(f"  feature_matrix.csv: {len(features)} records x {len(features.columns)} features")

    # Show feature coverage
    for col in features.columns:
        non_null = features[col].notna().sum()
        print(f"    {col}: {non_null} non-null ({non_null*100/len(features):.1f}%)")

    # Step 6: Quality report
    print("\n--- STEP 6: Quality report ---")
    generate_report(df)

    print("\n" + "=" * 70)
    print(f"DONE: {datetime.now()}")
    print(f"Final total: {len(df)} records")
    print(f"Model-ready (A+B+C): {len(model_df)} records")
    print(f"Rich (A+B): {len(rich_df)} records")
    print("=" * 70)


if __name__ == "__main__":
    main()
