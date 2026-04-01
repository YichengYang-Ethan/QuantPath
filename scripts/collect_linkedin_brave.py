#!/usr/bin/env python3
"""
Process LinkedIn profile data from Brave Search results into admission records.
This script processes hardcoded profile data extracted from Brave Search results.
"""

import csv
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path("/Users/ethanyang/QuantPath/data/admissions")
UNIFIED_CSV = BASE / "collected_multidim.csv"

UNIFIED_FIELDS = [
    "id", "source", "tier", "program", "result", "season",
    "gpa", "gpa_scale", "gre_quant", "gre_verbal", "toefl",
    "undergrad_school", "undergrad_tier", "undergrad_country",
    "major", "major_relevance",
    "intern_count", "intern_level", "intern_relevance",
    "has_paper", "has_research", "research_level",
    "gender", "nationality",
    "raw_text",
]


def make_id(source, *parts):
    raw = f"{source}:{'|'.join(str(p) for p in parts)}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def content_hash(text):
    norm = re.sub(r'\s+', ' ', str(text).lower().strip())[:500]
    return hashlib.md5(norm.encode()).hexdigest()[:16]


def compute_tier(rec):
    has_result = rec.get("result") and rec["result"] in ("accepted", "rejected", "waitlisted")
    rich_fields = 0
    if rec.get("gpa"): rich_fields += 1
    if rec.get("undergrad_tier"): rich_fields += 1
    if rec.get("intern_level"): rich_fields += 1
    if rec.get("research_level") and rec["research_level"] != "none": rich_fields += 1
    if rec.get("gender"): rich_fields += 1
    if rec.get("nationality"): rich_fields += 1
    if rec.get("major"): rich_fields += 1
    if rec.get("gre_quant"): rich_fields += 1
    if not has_result:
        return "D"
    if rich_fields >= 4:
        return "A"
    if rich_fields >= 2:
        return "B"
    if rec.get("gpa") or rec.get("gre_quant"):
        return "C"
    return "D"


# LinkedIn profiles extracted from Brave Search results
# Each entry: (name, program, undergrad, undergrad_tier, undergrad_country, season, intern, nationality_hint, major)
PROFILES = [
    # CMU MSCF
    ("Caroline Liu", "cmu-mscf", "Wellesley College", "T50", "US", "27Spring", "quant_intern", "chinese", "mathematics"),
    ("Jamie Ren", "cmu-mscf", "", "", "", "25Spring", "squarepoint", "chinese", ""),
    ("Yirun Wang", "cmu-mscf", "", "", "", "25Spring", "jpmorgan", "chinese", ""),
    ("Xinhan Qin", "cmu-mscf", "", "", "", "25Spring", "fidelity", "chinese", ""),
    ("Mengxin Cao", "cmu-mscf", "", "", "", "25Spring", "china_quant", "chinese", ""),
    ("Mingxuan Wu", "cmu-mscf", "", "", "", "25Spring", "citadel_securities", "chinese", ""),
    ("Yujing Zhang", "cmu-mscf", "", "", "", "25Spring", "bofa", "chinese", ""),
    ("Oscar Ma", "cmu-mscf", "", "", "", "25Spring", "mako_trading", "chinese", ""),
    ("Maria Jose Ruiz", "cmu-mscf", "", "", "", "25Spring", "calamos", "international", ""),
    ("Liye Zhu", "cmu-mscf", "Peking University", "C9", "CN", "26Spring", "millennium", "chinese", "finance"),
    ("Manjunath Bhat", "cmu-mscf", "", "", "IN", "25Spring", "millennium", "indian", ""),

    # Princeton MFin
    ("Pragnya Akella", "princeton-mfin", "IIT Madras", "IIT", "IN", "26Spring", "jpmorgan", "indian", ""),
    ("Yuchen Yang", "princeton-mfin", "Notre Dame", "T20", "US", "26Spring", "", "chinese", "mathematics"),
    ("Shiying Zhang", "princeton-mfin", "Tsinghua", "C9", "CN", "26Spring", "jpmorgan", "chinese", "finance"),
    ("Yang Ou", "princeton-mfin", "", "", "", "26Spring", "verition", "chinese", ""),
    ("Yuchen Yu", "princeton-mfin", "", "", "", "26Spring", "quant", "chinese", ""),
    ("Ivy M", "princeton-mfin", "", "", "", "26Spring", "susquehanna", "chinese", ""),
    ("Daniel Zhuang", "princeton-mfin", "U of Toronto", "T30", "CA", "27Spring", "", "chinese", ""),
    ("Ying Sun", "princeton-mfin", "CMU", "T30", "US", "26Spring", "verition", "chinese", "computer_science"),
    ("Siddhartha Kahali", "princeton-mfin", "IIT Delhi", "IIT", "IN", "25Spring", "arrowstreet", "indian", ""),
    ("Thomas Li", "princeton-mfin", "WashU", "T20", "US", "27Spring", "", "", "statistics"),
    ("Ran Lei", "princeton-mfin", "Peking University", "C9", "CN", "26Spring", "huatai", "chinese", "economics"),
    ("Jenny Jiang", "princeton-mfin", "UC Berkeley", "T30", "US", "24Spring", "bnp", "chinese", "mathematics"),
    ("Angelina Zhao", "princeton-mfin", "UC Berkeley Haas", "T30", "US", "25Spring", "graham_capital", "chinese", ""),
    ("Yu Ki Ho", "princeton-mfin", "", "", "", "25Spring", "balyasny", "", ""),
    ("Cathy Wu", "princeton-mfin", "Peking University", "C9", "CN", "25Spring", "", "chinese", ""),
    ("Jing W", "princeton-mfin", "", "", "", "23Spring", "ubs", "chinese", ""),
    ("Lucrezia Forcellini", "princeton-mfin", "", "", "EU", "25Spring", "", "european", ""),
    ("Jared Day", "mit-mfin", "NYU Stern", "T30", "US", "23Spring", "pjt", "us", "finance"),

    # Baruch MFE
    ("Dhruv Joshi", "baruch-mfe", "", "", "IN", "26Fall", "", "indian", ""),
    ("Andrew Verzino", "baruch-mfe", "", "", "US", "26Fall", "", "us", ""),
    ("Kai Zhang", "baruch-mfe", "", "", "", "25Spring", "hap_capital", "chinese", ""),
    ("Miriam Mariotti", "baruch-mfe", "", "", "", "25Spring", "", "european", ""),
    ("Oisin Kehoe", "baruch-mfe", "", "", "", "25Spring", "citadel", "european", ""),
    ("Yuxuan Wang", "baruch-mfe", "", "", "", "25Spring", "squarepoint", "chinese", ""),
    ("Gabriele Bernardino", "baruch-mfe", "Bocconi", "T30", "EU", "25Spring", "point72", "european", "economics"),
    ("Linting Wang", "baruch-mfe", "", "", "", "26Fall", "octaura", "chinese", ""),
    ("Yiyang Fan", "baruch-mfe", "", "", "", "25Spring", "", "chinese", ""),
    ("Jing Chen", "baruch-mfe", "", "", "", "25Spring", "axq_capital", "chinese", ""),
    ("Richard Griffiths", "baruch-mfe", "", "", "", "25Spring", "awr_capital", "", ""),

    # Columbia MSFE
    ("Chen Ye", "columbia-msfe", "", "", "", "26Spring", "ubs", "chinese", ""),
    ("Ella Sanderson", "columbia-msfe", "", "", "US", "26Spring", "", "us", "biology"),

    # Cornell MFE
    ("Jiaming Yao", "cornell-mfe", "", "", "", "25Spring", "one_william", "chinese", ""),
    ("Artem Ezhov", "cornell-mfe", "Cornell", "T20", "US", "24Fall", "jpmorgan", "international", "operations_research"),
    ("Riley Burke", "cornell-mfe", "", "", "US", "25Spring", "", "us", ""),
    ("David Niu", "cornell-mfe", "NUS", "T30", "SG", "26Spring", "", "chinese", "computer_science"),
    ("Sirui Zhao", "cornell-mfe", "Peking University", "C9", "CN", "25Spring", "", "chinese", "finance"),
    ("Xiyi Fan", "cornell-mfe", "", "", "", "25Spring", "morgan_stanley", "chinese", ""),
    ("Jiaxin Hu", "cornell-mfe", "", "", "", "25Spring", "", "chinese", ""),
    ("Chengyue Zhang", "cornell-mfe", "", "", "", "24Spring", "seven_grand", "chinese", ""),
    ("Linkun Lei", "cornell-mfe", "WHU", "985", "CN", "25Spring", "changjiang_sec", "chinese", ""),
    ("Sarvesh Sakhare", "cornell-mfe", "IIT Madras", "IIT", "IN", "25Spring", "alliancebernstein", "indian", ""),
    ("Ananya Mohapatra", "cornell-mfe", "BITS Pilani", "T50", "IN", "25Spring", "", "indian", "economics"),
    ("Xiyuan Feng", "cornell-mfe", "", "", "", "25Spring", "kraneshares", "chinese", ""),
    ("Fenghua Dong", "cornell-mfe", "", "", "", "25Spring", "gm_financial", "chinese", ""),
    ("Yunfan Yang", "cornell-mfe", "", "", "", "25Spring", "citi", "chinese", ""),
    ("Adarsh Pandey", "cornell-mfe", "IIT", "IIT", "IN", "25Spring", "simplify", "indian", ""),
    ("Kexin Deng", "cornell-mfe", "NYU Shanghai", "T30", "CN", "25Spring", "luoshu", "chinese", ""),
    ("Lewis Tian", "cornell-mfe", "", "", "", "23Spring", "bofa", "chinese", ""),
    ("Evelyn Shen", "cornell-mfe", "", "", "", "25Spring", "jpmorgan", "chinese", ""),
    ("Kayla Y", "cornell-mfe", "", "", "", "25Spring", "goldman_sachs", "chinese", ""),

    # NYU Tandon MFE
    ("Yujia Hou", "nyu-tandon-mfe", "", "", "CN", "26Spring", "zhongtai_sec", "chinese", ""),
    ("Sai Kishore", "nyu-tandon-mfe", "", "", "IN", "25Spring", "estee", "indian", ""),
    ("Edmure Windsor", "nyu-tandon-mfe", "", "", "", "24Spring", "capital_one", "", ""),
    ("Ramya Venkatesh", "nyu-tandon-mfe", "", "", "", "25Spring", "", "indian", ""),
    ("Li Ji", "nyu-tandon-mfe", "", "", "CN", "25Spring", "huatai", "chinese", ""),
    ("Suxiang Zhong", "nyu-tandon-mfe", "", "", "", "25Spring", "acadia", "chinese", ""),
    ("Wanzhen Tang", "nyu-tandon-mfe", "", "", "CN", "26Spring", "", "chinese", ""),
    ("Yijia Wang", "nyu-tandon-mfe", "", "", "", "25Spring", "", "chinese", ""),
    ("Harsh Patel", "nyu-tandon-mfe", "", "", "IN", "25Spring", "", "indian", ""),
    ("Gitika Daswani", "nyu-tandon-mfe", "", "", "IN", "25Spring", "orchestrade", "indian", ""),
    ("Aditya Shah", "nyu-tandon-mfe", "", "", "IN", "25Spring", "", "indian", ""),
    ("Barry Blecherman", "nyu-tandon-mfe", "", "", "US", "26Spring", "", "us", ""),
    ("Aditya Daftari", "nyu-tandon-mfe", "", "", "IN", "25Spring", "goldman_sachs", "indian", ""),
    ("Xin Huang", "nyu-tandon-mfe", "", "", "CN", "25Spring", "", "chinese", ""),
    ("Yixin Xu", "nyu-tandon-mfe", "", "", "CN", "25Spring", "goldman_sachs", "chinese", ""),
    ("Jiqiang Zhang", "nyu-tandon-mfe", "", "", "CN", "25Spring", "", "chinese", ""),

    # UCLA MFE
    ("Pranay Kumar", "ucla-mfe", "", "", "IN", "26Spring", "ucla", "indian", ""),
    ("Tianqi Han", "usc-msmf", "", "", "CN", "21Fall", "joinquant", "chinese", ""),

    # Berkeley MFE
    ("Yuan Yuan Gu", "berkeley-mfe", "UC Berkeley", "T30", "US", "25Spring", "workday", "chinese", ""),

    # Misc / additional from Shiying Zhang's offer list
    ("Shiying Zhang alt", "columbia-msfe", "Tsinghua", "C9", "CN", "26Spring", "jpmorgan", "chinese", "finance"),
    ("Shiying Zhang alt2", "mit-mfin", "Tsinghua", "C9", "CN", "26Spring", "jpmorgan", "chinese", "finance"),
    ("Shiying Zhang alt3", "oxford-mcf", "Tsinghua", "C9", "CN", "26Spring", "jpmorgan", "chinese", "finance"),
]


def process_profiles():
    records = []

    for profile in PROFILES:
        name, program, undergrad, undergrad_tier, undergrad_country, season, intern_hint, nationality, major = profile

        # Determine intern level and relevance
        intern_level = ""
        intern_relevance = ""
        intern_count = ""
        top_quant = ["citadel", "jane_street", "two_sigma", "de_shaw", "tower", "jump",
                     "hrt", "optiver", "imc", "virtu", "squarepoint", "susquehanna",
                     "aqr", "millennium", "point72", "bridgewater", "verition",
                     "balyasny", "mako_trading", "arrowstreet", "citadel_securities",
                     "graham_capital", "axq_capital", "seven_grand", "hap_capital",
                     "octaura"]
        finance = ["jpmorgan", "goldman_sachs", "morgan_stanley", "bofa", "citi",
                   "ubs", "barclays", "deutsche", "bnp", "pjt",
                   "huatai", "changjiang_sec", "zhongtai_sec", "luoshu",
                   "calamos", "alliancebernstein", "one_william",
                   "gm_financial", "kraneshares", "estee", "orchestrade", "acadia"]
        tech = ["capital_one", "workday"]

        if intern_hint:
            intern_count = "1"
            if any(q in intern_hint for q in top_quant):
                intern_level = "us_top_quant"
                intern_relevance = "quant_direct"
            elif "china_quant" in intern_hint or "joinquant" in intern_hint:
                intern_level = "china_top"
                intern_relevance = "quant_direct"
            elif any(f in intern_hint for f in finance):
                intern_level = "us_finance"
                intern_relevance = "somewhat_related"
            elif any(t in intern_hint for t in tech):
                intern_level = "us_tech"
                intern_relevance = "somewhat_related"
            elif intern_hint == "quant_intern" or intern_hint == "quant":
                intern_level = "us_top_quant"
                intern_relevance = "quant_direct"
            else:
                intern_level = "us_finance"
                intern_relevance = "somewhat_related"

        # Major mapping
        major_field = ""
        major_rel = ""
        if major:
            major_map = {
                "mathematics": ("mathematics", "quant_direct"),
                "statistics": ("statistics", "quant_direct"),
                "computer_science": ("computer_science", "quant_direct"),
                "physics": ("physics", "quant_direct"),
                "finance": ("finance", "somewhat_related"),
                "economics": ("economics", "somewhat_related"),
                "operations_research": ("operations_research", "quant_direct"),
                "biology": ("biology", "not_related"),
                "electrical_engineering": ("electrical_engineering", "somewhat_related"),
            }
            if major in major_map:
                major_field, major_rel = major_map[major]

        raw_text = f"LinkedIn profile: {name}, {program}, undergrad: {undergrad or 'unknown'}, season: {season}"

        rec = {
            "id": make_id("linkedin", name, program),
            "source": "linkedin",
            "program": program,
            "result": "accepted",  # They're enrolled, so accepted
            "season": season,
            "gpa": "",
            "gpa_scale": "",
            "gre_quant": "",
            "gre_verbal": "",
            "toefl": "",
            "undergrad_school": undergrad,
            "undergrad_tier": undergrad_tier,
            "undergrad_country": undergrad_country,
            "major": major_field,
            "major_relevance": major_rel,
            "intern_count": intern_count,
            "intern_level": intern_level,
            "intern_relevance": intern_relevance,
            "has_paper": "",
            "has_research": "",
            "research_level": "none",
            "gender": "",
            "nationality": nationality,
            "raw_text": raw_text,
        }
        rec["tier"] = compute_tier(rec)
        records.append(rec)

    return records


def main():
    print("Processing LinkedIn profiles from Brave Search...")
    records = process_profiles()

    # Load existing and deduplicate
    if UNIFIED_CSV.exists():
        existing = pd.read_csv(UNIFIED_CSV, dtype=str).fillna("")
        existing_ids = set(existing["id"].tolist())
        existing_hashes = set()
        for _, row in existing.iterrows():
            existing_hashes.add(content_hash(str(row.get("raw_text", ""))[:500]))
    else:
        existing = pd.DataFrame(columns=UNIFIED_FIELDS)
        existing_ids = set()
        existing_hashes = set()

    new_records = []
    for rec in records:
        raw_hash = content_hash(str(rec.get("raw_text", ""))[:500])
        if rec["id"] not in existing_ids and raw_hash not in existing_hashes:
            new_records.append(rec)
            existing_ids.add(rec["id"])
            existing_hashes.add(raw_hash)

    print(f"  Total profiles processed: {len(records)}")
    print(f"  New unique records: {len(new_records)}")

    if new_records:
        new_df = pd.DataFrame(new_records, columns=UNIFIED_FIELDS).fillna("")
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(UNIFIED_CSV, index=False)
        print(f"  Saved {len(combined)} total records to {UNIFIED_CSV}")

        # Count tiers
        for t in ["A", "B", "C", "D"]:
            cnt = sum(1 for r in new_records if r["tier"] == t)
            print(f"    Tier {t}: {cnt}")
    else:
        print("  No new records to add.")


if __name__ == "__main__":
    main()
