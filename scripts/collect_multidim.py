#!/usr/bin/env python3
# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""
QuantPath Multi-Dimensional MFE Admission Data Collection
Three pipelines: Reddit, GradCafe, 1Point3Acres offer
Outputs unified CSV with quality tiers.
"""

import csv
import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("/Users/ethanyang/QuantPath/data/admissions")
REDDIT_CSV = BASE / "reddit_mfe_posts.csv"
GRADCAFE_CSV = BASE / "gradcafe_mfe.csv"
OFFER_CSV = BASE / "offer_1p3a_enriched.csv"
UNIFIED_CSV = BASE / "collected_multidim.csv"
REPORT_MD = BASE / "data_quality_report.md"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36 QuantPath-Research/1.0"
}

# ── Known programs ─────────────────────────────────────────────────────────
MFE_PROGRAMS = {
    "baruch": "baruch-mfe",
    "cmu": "cmu-mscf",
    "mscf": "cmu-mscf",
    "columbia": "columbia-msfe",
    "princeton": "princeton-mfin",
    "mit": "mit-mfin",
    "berkeley": "berkeley-mfe",
    "uchicago": "uchicago-msfm",
    "chicago": "uchicago-msfm",
    "gatech": "gatech-qcf",
    "georgia tech": "gatech-qcf",
    "cornell": "cornell-mfe",
    "nyu": "nyu-mfe",
    "stanford": "stanford-mcf",
    "ucla": "ucla-mfe",
    "uiuc": "uiuc-msfe",
    "illinois": "uiuc-msfe",
    "rutgers": "rutgers-mqf",
    "bu": "bu-msmf",
    "boston university": "bu-msmf",
    "toronto": "toronto-mmf",
    "waterloo": "waterloo-mqf",
    "michigan": "michigan-qfr",
    "carnegie mellon": "cmu-mscf",
    "stony brook": "stonybrook-qf",
    "fordham": "fordham-msqf",
    "nyu tandon": "nyu-mfe",
}

T20_US = {
    "princeton", "mit", "harvard", "stanford", "yale", "caltech",
    "uchicago", "duke", "upenn", "johns hopkins", "northwestern",
    "columbia", "cornell", "rice", "dartmouth", "vanderbilt",
    "washington in st. louis", "notre dame", "emory", "georgetown",
}
T30_US = T20_US | {
    "michigan", "berkeley", "usc", "virginia", "nyu", "tufts",
    "unc", "florida", "wake forest", "carnegie mellon",
}
T50_US = T30_US | {
    "uiuc", "gatech", "wisconsin", "purdue", "ohio state", "penn state",
    "texas austin", "ucsb", "ucsd", "uc davis", "uc irvine",
    "boston university", "northeastern", "villanova", "lehigh",
    "tulane", "pepperdine", "stony brook", "rochester",
}

C9_CN = {"北大", "清华", "复旦", "上交", "浙大", "中科大", "南大", "哈工大", "西交"}
TOP985_CN = C9_CN | {"人大", "北航", "同济", "武大", "中山", "华科", "南开", "厦大", "天大", "中科院"}

# ── Unified record schema ────────────────────────────────────────────────
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


def safe_float(s):
    """Try to parse a float, return None on failure."""
    if not s:
        return None
    try:
        return float(str(s).strip())
    except (ValueError, TypeError):
        return None


def safe_int(s):
    if not s:
        return None
    try:
        return int(float(str(s).strip()))
    except (ValueError, TypeError):
        return None


def make_id(source, *parts):
    """Deterministic record id."""
    raw = f"{source}:{'|'.join(str(p) for p in parts)}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Text parsing helpers ───────────────────────────────────────────────────
RE_GPA = re.compile(r'(?:GPA|gpa|Gpa)[\s:=]*(\d\.\d{1,2})\b')
RE_GPA_100 = re.compile(r'(?:GPA|gpa|均分|绩点)[\s:=]*(\d{2,3}(?:\.\d{1,2})?)\s*/?\s*(?:100|4\.0|4)', re.I)
RE_GRE_TOTAL = re.compile(r'(?:GRE|gre)[\s:=]*(\d{3})\b')
RE_GRE_Q = re.compile(r'(?:GRE|gre)\s*(?:quant|Q|数学)[\s:=]*(\d{3})\b', re.I)
RE_GRE_V = re.compile(r'(?:GRE|gre)\s*(?:verbal|V|语文)[\s:=]*(\d{3})\b', re.I)
RE_TOEFL = re.compile(r'(?:TOEFL|toefl|托福)[\s:=]*(\d{2,3})\b', re.I)
RE_IELTS = re.compile(r'(?:IELTS|ielts|雅思)[\s:=]*(\d\.?\d?)\b', re.I)
RE_SEASON = re.compile(r'(20\d{2})\s*(Fall|Spring|Winter|Fa|Sp)', re.I)


def extract_gpa(text):
    """Return (gpa_float, scale_str) or (None, None)."""
    m = RE_GPA.search(text)
    if m:
        v = float(m.group(1))
        if 0 < v <= 4.0:
            return v, "4.0"
    m = RE_GPA_100.search(text)
    if m:
        v = float(m.group(1))
        if 50 <= v <= 100:
            return v, "100"
        elif 0 < v <= 4.0:
            return v, "4.0"
    return None, None


def extract_gre(text):
    """Return (quant, verbal) or (None, None)."""
    q = RE_GRE_Q.search(text)
    v = RE_GRE_V.search(text)
    quant = safe_int(q.group(1)) if q else None
    verbal = safe_int(v.group(1)) if v else None
    if quant is None and verbal is None:
        m = RE_GRE_TOTAL.search(text)
        if m:
            total = int(m.group(1))
            if 260 <= total <= 340:
                # Estimate: most MFE applicants have Q > V
                quant = min(170, total - 150)
                verbal = total - quant
    return quant, verbal


def extract_toefl(text):
    m = RE_TOEFL.search(text)
    if m:
        v = int(m.group(1))
        if 60 <= v <= 120:
            return v
    return None


def extract_result(text):
    t = text.lower()
    if any(w in t for w in ["accepted", "admitted", "offer", "ad无奖", "ad小奖", "ad", "got in", "i'm in"]):
        if any(w in t for w in ["rejected", "denied", "rej"]):
            pass  # ambiguous
        else:
            return "accepted"
    if any(w in t for w in ["rejected", "denied", "rej", "拒了", "拒"]):
        return "rejected"
    if any(w in t for w in ["waitlist", "waitlisted", "wl", "waiting"]):
        return "waitlisted"
    return None


def extract_program(text):
    t = text.lower()
    for key, prog in MFE_PROGRAMS.items():
        if key in t:
            return prog
    # Check for generic MFE
    if any(w in t for w in ["mfe", "financial engineering", "金融工程", "金工"]):
        return "mfe-unknown"
    if any(w in t for w in ["mscf", "computational finance", "计算金融"]):
        return "cmu-mscf"
    if any(w in t for w in ["msfm", "mathematical finance", "数学金融"]):
        return "msfm-unknown"
    return None


def extract_season(text):
    m = RE_SEASON.search(text)
    if m:
        year = m.group(1)
        sem = m.group(2).lower()
        if sem.startswith("f"):
            return f"{year} Fall"
        elif sem.startswith("sp") or sem.startswith("s"):
            return f"{year} Spring"
    # Check for shorthand: 26Fall, 25Fall
    m2 = re.search(r'(\d{2})(Fall|Spring|Fa|Sp)', text, re.I)
    if m2:
        year = "20" + m2.group(1)
        sem = m2.group(2).lower()
        if sem.startswith("f"):
            return f"{year} Fall"
        return f"{year} Spring"
    return None


def extract_nationality(text):
    t = text.lower()
    if any(w in t for w in ["chinese", "china", "中国", "大陆", "内地", "cn", "prc"]):
        return "chinese"
    if any(w in t for w in ["indian", "india"]):
        return "indian"
    if any(w in t for w in ["american", "us citizen", "domestic"]):
        return "us"
    if any(w in t for w in ["korean", "korea"]):
        return "korean"
    if any(w in t for w in ["international", "f1", "f-1"]):
        return "international"
    return None


def extract_gender(text):
    t = text.lower()
    # Simple heuristics from pronoun usage or explicit mention
    male_signals = len(re.findall(r'\b(he|him|his|male|男)\b', t))
    female_signals = len(re.findall(r'\b(she|her|hers|female|女)\b', t))
    if male_signals > female_signals and male_signals > 0:
        return "M"
    if female_signals > male_signals and female_signals > 0:
        return "F"
    return None


def extract_undergrad(text):
    """Return (school_name, tier, country)."""
    t = text.lower()
    # Chinese schools
    for s in C9_CN:
        if s in text:
            return s, "C9", "CN"
    for s in TOP985_CN:
        if s in text:
            return s, "985", "CN"
    if any(w in text for w in ["211", "两财一贸"]):
        return None, "211", "CN"
    if "985" in text:
        return None, "985", "CN"
    # US schools
    for s in T20_US:
        if s in t:
            return s, "T20", "US"
    for s in T30_US - T20_US:
        if s in t:
            return s, "T30", "US"
    for s in T50_US - T30_US:
        if s in t:
            return s, "T50", "US"
    return None, None, None


def extract_internships(text):
    """Return (count, level, relevance)."""
    t = text.lower()
    # Count internship mentions
    count = len(re.findall(r'(?:intern|实习|量化)', t))
    count = min(count, 5)  # cap
    if count == 0:
        numbers = re.findall(r'(\d)\s*(?:段|份|个)\s*(?:实习|intern)', text)
        if numbers:
            count = int(numbers[0])

    level = None
    relevance = None
    if any(w in t for w in ["citadel", "jane street", "two sigma", "de shaw", "tower",
                             "jump trading", "hrt", "hudson river", "optiver", "imc",
                             "virtu", "top quant", "顶级量化"]):
        level = "us_top_quant"
        relevance = "quant_direct"
    elif any(w in t for w in ["goldman", "morgan stanley", "jpmorgan", "jp morgan",
                               "citi", "boa", "barclays", "ubs", "deutsche",
                               "三中一华", "中金", "中信", "华泰", "国泰"]):
        level = "us_finance"
        relevance = "somewhat_related"
    elif any(w in t for w in ["百亿私募", "top私募", "头部私募", "top quant fund",
                               "幻方", "九坤", "明汯", "锐天", "量化"]):
        level = "china_top"
        relevance = "quant_direct"
    elif any(w in t for w in ["私募", "券商", "基金", "intern", "实习"]):
        level = "china_normal"
        relevance = "somewhat_related"

    return count or None, level, relevance


def extract_research(text):
    t = text.lower()
    has_paper = None
    has_research = None
    level = None  # None = not mentioned/unknown (distinct from "none" = confirmed no research)
    if any(w in t for w in ["published", "paper", "论文发表", "publication", "journal"]):
        has_paper = "yes"
        level = "published"
    if any(w in t for w in ["research", "科研", "研究", "ra ", "research assistant"]):
        has_research = "yes"
        if level is None:
            level = "relevant_experience"
    return has_paper, has_research, level


def extract_major(text):
    t = text.lower()
    if any(w in t for w in ["金工", "金融工程", "financial engineering"]):
        return "financial_engineering", "quant_direct"
    if any(w in t for w in ["数学", "math", "applied math"]):
        return "mathematics", "quant_direct"
    if any(w in t for w in ["统计", "stat", "statistics"]):
        return "statistics", "quant_direct"
    if any(w in t for w in ["cs", "computer science", "计算机"]):
        return "computer_science", "quant_direct"
    if any(w in t for w in ["physics", "物理"]):
        return "physics", "quant_direct"
    if any(w in t for w in ["econ", "经济"]):
        return "economics", "somewhat_related"
    if any(w in t for w in ["金融", "finance"]):
        return "finance", "somewhat_related"
    if any(w in t for w in ["ee", "electrical", "电气", "电子"]):
        return "electrical_engineering", "somewhat_related"
    return None, None


def compute_tier(rec):
    """Compute data quality tier A/B/C/D based on field coverage."""
    has_result = rec.get("result") and rec["result"] in ("accepted", "rejected", "waitlisted")
    rich_fields = 0
    if rec.get("gpa"):
        rich_fields += 1
    if rec.get("undergrad_tier"):
        rich_fields += 1
    if rec.get("intern_level"):
        rich_fields += 1
    if rec.get("research_level") and rec["research_level"] != "none":
        rich_fields += 1
    if rec.get("gender"):
        rich_fields += 1
    if rec.get("nationality"):
        rich_fields += 1

    if not has_result:
        return "D"
    if rich_fields >= 4:
        return "A"
    if rich_fields >= 2:
        return "B"
    if rec.get("gpa") or rec.get("gre_quant"):
        return "C"
    return "D"


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE 1: Reddit
# ════════════════════════════════════════════════════════════════════════════
def pipeline_reddit():
    print("\n" + "=" * 70)
    print("PIPELINE 1: Reddit")
    print("=" * 70)

    subreddits = [
        "gradadmissions", "QuantFinance", "quant",
        "financialengineering", "UIUC", "cmu",
        "GraduateAdmissions", "FinancialCareers",
    ]
    queries = [
        "MFE admission", "MFE accepted rejected",
        "financial engineering master", "MSCF CMU",
        "Baruch MFE", "Columbia MSFE", "Princeton MFin",
        "MFE profile evaluation", "MFE application results",
        "quantitative finance master admitted", "MFE GPA",
        "MFE offer", "MFE decision", "MFE profile review",
        "financial engineering application", "quant master admission",
        "MFE interview", "accepted MFE", "rejected MFE",
    ]

    records = []
    seen_ids = set()
    total_fetched = 0
    errors = 0

    for sub in subreddits:
        for qi, query in enumerate(queries):
            url = (
                f"https://www.reddit.com/r/{sub}/search.json"
                f"?q={quote_plus(query)}&restrict_sr=on&limit=100&sort=relevance&t=all"
            )
            try:
                resp = requests.get(url, headers=HEADERS, timeout=15)
                if resp.status_code == 429:
                    print(f"  Rate limited on r/{sub}, waiting 10s...")
                    time.sleep(10)
                    resp = requests.get(url, headers=HEADERS, timeout=15)
                if resp.status_code != 200:
                    print(f"  [WARN] r/{sub} q='{query}' -> HTTP {resp.status_code}")
                    errors += 1
                    time.sleep(2)
                    continue
                data = resp.json()
                posts = data.get("data", {}).get("children", [])
                total_fetched += len(posts)
                for p in posts:
                    pd_ = p.get("data", {})
                    post_id = pd_.get("id", "")
                    if post_id in seen_ids:
                        continue
                    seen_ids.add(post_id)

                    title = pd_.get("title", "")
                    selftext = pd_.get("selftext", "")
                    full_text = f"{title}\n{selftext}"

                    if len(full_text.strip()) < 30:
                        continue

                    gpa, gpa_scale = extract_gpa(full_text)
                    gre_q, gre_v = extract_gre(full_text)
                    toefl = extract_toefl(full_text)
                    result = extract_result(full_text)
                    program = extract_program(full_text)
                    season = extract_season(full_text)
                    nationality = extract_nationality(full_text)
                    gender = extract_gender(full_text)
                    school, tier, country = extract_undergrad(full_text)
                    intern_count, intern_level, intern_rel = extract_internships(full_text)
                    has_paper, has_research, research_level = extract_research(full_text)
                    major, major_rel = extract_major(full_text)

                    # Only keep posts that mention MFE/quant programs in some way
                    if not program and not result and not gpa:
                        continue

                    rec = {
                        "id": make_id("reddit", post_id),
                        "source": "reddit",
                        "program": program or "",
                        "result": result or "",
                        "season": season or "",
                        "gpa": gpa or "",
                        "gpa_scale": gpa_scale or "",
                        "gre_quant": gre_q or "",
                        "gre_verbal": gre_v or "",
                        "toefl": toefl or "",
                        "undergrad_school": school or "",
                        "undergrad_tier": tier or "",
                        "undergrad_country": country or "",
                        "major": major or "",
                        "major_relevance": major_rel or "",
                        "intern_count": intern_count or "",
                        "intern_level": intern_level or "",
                        "intern_relevance": intern_rel or "",
                        "has_paper": has_paper or "",
                        "has_research": has_research or "",
                        "research_level": research_level or "",
                        "gender": gender or "",
                        "nationality": nationality or "",
                        "raw_text": full_text[:2000].replace("\n", " ").replace(",", ";"),
                    }
                    rec["tier"] = compute_tier(rec)
                    records.append(rec)

            except Exception as e:
                print(f"  [ERROR] r/{sub} q='{query}': {e}")
                errors += 1

            time.sleep(2)  # Reddit rate limit

        print(f"  r/{sub}: processed ({len(records)} total unique records)")

    # Also search reddit globally (not restricted to subreddits)
    global_queries = [
        "MFE admitted profile", "financial engineering master results",
        "MFE acceptance rate profile", "Baruch MFE profile",
        "CMU MSCF accepted", "Berkeley MFE application",
    ]
    for query in global_queries:
        url = f"https://www.reddit.com/search.json?q={quote_plus(query)}&limit=100&sort=relevance&t=all"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                posts = data.get("data", {}).get("children", [])
                total_fetched += len(posts)
                for p in posts:
                    pd_ = p.get("data", {})
                    post_id = pd_.get("id", "")
                    if post_id in seen_ids:
                        continue
                    seen_ids.add(post_id)
                    title = pd_.get("title", "")
                    selftext = pd_.get("selftext", "")
                    full_text = f"{title}\n{selftext}"
                    if len(full_text.strip()) < 30:
                        continue
                    gpa, gpa_scale = extract_gpa(full_text)
                    gre_q, gre_v = extract_gre(full_text)
                    toefl = extract_toefl(full_text)
                    result = extract_result(full_text)
                    program = extract_program(full_text)
                    season = extract_season(full_text)
                    nationality = extract_nationality(full_text)
                    gender = extract_gender(full_text)
                    school, tier, country = extract_undergrad(full_text)
                    intern_count, intern_level, intern_rel = extract_internships(full_text)
                    has_paper, has_research, research_level = extract_research(full_text)
                    major, major_rel = extract_major(full_text)
                    if not program and not result and not gpa:
                        continue
                    rec = {
                        "id": make_id("reddit", post_id),
                        "source": "reddit",
                        "program": program or "",
                        "result": result or "",
                        "season": season or "",
                        "gpa": gpa or "",
                        "gpa_scale": gpa_scale or "",
                        "gre_quant": gre_q or "",
                        "gre_verbal": gre_v or "",
                        "toefl": toefl or "",
                        "undergrad_school": school or "",
                        "undergrad_tier": tier or "",
                        "undergrad_country": country or "",
                        "major": major or "",
                        "major_relevance": major_rel or "",
                        "intern_count": intern_count or "",
                        "intern_level": intern_level or "",
                        "intern_relevance": intern_rel or "",
                        "has_paper": has_paper or "",
                        "has_research": has_research or "",
                        "research_level": research_level or "",
                        "gender": gender or "",
                        "nationality": nationality or "",
                        "raw_text": full_text[:2000].replace("\n", " ").replace(",", ";"),
                    }
                    rec["tier"] = compute_tier(rec)
                    records.append(rec)
        except Exception as e:
            print(f"  [ERROR] global q='{query}': {e}")
            errors += 1
        time.sleep(2)

    # Save
    if records:
        df = pd.DataFrame(records, columns=UNIFIED_FIELDS)
        df.to_csv(REDDIT_CSV, index=False)
        print(f"\n  Saved {len(records)} records to {REDDIT_CSV}")
    else:
        print("\n  No records collected from Reddit.")

    print(f"  Total posts fetched: {total_fetched}, unique relevant: {len(records)}, errors: {errors}")
    return records


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE 2: GradCafe
# ════════════════════════════════════════════════════════════════════════════
def pipeline_gradcafe():
    print("\n" + "=" * 70)
    print("PIPELINE 2: GradCafe")
    print("=" * 70)

    records = []
    seen = set()

    # First try the old survey endpoint
    queries = [
        "financial+engineering", "computational+finance",
        "mathematical+finance", "quantitative+finance",
        "MFE", "MSCF", "financial+mathematics",
        "operations+research", "MFin",
    ]

    # Try the newer GradCafe search (they redesigned)
    # New URL format: https://www.thegradcafe.com/result/...
    # or the API: https://www.thegradcafe.com/survey/index.php

    for query in queries:
        for page in range(1, 30):
            # Try old endpoint
            url = f"https://www.thegradcafe.com/survey/index.php?q={query}&t=a&o=&pp=250&p={page}"
            try:
                resp = requests.get(url, headers=HEADERS, timeout=15)
                if resp.status_code == 403:
                    print(f"  GradCafe old endpoint returned 403 for '{query}', trying new format...")
                    break
                if resp.status_code != 200:
                    if page == 1:
                        print(f"  [WARN] GradCafe q='{query}' p={page} -> HTTP {resp.status_code}")
                    break

                soup = BeautifulSoup(resp.text, "html.parser")
                # Old GradCafe format: table with class 'submission-table' or similar
                rows = soup.select("table tr")
                if len(rows) <= 1:
                    # Try new format
                    rows = soup.select(".result-row, .submission")
                    if not rows:
                        break

                parsed_this_page = 0
                for row in rows[1:]:  # skip header
                    cells = row.find_all("td")
                    if len(cells) < 4:
                        continue
                    try:
                        institution = cells[0].get_text(strip=True)
                        program_name = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                        decision_text = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                        date_text = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                        gpa_text = cells[4].get_text(strip=True) if len(cells) > 4 else ""
                        gre_text = cells[5].get_text(strip=True) if len(cells) > 5 else ""

                        full_text = f"{institution} {program_name} {decision_text} {gpa_text} {gre_text}"
                        rec_key = f"{institution}|{program_name}|{decision_text}|{date_text}"
                        if rec_key in seen:
                            continue
                        seen.add(rec_key)

                        result = extract_result(decision_text)
                        program = extract_program(f"{institution} {program_name}")
                        gpa, gpa_scale = extract_gpa(gpa_text)
                        gre_q, gre_v = extract_gre(gre_text)
                        season = extract_season(date_text)
                        nationality = extract_nationality(decision_text)

                        rec = {
                            "id": make_id("gradcafe", rec_key),
                            "source": "gradcafe",
                            "program": program or "",
                            "result": result or "",
                            "season": season or "",
                            "gpa": gpa or "",
                            "gpa_scale": gpa_scale or "",
                            "gre_quant": gre_q or "",
                            "gre_verbal": gre_v or "",
                            "toefl": "",
                            "undergrad_school": "",
                            "undergrad_tier": "",
                            "undergrad_country": "",
                            "major": "",
                            "major_relevance": "",
                            "intern_count": "",
                            "intern_level": "",
                            "intern_relevance": "",
                            "has_paper": "",
                            "has_research": "",
                            "research_level": "",
                            "gender": "",
                            "nationality": nationality or "",
                            "raw_text": full_text[:2000].replace("\n", " ").replace(",", ";"),
                        }
                        rec["tier"] = compute_tier(rec)
                        records.append(rec)
                        parsed_this_page += 1
                    except Exception:
                        continue

                if parsed_this_page == 0:
                    break  # no more results
                print(f"    q='{query}' page {page}: {parsed_this_page} records")
            except requests.exceptions.Timeout:
                print(f"  [TIMEOUT] GradCafe q='{query}' p={page}")
                break
            except Exception as e:
                print(f"  [ERROR] GradCafe q='{query}' p={page}: {e}")
                break
            time.sleep(1.5)

    # Try new GradCafe search API
    print("  Trying GradCafe new search API...")
    new_queries = [
        "Financial Engineering", "Computational Finance",
        "Mathematical Finance", "Quantitative Finance",
        "MSCF", "MFE", "MFin",
    ]
    for query in new_queries:
        url = f"https://www.thegradcafe.com/result?q={quote_plus(query)}&t=a&pp=250"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"  New format q='{query}' -> HTTP {resp.status_code}")
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            # Parse whatever structure the new site uses
            result_cards = soup.select("[class*='result'], [class*='submission'], [class*='entry'], tr")
            parsed_new = 0
            for card in result_cards:
                text = card.get_text(" ", strip=True)
                if len(text) < 20:
                    continue
                rec_key = text[:200]
                if rec_key in seen:
                    continue
                seen.add(rec_key)

                result = extract_result(text)
                program = extract_program(text)
                gpa, gpa_scale = extract_gpa(text)
                gre_q, gre_v = extract_gre(text)
                season = extract_season(text)
                nationality = extract_nationality(text)

                if not program and not result:
                    continue

                rec = {
                    "id": make_id("gradcafe", rec_key),
                    "source": "gradcafe",
                    "program": program or "",
                    "result": result or "",
                    "season": season or "",
                    "gpa": gpa or "",
                    "gpa_scale": gpa_scale or "",
                    "gre_quant": gre_q or "",
                    "gre_verbal": gre_v or "",
                    "toefl": "",
                    "undergrad_school": "",
                    "undergrad_tier": "",
                    "undergrad_country": "",
                    "major": "",
                    "major_relevance": "",
                    "intern_count": "",
                    "intern_level": "",
                    "intern_relevance": "",
                    "has_paper": "",
                    "has_research": "",
                    "research_level": "",
                    "gender": "",
                    "nationality": nationality or "",
                    "raw_text": text[:2000].replace("\n", " ").replace(",", ";"),
                }
                rec["tier"] = compute_tier(rec)
                records.append(rec)
                parsed_new += 1
            if parsed_new:
                print(f"    New format q='{query}': {parsed_new} records")
        except Exception as e:
            print(f"  [ERROR] GradCafe new q='{query}': {e}")
        time.sleep(1.5)

    # Also try the GradCafe GitHub dataset
    print("  Trying GitHub GradCafe dataset...")
    github_urls = [
        "https://raw.githubusercontent.com/evansrjames/gradcafe-admissions-data/master/data/all_data.csv",
        "https://raw.githubusercontent.com/evansrjames/gradcafe-admissions-data/main/data/all_data.csv",
        "https://raw.githubusercontent.com/evansrjames/gradcafe-admissions-data/master/gradcafe_data.csv",
        "https://raw.githubusercontent.com/evansrjames/gradcafe-admissions-data/main/gradcafe_data.csv",
        # Another known dataset
        "https://raw.githubusercontent.com/deedy/gradcafe_data/master/all_uiuc_cleaned.csv",
        "https://raw.githubusercontent.com/deedy/gradcafe_data/master/all_schools.csv",
    ]
    for gurl in github_urls:
        try:
            resp = requests.get(gurl, headers=HEADERS, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 500:
                print(f"    Found GitHub dataset: {gurl}")
                # Parse CSV
                import io
                df_gh = pd.read_csv(io.StringIO(resp.text), low_memory=False)
                print(f"    Columns: {list(df_gh.columns)}")
                print(f"    Total rows: {len(df_gh)}")

                # Filter for finance/quant programs
                filter_terms = [
                    "financial engineering", "computational finance",
                    "mathematical finance", "quantitative finance",
                    "mfe", "mscf", "msfm", "mfin",
                    "operations research", "financial mathematics",
                ]
                mask = False
                for col in df_gh.columns:
                    if df_gh[col].dtype == object:
                        for term in filter_terms:
                            mask = mask | df_gh[col].str.contains(term, case=False, na=False)
                df_filtered = df_gh[mask]
                print(f"    Filtered to {len(df_filtered)} finance/quant rows")

                for _, row in df_filtered.iterrows():
                    row_text = " ".join(str(v) for v in row.values if pd.notna(v))
                    rec_key = row_text[:200]
                    if rec_key in seen:
                        continue
                    seen.add(rec_key)

                    result = extract_result(row_text)
                    program = extract_program(row_text)
                    gpa, gpa_scale = extract_gpa(row_text)
                    gre_q, gre_v = extract_gre(row_text)
                    season = extract_season(row_text)
                    nationality = extract_nationality(row_text)

                    # Try to get GPA from known column names
                    for c in ["gpa", "GPA", "ugrad_gpa"]:
                        if c in row and pd.notna(row[c]):
                            v = safe_float(row[c])
                            if v and 0 < v <= 4.0:
                                gpa = v
                                gpa_scale = "4.0"

                    for c in ["decision", "Decision", "status"]:
                        if c in row and pd.notna(row[c]):
                            result = result or extract_result(str(row[c]))

                    for c in ["institution", "Institution", "uni_name", "school"]:
                        if c in row and pd.notna(row[c]):
                            program = program or extract_program(str(row[c]))

                    rec = {
                        "id": make_id("gradcafe-gh", rec_key),
                        "source": "gradcafe",
                        "program": program or "",
                        "result": result or "",
                        "season": season or "",
                        "gpa": gpa or "",
                        "gpa_scale": gpa_scale or "",
                        "gre_quant": gre_q or "",
                        "gre_verbal": gre_v or "",
                        "toefl": "",
                        "undergrad_school": "",
                        "undergrad_tier": "",
                        "undergrad_country": "",
                        "major": "",
                        "major_relevance": "",
                        "intern_count": "",
                        "intern_level": "",
                        "intern_relevance": "",
                        "has_paper": "",
                        "has_research": "",
                        "research_level": "",
                        "gender": "",
                        "nationality": nationality or "",
                        "raw_text": row_text[:2000].replace("\n", " ").replace(",", ";"),
                    }
                    rec["tier"] = compute_tier(rec)
                    records.append(rec)
                break  # found one dataset, enough
        except Exception as e:
            print(f"    GitHub {gurl}: {e}")
            continue

    if records:
        df = pd.DataFrame(records, columns=UNIFIED_FIELDS)
        df.to_csv(GRADCAFE_CSV, index=False)
        print(f"\n  Saved {len(records)} records to {GRADCAFE_CSV}")
    else:
        print("\n  No records collected from GradCafe.")

    return records


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE 3: 1Point3Acres offer platform + existing data
# ════════════════════════════════════════════════════════════════════════════
def pipeline_1p3a():
    print("\n" + "=" * 70)
    print("PIPELINE 3: 1Point3Acres")
    print("=" * 70)

    records = []
    seen = set()

    # ── 3a: Parse existing offer_1p3a_results.csv ──────────────────────────
    results_csv = BASE / "offer_1p3a_results.csv"
    if results_csv.exists():
        print("  Parsing existing offer_1p3a_results.csv...")
        df_r = pd.read_csv(results_csv)
        print(f"    {len(df_r)} rows")
        for _, row in df_r.iterrows():
            row_text = " ".join(str(v) for v in row.values if pd.notna(v))
            rec_key = f"1p3a-offer|{row.get('program_id','')}|{row.get('result','')}|{row.get('season','')}|{row.get('tid','')}"
            if rec_key in seen:
                continue
            seen.add(rec_key)

            result_raw = str(row.get("result", ""))
            result = None
            if "AD" in result_raw or "ad" in result_raw.lower():
                result = "accepted"
            elif "Rej" in result_raw or "rej" in result_raw.lower():
                result = "rejected"
            elif "Wait" in result_raw or "wait" in result_raw.lower():
                result = "waitlisted"

            # Map program_id to program name
            pid = str(row.get("program_id", ""))
            school = str(row.get("school", ""))
            prog_name = str(row.get("program", ""))
            program = extract_program(f"{school} {prog_name}")

            season = str(row.get("season", ""))

            rec = {
                "id": make_id("1p3a-offer", rec_key),
                "source": "1p3a-offer",
                "program": program or "",
                "result": result or "",
                "season": season,
                "gpa": "",
                "gpa_scale": "",
                "gre_quant": "",
                "gre_verbal": "",
                "toefl": "",
                "undergrad_school": "",
                "undergrad_tier": "",
                "undergrad_country": "",
                "major": "",
                "major_relevance": "",
                "intern_count": "",
                "intern_level": "",
                "intern_relevance": "",
                "has_paper": "",
                "has_research": "",
                "research_level": "",
                "gender": "",
                "nationality": "chinese",  # 1p3a is predominantly Chinese users
                "raw_text": row_text[:2000].replace("\n", " ").replace(",", ";"),
            }
            rec["tier"] = compute_tier(rec)
            records.append(rec)

    # ── 3b: Parse existing offer_1p3a_backgrounds.csv ──────────────────────
    bg_csv = BASE / "offer_1p3a_backgrounds.csv"
    if bg_csv.exists():
        print("  Parsing existing offer_1p3a_backgrounds.csv...")
        df_bg = pd.read_csv(bg_csv)
        print(f"    {len(df_bg)} rows")
        for _, row in df_bg.iterrows():
            row_text = " ".join(str(v) for v in row.values if pd.notna(v))
            rec_key = f"1p3a-bg|{row_text[:100]}"
            if rec_key in seen:
                continue
            seen.add(rec_key)

            gpa_raw = str(row.get("gpa", ""))
            gpa, gpa_scale = None, None
            v = safe_float(gpa_raw)
            if v:
                if 0 < v <= 4.0:
                    gpa, gpa_scale = v, "4.0"
                elif 50 <= v <= 100:
                    gpa, gpa_scale = v, "100"

            gre_raw = str(row.get("gre", ""))
            gre_q, gre_v = extract_gre(f"GRE {gre_raw}")

            toefl_raw = str(row.get("toefl", ""))
            toefl = extract_toefl(f"TOEFL {toefl_raw}") if toefl_raw else None

            school_tier = str(row.get("school_tier", ""))
            undergrad_tier = None
            if any(w in school_tier for w in ["清华", "北大", "C9", "科大"]):
                undergrad_tier = "C9"
            elif "985" in school_tier:
                undergrad_tier = "985"
            elif "211" in school_tier:
                undergrad_tier = "211"
            elif "Top 30" in school_tier or "T30" in school_tier:
                undergrad_tier = "T30"
            elif "Top 50" in school_tier or "T50" in school_tier:
                undergrad_tier = "T50"

            major_raw = str(row.get("undergrad_major", ""))
            major, major_rel = extract_major(major_raw)
            intern_count, intern_level, intern_rel = extract_internships(str(row.get("work_exp", "")))
            has_paper, has_research, research_level = extract_research(str(row.get("research", "")))

            pid = str(row.get("program_id", ""))
            program = extract_program(row_text)

            rec = {
                "id": make_id("1p3a-bg", rec_key),
                "source": "1p3a-bg",
                "program": program or "",
                "result": "",
                "season": str(row.get("year", "")) + " " + str(row.get("term", "")),
                "gpa": gpa or "",
                "gpa_scale": gpa_scale or "",
                "gre_quant": gre_q or "",
                "gre_verbal": gre_v or "",
                "toefl": toefl or "",
                "undergrad_school": "",
                "undergrad_tier": undergrad_tier or "",
                "undergrad_country": "CN" if undergrad_tier in ("C9", "985", "211") else "",
                "major": major or "",
                "major_relevance": major_rel or "",
                "intern_count": intern_count or "",
                "intern_level": intern_level or "",
                "intern_relevance": intern_rel or "",
                "has_paper": has_paper or "",
                "has_research": has_research or "",
                "research_level": research_level or "",
                "gender": "",
                "nationality": "chinese",
                "raw_text": row_text[:2000].replace("\n", " ").replace(",", ";"),
            }
            rec["tier"] = compute_tier(rec)
            records.append(rec)

    # ── 3c: Parse existing collected.csv (the 6984-row file) ───────────────
    collected = BASE / "collected.csv"
    if collected.exists():
        print("  Parsing existing collected.csv...")
        df_c = pd.read_csv(collected, low_memory=False)
        print(f"    {len(df_c)} rows")
        for _, row in df_c.iterrows():
            row_text = " ".join(str(v) for v in row.values if pd.notna(v))
            rec_key = f"collected|{row.get('id','')}|{row.get('program','')}|{row.get('result','')}|{row.get('source','')}"
            if rec_key in seen:
                continue
            seen.add(rec_key)

            gpa_raw = row.get("gpa", "")
            gpa = safe_float(gpa_raw)
            gpa_scale_raw = str(row.get("gpa_scale", ""))
            gpa_scale = gpa_scale_raw if gpa_scale_raw and gpa_scale_raw != "nan" else None
            if gpa and not gpa_scale:
                gpa_scale = "4.0" if gpa <= 4.0 else "100"

            gre_raw = str(row.get("gre", ""))
            gre_q, gre_v = extract_gre(f"GRE {gre_raw}")

            toefl_raw = str(row.get("toefl", ""))
            toefl = extract_toefl(f"TOEFL {toefl_raw}") if toefl_raw and toefl_raw != "nan" else None
            # Some entries have TOEFL as plain number
            if not toefl and toefl_raw:
                tv = safe_int(toefl_raw)
                if tv and 60 <= tv <= 120:
                    toefl = tv

            result_raw = str(row.get("result", ""))
            result = extract_result(result_raw)

            program = str(row.get("program", ""))
            if program == "nan":
                program = ""

            season = str(row.get("season", ""))
            if season == "nan":
                season = ""

            gender_raw = str(row.get("gender", ""))
            gender = gender_raw if gender_raw in ("M", "F") else None

            bg_type = str(row.get("bg_type", ""))
            nationality_raw = str(row.get("nationality", ""))
            nationality = None
            if nationality_raw and nationality_raw != "nan":
                nationality = extract_nationality(nationality_raw)
            if not nationality and "中国" in bg_type:
                nationality = "chinese"

            # Extract undergrad tier from bg_type
            undergrad_tier = None
            undergrad_country = None
            if any(w in bg_type for w in ["清华", "北大"]):
                undergrad_tier = "C9"
                undergrad_country = "CN"
            elif "C9" in bg_type:
                undergrad_tier = "C9"
                undergrad_country = "CN"
            elif "985" in bg_type:
                undergrad_tier = "985"
                undergrad_country = "CN"
            elif "211" in bg_type or "两财一贸" in bg_type:
                undergrad_tier = "211"
                undergrad_country = "CN"
            elif any(w in bg_type for w in ["T20", "Top20", "Top 20"]):
                undergrad_tier = "T20"
                undergrad_country = "US"
            elif any(w in bg_type for w in ["T30", "Top30", "Top 30"]):
                undergrad_tier = "T30"
                undergrad_country = "US"
            elif any(w in bg_type for w in ["T50", "Top50", "Top 50"]):
                undergrad_tier = "T50"
                undergrad_country = "US"
            elif "海本" in bg_type:
                undergrad_country = "US"

            major_raw = str(row.get("major", ""))
            major, major_rel = extract_major(major_raw) if major_raw and major_raw != "nan" else (None, None)
            if not major and major_raw and major_raw != "nan":
                major = major_raw

            intern_raw = str(row.get("intern_desc", ""))
            intern_count, intern_level, intern_rel = extract_internships(intern_raw) if intern_raw and intern_raw != "nan" else (None, None, None)

            has_paper_raw = str(row.get("has_paper", ""))
            has_research_raw = str(row.get("has_research", ""))
            hp = has_paper_raw if has_paper_raw not in ("", "nan", "不明") else None
            hr = has_research_raw if has_research_raw not in ("", "nan", "不明") else None
            research_level = None  # unknown until evidence found
            if hp and hp in ("yes", "是"):
                research_level = "published"
            elif hr and hr in ("yes", "是"):
                research_level = "relevant_experience"

            source_raw = str(row.get("source", ""))
            source = source_raw if source_raw and source_raw != "nan" else "unknown"

            rec = {
                "id": make_id("collected", rec_key),
                "source": source,
                "program": program,
                "result": result or "",
                "season": season,
                "gpa": gpa or "",
                "gpa_scale": gpa_scale or "",
                "gre_quant": gre_q or "",
                "gre_verbal": gre_v or "",
                "toefl": toefl or "",
                "undergrad_school": bg_type if bg_type != "nan" else "",
                "undergrad_tier": undergrad_tier or "",
                "undergrad_country": undergrad_country or "",
                "major": major or "",
                "major_relevance": major_rel or "",
                "intern_count": intern_count or "",
                "intern_level": intern_level or "",
                "intern_relevance": intern_rel or "",
                "has_paper": hp if hp and hp in ("yes", "是") else "",
                "has_research": hr if hr and hr in ("yes", "是") else "",
                "research_level": research_level or "",
                "gender": gender or "",
                "nationality": nationality or "",
                "raw_text": row_text[:2000].replace("\n", " ").replace(",", ";"),
            }
            rec["tier"] = compute_tier(rec)
            records.append(rec)

    # ── 3d: Parse existing parsed_threads.csv ──────────────────────────────
    threads_csv = BASE / "parsed_threads.csv"
    if threads_csv.exists():
        print("  Parsing existing parsed_threads.csv...")
        df_t = pd.read_csv(threads_csv)
        print(f"    {len(df_t)} rows")
        for _, row in df_t.iterrows():
            row_text = " ".join(str(v) for v in row.values if pd.notna(v))
            rec_key = f"thread|{row.get('tid','')}|{row.get('program','')}|{row.get('result','')}"
            if rec_key in seen:
                continue
            seen.add(rec_key)

            gpa_raw = str(row.get("gpa", ""))
            gpa = safe_float(gpa_raw) if gpa_raw != "nan" else None
            gpa_scale_raw = str(row.get("gpa_scale", ""))
            gpa_scale = gpa_scale_raw if gpa_scale_raw not in ("", "nan") else None

            result = extract_result(str(row.get("result", "")))
            program = str(row.get("program", ""))
            if program == "nan":
                program = ""

            bg_type = str(row.get("bg_type", ""))
            undergrad_tier = None
            undergrad_country = None
            if "C9" in bg_type:
                undergrad_tier = "C9"
                undergrad_country = "CN"
            elif "985" in bg_type:
                undergrad_tier = "985"
                undergrad_country = "CN"
            elif "211" in bg_type:
                undergrad_tier = "211"
                undergrad_country = "CN"

            nationality = extract_nationality(str(row.get("nationality", "")))
            intern_raw = str(row.get("intern_desc", ""))
            intern_count, intern_level, intern_rel = extract_internships(intern_raw) if intern_raw != "nan" else (None, None, None)
            has_paper_raw = str(row.get("has_paper", ""))
            has_research_raw = str(row.get("has_research", ""))
            hp = "yes" if has_paper_raw in ("是", "yes") else ""
            hr = "yes" if has_research_raw in ("是", "yes") else ""
            research_level = "published" if hp else ("relevant_experience" if hr else "")
            season = str(row.get("season", ""))
            if season == "nan":
                season = ""

            rec = {
                "id": make_id("1p3a-thread", rec_key),
                "source": "1p3a-thread",
                "program": program,
                "result": result or "",
                "season": season,
                "gpa": gpa or "",
                "gpa_scale": gpa_scale or "",
                "gre_quant": "",
                "gre_verbal": "",
                "toefl": "",
                "undergrad_school": bg_type if bg_type != "nan" else "",
                "undergrad_tier": undergrad_tier or "",
                "undergrad_country": undergrad_country or "",
                "major": "",
                "major_relevance": "",
                "intern_count": intern_count or "",
                "intern_level": intern_level or "",
                "intern_relevance": intern_rel or "",
                "has_paper": hp,
                "has_research": hr,
                "research_level": research_level or "",
                "gender": "",
                "nationality": nationality or "chinese",
                "raw_text": row_text[:2000].replace("\n", " ").replace(",", ";"),
            }
            rec["tier"] = compute_tier(rec)
            records.append(rec)

    # ── 3e: Parse bbs_threads JSON files ───────────────────────────────────
    bbs_dir = BASE / "bbs_threads"
    if bbs_dir.exists():
        json_files = list(bbs_dir.glob("*.json"))
        print(f"  Parsing {len(json_files)} bbs_thread JSON files...")
        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                posts = data if isinstance(data, list) else [data]
                for post in posts:
                    if isinstance(post, str):
                        text = post
                    elif isinstance(post, dict):
                        text = post.get("content", "") or post.get("body", "") or post.get("text", "") or json.dumps(post, ensure_ascii=False)
                    else:
                        continue
                    if len(text) < 30:
                        continue
                    rec_key = f"bbs|{jf.stem}|{text[:80]}"
                    if rec_key in seen:
                        continue
                    seen.add(rec_key)

                    gpa, gpa_scale = extract_gpa(text)
                    gre_q, gre_v = extract_gre(text)
                    toefl = extract_toefl(text)
                    result = extract_result(text)
                    program = extract_program(text)
                    season = extract_season(text)
                    nationality = extract_nationality(text)
                    gender = extract_gender(text)
                    school, tier, country = extract_undergrad(text)
                    intern_count, intern_level, intern_rel = extract_internships(text)
                    has_paper, has_research, research_level = extract_research(text)
                    major, major_rel = extract_major(text)

                    if not program and not result and not gpa:
                        continue

                    rec = {
                        "id": make_id("1p3a-bbs", rec_key),
                        "source": "1p3a-bbs",
                        "program": program or "",
                        "result": result or "",
                        "season": season or "",
                        "gpa": gpa or "",
                        "gpa_scale": gpa_scale or "",
                        "gre_quant": gre_q or "",
                        "gre_verbal": gre_v or "",
                        "toefl": toefl or "",
                        "undergrad_school": school or "",
                        "undergrad_tier": tier or "",
                        "undergrad_country": country or "",
                        "major": major or "",
                        "major_relevance": major_rel or "",
                        "intern_count": intern_count or "",
                        "intern_level": intern_level or "",
                        "intern_relevance": intern_rel or "",
                        "has_paper": has_paper or "",
                        "has_research": has_research or "",
                        "research_level": research_level or "",
                        "gender": gender or "",
                        "nationality": nationality or "chinese",
                        "raw_text": text[:2000].replace("\n", " ").replace(",", ";"),
                    }
                    rec["tier"] = compute_tier(rec)
                    records.append(rec)
            except Exception as e:
                print(f"    [ERROR] {jf.name}: {e}")

    # ── 3f: Parse raw_1p3a text files ──────────────────────────────────────
    raw_dir = BASE / "raw_1p3a"
    if raw_dir.exists():
        txt_files = list(raw_dir.glob("*.txt"))
        print(f"  Parsing {len(txt_files)} raw_1p3a text files...")
        for tf in txt_files:
            try:
                text = tf.read_text(errors="replace")
                if len(text) < 30:
                    continue
                # Split into individual posts/sections
                sections = re.split(r'\n{3,}|={5,}|-{5,}', text)
                for si, section in enumerate(sections):
                    if len(section.strip()) < 30:
                        continue
                    rec_key = f"raw1p3a|{tf.stem}|{si}|{section[:60]}"
                    if rec_key in seen:
                        continue
                    seen.add(rec_key)

                    gpa, gpa_scale = extract_gpa(section)
                    gre_q, gre_v = extract_gre(section)
                    toefl = extract_toefl(section)
                    result = extract_result(section)
                    program = extract_program(section)
                    season = extract_season(section)
                    nationality = extract_nationality(section)
                    gender = extract_gender(section)
                    school, tier, country = extract_undergrad(section)
                    intern_count, intern_level, intern_rel = extract_internships(section)
                    has_paper, has_research, research_level = extract_research(section)
                    major, major_rel = extract_major(section)

                    if not program and not result and not gpa:
                        continue

                    rec = {
                        "id": make_id("1p3a-raw", rec_key),
                        "source": "1p3a-raw",
                        "program": program or "",
                        "result": result or "",
                        "season": season or "",
                        "gpa": gpa or "",
                        "gpa_scale": gpa_scale or "",
                        "gre_quant": gre_q or "",
                        "gre_verbal": gre_v or "",
                        "toefl": toefl or "",
                        "undergrad_school": school or "",
                        "undergrad_tier": tier or "",
                        "undergrad_country": country or "",
                        "major": major or "",
                        "major_relevance": major_rel or "",
                        "intern_count": intern_count or "",
                        "intern_level": intern_level or "",
                        "intern_relevance": intern_rel or "",
                        "has_paper": has_paper or "",
                        "has_research": has_research or "",
                        "research_level": research_level or "",
                        "gender": gender or "",
                        "nationality": nationality or "chinese",
                        "raw_text": section[:2000].replace("\n", " ").replace(",", ";"),
                    }
                    rec["tier"] = compute_tier(rec)
                    records.append(rec)
            except Exception as e:
                print(f"    [ERROR] {tf.name}: {e}")

    # ── 3g: Try to fetch from offer.1point3acres.com live ──────────────────
    print("  Trying offer.1point3acres.com live fetch...")
    program_ids = {
        46: "berkeley-mfe",
        87: "uchicago-msfm",
        129: "cmu-mscf",
        162: "cornell-mfe",
        170: "columbia-msfe",
        180: "nyu-mfe",
        230: "gatech-qcf",
        324: "mit-mfin",
        511: "baruch-mfe",
        620: "princeton-mfin",
        1017: "uiuc-msfe",
        1700: "stanford-mcf",
    }
    for pid, pname in program_ids.items():
        url = f"https://offer.1point3acres.com/program/{pid}/results"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"    Program {pid} ({pname}): HTTP {resp.status_code}")
                continue
            # Try to extract __NEXT_DATA__
            soup = BeautifulSoup(resp.text, "html.parser")
            script_tag = soup.find("script", id="__NEXT_DATA__")
            if script_tag:
                try:
                    next_data = json.loads(script_tag.string)
                    # Navigate the nested structure
                    props = next_data.get("props", {}).get("pageProps", {})
                    results_data = props.get("results", props.get("data", []))
                    if isinstance(results_data, dict):
                        results_data = results_data.get("data", results_data.get("results", []))
                    if isinstance(results_data, list):
                        new_count = 0
                        for item in results_data:
                            if not isinstance(item, dict):
                                continue
                            item_text = json.dumps(item, ensure_ascii=False)
                            rec_key = f"1p3a-live|{pid}|{item.get('id','')}|{item_text[:60]}"
                            if rec_key in seen:
                                continue
                            seen.add(rec_key)

                            result_raw = item.get("result", item.get("decision", ""))
                            result = extract_result(str(result_raw))
                            season_raw = item.get("season", item.get("year", ""))
                            gpa = safe_float(item.get("gpa", ""))
                            gre = safe_int(item.get("gre", item.get("gre_total", "")))
                            gre_q = None
                            gre_v = None
                            if gre and 260 <= gre <= 340:
                                gre_q = min(170, gre - 150)
                                gre_v = gre - gre_q

                            rec = {
                                "id": make_id("1p3a-live", rec_key),
                                "source": "1p3a-live",
                                "program": pname,
                                "result": result or "",
                                "season": str(season_raw),
                                "gpa": gpa or "",
                                "gpa_scale": "4.0" if gpa and gpa <= 4.0 else ("100" if gpa else ""),
                                "gre_quant": gre_q or "",
                                "gre_verbal": gre_v or "",
                                "toefl": "",
                                "undergrad_school": "",
                                "undergrad_tier": "",
                                "undergrad_country": "",
                                "major": "",
                                "major_relevance": "",
                                "intern_count": "",
                                "intern_level": "",
                                "intern_relevance": "",
                                "has_paper": "",
                                "has_research": "",
                                "research_level": "",
                                "gender": "",
                                "nationality": "chinese",
                                "raw_text": item_text[:2000].replace("\n", " ").replace(",", ";"),
                            }
                            rec["tier"] = compute_tier(rec)
                            records.append(rec)
                            new_count += 1
                        if new_count:
                            print(f"    Program {pid} ({pname}): {new_count} new records from __NEXT_DATA__")
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"    Program {pid} ({pname}): __NEXT_DATA__ parse error: {e}")
            else:
                print(f"    Program {pid} ({pname}): no __NEXT_DATA__ found")
        except Exception as e:
            print(f"    Program {pid} ({pname}): {e}")
        time.sleep(1.5)

    # ── 3h: Parse offer_backgrounds_member.json ────────────────────────────
    member_json = BASE / "offer_backgrounds_member.json"
    if member_json.exists():
        print("  Parsing offer_backgrounds_member.json...")
        try:
            with open(member_json) as f:
                member_data = json.load(f)
            if isinstance(member_data, list):
                items = member_data
            elif isinstance(member_data, dict):
                items = member_data.get("data", member_data.get("results", [member_data]))
            else:
                items = []
            print(f"    {len(items)} entries")
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_text = json.dumps(item, ensure_ascii=False)
                rec_key = f"1p3a-member|{item_text[:100]}"
                if rec_key in seen:
                    continue
                seen.add(rec_key)

                gpa = safe_float(item.get("gpa", ""))
                gre = safe_int(item.get("gre", ""))
                toefl = safe_int(item.get("toefl", ""))

                rec = {
                    "id": make_id("1p3a-member", rec_key),
                    "source": "1p3a-member",
                    "program": extract_program(item_text) or "",
                    "result": extract_result(item_text) or "",
                    "season": "",
                    "gpa": gpa or "",
                    "gpa_scale": "4.0" if gpa and gpa <= 4.0 else ("100" if gpa else ""),
                    "gre_quant": "",
                    "gre_verbal": "",
                    "toefl": toefl if toefl and 60 <= toefl <= 120 else "",
                    "undergrad_school": "",
                    "undergrad_tier": "",
                    "undergrad_country": "",
                    "major": "",
                    "major_relevance": "",
                    "intern_count": "",
                    "intern_level": "",
                    "intern_relevance": "",
                    "has_paper": "",
                    "has_research": "",
                    "research_level": "",
                    "gender": "",
                    "nationality": "chinese",
                    "raw_text": item_text[:2000].replace("\n", " ").replace(",", ";"),
                }
                rec["tier"] = compute_tier(rec)
                records.append(rec)
        except Exception as e:
            print(f"    [ERROR] member json: {e}")

    # Save enriched 1p3a data
    if records:
        df = pd.DataFrame(records, columns=UNIFIED_FIELDS)
        df.to_csv(OFFER_CSV, index=False)
        print(f"\n  Saved {len(records)} records to {OFFER_CSV}")
    else:
        print("\n  No records collected from 1Point3Acres.")

    return records


# ════════════════════════════════════════════════════════════════════════════
# MERGE & QUALITY REPORT
# ════════════════════════════════════════════════════════════════════════════
def merge_and_report(reddit_recs, gradcafe_recs, offer_recs):
    print("\n" + "=" * 70)
    print("MERGING & QUALITY REPORT")
    print("=" * 70)

    all_records = reddit_recs + gradcafe_recs + offer_recs
    print(f"  Total records before dedup: {len(all_records)}")

    # Deduplicate by id
    seen_ids = set()
    deduped = []
    for rec in all_records:
        if rec["id"] not in seen_ids:
            seen_ids.add(rec["id"])
            deduped.append(rec)
    all_records = deduped
    print(f"  After dedup: {len(all_records)}")

    # Save unified CSV
    df = pd.DataFrame(all_records, columns=UNIFIED_FIELDS)
    df.to_csv(UNIFIED_CSV, index=False)
    print(f"  Saved unified CSV: {UNIFIED_CSV}")

    # ── Generate quality report ────────────────────────────────────────────
    sources = df["source"].value_counts()
    tiers = df["tier"].value_counts()

    # Per-source tier distribution
    source_tier = df.groupby(["source", "tier"]).size().unstack(fill_value=0)

    # Field coverage
    coverage = {}
    for col in UNIFIED_FIELDS:
        if col in ("id", "source", "tier", "raw_text"):
            continue
        non_empty = df[col].astype(str).apply(lambda x: x not in ("", "nan", "None")).sum()
        coverage[col] = non_empty / len(df) * 100

    # Top programs
    prog_counts = df[df["program"] != ""]["program"].value_counts().head(20)

    # Result distribution
    result_dist = df[df["result"] != ""]["result"].value_counts()

    report = f"""# MFE Admission Data Quality Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Records**: {len(df)}

## Records by Source

| Source | Count | % |
|--------|------:|--:|
"""
    for src, cnt in sources.items():
        report += f"| {src} | {cnt} | {cnt/len(df)*100:.1f}% |\n"

    report += f"""
## Quality Tier Distribution

| Tier | Description | Count | % |
|------|-------------|------:|--:|
| A (Gold) | 4+ rich fields + result | {tiers.get('A', 0)} | {tiers.get('A', 0)/len(df)*100:.1f}% |
| B (Silver) | 2-3 rich fields + result | {tiers.get('B', 0)} | {tiers.get('B', 0)/len(df)*100:.1f}% |
| C (Bronze) | GPA/GRE + result | {tiers.get('C', 0)} | {tiers.get('C', 0)/len(df)*100:.1f}% |
| D (Basic) | Result only or no result | {tiers.get('D', 0)} | {tiers.get('D', 0)/len(df)*100:.1f}% |

## Tier Distribution by Source

| Source | A | B | C | D |
|--------|--:|--:|--:|--:|
"""
    for src in source_tier.index:
        row = source_tier.loc[src]
        report += f"| {src} | {row.get('A', 0)} | {row.get('B', 0)} | {row.get('C', 0)} | {row.get('D', 0)} |\n"

    report += "\n## Field Coverage\n\n| Field | Records with data | Coverage % |\n|-------|------------------:|-----------:|\n"
    for field, pct in sorted(coverage.items(), key=lambda x: -x[1]):
        cnt = int(pct * len(df) / 100)
        report += f"| {field} | {cnt} | {pct:.1f}% |\n"

    report += "\n## Top Programs by Record Count\n\n| Program | Count |\n|---------|------:|\n"
    for prog, cnt in prog_counts.items():
        report += f"| {prog} | {cnt} |\n"

    report += "\n## Result Distribution\n\n| Result | Count | % |\n|--------|------:|--:|\n"
    total_with_result = result_dist.sum()
    for res, cnt in result_dist.items():
        report += f"| {res} | {cnt} | {cnt/total_with_result*100:.1f}% |\n"

    report += f"\n## Model-Readiness Summary\n\n"
    report += f"- **Tier A+B (model-ready with rich features)**: {tiers.get('A', 0) + tiers.get('B', 0)} records ({(tiers.get('A', 0) + tiers.get('B', 0))/len(df)*100:.1f}%)\n"
    report += f"- **Tier C (basic features)**: {tiers.get('C', 0)} records ({tiers.get('C', 0)/len(df)*100:.1f}%)\n"
    report += f"- **Tier D (needs enrichment)**: {tiers.get('D', 0)} records ({tiers.get('D', 0)/len(df)*100:.1f}%)\n"
    report += f"- **Records with accept/reject label**: {total_with_result} ({total_with_result/len(df)*100:.1f}%)\n"

    with open(REPORT_MD, "w") as f:
        f.write(report)
    print(f"  Saved quality report: {REPORT_MD}")

    return df


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("QuantPath Multi-Dimensional MFE Data Collection")
    print(f"Start time: {datetime.now()}")

    reddit_recs = pipeline_reddit()
    gradcafe_recs = pipeline_gradcafe()
    offer_recs = pipeline_1p3a()

    df = merge_and_report(reddit_recs, gradcafe_recs, offer_recs)

    print(f"\nDone. {datetime.now()}")
    print(f"Total unified records: {len(df)}")
