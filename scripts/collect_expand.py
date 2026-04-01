#!/usr/bin/env python3
# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""
QuantPath Expanded Multi-Source MFE Admission Data Collection
Exhaustive collection from: Reddit (expanded), GradCafe (direct), QuantNet,
Xiaohongshu, ChaseDream, Quora, 1Point3Acres, GitHub datasets.

Appends to existing collected_multidim.csv with deduplication.
"""

import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("/Users/ethanyang/QuantPath/data/admissions")
UNIFIED_CSV = BASE / "collected_multidim.csv"
REPORT_MD = BASE / "data_quality_report.md"
STAGING_DIR = BASE / "staging"
STAGING_DIR.mkdir(exist_ok=True)

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
    "nyu tandon": "nyu-tandon-mfe",
    "stanford": "stanford-mcf",
    "ucla": "ucla-mfe",
    "uiuc": "uiuc-msfe",
    "illinois": "uiuc-msfe",
    "rutgers": "rutgers-mqf",
    "bu ": "bu-msmf",
    "boston university": "bu-msmf",
    "toronto": "toronto-mmf",
    "waterloo": "waterloo-mqf",
    "michigan": "michigan-qfr",
    "carnegie mellon": "cmu-mscf",
    "stony brook": "stonybrook-qf",
    "fordham": "fordham-msqf",
    "ncstate": "ncstate-mfm",
    "north carolina state": "ncstate-mfm",
    "jhu": "jhu-mfm",
    "johns hopkins": "jhu-mfm",
    "usc": "usc-msmf",
    "uwash": "uwash-cfrm",
    "washington": "uwash-cfrm",
    "stevens": "stevens-mfe",
    "claremont": "claremont-mfe",
    "oxford": "oxford-mcf",
    "lse": "lse-mfe",
    "imperial": "imperial-mfe",
    "eth": "eth-qf",
    "hkust": "hkust-mfe",
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

IIT_SCHOOLS = {"iit bombay", "iit delhi", "iit kanpur", "iit madras", "iit kharagpur",
               "iit roorkee", "iit guwahati", "iit hyderabad"}

C9_CN = {"北大", "清华", "复旦", "上交", "浙大", "中科大", "南大", "哈工大", "西交",
         "peking", "tsinghua", "fudan", "sjtu", "zju", "ustc", "nju"}
TOP985_CN = C9_CN | {"人大", "北航", "同济", "武大", "中山", "华科", "南开", "厦大", "天大", "中科院",
                     "beihang", "tongji", "wuhan", "sysu", "hust", "nankai"}

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
    raw = f"{source}:{'|'.join(str(p) for p in parts)}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def content_hash(text):
    """Hash of normalized text content for dedup."""
    norm = re.sub(r'\s+', ' ', str(text).lower().strip())[:500]
    return hashlib.md5(norm.encode()).hexdigest()[:16]


# ── Text parsing helpers ───────────────────────────────────────────────────
RE_GPA = re.compile(r'(?:GPA|gpa|Gpa)[\s:=]*(\d\.\d{1,2})\b')
RE_GPA_100 = re.compile(r'(?:GPA|gpa|均分|绩点|grade)[\s:=]*(\d{2,3}(?:\.\d{1,2})?)\s*/?\s*(?:100|4\.0|4)', re.I)
RE_GPA_LOOSE = re.compile(r'(\d\.\d{1,2})\s*(?:/\s*4\.0|GPA|gpa)', re.I)
RE_GRE_TOTAL = re.compile(r'(?:GRE|gre)[\s:=]*(\d{3})\b')
RE_GRE_Q = re.compile(r'(?:GRE|gre)\s*(?:quant|Q|数学|Quant)[\s:=]*(\d{3})\b', re.I)
RE_GRE_V = re.compile(r'(?:GRE|gre)\s*(?:verbal|V|语文|Verbal)[\s:=]*(\d{3})\b', re.I)
RE_TOEFL = re.compile(r'(?:TOEFL|toefl|托福)[\s:=]*(\d{2,3})\b', re.I)
RE_SEASON = re.compile(r'(20\d{2})\s*(Fall|Spring|Winter|Fa|Sp)', re.I)


def extract_gpa(text):
    m = RE_GPA.search(text)
    if m:
        v = float(m.group(1))
        if 0 < v <= 4.0:
            return v, "4.0"
    m = RE_GPA_LOOSE.search(text)
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
    q = RE_GRE_Q.search(text)
    v = RE_GRE_V.search(text)
    quant = safe_int(q.group(1)) if q else None
    verbal = safe_int(v.group(1)) if v else None
    if quant and not (130 <= quant <= 170):
        quant = None
    if verbal and not (130 <= verbal <= 170):
        verbal = None
    if quant is None and verbal is None:
        m = RE_GRE_TOTAL.search(text)
        if m:
            total = int(m.group(1))
            if 260 <= total <= 340:
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
    acc = any(w in t for w in ["accepted", "admitted", "offer", "ad无奖", "ad小奖",
                                "got in", "i'm in", "录取", "录了", "拿到offer",
                                "admission", "admit", "congratulations"])
    rej = any(w in t for w in ["rejected", "denied", "rej", "拒了", "拒",
                                "not accepted", "unsuccessful", "decline"])
    wl = any(w in t for w in ["waitlist", "waitlisted", "wl", "waiting list"])
    if wl:
        return "waitlisted"
    if acc and not rej:
        return "accepted"
    if rej and not acc:
        return "rejected"
    if acc and rej:
        # Both present - look at context
        return None
    return None


def extract_program(text):
    t = text.lower()
    for key, prog in sorted(MFE_PROGRAMS.items(), key=lambda x: -len(x[0])):
        if key in t:
            return prog
    if any(w in t for w in ["mfe", "financial engineering", "金融工程", "金工"]):
        return "mfe-unknown"
    if any(w in t for w in ["mscf", "computational finance", "计算金融"]):
        return "cmu-mscf"
    if any(w in t for w in ["msfm", "mathematical finance", "数学金融"]):
        return "msfm-unknown"
    if any(w in t for w in ["quantitative finance", "quant finance", "量化金融"]):
        return "finance-unknown"
    if any(w in t for w in ["master of finance", "master in finance", "mfin"]):
        return "finance-unknown"
    return None


def extract_season(text):
    m = RE_SEASON.search(text)
    if m:
        year = m.group(1)
        sem = m.group(2).lower()
        if sem.startswith("f"):
            return f"{year[2:]}Fall"
        elif sem.startswith("sp") or sem.startswith("s"):
            return f"{year[2:]}Spring"
    m2 = re.search(r'(\d{2})(Fall|Spring|Fa|Sp)', text, re.I)
    if m2:
        yr = m2.group(1)
        sem = m2.group(2).lower()
        if sem.startswith("f"):
            return f"{yr}Fall"
        return f"{yr}Spring"
    # Try just year mention
    m3 = re.search(r'class of (20\d{2})', text, re.I)
    if m3:
        return f"{m3.group(1)[2:]}Spring"
    return None


def extract_nationality(text):
    t = text.lower()
    if any(w in t for w in ["chinese", "china", "中国", "大陆", "内地", "cn", "prc"]):
        return "chinese"
    if any(w in t for w in ["indian", "india", "iit"]):
        return "indian"
    if any(w in t for w in ["american", "us citizen", "domestic", "from the us"]):
        return "us"
    if any(w in t for w in ["korean", "korea"]):
        return "korean"
    if any(w in t for w in ["taiwanese", "taiwan"]):
        return "taiwanese"
    if any(w in t for w in ["canadian", "canada"]):
        return "canadian"
    if any(w in t for w in ["european", "french", "german", "british", "uk"]):
        return "european"
    if any(w in t for w in ["international", "f1", "f-1", "visa"]):
        return "international"
    return None


def extract_gender(text):
    t = text.lower()
    male_signals = len(re.findall(r'\b(he|him|his|male|男|guy)\b', t))
    female_signals = len(re.findall(r'\b(she|her|hers|female|女|girl)\b', t))
    if male_signals > female_signals and male_signals > 0:
        return "M"
    if female_signals > male_signals and female_signals > 0:
        return "F"
    return None


def extract_undergrad(text):
    t = text.lower()
    # IIT
    for s in IIT_SCHOOLS:
        if s in t:
            return s, "IIT", "IN"
    # Chinese schools
    for s in C9_CN:
        if s in text or s in t:
            return s, "C9", "CN"
    for s in TOP985_CN:
        if s in text or s in t:
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
    t = text.lower()
    count = len(re.findall(r'(?:intern|实习)', t))
    count = min(count, 5)
    if count == 0:
        numbers = re.findall(r'(\d)\s*(?:段|份|个)\s*(?:实习|intern)', text)
        if numbers:
            count = int(numbers[0])
    level = None
    relevance = None
    if any(w in t for w in ["citadel", "jane street", "two sigma", "de shaw", "tower",
                             "jump trading", "hrt", "hudson river", "optiver", "imc",
                             "virtu", "top quant", "顶级量化", "aqr", "millennium",
                             "point72", "bridgewater"]):
        level = "us_top_quant"
        relevance = "quant_direct"
    elif any(w in t for w in ["goldman", "morgan stanley", "jpmorgan", "jp morgan",
                               "citi", "boa", "barclays", "ubs", "deutsche",
                               "三中一华", "中金", "中信", "华泰", "国泰",
                               "bank of america", "credit suisse", "nomura"]):
        level = "us_finance"
        relevance = "somewhat_related"
    elif any(w in t for w in ["百亿私募", "top私募", "头部私募", "top quant fund",
                               "幻方", "九坤", "明汯", "锐天", "量化"]):
        level = "china_top"
        relevance = "quant_direct"
    elif any(w in t for w in ["私募", "券商", "基金", "intern", "实习",
                               "big 4", "big four", "deloitte", "pwc", "ey", "kpmg"]):
        level = "china_normal"
        relevance = "somewhat_related"
    elif any(w in t for w in ["google", "meta", "amazon", "apple", "microsoft",
                               "faang", "maang", "tech intern"]):
        level = "us_tech"
        relevance = "somewhat_related"
    return count or None, level, relevance


def extract_research(text):
    t = text.lower()
    has_paper = None
    has_research = None
    level = "none"
    if any(w in t for w in ["published", "paper", "论文发表", "publication", "journal",
                             "conference paper", "proceedings"]):
        has_paper = "yes"
        level = "published"
    if any(w in t for w in ["research", "科研", "研究", "ra ", "research assistant",
                             "thesis", "论文", "capstone"]):
        has_research = "yes"
        if level == "none":
            level = "relevant_experience"
    return has_paper, has_research, level


def extract_major(text):
    t = text.lower()
    if any(w in t for w in ["金工", "金融工程", "financial engineering"]):
        return "financial_engineering", "quant_direct"
    if any(w in t for w in ["数学", "math", "applied math", "mathematics"]):
        return "mathematics", "quant_direct"
    if any(w in t for w in ["统计", "stat", "statistics", "biostatistics"]):
        return "statistics", "quant_direct"
    if any(w in t for w in ["cs ", "computer science", "计算机", "cs,"]):
        return "computer_science", "quant_direct"
    if any(w in t for w in ["physics", "物理"]):
        return "physics", "quant_direct"
    if any(w in t for w in ["econ", "经济"]):
        return "economics", "somewhat_related"
    if any(w in t for w in ["金融", "finance"]):
        return "finance", "somewhat_related"
    if any(w in t for w in ["ee", "electrical", "电气", "电子"]):
        return "electrical_engineering", "somewhat_related"
    if any(w in t for w in ["operations research", "or ", "运筹"]):
        return "operations_research", "quant_direct"
    if any(w in t for w in ["actuarial", "精算"]):
        return "actuarial_science", "somewhat_related"
    return None, None


def compute_tier(rec):
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
    if rec.get("major"):
        rich_fields += 1
    if rec.get("gre_quant"):
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


def make_record(source, text, extra=None):
    """Parse a text block into a record dict."""
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

    rec = {
        "id": make_id(source, content_hash(text)),
        "source": source,
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
        "raw_text": text[:2000].replace("\n", " ").replace(",", ";"),
    }
    # Override with explicit extra fields
    if extra:
        for k, v in extra.items():
            if v:
                rec[k] = v
    rec["tier"] = compute_tier(rec)
    return rec


def fetch_json(url, retries=2, delay=2):
    """Fetch URL and return JSON, with retry logic."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                return resp.json()
            else:
                if attempt == 0:
                    print(f"    HTTP {resp.status_code} for {url[:80]}...")
                return None
        except Exception as e:
            if attempt == retries:
                print(f"    Error fetching {url[:80]}: {e}")
            time.sleep(delay)
    return None


def fetch_html(url, retries=2, delay=1.5):
    """Fetch URL and return BeautifulSoup."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser"), resp.text
            else:
                return None, None
        except Exception as e:
            if attempt == retries:
                print(f"    Error: {e}")
            time.sleep(delay)
    return None, None


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE 1: Reddit EXPANDED
# ════════════════════════════════════════════════════════════════════════════
def pipeline_reddit_expanded():
    print("\n" + "=" * 70)
    print("PIPELINE 1: Reddit EXPANDED (15 subreddits, 30+ queries, pagination)")
    print("=" * 70)

    subreddits = [
        # Original 8
        "gradadmissions", "QuantFinance", "quant",
        "financialengineering", "UIUC", "cmu",
        "GraduateAdmissions", "FinancialCareers",
        # New 15
        "MBA", "ApplyingToCollege", "math", "statistics",
        "datascience", "cscareerquestions", "wallstreetbets",
        "chicago", "nyu", "columbia", "berkeley", "mit",
        "princeton", "gatech", "AskAcademia",
    ]

    queries = [
        # Original queries
        "MFE admission", "MFE accepted rejected",
        "financial engineering master", "MSCF CMU",
        "Baruch MFE", "Columbia MSFE", "Princeton MFin",
        "MFE profile evaluation", "MFE application results",
        "quantitative finance master admitted", "MFE GPA",
        "MFE offer", "MFE decision", "MFE profile review",
        "financial engineering application", "quant master admission",
        "MFE interview", "accepted MFE", "rejected MFE",
        # NEW expanded queries
        "MFE profile review", "MFE chances", "MFE 2024 results",
        "MFE 2025 results", "MFE 2026 results",
        "MSCF accepted", "MSCF rejected", "Baruch MFE accepted",
        "financial engineering GPA", "quant master GPA 3.5",
        "quant master GPA 3.8", "MFE international student",
        "MFE from India", "MFE from China",
        "quantitative finance master profile",
        "MFE work experience", "financial engineering interview",
        "MFE scholarship", "should I do MFE", "MFE vs PhD",
        "MFE worth it", "Berkeley MFE accepted", "UCLA MFE",
        "GaTech QCF admitted", "UChicago MSFM",
        "Cornell MFE admitted", "NYU Tandon MFE",
        "UIUC MSFE", "Stanford MCF", "financial mathematics master",
        "computational finance master",
    ]

    records = []
    seen_ids = set()
    total_fetched = 0
    errors = 0

    def process_posts(posts, source_tag):
        nonlocal total_fetched
        count = 0
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

            program = extract_program(full_text)
            result = extract_result(full_text)
            gpa, _ = extract_gpa(full_text)

            if not program and not result and not gpa:
                continue

            rec = make_record("reddit", full_text)
            rec["id"] = make_id("reddit", post_id)
            records.append(rec)
            count += 1

        total_fetched += len(posts)
        return count

    # Search each subreddit with each query, with pagination
    for si, sub in enumerate(subreddits):
        sub_count = 0
        for qi, query in enumerate(queries):
            after = None
            pages = 0
            while pages < 3:  # Up to 3 pages per query per subreddit
                url = (
                    f"https://www.reddit.com/r/{sub}/search.json"
                    f"?q={quote_plus(query)}&restrict_sr=on&limit=100&sort=new&t=all"
                )
                if after:
                    url += f"&after={after}"

                data = fetch_json(url, retries=1, delay=2)
                if not data:
                    break

                posts = data.get("data", {}).get("children", [])
                if not posts:
                    break

                new = process_posts(posts, f"r/{sub}")
                sub_count += new

                after = data.get("data", {}).get("after")
                if not after:
                    break
                pages += 1
                time.sleep(2.5)

            time.sleep(2)

        print(f"  [{si+1}/{len(subreddits)}] r/{sub}: {sub_count} new records")

    # Global searches with pagination
    print("\n  Global Reddit search...")
    global_queries = [
        "MFE admitted profile", "financial engineering master results",
        "MFE acceptance rate profile", "Baruch MFE profile",
        "CMU MSCF accepted", "Berkeley MFE application",
        "MFE admitted 2025", "MFE admitted 2026",
        "quant master admitted stats", "financial engineering GPA GRE",
        "MFE decision thread", "MSCF financial engineering",
        "quant finance master degree admission results",
        "MFE profile evaluation subreddit",
        "MFE application timeline results",
    ]
    for query in global_queries:
        after = None
        for page in range(3):
            url = f"https://www.reddit.com/search.json?q={quote_plus(query)}&limit=100&sort=relevance&t=all"
            if after:
                url += f"&after={after}"
            data = fetch_json(url, retries=1, delay=2)
            if not data:
                break
            posts = data.get("data", {}).get("children", [])
            if not posts:
                break
            process_posts(posts, "global")
            after = data.get("data", {}).get("after")
            if not after:
                break
            time.sleep(2.5)
        time.sleep(2)

    # Also try Reddit comment search via Pushshift-style API (if available)
    print(f"\n  Reddit total: {len(records)} new unique records, {total_fetched} posts checked, {errors} errors")
    return records


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE 2: GradCafe Direct Scraping
# ════════════════════════════════════════════════════════════════════════════
def pipeline_gradcafe():
    print("\n" + "=" * 70)
    print("PIPELINE 2: GradCafe Direct Scraping")
    print("=" * 70)

    records = []
    seen = set()

    queries = [
        "financial+engineering", "computational+finance",
        "mathematical+finance", "quantitative+finance",
        "MFE", "MSCF", "financial+mathematics",
        "master+of+finance", "quant+finance",
        "operations+research", "financial+math",
        "applied+math+finance", "math+finance",
    ]

    # Try old survey endpoint
    print("  Trying old GradCafe survey endpoint...")
    for query in queries:
        for page in range(1, 50):
            url = f"https://www.thegradcafe.com/survey/index.php?q={query}&t=a&o=&pp=250&p={page}"
            soup, raw_html = fetch_html(url, retries=1)
            if not soup:
                break
            if raw_html and len(raw_html) < 500:
                break

            rows = soup.select("table tr")
            if len(rows) <= 1:
                break

            parsed = 0
            for row in rows[1:]:
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
                    comments = cells[6].get_text(strip=True) if len(cells) > 6 else ""

                    full_text = f"{institution} {program_name} {decision_text} {gpa_text} {gre_text} {comments}"
                    rec_key = f"{institution}|{program_name}|{decision_text}|{date_text}"
                    if rec_key in seen:
                        continue
                    seen.add(rec_key)

                    rec = make_record("gradcafe", full_text)
                    rec["id"] = make_id("gradcafe", rec_key)
                    records.append(rec)
                    parsed += 1
                except Exception:
                    continue

            if parsed == 0:
                break
            time.sleep(1.5)

    # Try new GradCafe search pages
    print("  Trying new GradCafe search...")
    new_queries = [
        "Financial Engineering", "Computational Finance",
        "Mathematical Finance", "Quantitative Finance",
        "MSCF", "MFE", "MFin", "Financial Mathematics",
    ]
    for query in new_queries:
        for page in range(1, 20):
            url = f"https://www.thegradcafe.com/result?q={quote_plus(query)}&t=a&pp=250&p={page}"
            soup, raw_html = fetch_html(url, retries=1)
            if not soup:
                break

            # Try various selectors
            result_items = soup.select("[class*='result'], [class*='submission'], [class*='entry'], .tw-flex, article")
            if not result_items:
                # Try parsing the whole page text
                text_blocks = soup.find_all("div", recursive=True)
                result_items = [b for b in text_blocks if len(b.get_text(strip=True)) > 50]

            parsed = 0
            for item in result_items:
                text = item.get_text(" ", strip=True)
                if len(text) < 20:
                    continue
                ch = content_hash(text)
                if ch in seen:
                    continue
                seen.add(ch)

                program = extract_program(text)
                result = extract_result(text)
                if not program and not result:
                    continue

                rec = make_record("gradcafe", text)
                rec["id"] = make_id("gradcafe", ch)
                records.append(rec)
                parsed += 1

            if parsed == 0:
                break
            time.sleep(1.5)

    # Try GradCafe API (newer versions use JSON API)
    print("  Trying GradCafe JSON API...")
    api_urls = [
        "https://www.thegradcafe.com/api/results?q=financial+engineering&limit=100",
        "https://www.thegradcafe.com/api/results?q=MFE&limit=100",
        "https://www.thegradcafe.com/api/results?q=MSCF&limit=100",
        "https://www.thegradcafe.com/api/survey?q=financial+engineering&limit=100",
    ]
    for aurl in api_urls:
        data = fetch_json(aurl, retries=1)
        if data and isinstance(data, (list, dict)):
            items = data if isinstance(data, list) else data.get("results", data.get("data", []))
            if isinstance(items, list):
                for item in items:
                    text = json.dumps(item) if isinstance(item, dict) else str(item)
                    ch = content_hash(text)
                    if ch in seen:
                        continue
                    seen.add(ch)
                    rec = make_record("gradcafe", text)
                    records.append(rec)
                print(f"    API found {len(items)} items from {aurl[:60]}")
        time.sleep(1)

    print(f"  GradCafe total: {len(records)} records")
    return records


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE 3: QuantNet Forum Scraping
# ════════════════════════════════════════════════════════════════════════════
def pipeline_quantnet():
    print("\n" + "=" * 70)
    print("PIPELINE 3: QuantNet Forum Scraping")
    print("=" * 70)

    records = []
    seen = set()

    # QuantNet uses XenForo — search and thread listing
    search_queries = [
        "MFE profile evaluation", "MSCF admitted", "financial engineering accepted",
        "profile evaluation", "MFE application results", "admitted 2025",
        "admitted 2026", "Baruch MFE profile", "MFE chances",
        "profile review", "application result", "GPA GRE MFE",
    ]

    # Search page
    for query in search_queries:
        url = f"https://quantnet.com/search/?q={quote_plus(query)}&o=date"
        soup, _ = fetch_html(url)
        if not soup:
            time.sleep(2)
            continue

        # XenForo search results
        result_links = soup.select("a[href*='/threads/']")
        thread_urls = set()
        for link in result_links:
            href = link.get("href", "")
            if "/threads/" in href:
                full_url = urljoin("https://quantnet.com", href)
                thread_urls.add(full_url)

        print(f"  q='{query}': {len(thread_urls)} threads found")

        for turl in list(thread_urls)[:20]:  # Cap at 20 threads per query
            tsoup, _ = fetch_html(turl)
            if not tsoup:
                time.sleep(1.5)
                continue

            # Parse thread posts
            posts = tsoup.select(".message-body, .messageText, .bbWrapper, article")
            for post in posts:
                text = post.get_text(" ", strip=True)
                if len(text) < 50:
                    continue
                ch = content_hash(text)
                if ch in seen:
                    continue
                seen.add(ch)

                program = extract_program(text)
                result = extract_result(text)
                gpa, _ = extract_gpa(text)
                if not program and not result and not gpa:
                    continue

                rec = make_record("quantnet", text)
                rec["id"] = make_id("quantnet", ch)
                records.append(rec)

            time.sleep(1.5)
        time.sleep(2)

    # Also try QuantNet tagged threads
    tagged_urls = [
        "https://quantnet.com/threads/tagged/profile-evaluation",
        "https://quantnet.com/threads/tagged/mfe",
        "https://quantnet.com/threads/tagged/admission",
        "https://quantnet.com/forums/quant-pair-review.pair-review/",
        "https://quantnet.com/forums/pair-review.pair-review/",
    ]
    for turl in tagged_urls:
        soup, _ = fetch_html(turl)
        if not soup:
            time.sleep(1.5)
            continue

        thread_links = soup.select("a[href*='/threads/']")
        thread_urls = set()
        for link in thread_links[:30]:
            href = link.get("href", "")
            if "/threads/" in href:
                thread_urls.add(urljoin("https://quantnet.com", href))

        print(f"  Tagged page: {len(thread_urls)} threads")

        for tu in thread_urls:
            tsoup, _ = fetch_html(tu)
            if not tsoup:
                time.sleep(1.5)
                continue
            posts = tsoup.select(".message-body, .messageText, .bbWrapper, article")
            for post in posts:
                text = post.get_text(" ", strip=True)
                if len(text) < 50:
                    continue
                ch = content_hash(text)
                if ch in seen:
                    continue
                seen.add(ch)
                program = extract_program(text)
                result = extract_result(text)
                gpa, _ = extract_gpa(text)
                if not program and not result and not gpa:
                    continue
                rec = make_record("quantnet", text)
                rec["id"] = make_id("quantnet", ch)
                records.append(rec)
            time.sleep(1.5)

    print(f"  QuantNet total: {len(records)} records")
    return records


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE 4: ChaseDream Forum
# ════════════════════════════════════════════════════════════════════════════
def pipeline_chasedream():
    print("\n" + "=" * 70)
    print("PIPELINE 4: ChaseDream Forum")
    print("=" * 70)

    records = []
    seen = set()

    # ChaseDream is a Discuz forum
    # Search for MFE/金工 threads
    search_queries = [
        "金融工程", "MFE", "MSCF", "金工录取", "金工申请",
        "financial engineering", "MFE offer", "MFE 录取",
        "quant master", "金工面试", "MFE 面试",
    ]

    for query in search_queries:
        # Discuz search URL
        url = f"https://forum.chasedream.com/search.php?searchsubmit=yes&srchtxt={quote_plus(query)}&searchfield=title"
        soup, _ = fetch_html(url)
        if not soup:
            # Try alternative search
            url2 = f"https://forum.chasedream.com/search.php?mod=forum&srchtxt={quote_plus(query)}&formhash=&srchtype=title&srhfid=&srhlocality=forum%3A%3Aforum"
            soup, _ = fetch_html(url2)
        if not soup:
            time.sleep(2)
            continue

        # Find thread links
        thread_links = soup.select("a[href*='thread-']")
        if not thread_links:
            thread_links = soup.select("a[href*='viewthread']")
        thread_urls = set()
        for link in thread_links[:25]:
            href = link.get("href", "")
            if "thread" in href:
                thread_urls.add(urljoin("https://forum.chasedream.com/", href))

        print(f"  q='{query}': {len(thread_urls)} threads")

        for turl in thread_urls:
            tsoup, _ = fetch_html(turl)
            if not tsoup:
                time.sleep(1.5)
                continue

            # Discuz post content
            posts = tsoup.select(".t_f, .postmessage, .pcb, td[id^='postmessage']")
            if not posts:
                posts = tsoup.select("div[id^='post_']")
            for post in posts:
                text = post.get_text(" ", strip=True)
                if len(text) < 50:
                    continue
                ch = content_hash(text)
                if ch in seen:
                    continue
                seen.add(ch)

                program = extract_program(text)
                result = extract_result(text)
                gpa, _ = extract_gpa(text)
                if not program and not result and not gpa:
                    continue

                rec = make_record("chasedream", text)
                rec["id"] = make_id("chasedream", ch)
                records.append(rec)

            time.sleep(1.5)
        time.sleep(2)

    # Direct forum section URLs for MFE discussions
    forum_sections = [
        "https://forum.chasedream.com/forum-38-1.html",  # Master申请
        "https://forum.chasedream.com/forum-114-1.html",  # 金融工程
        "https://forum.chasedream.com/forum-147-1.html",  # 申请总结
    ]
    for furl in forum_sections:
        for page in range(1, 10):
            purl = furl.replace("-1.html", f"-{page}.html")
            soup, _ = fetch_html(purl)
            if not soup:
                break

            thread_links = soup.select("a[href*='thread-']")
            thread_urls = set()
            for link in thread_links:
                href = link.get("href", "")
                title_text = link.get_text(strip=True).lower()
                if "thread" in href and any(kw in title_text for kw in
                    ["mfe", "金工", "金融工程", "mscf", "quant", "量化", "financial engineering"]):
                    thread_urls.add(urljoin("https://forum.chasedream.com/", href))

            if not thread_urls:
                break

            for turl in thread_urls:
                tsoup, _ = fetch_html(turl)
                if not tsoup:
                    time.sleep(1)
                    continue
                posts = tsoup.select(".t_f, .postmessage, .pcb, td[id^='postmessage']")
                if not posts:
                    posts = tsoup.select("div[id^='post_']")
                for post in posts:
                    text = post.get_text(" ", strip=True)
                    if len(text) < 50:
                        continue
                    ch = content_hash(text)
                    if ch in seen:
                        continue
                    seen.add(ch)
                    program = extract_program(text)
                    result = extract_result(text)
                    gpa, _ = extract_gpa(text)
                    if not program and not result and not gpa:
                        continue
                    rec = make_record("chasedream", text)
                    rec["id"] = make_id("chasedream", ch)
                    records.append(rec)
                time.sleep(1)
            time.sleep(1.5)

    print(f"  ChaseDream total: {len(records)} records")
    return records


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE 5: 1Point3Acres Systematic
# ════════════════════════════════════════════════════════════════════════════
def pipeline_1p3a():
    print("\n" + "=" * 70)
    print("PIPELINE 5: 1Point3Acres Systematic Enumeration")
    print("=" * 70)

    records = []
    seen = set()

    # Try various URL patterns for offer pages
    url_patterns = [
        "https://offer.1point3acres.com/db/results?major=financial_engineering",
        "https://offer.1point3acres.com/db/results?major=mfe",
        "https://offer.1point3acres.com/db/results?program=financial+engineering",
        "https://offer.1point3acres.com/results?q=MFE",
        "https://offer.1point3acres.com/results?q=financial+engineering",
        "https://offer.1point3acres.com/results?q=MSCF",
        "https://offer.1point3acres.com/results?q=quantitative+finance",
    ]

    for url in url_patterns:
        soup, raw = fetch_html(url)
        if not soup:
            time.sleep(1)
            continue

        # Try to find __NEXT_DATA__
        script_tags = soup.select("script#__NEXT_DATA__")
        if script_tags:
            try:
                ndata = json.loads(script_tags[0].string)
                # Navigate the Next.js data structure
                props = ndata.get("props", {}).get("pageProps", {})
                results = props.get("results", props.get("offers", props.get("data", [])))
                if isinstance(results, list):
                    for item in results:
                        text = json.dumps(item, ensure_ascii=False)
                        ch = content_hash(text)
                        if ch in seen:
                            continue
                        seen.add(ch)
                        rec = make_record("1p3a-offer", text)
                        rec["id"] = make_id("1p3a-offer", ch)
                        records.append(rec)
                    print(f"    __NEXT_DATA__: {len(results)} items from {url[:60]}")
            except json.JSONDecodeError:
                pass

        time.sleep(1.5)

    # Try the BBS search on 1point3acres
    bbs_queries = [
        "金融工程 录取", "MFE offer", "MSCF 录取", "金工 申请",
        "MFE 申请结果", "financial engineering admitted",
        "quant master 录取", "MFE 面试",
    ]
    for query in bbs_queries:
        url = f"https://www.1point3acres.com/bbs/search.php?mod=forum&srchtxt={quote_plus(query)}&srchtype=title"
        soup, _ = fetch_html(url)
        if not soup:
            time.sleep(2)
            continue

        thread_links = soup.select("a[href*='thread-']")
        thread_urls = set()
        for link in thread_links[:20]:
            href = link.get("href", "")
            if "thread" in href:
                thread_urls.add(urljoin("https://www.1point3acres.com/bbs/", href))

        print(f"  BBS q='{query}': {len(thread_urls)} threads")

        for turl in thread_urls:
            tsoup, _ = fetch_html(turl)
            if not tsoup:
                time.sleep(1)
                continue
            posts = tsoup.select(".t_f, td[id^='postmessage']")
            for post in posts:
                text = post.get_text(" ", strip=True)
                if len(text) < 50:
                    continue
                ch = content_hash(text)
                if ch in seen:
                    continue
                seen.add(ch)
                program = extract_program(text)
                result = extract_result(text)
                gpa, _ = extract_gpa(text)
                if not program and not result and not gpa:
                    continue
                rec = make_record("1p3a-bbs", text)
                rec["id"] = make_id("1p3a-bbs", ch)
                records.append(rec)
            time.sleep(1.5)
        time.sleep(2)

    print(f"  1Point3Acres total: {len(records)} records")
    return records


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE 6: Xiaohongshu (小红书)
# ════════════════════════════════════════════════════════════════════════════
def pipeline_xiaohongshu():
    print("\n" + "=" * 70)
    print("PIPELINE 6: Xiaohongshu (小红书)")
    print("=" * 70)

    records = []
    seen = set()

    keywords = [
        "MFE录取", "金工offer", "金融工程申请", "MSCF录取",
        "Baruch MFE", "quant master 申请结果", "金工申请总结",
        "MFE选校", "金融工程录取", "MFE面试", "金工面经",
        "金融工程offer", "量化金融master", "MFE申请",
        "金融工程硕士", "MFE GPA", "金工选校",
    ]

    for kw in keywords:
        url = f"https://www.xiaohongshu.com/search_result?keyword={quote_plus(kw)}&source=web_search_result_notes"
        soup, raw = fetch_html(url)
        if not soup:
            time.sleep(2)
            continue

        # XHS uses React/SSR - try to find data in script tags
        scripts = soup.find_all("script")
        for script in scripts:
            stext = script.string or ""
            if "window.__INITIAL_STATE__" in stext or "window.__INITIAL_SSR_STATE__" in stext:
                # Try to parse JSON from the script
                try:
                    json_match = re.search(r'=\s*(\{.+?\})\s*;?\s*$', stext, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(1))
                        # Navigate the data structure for note content
                        notes = []
                        for key, val in data.items():
                            if isinstance(val, dict):
                                for k2, v2 in val.items():
                                    if isinstance(v2, dict) and "desc" in v2:
                                        notes.append(v2)
                                    elif isinstance(v2, list):
                                        notes.extend([x for x in v2 if isinstance(x, dict)])
                        for note in notes:
                            desc = note.get("desc", "") or note.get("title", "") or note.get("content", "")
                            if len(desc) < 30:
                                continue
                            ch = content_hash(desc)
                            if ch in seen:
                                continue
                            seen.add(ch)
                            program = extract_program(desc)
                            result = extract_result(desc)
                            if not program and not result:
                                continue
                            rec = make_record("小红书", desc)
                            rec["id"] = make_id("小红书", ch)
                            records.append(rec)
                except (json.JSONDecodeError, AttributeError):
                    pass

        # Also try parsing visible text
        note_cards = soup.select("[class*='note'], [class*='card'], article, .content")
        for card in note_cards:
            text = card.get_text(" ", strip=True)
            if len(text) < 50:
                continue
            ch = content_hash(text)
            if ch in seen:
                continue
            seen.add(ch)
            program = extract_program(text)
            result = extract_result(text)
            if not program and not result:
                continue
            rec = make_record("小红书", text)
            rec["id"] = make_id("小红书", ch)
            records.append(rec)

        time.sleep(2)

    print(f"  Xiaohongshu total: {len(records)} records")
    return records


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE 7: Quora
# ════════════════════════════════════════════════════════════════════════════
def pipeline_quora():
    print("\n" + "=" * 70)
    print("PIPELINE 7: Quora MFE Discussions")
    print("=" * 70)

    records = []
    seen = set()

    queries = [
        "MFE admission profile", "financial engineering master accepted",
        "MFE program chances", "MFE GPA requirements",
        "best MFE programs", "MFE application experience",
        "Baruch MFE admission", "CMU MSCF admission profile",
        "financial engineering GRE score", "MFE from India",
        "MFE international student", "MFE worth it",
        "quantitative finance master admission", "MFE interview experience",
    ]

    for query in queries:
        url = f"https://www.quora.com/search?q={quote_plus(query)}"
        soup, _ = fetch_html(url)
        if not soup:
            time.sleep(2)
            continue

        # Quora answers
        answers = soup.select(".q-text, .qu-wordBreak, [class*='answer'], .spacing_log_answer_content")
        if not answers:
            # Try broader selectors
            answers = soup.select("div span[class]")

        for ans in answers:
            text = ans.get_text(" ", strip=True)
            if len(text) < 100:
                continue
            ch = content_hash(text)
            if ch in seen:
                continue
            seen.add(ch)

            program = extract_program(text)
            result = extract_result(text)
            gpa, _ = extract_gpa(text)
            if not program and not result and not gpa:
                continue

            rec = make_record("quora", text)
            rec["id"] = make_id("quora", ch)
            records.append(rec)

        time.sleep(2)

    print(f"  Quora total: {len(records)} records")
    return records


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE 8: GitHub Datasets
# ════════════════════════════════════════════════════════════════════════════
def pipeline_github_datasets():
    print("\n" + "=" * 70)
    print("PIPELINE 8: GitHub Admission Datasets")
    print("=" * 70)

    records = []
    seen = set()

    # Search for repos with gradcafe/admission data
    search_queries = [
        "gradcafe admission data",
        "MFE admission",
        "graduate admission dataset",
        "grad cafe data scrape",
        "gradcafe csv",
        "graduate school admission results",
        "financial engineering admission",
        "grad admission data",
    ]

    repo_urls = set()
    for query in search_queries:
        url = f"https://api.github.com/search/repositories?q={quote_plus(query)}&sort=stars&per_page=30"
        data = fetch_json(url, retries=1)
        if data and "items" in data:
            for item in data["items"]:
                full_name = item.get("full_name", "")
                default_branch = item.get("default_branch", "main")
                repo_urls.add((full_name, default_branch))
            print(f"  q='{query}': {len(data['items'])} repos")
        time.sleep(1)

    print(f"  Total unique repos: {len(repo_urls)}")

    # Try to find CSV/JSON data files in each repo
    csv_urls_tried = set()
    for full_name, branch in repo_urls:
        # Search for CSV files in the repo
        api_url = f"https://api.github.com/search/code?q=extension:csv+repo:{full_name}"
        data = fetch_json(api_url, retries=1)
        if data and "items" in data:
            for item in data["items"]:
                path = item.get("path", "")
                if path.endswith(".csv"):
                    raw_url = f"https://raw.githubusercontent.com/{full_name}/{branch}/{path}"
                    if raw_url in csv_urls_tried:
                        continue
                    csv_urls_tried.add(raw_url)

                    try:
                        resp = requests.get(raw_url, headers=HEADERS, timeout=30)
                        if resp.status_code == 200 and len(resp.text) > 200:
                            df = pd.read_csv(io.StringIO(resp.text), low_memory=False, nrows=50000)
                            cols_lower = [c.lower() for c in df.columns]
                            cols_str = " ".join(cols_lower)

                            # Check if this looks like admission data
                            is_admission = any(kw in cols_str for kw in
                                ["admit", "decision", "accept", "reject", "gpa", "gre",
                                 "institution", "program", "result", "status"])

                            if not is_admission:
                                continue

                            print(f"    Parsing: {full_name}/{path} ({len(df)} rows, cols: {list(df.columns)[:6]})")

                            # Try to filter for finance/quant programs
                            text_cols = [c for c in df.columns if df[c].dtype == object]
                            if text_cols:
                                mask = df[text_cols].apply(
                                    lambda col: col.str.contains(
                                        'financ|quant|mfe|mscf|mfin|msfm|computational|math.*finance|operation.*research',
                                        case=False, na=False
                                    )
                                ).any(axis=1)
                                df_filtered = df[mask]
                            else:
                                df_filtered = df

                            if len(df_filtered) == 0:
                                continue

                            print(f"      Filtered to {len(df_filtered)} finance/quant rows")

                            for _, row in df_filtered.iterrows():
                                text = " ".join(str(v) for v in row.values if pd.notna(v))
                                ch = content_hash(text)
                                if ch in seen:
                                    continue
                                seen.add(ch)

                                rec = make_record("github", text)
                                rec["id"] = make_id("github", ch)

                                # Try to extract structured fields from known column names
                                for col in df.columns:
                                    cl = col.lower().strip()
                                    val = str(row[col]).strip() if pd.notna(row[col]) else ""
                                    if not val or val == "nan":
                                        continue
                                    if cl in ("gpa", "ugpa", "undergrad_gpa") and not rec["gpa"]:
                                        gpa = safe_float(val)
                                        if gpa and 0 < gpa <= 4.0:
                                            rec["gpa"] = gpa
                                            rec["gpa_scale"] = "4.0"
                                        elif gpa and 50 <= gpa <= 100:
                                            rec["gpa"] = gpa
                                            rec["gpa_scale"] = "100"
                                    if cl in ("gre_q", "gre_quant", "greq") and not rec["gre_quant"]:
                                        q = safe_int(val)
                                        if q and 130 <= q <= 170:
                                            rec["gre_quant"] = q
                                    if cl in ("gre_v", "gre_verbal", "grev") and not rec["gre_verbal"]:
                                        v = safe_int(val)
                                        if v and 130 <= v <= 170:
                                            rec["gre_verbal"] = v
                                    if cl in ("decision", "result", "status", "admit") and not rec["result"]:
                                        rec["result"] = extract_result(val) or rec["result"]
                                    if cl in ("institution", "school", "uni", "university") and not rec["undergrad_school"]:
                                        rec["undergrad_school"] = val[:100]
                                    if cl in ("season", "semester", "term") and not rec["season"]:
                                        rec["season"] = extract_season(val) or val[:10]

                                rec["tier"] = compute_tier(rec)
                                records.append(rec)

                    except Exception as e:
                        print(f"      Error parsing {path}: {e}")
                    time.sleep(0.5)
        time.sleep(1)

    # Also try known dataset URLs directly
    known_datasets = [
        "https://raw.githubusercontent.com/be-green/gradcafe/main/data/gradcafe.csv",
        "https://raw.githubusercontent.com/be-green/gradcafe/master/data/gradcafe.csv",
        "https://raw.githubusercontent.com/evansrjames/gradcafe-admissions-data/main/data/all_data.csv",
        "https://raw.githubusercontent.com/evansrjames/gradcafe-admissions-data/master/data/all_data.csv",
        "https://raw.githubusercontent.com/deedy/gradcafe_data/master/all_schools.csv",
        "https://raw.githubusercontent.com/deedy/gradcafe_data/master/all.csv",
        "https://raw.githubusercontent.com/lizzij/GradCafe/master/data/gradcafe_data.csv",
        "https://raw.githubusercontent.com/zthorson/gradcafe-data/master/gradcafe_all.csv",
        "https://raw.githubusercontent.com/zthorson/gradcafe-data/main/gradcafe_all.csv",
    ]

    for durl in known_datasets:
        if durl in csv_urls_tried:
            continue
        csv_urls_tried.add(durl)
        try:
            resp = requests.get(durl, headers=HEADERS, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 500:
                print(f"  Found dataset: {durl.split('/')[-1]}")
                df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
                print(f"    Columns: {list(df.columns)[:8]}, Rows: {len(df)}")

                # Filter for finance/quant
                text_cols = [c for c in df.columns if df[c].dtype == object]
                if text_cols:
                    mask = df[text_cols].apply(
                        lambda col: col.str.contains(
                            'financ|quant|mfe|mscf|mfin|msfm|computational|math.*finance|operation.*research',
                            case=False, na=False
                        )
                    ).any(axis=1)
                    df_filtered = df[mask]
                else:
                    df_filtered = pd.DataFrame()

                print(f"    Finance/quant rows: {len(df_filtered)}")

                for _, row in df_filtered.iterrows():
                    text = " ".join(str(v) for v in row.values if pd.notna(v))
                    ch = content_hash(text)
                    if ch in seen:
                        continue
                    seen.add(ch)
                    rec = make_record("github", text)
                    rec["id"] = make_id("github", ch)

                    for col in df.columns:
                        cl = col.lower().strip()
                        val = str(row[col]).strip() if pd.notna(row[col]) else ""
                        if not val or val == "nan":
                            continue
                        if "gpa" in cl and not rec["gpa"]:
                            gpa = safe_float(val)
                            if gpa and 0 < gpa <= 4.0:
                                rec["gpa"] = gpa
                                rec["gpa_scale"] = "4.0"
                        if "gre" in cl and "q" in cl and not rec["gre_quant"]:
                            q = safe_int(val)
                            if q and 130 <= q <= 170:
                                rec["gre_quant"] = q
                        if "gre" in cl and "v" in cl and not rec["gre_verbal"]:
                            v = safe_int(val)
                            if v and 130 <= v <= 170:
                                rec["gre_verbal"] = v
                        if cl in ("decision", "result", "status", "admit", "st"):
                            r = extract_result(val)
                            if r:
                                rec["result"] = r

                    rec["tier"] = compute_tier(rec)
                    records.append(rec)

        except Exception as e:
            print(f"  Error with {durl.split('/')[-1]}: {e}")
        time.sleep(1)

    print(f"  GitHub datasets total: {len(records)} records")
    return records


# ════════════════════════════════════════════════════════════════════════════
# MERGE & DEDUPLICATE
# ════════════════════════════════════════════════════════════════════════════
def merge_and_save(all_new_records):
    print("\n" + "=" * 70)
    print("MERGING & DEDUPLICATING")
    print("=" * 70)

    # Load existing data
    if UNIFIED_CSV.exists():
        existing = pd.read_csv(UNIFIED_CSV, dtype=str).fillna("")
        print(f"  Existing records: {len(existing)}")
        existing_hashes = set()
        for _, row in existing.iterrows():
            raw = str(row.get("raw_text", ""))[:500]
            existing_hashes.add(content_hash(raw))
        existing_ids = set(existing["id"].tolist())
    else:
        existing = pd.DataFrame(columns=UNIFIED_FIELDS)
        existing_hashes = set()
        existing_ids = set()

    # Deduplicate new records
    new_unique = []
    for rec in all_new_records:
        raw_hash = content_hash(str(rec.get("raw_text", ""))[:500])
        if rec["id"] in existing_ids:
            continue
        if raw_hash in existing_hashes:
            continue
        existing_ids.add(rec["id"])
        existing_hashes.add(raw_hash)
        new_unique.append(rec)

    print(f"  New unique records after dedup: {len(new_unique)}")

    if not new_unique:
        print("  No new records to add.")
        return existing

    # Create DataFrame and append
    new_df = pd.DataFrame(new_unique, columns=UNIFIED_FIELDS).fillna("")
    combined = pd.concat([existing, new_df], ignore_index=True)

    # Save
    combined.to_csv(UNIFIED_CSV, index=False)
    print(f"  Saved {len(combined)} total records to {UNIFIED_CSV}")
    print(f"  Added {len(new_unique)} new records")

    return combined


def generate_report(df):
    """Generate data quality report."""
    print("\n  Generating data quality report...")

    total = len(df)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# MFE Admission Data Quality Report\n",
        f"**Generated**: {now}",
        f"**Total Records**: {total}\n",
    ]

    # By source
    lines.append("## Records by Source\n")
    lines.append("| Source | Count | % |")
    lines.append("|--------|------:|--:|")
    source_counts = df["source"].value_counts()
    for src, cnt in source_counts.items():
        lines.append(f"| {src} | {cnt} | {cnt*100/total:.1f}% |")

    # By tier
    lines.append("\n## Quality Tier Distribution\n")
    lines.append("| Tier | Description | Count | % |")
    lines.append("|------|-------------|------:|--:|")
    tier_desc = {"A": "4+ rich fields + result", "B": "2-3 rich fields + result",
                 "C": "GPA/GRE + result", "D": "Result only or no result"}
    tier_names = {"A": "A (Gold)", "B": "B (Silver)", "C": "C (Bronze)", "D": "D (Basic)"}
    for t in ["A", "B", "C", "D"]:
        cnt = len(df[df["tier"] == t])
        lines.append(f"| {tier_names[t]} | {tier_desc[t]} | {cnt} | {cnt*100/total:.1f}% |")

    # Tier by source
    lines.append("\n## Tier Distribution by Source\n")
    lines.append("| Source | A | B | C | D | Total |")
    lines.append("|--------|--:|--:|--:|--:|------:|")
    for src in source_counts.index:
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
    for field in UNIFIED_FIELDS:
        if field in ("id", "source", "tier", "raw_text"):
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
    result_counts = result_df["result"].value_counts()
    for r, cnt in result_counts.items():
        lines.append(f"| {r} | {cnt} | {cnt*100/len(result_df):.1f}% |")

    # Top seasons
    lines.append("\n## Top Seasons\n")
    lines.append("| Season | Count |")
    lines.append("|--------|------:|")
    season_df = df[df["season"].astype(str).str.strip().ne("")]
    season_counts = season_df["season"].value_counts().head(10)
    for s, cnt in season_counts.items():
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

    report = "\n".join(lines)
    REPORT_MD.write_text(report)
    print(f"  Report saved to {REPORT_MD}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("QuantPath EXPANDED Data Collection")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_records = []
    pipeline_stats = {}

    # Pipeline 1: Reddit Expanded
    try:
        recs = pipeline_reddit_expanded()
        all_records.extend(recs)
        pipeline_stats["reddit_expanded"] = len(recs)
    except Exception as e:
        print(f"  [FATAL] Reddit pipeline: {e}")
        traceback.print_exc()
        pipeline_stats["reddit_expanded"] = 0

    # Pipeline 2: GradCafe
    try:
        recs = pipeline_gradcafe()
        all_records.extend(recs)
        pipeline_stats["gradcafe"] = len(recs)
    except Exception as e:
        print(f"  [FATAL] GradCafe pipeline: {e}")
        traceback.print_exc()
        pipeline_stats["gradcafe"] = 0

    # Pipeline 3: QuantNet
    try:
        recs = pipeline_quantnet()
        all_records.extend(recs)
        pipeline_stats["quantnet"] = len(recs)
    except Exception as e:
        print(f"  [FATAL] QuantNet pipeline: {e}")
        traceback.print_exc()
        pipeline_stats["quantnet"] = 0

    # Pipeline 4: ChaseDream
    try:
        recs = pipeline_chasedream()
        all_records.extend(recs)
        pipeline_stats["chasedream"] = len(recs)
    except Exception as e:
        print(f"  [FATAL] ChaseDream pipeline: {e}")
        traceback.print_exc()
        pipeline_stats["chasedream"] = 0

    # Pipeline 5: 1Point3Acres
    try:
        recs = pipeline_1p3a()
        all_records.extend(recs)
        pipeline_stats["1p3a"] = len(recs)
    except Exception as e:
        print(f"  [FATAL] 1P3A pipeline: {e}")
        traceback.print_exc()
        pipeline_stats["1p3a"] = 0

    # Pipeline 6: Xiaohongshu
    try:
        recs = pipeline_xiaohongshu()
        all_records.extend(recs)
        pipeline_stats["xiaohongshu"] = len(recs)
    except Exception as e:
        print(f"  [FATAL] Xiaohongshu pipeline: {e}")
        traceback.print_exc()
        pipeline_stats["xiaohongshu"] = 0

    # Pipeline 7: Quora
    try:
        recs = pipeline_quora()
        all_records.extend(recs)
        pipeline_stats["quora"] = len(recs)
    except Exception as e:
        print(f"  [FATAL] Quora pipeline: {e}")
        traceback.print_exc()
        pipeline_stats["quora"] = 0

    # Pipeline 8: GitHub Datasets
    try:
        recs = pipeline_github_datasets()
        all_records.extend(recs)
        pipeline_stats["github"] = len(recs)
    except Exception as e:
        print(f"  [FATAL] GitHub pipeline: {e}")
        traceback.print_exc()
        pipeline_stats["github"] = 0

    # Summary before merge
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY (before dedup)")
    print("=" * 70)
    for name, count in pipeline_stats.items():
        print(f"  {name}: {count} records")
    print(f"  TOTAL: {sum(pipeline_stats.values())} records")

    # Merge and save
    combined = merge_and_save(all_records)

    # Generate report
    generate_report(combined)

    print("\n" + "=" * 70)
    print(f"DONE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Final total: {len(combined)} records")
    print("=" * 70)


if __name__ == "__main__":
    main()
