#!/usr/bin/env python3
"""
1point3acres MFE Admission Report Scraper + Parser

Features:
  1. scrape  — Playwright automated scraping of admission report posts, saves raw text
  2. parse   — Regex parsing of saved raw text -> CSV (no API key needed)
  3. manual  — Manually paste/input post text for direct parsing

Usage:
  # Scrape (requires: pip install playwright && playwright install chromium)
  python tools/scrape_1p3a.py scrape --pages 10

  # Parse already-scraped raw text
  python tools/scrape_1p3a.py parse

  # Manually paste a single post
  python tools/scrape_1p3a.py manual --season 26Fall

  # Parse from file
  python tools/scrape_1p3a.py manual --input post.txt --season 26Fall

  # Statistics
  python tools/scrape_1p3a.py stats
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TOOLS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TOOLS_DIR.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "admissions" / "raw_1p3a"
_OUTPUT_CSV = _PROJECT_ROOT / "data" / "admissions" / "collected.csv"

CSV_FIELDS = [
    "id", "gender", "bg_type", "nationality", "gpa", "gpa_scale",
    "gre", "toefl", "major", "intern_desc", "has_paper", "has_research",
    "courses_note", "program", "result", "season", "source",
]

# ---------------------------------------------------------------------------
# Program name → ID mapping (extends collect_data.py)
# ---------------------------------------------------------------------------

PROGRAM_MAP: dict[str, str] = {
    # English
    "baruch": "baruch-mfe", "baruch mfe": "baruch-mfe",
    "cmu": "cmu-mscf", "cmu mscf": "cmu-mscf", "mscf": "cmu-mscf",
    "carnegie mellon": "cmu-mscf",
    "princeton": "princeton-mfin", "princeton mfin": "princeton-mfin",
    "berkeley": "berkeley-mfe", "berkeley mfe": "berkeley-mfe",
    "uc berkeley": "berkeley-mfe", "ucb": "berkeley-mfe",
    "columbia msfe": "columbia-msfe", "columbia mfe": "columbia-msfe",
    "columbia mafn": "columbia-mafn", "columbia": "columbia-msfe",
    "mit": "mit-mfin", "mit mfin": "mit-mfin",
    "stanford": "stanford-mcf", "stanford mcf": "stanford-mcf",
    "uchicago": "uchicago-msfm", "chicago": "uchicago-msfm",
    "uchicago msfm": "uchicago-msfm",
    "cornell": "cornell-mfe", "cornell mfe": "cornell-mfe",
    "nyu courant": "nyu-courant", "courant": "nyu-courant",
    "nyu tandon": "nyu-tandon-mfe", "nyu mfe": "nyu-tandon-mfe",
    "gatech": "gatech-qcf", "georgia tech": "gatech-qcf", "qcf": "gatech-qcf",
    "rutgers": "rutgers-mqf", "ucla": "ucla-mfe",
    "uiuc": "uiuc-msfe", "illinois": "uiuc-msfe",
    "northwestern": "northwestern-mfe", "jhu": "jhu-mfm",
    "johns hopkins": "jhu-mfm", "fordham": "fordham-msqf",
    "uwash": "uwash-cfrm", "stevens": "stevens-mfe",
    "usc": "usc-msmf", "bu": "bu-msmf", "boston university": "bu-msmf",
    "umich": "umich-mfe", "michigan": "umich-mfe",
    "umn": "uminn-mfm", "minnesota": "uminn-mfm",
    "utoronto": "utoronto-mmf", "toronto": "utoronto-mmf",
    "ncstate": "ncstate-mfm", "nc state": "ncstate-mfm",
    # Chinese aliases
    "巴鲁克": "baruch-mfe", "baruch金工": "baruch-mfe",
    "卡梅": "cmu-mscf", "卡内基梅隆": "cmu-mscf",
    "普林斯顿": "princeton-mfin", "普林": "princeton-mfin",
    "伯克利": "berkeley-mfe",
    "哥大": "columbia-msfe", "哥伦比亚": "columbia-msfe",
    "麻省理工": "mit-mfin",
    "斯坦福": "stanford-mcf",
    "芝大": "uchicago-msfm", "芝加哥": "uchicago-msfm",
    "康奈尔": "cornell-mfe",
    "纽大": "nyu-tandon-mfe",
    "佐治亚理工": "gatech-qcf",
    "罗格斯": "rutgers-mqf",
    "西北": "northwestern-mfe",
    "约翰霍普金斯": "jhu-mfm",
}

# Build a sorted list for regex (longest first to avoid partial matches)
_PROGRAM_NAMES = sorted(PROGRAM_MAP.keys(), key=len, reverse=True)
# NOTE: no \b — Chinese characters are all "word chars" so \b fails between
# adjacent Chinese tokens like "和卡梅" (and CMU) or "被普林斯顿" (by Princeton).
_PROGRAM_RE = re.compile(
    r"(?i)(" + "|".join(re.escape(n) for n in _PROGRAM_NAMES) + r")"
    r"(?:\s*(?:MFE|MSCF|MFin|MSFE|MAFN|MSFM|QCF|MQF|MCF|金工|金融工程))?"
)

# ---------------------------------------------------------------------------
# Result keyword patterns
# ---------------------------------------------------------------------------

_RESULT_KEYWORDS = {
    # accepted
    "ad": "accepted", "offer": "accepted", "admitted": "accepted",
    "accepted": "accepted", "录取": "accepted", "录了": "accepted",
    "拿到": "accepted", "收到offer": "accepted", "admit": "accepted",
    "got in": "accepted",
    # rejected
    "rej": "rejected", "rejected": "rejected", "拒": "rejected",
    "拒了": "rejected", "被拒": "rejected", "reject": "rejected",
    "decline": "rejected",
    # waitlisted
    "wl": "waitlisted", "waitlist": "waitlisted", "waitlisted": "waitlisted",
    "候补": "waitlisted", "waiting list": "waitlisted",
}


# ===================================================================
# REGEX PARSER — zero dependencies, no API key needed
# ===================================================================


def _extract_bg_type(text: str) -> str:
    """Extract undergraduate background type from post text."""
    text_lower = text.lower()

    # Specific tier keywords
    tier_patterns = [
        (r"(?:C9|c9|清华|北大|浙大|上交|复旦|中科大|南大|西交|哈工大)", "C9"),
        (r"(?:美本|美国本科)\s*(?:top\s*10|Top10)", "海本(Top10)"),
        (r"(?:美本|海本)\s*(?:top\s*20|Top20)", "海本(Top20)"),
        (r"(?:美本|海本)\s*(?:top\s*30|Top30)", "海本(Top30)"),
        (r"(?:美本|海本)\s*(?:top\s*50|Top50)", "海本(Top50)"),
        (r"(?:美本|海本)\s*(?:top\s*100|Top100)", "海本(Top100)"),
        (r"海本\s*(?:top\s*10|Top10|TOP10)", "海本(Top10)"),
        (r"海本\s*(?:top\s*20|Top20|TOP20)", "海本(Top20)"),
        (r"海本\s*(?:top\s*30|Top30|TOP30)", "海本(Top30)"),
        (r"海本\s*(?:top\s*50|Top50|TOP50)", "海本(Top50)"),
        (r"海本\s*(?:top\s*100|Top100|TOP100)", "海本(Top100)"),
        (r"美本|海本|海外本科|英本|加本|港本|overseas|陆本top", "海本"),
        (r"两财一贸", "两财一贸(211)"),
        (r"中外合办", "211"),
        (r"(?<!非)985", "985"),
        (r"(?<!非)211", "211"),
        (r"双非一本|双非\s*一本", "双非一本"),
        (r"双非|二本|普通本科|民办", "双非"),
        # English school names as tier hints
        (r"(?i)cornell|cmu|columbia|nyu|berkeley|umich|uiuc", "海本(Top30)"),
    ]

    for pattern, tier in tier_patterns:
        if re.search(pattern, text):
            return tier

    return ""


def _extract_gpa(text: str) -> tuple[Optional[float], float]:
    """Extract GPA and scale. Returns (gpa, scale) or (None, 4.0)."""
    patterns = [
        # GPA: 3.85/4.0 or GPA 3.85/4
        r"(?:GPA|gpa|绩点|均分)[：:\s]*(\d+\.?\d*)\s*/\s*(\d+\.?\d*)",
        # GPA: 3.85 (not followed by /xxx)
        r"(?:GPA|gpa|绩点)[：:\s]*(\d+\.?\d*)",
        # Average score 91.8
        r"均分[：:\s]*(\d+\.?\d*)",
        # Freeform: "gpa只有3.8" or "gpa3.9" or "3.9GPA" or "3.9的GPA"
        r"[Gg][Pp][Aa]\s*(?:只有|大概|约|是|为|有)?\s*(\d+\.?\d*)",
        r"(\d\.\d{1,2})\s*(?:的)?[Gg][Pp][Aa]",
        # Chinese: "绩点89" or "本科绩点3.8"
        r"(?:本科)?绩点\s*(?:只有|大概|约|是|为)?\s*(\d+\.?\d*)",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            gpa = float(m.group(1))
            if m.lastindex and m.lastindex >= 2:
                scale = float(m.group(2))
            elif gpa > 10:
                scale = 100.0
            elif gpa > 4.3:
                scale = 5.0
            else:
                scale = 4.0
            # Sanity check
            if (scale == 4.0 and 2.0 <= gpa <= 4.0) or \
               (scale == 100.0 and 50 <= gpa <= 100) or \
               (scale == 5.0 and 2.0 <= gpa <= 5.0):
                return gpa, scale

    return None, 4.0


def _extract_gre(text: str) -> Optional[str]:
    """Extract GRE score."""
    patterns = [
        r"GRE[：:\s]*(\d{3})\+?",          # GRE: 331 or GRE 331+
        r"GRE[：:\s]*V\d+\s*Q(\d+)",        # GRE: V160 Q170
        r"GRE[：:\s]*Q(\d+)",               # GRE: Q170
        r"(?:V|Verbal)\s*(\d{3})\s*(?:Q|Quant)\s*(\d{3})",  # V160 Q170
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            if m.lastindex == 2:
                return str(int(m.group(1)) + int(m.group(2)))
            return m.group(1)
    return None


def _extract_toefl(text: str) -> Optional[str]:
    """Extract TOEFL score."""
    patterns = [
        r"(?:TOEFL|托福|toefl)[：:\s]*(\d{2,3})\+?",
        r"(?:T|t)\s*(\d{3})\b",  # T110
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            score = m.group(1)
            if 70 <= int(score) <= 120:
                return score
    return None


def _extract_gender(text: str) -> str:
    """Extract gender from post text."""
    if re.search(r"\b(?:男|male|他)\b", text, re.IGNORECASE):
        return "M"
    if re.search(r"\b(?:女|female|她)\b", text, re.IGNORECASE):
        return "F"
    return "unknown"


def _extract_major(text: str) -> str:
    """Extract undergraduate major."""
    patterns = [
        r"(?:专业|major|本科专业)[：:\s]*([^\n,，。;；]{2,20})",
        r"(?:数学|统计|金融|金工|计算机|CS|Math|Statistics|Finance|Economics|经济)",
    ]
    m = re.search(patterns[0], text)
    if m:
        return m.group(1).strip()
    # Fallback: just detect field
    field_map = {
        "数学": "数学", "统计": "统计", "金融": "金融", "金工": "金融工程",
        "计算机": "计算机", "经济": "经济", "物理": "物理",
        "math": "数学", "stat": "统计", "finance": "金融",
        "cs": "计算机", "econ": "经济", "physics": "物理",
    }
    text_lower = text.lower()
    for kw, label in field_map.items():
        if kw in text_lower:
            return label
    return ""


def _extract_internships(text: str) -> str:
    """Extract internship description."""
    # Try to find explicit internship section
    patterns = [
        r"(?:实习|intern|工作经[历验])[：:\s]*([^\n]{5,100})",
        r"(\d)\s*段\s*(?:实习|intern)",
    ]
    m = re.search(patterns[0], text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(patterns[1], text, re.IGNORECASE)
    if m:
        n = m.group(1)
        # Try to find quality context nearby
        ctx_match = re.search(
            rf"{n}\s*段[^\n]{{0,80}}", text
        )
        if ctx_match:
            return ctx_match.group(0).strip()
        return f"{n}段实习"

    # Detect internship keywords without explicit label
    intern_kws = ["量化实习", "投行实习", "私募实习", "quant intern", "研究实习"]
    found = [kw for kw in intern_kws if kw in text.lower()]
    if found:
        return "、".join(found)

    return ""


def _extract_research(text: str) -> tuple[str, str]:
    """Extract research and paper info. Returns (has_paper, has_research)."""
    text_lower = text.lower()

    has_paper = "不明"
    has_research = "不明"

    paper_kws = ["论文", "paper", "publication", "发表", "一作", "first author",
                 "NeurIPS", "ICML", "AAAI", "顶会", "顶刊", "SCI", "期刊"]
    research_kws = ["科研", "research", "RA", "研究经历", "研究项目",
                    "导师", "实验室", "lab"]

    if any(kw in text_lower for kw in paper_kws):
        has_paper = "是"
        has_research = "是"
    elif re.search(r"(?:无|没有|0)\s*(?:论文|paper)", text_lower):
        has_paper = "否"

    if has_research == "不明":
        if any(kw in text_lower for kw in research_kws):
            has_research = "是"
        elif re.search(r"(?:无|没有|0)\s*(?:科研|research)", text_lower):
            has_research = "否"

    return has_paper, has_research


def _extract_courses(text: str) -> str:
    """Extract notable courses mentioned."""
    courses = []
    course_kws = {
        "随机微积分": "随机微积分", "stochastic calculus": "随机微积分",
        "实分析": "实分析", "real analysis": "实分析",
        "测度论": "测度论", "measure theory": "测度论",
        "pde": "PDE", "偏微分方程": "PDE",
        "ode": "ODE", "常微分方程": "ODE",
        "c++": "C++", "数值分析": "数值分析", "numerical": "数值方法",
        "随机过程": "随机过程", "stochastic process": "随机过程",
        "机器学习": "ML", "machine learning": "ML",
        "深度学习": "DL", "deep learning": "DL",
        "时间序列": "时间序列", "time series": "时间序列",
    }
    text_lower = text.lower()
    seen = set()
    for kw, label in course_kws.items():
        if kw in text_lower and label not in seen:
            courses.append(label)
            seen.add(label)
    return "+".join(courses)


def _extract_season(text: str, default: str = "26Fall") -> str:
    """Extract application season (e.g. 26Fall, 25Fall)."""
    m = re.search(r"(\d{2})\s*(?:Fall|fall|秋)", text)
    if m:
        return f"{m.group(1)}Fall"
    m = re.search(r"(\d{4})\s*(?:Fall|fall|秋)", text)
    if m:
        return f"{m.group(1)[2:]}Fall"
    m = re.search(r"(\d{2})\s*(?:Spring|spring|春)", text)
    if m:
        return f"{m.group(1)}Spring"
    return default


def _extract_results(text: str) -> list[dict[str, str]]:
    """Extract admission results (program_id, result) pairs.

    This is the most complex extraction — handles multiple formats:
    1. 'Baruch MFE: AD' or 'Baruch: AD' (key-value)
    2. 'AD Baruch' or 'Rej CMU' (result-first)
    3. Chinese narrative format for acceptances (e.g. got offer from Baruch and CMU)
    4. Chinese narrative format for rejections (e.g. rejected by Princeton)
    """
    results: list[dict[str, str]] = []
    seen_programs: set[str] = set()  # dedup by program_id (first result wins)

    def _add(program_id: str, result: str) -> None:
        if program_id in seen_programs:
            return  # first pass to assign a result wins
        seen_programs.add(program_id)
        results.append({"program_id": program_id, "result": result})

    def _resolve(name: str) -> Optional[str]:
        name_lower = name.lower().strip()
        if name_lower in PROGRAM_MAP:
            return PROGRAM_MAP[name_lower]
        for k, v in PROGRAM_MAP.items():
            if k in name_lower or name_lower in k:
                return v
        return None

    # --- Pass 1: Structured "Program: Result" ---
    # e.g. "Baruch MFE: AD", "CMU MSCF → Rej", "Columbia MSFE - WL"
    p1 = re.compile(
        r"(" + "|".join(re.escape(n) for n in _PROGRAM_NAMES) + r")"
        r"(?:\s*(?:MFE|MSCF|MFin|MSFE|MAFN|MSFM|QCF|金工))?"
        r"\s*[：:→\-—~]\s*"
        r"(AD|Rej|WL|Offer|Admit|Reject|Waitlist|录取|拒绝?|候补|accepted|rejected|waitlisted)",
        re.IGNORECASE,
    )
    for m in p1.finditer(text):
        pid = _resolve(m.group(1))
        result_raw = m.group(2).lower().strip()
        result = _RESULT_KEYWORDS.get(result_raw, result_raw)
        if pid and result in ("accepted", "rejected", "waitlisted"):
            _add(pid, result)

    # --- Pass 2: "Result: Program1, Program2, ..." (result first, comma list) ---
    # e.g. "AD: Baruch, Berkeley, Columbia MSFE" or "Rej: Princeton, MIT"
    p2_line = re.compile(
        r"(AD|Rej|WL|Offer|Reject)\s*[：:]\s*(.+?)(?:\n|$)",
        re.IGNORECASE,
    )
    for m in p2_line.finditer(text):
        result_raw = m.group(1).lower().strip()
        result = _RESULT_KEYWORDS.get(result_raw, result_raw)
        if result not in ("accepted", "rejected", "waitlisted"):
            continue
        chunk = m.group(2)
        for pm in _PROGRAM_RE.finditer(chunk):
            pid = _resolve(pm.group(1))
            if pid:
                _add(pid, result)

    # --- Pass 3: Sentence-level analysis ---
    # Split into sentences (period/excl-mark/newline are boundaries; comma is NOT)
    # Classify each sentence as accept/reject/wl, then assign to all programs in it.
    _ACCEPT_KWS = {"拿到", "录取", "录了", "收到offer", "offer", "admit"}
    _REJECT_KWS = {"拒", "拒了", "拒绝", "被拒", "reject", "rej"}
    _WL_KWS = {"wl", "waitlist", "候补"}

    sentences = re.split(r"[。！?\n]+", text)
    for sent in sentences:
        sent_lower = sent.lower()
        # Determine sentence sentiment
        is_accept = (
            any(kw in sent_lower for kw in _ACCEPT_KWS)
            or bool(re.search(r"\bAD\b", sent))  # uppercase AD as standalone word
        )
        is_reject = any(kw in sent_lower for kw in _REJECT_KWS)
        is_wl = any(kw in sent_lower for kw in _WL_KWS)

        if not (is_accept or is_reject or is_wl):
            continue

        for pm in _PROGRAM_RE.finditer(sent):
            pid = _resolve(pm.group(1))
            if not pid:
                continue
            if is_reject:
                _add(pid, "rejected")
            elif is_wl:
                _add(pid, "waitlisted")
            elif is_accept:
                _add(pid, "accepted")

    # --- Pass 4: Adjacency fallback (only if nothing found yet) ---
    if not results:
        for pm in _PROGRAM_RE.finditer(text):
            pid = _resolve(pm.group(1))
            if not pid or pid in seen_programs:
                continue
            start = max(0, pm.start() - 20)
            end = min(len(text), pm.end() + 20)
            window = text[start:end].lower()
            for kw, res in _RESULT_KEYWORDS.items():
                if kw in window and res in ("accepted", "rejected", "waitlisted"):
                    _add(pid, res)
                    break

    return results


def parse_post_text(
    text: str,
    season: str = "26Fall",
    source: str = "1p3a",
) -> list[dict]:
    """Parse a single post's text into structured admission records.

    Returns a list of CSV-row dicts (one per program result).
    Returns empty list if no MFE results found.
    """
    if not text or len(text.strip()) < 20:
        return []

    # Extract all fields
    bg_type = _extract_bg_type(text)
    gpa, gpa_scale = _extract_gpa(text)
    gre = _extract_gre(text)
    toefl = _extract_toefl(text)
    gender = _extract_gender(text)
    major = _extract_major(text)
    intern_desc = _extract_internships(text)
    has_paper, has_research = _extract_research(text)
    courses = _extract_courses(text)
    detected_season = _extract_season(text, default=season)
    admission_results = _extract_results(text)

    if not admission_results:
        return []

    rows = []
    for r in admission_results:
        rows.append({
            "gender": gender,
            "bg_type": bg_type,
            "nationality": "中国大陆",  # default for 1p3a
            "gpa": str(gpa) if gpa is not None else "",
            "gpa_scale": str(int(gpa_scale)) if gpa_scale == int(gpa_scale) else str(gpa_scale),
            "gre": gre or "",
            "toefl": toefl or "",
            "major": major,
            "intern_desc": intern_desc,
            "has_paper": has_paper,
            "has_research": has_research,
            "courses_note": courses,
            "program": r["program_id"],
            "result": r["result"],
            "season": detected_season,
            "source": source,
        })
    return rows


# ===================================================================
# CSV OUTPUT — compatible with existing collected.csv
# ===================================================================


def get_next_id(output_path: Path = _OUTPUT_CSV) -> int:
    """Read existing CSV, return next available ID."""
    if not output_path.exists():
        return 1
    with output_path.open() as f:
        reader = csv.DictReader(f)
        ids = [int(row["id"]) for row in reader if row.get("id", "").isdigit()]
    return max(ids, default=0) + 1


def write_rows(rows: list[dict], output_path: Path = _OUTPUT_CSV) -> int:
    """Append parsed rows to CSV. Returns number written."""
    if not rows:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    next_id = get_next_id(output_path)

    # Group rows by applicant (all rows from one parse_post_text call share an ID)
    written = 0
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            row["id"] = next_id
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
            written += 1

    return written


# ===================================================================
# PLAYWRIGHT SCRAPER
# ===================================================================

# Search queries — use actual program names from QuantNet top 15.
MFE_SEARCH_QUERIES = [
    # Top 10 programs by name
    "Baruch MFE",
    "Princeton MFin",
    "CMU MSCF",
    "Columbia MSFE",
    "MIT MFin",
    "Berkeley MFE",
    "UChicago MSFM",
    "GaTech QCF",
    "Cornell MFE",
    "NYU MFE",
    # Broader keywords
    "MFE",
    "金融工程",
    "MSCF",
    "金工",
]

# 1p3a search URL template — searches within forum-82 (Admission Reports: Graduate)
_SEARCH_URL = (
    "https://www.1point3acres.com/bbs/search.php"
    "?mod=forum&searchsubmit=yes&srchtxt={query}"
    "&formhash=&searchtype=title"
    "&srchfid%5B0%5D=82"  # restrict to forum-82
    "&page={page}"
)

# Login entry point
_LOGIN_ENTRY = "https://www.1point3acres.com/bbs/forum-82-1.html"

# Persistent browser state so login survives between commands
_STATE_DIR = _RAW_DIR / ".browser_state"


def _extract_thread_links(page) -> list[tuple[str, str, str]]:
    """Extract (title, url, tid) from current search results or forum page."""
    threads: list[tuple[str, str, str]] = []

    # Try search result selectors first, then forum page selectors
    selectors = [
        "li.pbw h3 a",              # search results
        "a.s.xst",                  # search results (alt)
        "th a.xst, td a.xst",      # forum listing
        "a[href*='thread-']",       # broad fallback
    ]
    thread_els = []
    for sel in selectors:
        thread_els = page.query_selector_all(sel)
        if thread_els:
            break

    for el in thread_els:
        title = (el.inner_text() or "").strip()
        href = el.get_attribute("href") or ""
        if not title or not href:
            continue
        tid_match = re.search(r"thread-(\d+)", href)
        if not tid_match:
            continue
        full_url = href if href.startswith("http") else f"https://www.1point3acres.com/bbs/{href}"
        threads.append((title, full_url, tid_match.group(1)))

    return threads


def _extract_post_content(page) -> str:
    """Extract main post text from a thread page."""
    # Multiple selector strategies for 1p3a post content
    selectors = [
        "td.t_f",                   # classic Discuz template
        "div.t_fsz",                # alternative content wrapper
        "#postlist .t_f",           # postlist first post
        ".forum-post-content",      # newer template
        "article .message-body",    # another variant
    ]
    for sel in selectors:
        el = page.query_selector(sel)
        if el:
            text = el.inner_text().strip()
            if len(text) > 30:
                return text
    return ""


def _launch_browser(headless: bool = False):
    """Launch playwright browser with persistent state (keeps login cookies)."""
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    context = pw.chromium.launch_persistent_context(
        user_data_dir=str(_STATE_DIR),
        headless=headless,
        viewport={"width": 1280, "height": 900},
    )
    return pw, context


def do_login() -> None:
    """Open browser for user to log in. Saves session for later scraping."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        print("Error: pip install playwright && playwright install chromium")
        sys.exit(1)

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    pw, context = _launch_browser(headless=False)
    pg = context.new_page()

    print("Opening 1point3acres...")
    pg.goto(_LOGIN_ENTRY, wait_until="domcontentloaded")
    time.sleep(2)

    if "auth." in pg.url or "login" in pg.url:
        print("\n" + "=" * 50)
        print("  Please log in to 1point3acres in the browser window")
        print("  WeChat QR code / verification code login supported")
        print("  Login will be auto-detected once successful")
        print("=" * 50 + "\n")
        try:
            pg.wait_for_url("**/bbs/**", timeout=300_000)
            time.sleep(2)
            print("Login successful! Session saved.")
            print("  You can now run: python3 tools/scrape_1p3a.py scrape")
        except Exception:
            print("Login timed out (5 minutes)")
    else:
        print("Already logged in (reusing previous session)")

    context.close()
    pw.stop()


def scrape_posts(max_pages: int = 3, headless: bool = False) -> int:
    """Scrape MFE admission report posts using keyword search.

    Requires prior login via `do_login()` — uses saved browser state.
    Searches for each program name within the Admission Reports section (forum-82).
    Returns number of posts saved.
    """
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        print("Error: pip install playwright && playwright install chromium")
        sys.exit(1)

    if not _STATE_DIR.exists():
        print("Error: please run the login command first")
        print("  python3 tools/scrape_1p3a.py login")
        return 0

    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Load already-scraped post IDs
    seen_file = _RAW_DIR / ".seen_ids.txt"
    seen_ids: set[str] = set()
    if seen_file.exists():
        seen_ids = set(seen_file.read_text().splitlines())

    saved = 0
    pw, context = _launch_browser(headless=headless)
    pg = context.new_page()

    # Quick login check
    pg.goto(_LOGIN_ENTRY, wait_until="domcontentloaded")
    time.sleep(2)
    if "auth." in pg.url or "login" in pg.url:
        print("Not logged in. Please run first: python3 tools/scrape_1p3a.py login")
        context.close()
        pw.stop()
        return 0
    print("Logged in. Starting MFE post search...\n")

    # --- Search for MFE posts ---
    for query in MFE_SEARCH_QUERIES:
        print(f"\n{'─'*50}")
        print(f"Search: \"{query}\" (Admission Reports forum)")
        print(f"{'─'*50}")

        for search_page in range(1, max_pages + 1):
            url = _SEARCH_URL.format(
                query=query.replace(" ", "+"),
                page=search_page,
            )
            try:
                pg.goto(url, wait_until="domcontentloaded")
                time.sleep(2)
            except Exception as e:
                print(f"  Search page load failed: {e}")
                break

            threads = _extract_thread_links(pg)
            new_threads = [(t, u, tid) for t, u, tid in threads if tid not in seen_ids]

            if not threads:
                print(f"  Page {search_page}: no results, moving to next keyword")
                break

            print(f"  Page {search_page}: {len(threads)} results, {len(new_threads)} new posts")

            if not new_threads:
                continue

            for title, thread_url, tid_str in new_threads:
                try:
                    pg.goto(thread_url, wait_until="domcontentloaded")
                    time.sleep(1.5)

                    text = _extract_post_content(pg)
                    if text:
                        raw_file = _RAW_DIR / f"{tid_str}.txt"
                        metadata = {
                            "title": title,
                            "url": thread_url,
                            "tid": tid_str,
                            "query": query,
                        }
                        raw_file.write_text(
                            json.dumps(metadata, ensure_ascii=False) + "\n"
                            + "---\n"
                            + text,
                            encoding="utf-8",
                        )
                        saved += 1
                        seen_ids.add(tid_str)
                        print(f"    ✓ [{tid_str}] {title[:50]}")
                    else:
                        print(f"    ✗ [{tid_str}] Failed to extract content")

                except Exception as e:
                    print(f"    ✗ [{tid_str}] Error: {e}")

                time.sleep(1.5 + (saved % 3))

            seen_file.write_text("\n".join(sorted(seen_ids)), encoding="utf-8")

        time.sleep(2)

    context.close()
    pw.stop()

    print(f"\n{'='*50}")
    print(f"  Scraping complete: saved {saved} MFE posts")
    print(f"  Raw text: {_RAW_DIR}")
    print(f"  Next step: python3 tools/scrape_1p3a.py parse")
    print(f"{'='*50}")
    return saved


# ===================================================================
# BATCH PARSE — process all raw text files
# ===================================================================


def parse_all_raw(season: str = "26Fall") -> int:
    """Parse all raw text files in raw_1p3a/ directory."""
    if not _RAW_DIR.exists():
        print(f"Directory does not exist: {_RAW_DIR}")
        return 0

    files = sorted(_RAW_DIR.glob("*.txt"))
    if not files:
        print("No raw text files found")
        return 0

    total_rows = 0
    total_posts = 0
    next_id = get_next_id()

    all_rows: list[dict] = []

    for f in files:
        content = f.read_text(encoding="utf-8")
        # Split metadata header from content
        parts = content.split("---\n", 1)
        text = parts[1] if len(parts) > 1 else content

        rows = parse_post_text(text, season=season, source="1p3a")
        if rows:
            # Assign same ID to all rows from one post
            for row in rows:
                row["id"] = next_id
            next_id += 1
            all_rows.extend(rows)
            total_posts += 1
            total_rows += len(rows)
            print(f"  ✓ {f.name}: {len(rows)} results")
        else:
            print(f"  - {f.name}: no MFE results extracted")

    if all_rows:
        written = write_rows(all_rows)
        print(f"\nParsed {total_posts} posts, wrote {written} records")
    else:
        print("\nNo results extracted")

    return total_rows


# ===================================================================
# MANUAL INPUT MODE
# ===================================================================


def manual_parse(
    input_path: Optional[str] = None,
    season: str = "26Fall",
    dry_run: bool = False,
) -> int:
    """Parse manually provided text (stdin or file)."""
    if input_path:
        text = Path(input_path).read_text(encoding="utf-8")
        print(f"Reading file: {input_path}")
    else:
        print("Paste post content (press Ctrl+D when done):")
        text = sys.stdin.read()

    if not text.strip():
        print("Error: no input content")
        return 0

    rows = parse_post_text(text, season=season, source="1p3a-manual")

    if not rows:
        print("\nNo MFE admission results extracted")
        print("Hint: post must contain program names (e.g. Baruch/CMU) and results (e.g. AD/Rej)")
        return 0

    # Display parsed results
    print(f"\nExtracted {len(rows)} results:")
    print("-" * 60)
    for r in rows:
        print(f"  {r['program']:20} {r['result']:10} "
              f"GPA={r['gpa'] or '?'} bg={r['bg_type'] or '?'}")
    print("-" * 60)

    if dry_run:
        print("(dry run, CSV not written)")
        return len(rows)

    next_id = get_next_id()
    for row in rows:
        row["id"] = next_id
    written = write_rows(rows)
    print(f"Wrote {written} records to {_OUTPUT_CSV}")
    return written


# ===================================================================
# STATS
# ===================================================================


def print_stats() -> None:
    """Print dataset statistics."""
    from collections import Counter

    if not _OUTPUT_CSV.exists():
        print("Data file does not exist")
        return

    programs: Counter[str] = Counter()
    results: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    bg_types: Counter[str] = Counter()
    applicant_ids: set[str] = set()

    with _OUTPUT_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            programs[row.get("program", "")] += 1
            results[row.get("result", "")] += 1
            sources[row.get("source", "")] += 1
            bg_types[row.get("bg_type", "")] += 1
            applicant_ids.add(row.get("id", ""))

    total = sum(programs.values())
    print(f"\n{'='*55}")
    print(f"  QuantPath Admission Dataset Statistics")
    print(f"{'='*55}")
    print(f"  Applicants:   {len(applicant_ids):,}")
    print(f"  Total records: {total:,}")
    print(f"\n  Result distribution:")
    for k, v in results.most_common():
        pct = v / total * 100
        print(f"    {k:15} {v:5} ({pct:.1f}%)")
    print(f"\n  Top 10 programs:")
    for k, v in programs.most_common(10):
        print(f"    {k:22} {v:5}")
    print(f"\n  Data sources:")
    for k, v in sources.most_common():
        print(f"    {k:15} {v:5}")
    print(f"\n  Background distribution (Top 8):")
    for k, v in bg_types.most_common(8):
        print(f"    {k or '(empty)':20} {v:5}")

    # Raw files stats
    if _RAW_DIR.exists():
        raw_count = len(list(_RAW_DIR.glob("*.txt")))
        print(f"\n  Scraped raw posts: {raw_count}")

    print(f"{'='*55}")


# ===================================================================
# CLI
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="1point3acres MFE admission data scraper + parser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # login (step 1)
    sub.add_parser("login", help="Open browser to log in to 1point3acres (saves session)")

    # scrape (step 2)
    sp_scrape = sub.add_parser("scrape", help="Search and scrape MFE admission posts (login required)")
    sp_scrape.add_argument("--pages", type=int, default=3, help="Search pages per query (default: 3)")
    sp_scrape.add_argument("--headless", action="store_true", help="Headless mode")

    # parse
    sp_parse = sub.add_parser("parse", help="Parse scraped raw text -> CSV")
    sp_parse.add_argument("--season", default="26Fall", help="Default application season")

    # manual
    sp_manual = sub.add_parser("manual", help="Manually paste/input post text")
    sp_manual.add_argument("--input", "-i", help="Input file path")
    sp_manual.add_argument("--season", default="26Fall", help="Application season")
    sp_manual.add_argument("--dry-run", action="store_true", help="Display parsed results only, do not write CSV")

    # stats
    sub.add_parser("stats", help="Show dataset statistics")

    args = parser.parse_args()

    if args.command == "login":
        do_login()
    elif args.command == "scrape":
        scrape_posts(max_pages=args.pages, headless=args.headless)
    elif args.command == "parse":
        parse_all_raw(season=args.season)
    elif args.command == "manual":
        manual_parse(input_path=args.input, season=args.season, dry_run=args.dry_run)
    elif args.command == "stats":
        print_stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
