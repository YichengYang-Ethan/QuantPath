#!/usr/bin/env python3
# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""
GradCafe + QuantNet MFE 数据爬虫（无需 Claude API）

用正则解析 GradCafe 已结构化的表格数据，
同时爬取 QuantNet profile 帖子中的结构化结果。

用法:
    python tools/scrape_gradcafe.py
    python tools/scrape_gradcafe.py --pages 20 --output data/admissions/collected.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import time
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
OUTPUT_CSV = Path(__file__).parent.parent / "data" / "admissions" / "collected.csv"
SEEN_URLS  = Path(__file__).parent.parent / "data" / "admissions" / ".seen_urls.txt"
CSV_FIELDS = [
    "id","gender","bg_type","nationality","gpa","gpa_scale",
    "gre","toefl","major","intern_desc","has_paper","has_research",
    "courses_note","program","result","season","source",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# ---------------------------------------------------------------------------
# GradCafe 搜索词 → 目标 program_id 列表
# ---------------------------------------------------------------------------
GRADCAFE_TARGETS = [
    # (搜索词, 匹配的 institution 关键词, program_id)
    ("baruch+college+financial+engineering",   "baruch",          "baruch-mfe"),
    ("carnegie+mellon+computational+finance",  "carnegie mellon", "cmu-mscf"),
    ("princeton+financial+engineering",        "princeton",       "princeton-mfin"),
    ("princeton+master+finance",               "princeton",       "princeton-mfin"),
    ("berkeley+financial+engineering",         "berkeley",        "berkeley-mfe"),
    ("columbia+financial+engineering",         "columbia",        "columbia-msfe"),
    ("columbia+mathematical+finance",          "columbia",        "columbia-mafn"),
    ("mit+master+finance",                     "massachusetts",   "mit-mfin"),
    ("stanford+mathematical+finance",          "stanford",        "stanford-mcf"),
    ("chicago+financial+mathematics",          "chicago",         "uchicago-msfm"),
    ("cornell+financial+engineering",          "cornell",         "cornell-mfe"),
    ("nyu+financial+engineering",              "nyu",             "nyu-tandon-mfe"),
    ("nyu+mathematics+finance",                "nyu",             "nyu-courant"),
    ("georgia+tech+quantitative",              "georgia",         "gatech-qcf"),
    ("rutgers+quantitative+finance",           "rutgers",         "rutgers-mqf"),
    ("ucla+financial+engineering",             "los angeles",     "ucla-mfe"),
    ("uiuc+financial+engineering",             "illinois",        "uiuc-msfe"),
]

# 更广泛的搜索，用来批量抓取所有 MFE 相关结果
BROAD_QUERIES = [
    "financial+engineering+masters",
    "computational+finance+masters",
    "mathematical+finance+masters",
    "quantitative+finance+masters",
]

# institution 关键词 → program_id（用于广泛搜索时的映射）
INST_MAP = {
    "baruch":          "baruch-mfe",
    "carnegie mellon": "cmu-mscf",
    "cmu":             "cmu-mscf",
    "princeton":       "princeton-mfin",
    "berkeley":        "berkeley-mfe",
    "columbia":        "columbia-msfe",
    "mit ":            "mit-mfin",
    "massachusetts":   "mit-mfin",
    "stanford":        "stanford-mcf",
    "chicago":         "uchicago-msfm",
    "cornell":         "cornell-mfe",
    "nyu tandon":      "nyu-tandon-mfe",
    "new york university": "nyu-tandon-mfe",
    "nyu":             "nyu-tandon-mfe",
    "georgia tech":    "gatech-qcf",
    "georgia institute": "gatech-qcf",
    "rutgers":         "rutgers-mqf",
    "ucla":            "ucla-mfe",
    "los angeles":     "ucla-mfe",
    "illinois":        "uiuc-msfe",
    "northwestern":    "northwestern-mfe",
    "johns hopkins":   "jhu-mfm",
    "nc state":        "ncstate-mfm",
    "north carolina state": "ncstate-mfm",
    "fordham":         "fordham-msqf",
    "washington":      "uwash-cfrm",
    "stevens":         "stevens-mfe",
    "usc":             "usc-msmf",
    "southern california": "usc-msmf",
    "boston university": "bu-msmf",
    "michigan":        "umich-mfe",
    "minnesota":       "uminn-mfm",
    "toronto":         "utoronto-mmf",
    "charlotte":       "uncc-msmf",
}

PROGRAM_KEYWORDS = {
    "financial engineering": True,
    "computational finance": True,
    "mathematical finance":  True,
    "quantitative finance":  True,
    "mscf": True,
    "mfin": True,
    "mfe":  True,
    "msfe": True,
    "mafn": True,
    "msfm": True,
    "qcf":  True,
    "mqf":  True,
    "mmf":  True,
}

# ---------------------------------------------------------------------------
# GradCafe 解析
# ---------------------------------------------------------------------------

def map_institution_to_program(institution: str, program_raw: str) -> str | None:
    inst_lower  = institution.lower()
    prog_lower  = program_raw.lower()

    # 先检查 program 名称是否包含 MFE 相关词
    is_mfe_program = any(kw in prog_lower for kw in PROGRAM_KEYWORDS)
    if not is_mfe_program:
        return None

    # 再根据学校名匹配
    for key, pid in INST_MAP.items():
        if key in inst_lower:
            # Columbia 两个项目区分
            if "columbia" in inst_lower:
                if "mafn" in prog_lower or "mathematical" in prog_lower:
                    return "columbia-mafn"
                return "columbia-msfe"
            # NYU 两个项目区分
            if "nyu" in inst_lower or "new york university" in inst_lower:
                if "courant" in prog_lower or "mathematics in finance" in prog_lower:
                    return "nyu-courant"
                return "nyu-tandon-mfe"
            return pid
    return None


def parse_gradcafe_season(season_str: str) -> str:
    """'F25', 'Fall 2025', 'F26', 'Fall 2026' → '25Fall' / '26Fall'"""
    s = season_str.strip()
    m = re.search(r'(\d{2,4})', s)
    if not m:
        return ""
    year = m.group(1)
    if len(year) == 4:
        year = year[2:]
    return f"{year}Fall"


def parse_gradcafe_page(html: str, forced_program_id: str | None = None) -> list[dict]:
    """解析 GradCafe 一页 HTML，返回结构化记录列表"""
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")
    records = []
    i = 0

    while i < len(rows):
        row = rows[i]
        cells = row.find_all("td")

        # 主行：有5列 → institution | program | date | result | comments
        if len(cells) >= 4:
            inst_cell = cells[0].get_text(strip=True)
            prog_cell = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            result_cell = cells[3].get_text(strip=True) if len(cells) > 3 else ""

            program_id = forced_program_id or map_institution_to_program(inst_cell, prog_cell)
            if not program_id:
                i += 1
                continue

            # 解析 result
            result_lower = result_cell.lower()
            if "accepted" in result_lower:
                result = "accepted"
            elif "rejected" in result_lower or "denied" in result_lower:
                result = "rejected"
            elif "wait" in result_lower:
                result = "waitlisted"
            elif "interview" in result_lower:
                result = "interview"
            else:
                i += 1
                continue

            # 下一行通常有 details：nationality, GPA, GRE, season
            gpa, gpa_scale, gre, toefl, nationality, season = None, 4, None, None, "", ""
            if i + 1 < len(rows):
                detail_row = rows[i + 1]
                detail_text = detail_row.get_text(separator=" ", strip=True)

                # GPA
                gpa_m = re.search(r'GPA\s+([\d.]+)', detail_text)
                if gpa_m:
                    gpa = float(gpa_m.group(1))
                    gpa_scale = 100 if gpa > 10 else 4

                # GRE
                gre_m = re.search(r'GRE\s+(\d{3})', detail_text)
                if gre_m:
                    gre = int(gre_m.group(1))

                # TOEFL
                toefl_m = re.search(r'TOEFL\s+(\d+)', detail_text)
                if toefl_m:
                    toefl = int(toefl_m.group(1))

                # Nationality
                if "american" in detail_text.lower() or "domestic" in detail_text.lower():
                    nationality = "美籍"
                elif "international" in detail_text.lower():
                    nationality = "其他国际"

                # Season: F25, Fall 2025, Fall 2026 等
                season_m = re.search(r'(F\d{2}|Fall\s+20\d{2}|\d{2}Fall)', detail_text)
                if season_m:
                    season = parse_gradcafe_season(season_m.group(1))

            records.append({
                "program_id": program_id,
                "result": result,
                "gpa": gpa,
                "gpa_scale": gpa_scale,
                "gre": gre,
                "toefl": toefl,
                "nationality": nationality,
                "season": season,
                "source": "gradcafe",
            })

        i += 1

    return records


def scrape_gradcafe(query: str, max_pages: int = 10,
                    forced_program_id: str | None = None) -> list[dict]:
    all_records = []
    for page in range(1, max_pages + 1):
        url = f"https://www.thegradcafe.com/survey/index.php?q={query}&t=a&o=&p={page}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            records = parse_gradcafe_page(resp.text, forced_program_id)
            if not records and page > 1:
                break  # 没有更多数据
            all_records.extend(records)
            print(f"    第{page}页: {len(records)} 条", end="\r")
        except Exception as e:
            print(f"\n  [错误] {e}")
            break
        time.sleep(1.2)
    print()
    return all_records


# ---------------------------------------------------------------------------
# QuantNet 爬虫（解析 profile 帖里的结构化结果）
# ---------------------------------------------------------------------------

RESULT_PATTERN = re.compile(
    r'(baruch|cmu|mscf|princeton|berkeley|columbia|mit|stanford|uchicago|chicago|'
    r'cornell|nyu|gatech|georgia tech|rutgers|ucla|uiuc|illinois|northwestern|'
    r'johns hopkins|jhu|nc state|fordham|rutgers|stevens|usc|boston|michigan|'
    r'minnesota|toronto|mfin|mfe|msfe|mafn|msfm|qcf|mqf)'
    r'[\s\w]*'
    r'[\(\[\s]*'
    r'(accepted|rejected|waitlisted|waitlist|admitted|admit|declined|denied|interview)'
    r'[\)\]\s]*',
    re.IGNORECASE
)

PROGRAM_NAME_MAP = {
    "baruch": "baruch-mfe",
    "cmu": "cmu-mscf", "mscf": "cmu-mscf", "carnegie": "cmu-mscf",
    "princeton": "princeton-mfin",
    "berkeley": "berkeley-mfe",
    "columbia msfe": "columbia-msfe", "columbia mfe": "columbia-msfe",
    "columbia mafn": "columbia-mafn", "columbia": "columbia-msfe",
    "mit": "mit-mfin",
    "stanford": "stanford-mcf",
    "uchicago": "uchicago-msfm", "chicago": "uchicago-msfm", "msfm": "uchicago-msfm",
    "cornell": "cornell-mfe",
    "nyu tandon": "nyu-tandon-mfe", "nyu courant": "nyu-courant", "nyu": "nyu-tandon-mfe",
    "gatech": "gatech-qcf", "georgia tech": "gatech-qcf", "qcf": "gatech-qcf",
    "rutgers": "rutgers-mqf",
    "ucla": "ucla-mfe",
    "uiuc": "uiuc-msfe", "illinois": "uiuc-msfe",
    "northwestern": "northwestern-mfe",
    "jhu": "jhu-mfm", "johns hopkins": "jhu-mfm",
    "nc state": "ncstate-mfm",
    "fordham": "fordham-msqf",
    "stevens": "stevens-mfe",
    "usc": "usc-msmf",
    "boston": "bu-msmf",
    "michigan": "umich-mfe",
    "minnesota": "uminn-mfm",
    "toronto": "utoronto-mmf",
}


def map_name_to_program_id(name: str) -> str | None:
    name_l = name.lower()
    # 精确匹配
    for k, v in PROGRAM_NAME_MAP.items():
        if name_l.startswith(k):
            return v
    # 子串匹配
    for k, v in PROGRAM_NAME_MAP.items():
        if k in name_l:
            return v
    return None


def normalize_result(r: str) -> str:
    r_l = r.lower()
    if any(x in r_l for x in ("accepted", "admitted", "admit")):
        return "accepted"
    if any(x in r_l for x in ("rejected", "denied", "declined")):
        return "rejected"
    if "waitlist" in r_l or "waitlisted" in r_l:
        return "waitlisted"
    if "interview" in r_l:
        return "interview"
    return r_l.strip()


def extract_profile_from_thread(text: str) -> dict:
    """从帖子文本提取GPA、GRE等信息"""
    gpa, gpa_scale, gre, toefl = None, 4, None, None

    gpa_m = re.search(r'GPA[:\s]+([0-9.]+)\s*(?:/\s*([0-9.]+))?', text, re.IGNORECASE)
    if gpa_m:
        gpa = float(gpa_m.group(1))
        if gpa_m.group(2):
            gpa_scale = float(gpa_m.group(2))
            if gpa_scale > 10:
                gpa_scale = 100
            elif gpa_scale > 4:
                gpa_scale = 10
        elif gpa > 10:
            gpa_scale = 100

    gre_m = re.search(r'GRE[:\s]+(\d{3})', text, re.IGNORECASE)
    gre_q_m = re.search(r'GRE[^0-9]*Q[:\s]*(\d{3})|Q\s*(\d{3})\s*[/,]', text, re.IGNORECASE)
    if gre_q_m:
        gre = int(gre_q_m.group(1) or gre_q_m.group(2))
    elif gre_m:
        gre = int(gre_m.group(1))

    toefl_m = re.search(r'TOEFL[:\s]+(\d+)', text, re.IGNORECASE)
    if toefl_m:
        toefl = int(toefl_m.group(1))

    nationality = ""
    if re.search(r'\b(domestic|US citizen|american)\b', text, re.IGNORECASE):
        nationality = "美籍"
    elif re.search(r'\bIIT\b', text):
        nationality = "印度"
    elif re.search(r'\binternational\b', text, re.IGNORECASE):
        nationality = "其他国际"

    season_m = re.search(r'(Fall\s+20\d{2}|20\d{2}\s+Fall|\d{2}Fall)', text)
    season = ""
    if season_m:
        season = parse_gradcafe_season(season_m.group(1))

    return {"gpa": gpa, "gpa_scale": gpa_scale, "gre": gre, "toefl": toefl,
            "nationality": nationality, "season": season}


def scrape_quantnet_thread(url: str) -> list[dict]:
    """爬取一个 QuantNet 帖子，提取结构化结果"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 提取前3楼
        posts = soup.find_all("article", class_="message")[:3]
        full_text = "\n".join(
            p.find("div", class_="bbWrapper").get_text("\n", strip=True)
            for p in posts
            if p.find("div", class_="bbWrapper")
        )

        profile = extract_profile_from_thread(full_text)

        # 找 "Program (Result)" 格式
        results = []
        # 格式1: Princeton MFin (Rejected)
        for m in re.finditer(
            r'([A-Za-z\s]+(?:MFE|MFin|MSCF|MSFE|MAFN|MSFM|QCF|MQF|MCF|MMF|Quant\w*)[\s\w]*)'
            r'\s*[\(\[]\s*(accepted|rejected|waitlisted|admitted|denied)\s*[\)\]]',
            full_text, re.IGNORECASE
        ):
            pid = map_name_to_program_id(m.group(1))
            if pid:
                results.append({"program_id": pid, "result": normalize_result(m.group(2))})

        # 格式2: ✅ Baruch | ❌ Princeton
        for m in re.finditer(
            r'(✅|❌|✓|×)\s*([A-Za-z\s]+?)(?:\||$|\n)',
            full_text
        ):
            symbol, name = m.group(1), m.group(2).strip()
            pid = map_name_to_program_id(name)
            if pid:
                result = "accepted" if m.group(1) in ("✅", "✓") else "rejected"
                results.append({"program_id": pid, "result": result})

        if not results:
            return []

        return [{**profile, "source": "quantnet", "results": results}]

    except Exception as e:
        print(f"  [错误] {url}: {e}", file=sys.stderr)
        return []


QUANTNET_PROFILE_URLS = [
    "https://quantnet.com/threads/mfe-2026-profile-evaluation.61831/",
    "https://quantnet.com/threads/profile-for-2026-mfe.57865/",
    "https://quantnet.com/threads/mfe-2026-profile-review.62928/",
    "https://quantnet.com/threads/mfe-2026-applicants-%E2%80%94-which-schools-are-you-applying-to.62839/",
    "https://quantnet.com/threads/profile-review-%E2%80%93-fall-2026-mfe-applications.62425/",
    "https://quantnet.com/threads/mfe-profile-advice.61280/",
    "https://quantnet.com/threads/profile-evaluation-fall-2025-mfe.59249/",
    "https://quantnet.com/threads/profile-evaluation-for-2025-mfe.58244/",
    "https://quantnet.com/threads/profile-evaluation-for-fall-2025-admission.59666/",
    "https://quantnet.com/threads/profile-evaluation-mfe-fall-2025.59778/",
]


def scrape_quantnet_all() -> list[dict]:
    all_records = []
    for url in QUANTNET_PROFILE_URLS:
        print(f"  QuantNet: {url.split('/')[-2][:50]}")
        records = scrape_quantnet_thread(url)
        all_records.extend(records)
        time.sleep(1.5)
    return all_records


# ---------------------------------------------------------------------------
# CSV 写入
# ---------------------------------------------------------------------------

def load_seen() -> set:
    if SEEN_URLS.exists():
        return set(SEEN_URLS.read_text().splitlines())
    return set()


def get_next_id(path: Path) -> int:
    if not path.exists():
        return 1
    with path.open() as f:
        ids = [int(r["id"]) for r in csv.DictReader(f) if r.get("id","").isdigit()]
    return max(ids, default=0) + 1


def write_to_csv(flat_records: list[dict], output: Path) -> int:
    """
    flat_records: 每条已是扁平化的行（含 program_id, result, gpa...）
    或者带 results 列表的嵌套格式（来自 quantnet）
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()
    next_id = get_next_id(output)
    written = 0

    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for rec in flat_records:
            # 处理嵌套 results 格式（来自 QuantNet）
            if "results" in rec:
                results = rec.pop("results")
                gpa = rec.get("gpa", "")
                app_id = next_id
                next_id += 1
                for r in results:
                    row = {
                        "id": app_id,
                        "gender": rec.get("gender", "unknown"),
                        "bg_type": rec.get("bg_type", ""),
                        "nationality": rec.get("nationality", ""),
                        "gpa": gpa or "",
                        "gpa_scale": rec.get("gpa_scale", 4),
                        "gre": rec.get("gre", ""),
                        "toefl": rec.get("toefl", ""),
                        "major": rec.get("major", ""),
                        "intern_desc": "",
                        "has_paper": "不明",
                        "has_research": "不明",
                        "courses_note": "",
                        "program": r.get("program_id", ""),
                        "result": r.get("result", ""),
                        "season": rec.get("season", ""),
                        "source": rec.get("source", ""),
                    }
                    if row["program"] and row["result"]:
                        writer.writerow(row)
                        written += 1
            else:
                # 扁平记录（来自 GradCafe）
                row = {
                    "id": next_id,
                    "gender": "unknown",
                    "bg_type": "",
                    "nationality": rec.get("nationality", ""),
                    "gpa": rec.get("gpa", "") or "",
                    "gpa_scale": rec.get("gpa_scale", 4),
                    "gre": rec.get("gre", "") or "",
                    "toefl": rec.get("toefl", "") or "",
                    "major": "",
                    "intern_desc": "",
                    "has_paper": "不明",
                    "has_research": "不明",
                    "courses_note": "",
                    "program": rec.get("program_id", ""),
                    "result": rec.get("result", ""),
                    "season": rec.get("season", ""),
                    "source": rec.get("source", "gradcafe"),
                }
                if row["program"] and row["result"]:
                    writer.writerow(row)
                    next_id += 1
                    written += 1

    return written


def merge_sample(output: Path) -> int:
    sample = output.parent / "sample.csv"
    if not sample.exists():
        return 0
    if output.exists():
        content = output.read_text()
        if "小红书" in content:
            return 0  # 已合并
    print("合并 sample.csv...")
    rows = list(csv.DictReader(sample.open()))
    write_header = not output.exists()
    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    print(f"  合并了 {len(rows)} 条历史数据")
    return len(rows)


def print_stats(output: Path) -> None:
    if not output.exists():
        print("数据文件不存在")
        return
    from collections import Counter
    programs: Counter = Counter()
    results: Counter = Counter()
    ids: set = set()
    with output.open() as f:
        for row in csv.DictReader(f):
            programs[row.get("program", "")] += 1
            results[row.get("result", "")] += 1
            ids.add(row.get("id", ""))
    print(f"\n{'='*50}")
    print(f"数据集统计")
    print(f"{'='*50}")
    print(f"申请人数: {len(ids)}")
    print(f"记录总数: {sum(programs.values())}")
    print(f"\n结果分布:")
    for k, v in results.most_common():
        if k:
            print(f"  {k:15} {v}")
    print(f"\n项目分布（前15）:")
    for k, v in programs.most_common(15):
        if k:
            print(f"  {k:25} {v}")


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GradCafe + QuantNet MFE 数据爬虫")
    parser.add_argument("--pages", type=int, default=15, help="每个搜索词爬取页数")
    parser.add_argument("--output", default=str(OUTPUT_CSV))
    parser.add_argument("--no-gradcafe", action="store_true")
    parser.add_argument("--no-quantnet", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    total = 0

    # 0. 先合并 sample.csv
    total += merge_sample(output)

    # 1. GradCafe 广泛搜索
    if not args.no_gradcafe:
        print("\n[GradCafe] 开始爬取...")
        all_gc = []
        for query in BROAD_QUERIES:
            print(f"  搜索: {query}")
            records = scrape_gradcafe(query, max_pages=args.pages)
            print(f"  → {len(records)} 条")
            all_gc.extend(records)
        # 定向搜索（更精确）
        for query, inst, pid in GRADCAFE_TARGETS:
            print(f"  定向: {query[:40]}")
            records = scrape_gradcafe(query, max_pages=5, forced_program_id=pid)
            print(f"  → {len(records)} 条")
            all_gc.extend(records)
        if all_gc:
            n = write_to_csv(all_gc, output)
            print(f"\nGradCafe 写入 {n} 条有效记录")
            total += n

    # 2. QuantNet profile 帖子
    if not args.no_quantnet:
        print("\n[QuantNet] 开始爬取...")
        qn_records = scrape_quantnet_all()
        if qn_records:
            n = write_to_csv(qn_records, output)
            print(f"QuantNet 写入 {n} 条有效记录")
            total += n

    print(f"\n本次共写入 {total} 条记录")
    print_stats(output)


if __name__ == "__main__":
    main()
