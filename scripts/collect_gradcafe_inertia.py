#!/usr/bin/env python3
# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""
Scrape GradCafe via Inertia.js data-page JSON embedded in HTML.
Paginates through all pages for multiple program queries.
"""

import csv
import hashlib
import html
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

BASE = Path("/Users/ethanyang/QuantPath/data/admissions")
UNIFIED_CSV = BASE / "collected_multidim.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
}

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

MFE_PROGRAMS = {
    "baruch": "baruch-mfe", "cmu": "cmu-mscf", "mscf": "cmu-mscf",
    "columbia": "columbia-msfe", "princeton": "princeton-mfin",
    "mit": "mit-mfin", "berkeley": "berkeley-mfe",
    "uchicago": "uchicago-msfm", "chicago": "uchicago-msfm",
    "gatech": "gatech-qcf", "georgia tech": "gatech-qcf",
    "cornell": "cornell-mfe", "nyu": "nyu-mfe",
    "nyu tandon": "nyu-tandon-mfe",
    "stanford": "stanford-mcf", "ucla": "ucla-mfe",
    "uiuc": "uiuc-msfe", "illinois": "uiuc-msfe",
    "rutgers": "rutgers-mqf", "boston university": "bu-msmf",
    "toronto": "toronto-mmf", "michigan": "michigan-qfr",
    "carnegie mellon": "cmu-mscf", "stony brook": "stonybrook-qf",
    "fordham": "fordham-msqf", "johns hopkins": "jhu-mfm",
    "north carolina state": "ncstate-mfm",
    "usc": "usc-msmf", "washington": "uwash-cfrm",
    "stevens": "stevens-mfe", "oxford": "oxford-mcf",
    "hec montreal": "hec-mfe",
}


def make_id(source, *parts):
    raw = f"{source}:{'|'.join(str(p) for p in parts)}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def content_hash(text):
    norm = re.sub(r'\s+', ' ', str(text).lower().strip())[:500]
    return hashlib.md5(norm.encode()).hexdigest()[:16]


def map_program(school, program_name):
    """Map GradCafe school+program to our program ID."""
    combined = f"{school} {program_name}".lower()
    for key, prog_id in sorted(MFE_PROGRAMS.items(), key=lambda x: -len(x[0])):
        if key in combined:
            return prog_id
    if "financial engineering" in combined:
        return "mfe-unknown"
    if "financial math" in combined or "mathematical finance" in combined:
        return "msfm-unknown"
    if "computational finance" in combined:
        return "cmu-mscf" if "carnegie" in combined else "mfe-unknown"
    if "quantitative finance" in combined:
        return "finance-unknown"
    if "operations research" in combined:
        return "or-unknown"
    if "finance" in combined:
        return "finance-unknown"
    return None


def map_result(decision):
    d = decision.lower()
    if "accepted" in d or "admitted" in d:
        return "accepted"
    if "rejected" in d or "denied" in d:
        return "rejected"
    if "waitlisted" in d or "wait" in d:
        return "waitlisted"
    if "interview" in d:
        return None  # interview is not a final result
    return None


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


def extract_page_data(html_text):
    """Extract Inertia.js page data from HTML."""
    idx = html_text.find('data-page="')
    if idx < 0:
        return None

    raw = html_text[idx + 11:]
    # Find end of attribute
    end = raw.find('" id="app"')
    if end < 0:
        end = raw.find('"></div>')
    if end < 0:
        end = raw.find('">')
    if end < 0:
        return None

    encoded = raw[:end]
    decoded = html.unescape(encoded)

    # Extract results data section using regex (safer than full JSON parse)
    results = []
    # Find the data array
    data_start = decoded.find('"data":[{')
    if data_start < 0:
        return {"results": [], "last_page": 1, "total": 0}

    # Find total and last_page
    total_m = re.search(r'"total":(\d+)', decoded)
    last_page_m = re.search(r'"last_page":(\d+)', decoded)
    total = int(total_m.group(1)) if total_m else 0
    last_page = int(last_page_m.group(1)) if last_page_m else 1

    # Extract individual records using regex
    # Each record starts with {"id": and ends before the next {"id":
    record_pattern = re.compile(
        r'\{"id":(\d+),'
        r'"school":"([^"]*)",'
        r'"program":"([^"]*)",'
        r'"level":"([^"]*)",'
        r'"how":"([^"]*)",'
        r'"decision":"([^"]*)",'
        r'"date_of_notification":"([^"]*)",'
        r'"created_at":"([^"]*)",'
        r'"notes":"((?:[^"\\]|\\.)*)",'  # notes can have escaped chars
        r'"status":"([^"]*)",'
        r'"season":"([^"]*)",'
        r'"ugpa":"([^"]*)",'
        r'"greq":(\d+),'
        r'"grev":(\d+),'
    )

    for m in record_pattern.finditer(decoded):
        results.append({
            "gc_id": int(m.group(1)),
            "school": m.group(2),
            "program": m.group(3),
            "level": m.group(4),
            "how": m.group(5),
            "decision": m.group(6),
            "date": m.group(7),
            "created": m.group(8),
            "notes": m.group(9),
            "status": m.group(10),
            "season": m.group(11),
            "ugpa": m.group(12),
            "greq": int(m.group(13)),
            "grev": int(m.group(14)),
        })

    return {"results": results, "last_page": last_page, "total": total}


def scrape_gradcafe():
    queries = [
        "Financial Engineering",
        "Computational Finance",
        "Mathematical Finance",
        "Quantitative Finance",
        "Financial Mathematics",
        "Operations Research",
        "MFE",
    ]

    # Also search by institution
    institutions = [
        "Carnegie Mellon University",
        "Columbia University",
        "New York University",
        "University of California Berkeley",
        "Princeton University",
        "Baruch College",
        "University of Chicago",
        "Georgia Institute of Technology",
        "Cornell University",
        "Massachusetts Institute of Technology",
        "Stanford University",
        "UCLA",
        "University of Illinois",
        "Boston University",
        "Rutgers University",
    ]

    all_records = []
    seen_gc_ids = set()

    # Search by program
    for query in queries:
        print(f"\n  Querying: {query}")
        for page in range(1, 200):
            url = f"https://www.thegradcafe.com/survey?q={query}&page={page}"
            try:
                resp = requests.get(url, headers=HEADERS, timeout=20)
                if resp.status_code != 200:
                    print(f"    Page {page}: HTTP {resp.status_code}")
                    break

                page_data = extract_page_data(resp.text)
                if not page_data or not page_data["results"]:
                    break

                new_count = 0
                for gc_rec in page_data["results"]:
                    if gc_rec["gc_id"] in seen_gc_ids:
                        continue
                    seen_gc_ids.add(gc_rec["gc_id"])

                    program = map_program(gc_rec["school"], gc_rec["program"])
                    if not program:
                        continue

                    result = map_result(gc_rec["decision"])
                    gpa = None
                    gpa_scale = ""
                    try:
                        gpa_val = float(gc_rec["ugpa"])
                        if 0 < gpa_val <= 4.0:
                            gpa = gpa_val
                            gpa_scale = "4.0"
                        elif 50 <= gpa_val <= 100:
                            gpa = gpa_val
                            gpa_scale = "100"
                    except (ValueError, TypeError):
                        pass

                    gre_q = gc_rec["greq"] if gc_rec["greq"] and 130 <= gc_rec["greq"] <= 170 else None
                    gre_v = gc_rec["grev"] if gc_rec["grev"] and 130 <= gc_rec["grev"] <= 170 else None

                    nationality = ""
                    status = gc_rec["status"].lower()
                    if "international" in status:
                        nationality = "international"
                    elif "american" in status or "us" in status or "domestic" in status:
                        nationality = "us"

                    season = gc_rec["season"] if gc_rec["season"] else ""
                    # Normalize season
                    sm = re.match(r'(Fall|Spring)\s*(20\d{2})', season)
                    if sm:
                        yr = sm.group(2)[2:]
                        sem = sm.group(1)
                        season = f"{yr}{sem}"

                    raw_text = f"{gc_rec['school']} {gc_rec['program']} {gc_rec['decision']} GPA:{gc_rec['ugpa']} GRE Q:{gc_rec['greq']} V:{gc_rec['grev']} {gc_rec['status']} {gc_rec['season']} {gc_rec['notes']}"

                    rec = {
                        "id": make_id("gradcafe", gc_rec["gc_id"]),
                        "source": "gradcafe",
                        "program": program,
                        "result": result or "",
                        "season": season,
                        "gpa": gpa or "",
                        "gpa_scale": gpa_scale,
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
                        "nationality": nationality,
                        "raw_text": raw_text[:2000].replace("\n", " ").replace(",", ";"),
                    }
                    rec["tier"] = compute_tier(rec)
                    all_records.append(rec)
                    new_count += 1

                if page == 1:
                    print(f"    Total available: {page_data['total']}, pages: {page_data['last_page']}")
                if page % 10 == 0:
                    print(f"    Page {page}/{page_data['last_page']}: {new_count} new ({len(all_records)} total)")

                if page >= page_data["last_page"]:
                    break

            except Exception as e:
                print(f"    Page {page} error: {e}")
                break

            time.sleep(1.0)

    # Search by institution
    for inst in institutions:
        print(f"\n  Institution: {inst}")
        for page in range(1, 50):
            url = f"https://www.thegradcafe.com/survey?institution={inst}&page={page}"
            try:
                resp = requests.get(url, headers=HEADERS, timeout=20)
                if resp.status_code != 200:
                    break

                page_data = extract_page_data(resp.text)
                if not page_data or not page_data["results"]:
                    break

                new_count = 0
                for gc_rec in page_data["results"]:
                    if gc_rec["gc_id"] in seen_gc_ids:
                        continue
                    seen_gc_ids.add(gc_rec["gc_id"])

                    # Filter for finance/quant programs
                    prog_text = f"{gc_rec['school']} {gc_rec['program']}".lower()
                    is_quant = any(kw in prog_text for kw in [
                        "financ", "quant", "mfe", "mscf", "mfin", "msfm",
                        "computational", "math", "operations research",
                    ])
                    if not is_quant:
                        continue

                    program = map_program(gc_rec["school"], gc_rec["program"])
                    if not program:
                        continue

                    result = map_result(gc_rec["decision"])
                    gpa = None
                    gpa_scale = ""
                    try:
                        gpa_val = float(gc_rec["ugpa"])
                        if 0 < gpa_val <= 4.0:
                            gpa = gpa_val
                            gpa_scale = "4.0"
                    except (ValueError, TypeError):
                        pass

                    gre_q = gc_rec["greq"] if gc_rec["greq"] and 130 <= gc_rec["greq"] <= 170 else None
                    gre_v = gc_rec["grev"] if gc_rec["grev"] and 130 <= gc_rec["grev"] <= 170 else None

                    nationality = ""
                    if "international" in gc_rec["status"].lower():
                        nationality = "international"
                    elif "american" in gc_rec["status"].lower():
                        nationality = "us"

                    season = gc_rec["season"] if gc_rec["season"] else ""
                    sm = re.match(r'(Fall|Spring)\s*(20\d{2})', season)
                    if sm:
                        season = f"{sm.group(2)[2:]}{sm.group(1)}"

                    raw_text = f"{gc_rec['school']} {gc_rec['program']} {gc_rec['decision']} GPA:{gc_rec['ugpa']} GRE Q:{gc_rec['greq']} V:{gc_rec['grev']} {gc_rec['status']} {gc_rec['season']} {gc_rec['notes']}"

                    rec = {
                        "id": make_id("gradcafe", gc_rec["gc_id"]),
                        "source": "gradcafe",
                        "program": program,
                        "result": result or "",
                        "season": season,
                        "gpa": gpa or "",
                        "gpa_scale": gpa_scale,
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
                        "nationality": nationality,
                        "raw_text": raw_text[:2000].replace("\n", " ").replace(",", ";"),
                    }
                    rec["tier"] = compute_tier(rec)
                    all_records.append(rec)
                    new_count += 1

                if page == 1 and page_data["total"] > 0:
                    print(f"    Total: {page_data['total']}")
                if page >= page_data["last_page"]:
                    break

            except Exception as e:
                print(f"    Page {page} error: {e}")
                break

            time.sleep(1.0)

    print(f"\n  GradCafe total scraped: {len(all_records)} records")
    return all_records


def main():
    print("=" * 70)
    print("GradCafe Inertia.js Scraping")
    print(f"Started: {datetime.now()}")
    print("=" * 70)

    records = scrape_gradcafe()

    # Load existing and merge
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

    print(f"\n  New unique records after dedup: {len(new_records)}")

    if new_records:
        new_df = pd.DataFrame(new_records, columns=UNIFIED_FIELDS).fillna("")
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(UNIFIED_CSV, index=False)
        print(f"  Saved {len(combined)} total records")

        for t in ["A", "B", "C", "D"]:
            cnt = sum(1 for r in new_records if r["tier"] == t)
            print(f"    Tier {t}: {cnt}")
    else:
        print("  No new records to add.")

    print(f"\nDone: {datetime.now()}")


if __name__ == "__main__":
    main()
