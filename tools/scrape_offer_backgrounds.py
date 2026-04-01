#!/usr/bin/env python3
# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""Scrape individual applicant backgrounds + thread content from offer.1point3acres.com.

Flow per program:
  1. Go to /program/{id}/results
  2. For each page of results, click each 详情 link
  3. Read structured background from modal (GPA, school tier, etc.)
  4. Get thread URL from "前往论坛" button
  5. Close modal, move to next entry
  6. After all modals, visit each thread URL and extract post text

Output: data/admissions/offer_backgrounds_full.csv
"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "admissions"
BG_CSV = OUTPUT_DIR / "offer_backgrounds_full.csv"
THREADS_DIR = OUTPUT_DIR / "offer_threads"

PROGRAMS = [
    46, 87, 129, 162, 170, 180, 230, 324, 511, 620, 1017, 1700,
]

BG_FIELDS = [
    "program_id", "result_index", "year", "term", "major", "degree",
    "school_tier", "undergrad_major", "gpa_raw", "toefl_raw", "gre_raw",
    "work_exp", "research", "awards", "thread_url", "has_data",
]

FIELD_LABELS = [
    ("year", "申入学年度"),
    ("term", "入学学期"),
    ("major", "申请专业"),
    ("degree", "申请学位"),
    ("school_tier", "本科学校档次"),
    ("undergrad_major", "本科专业"),
    ("gpa_raw", "本科成绩和算法"),
    ("toefl_raw", "托福/雅思"),
    ("gre_raw", "GRE/GMAT"),
    ("work_exp", "相关工作经验范围"),
    ("research", "论文/科研经历"),
    ("awards", "学科竞赛/奖励"),
]


def parse_modal_text(text: str) -> dict:
    """Parse the structured modal text into a dict of fields."""
    row = {}
    text = text.replace("\n", " ").strip()

    positions = []
    for key, label in FIELD_LABELS:
        idx = text.find(label)
        if idx >= 0:
            positions.append((idx, key, label))
    positions.sort()

    for i, (pos, key, label) in enumerate(positions):
        start = pos + len(label)
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        val = text[start:end]
        val = re.sub(r"^[，,排名\s总分\+单项()（）R\+L\+S\+WV\+Q\+AW]+", "", val).strip()
        val = val.rstrip("查看申请背景").strip()
        row[key] = val

    return row


def scrape_all():
    from playwright.sync_api import sync_playwright

    state_dir = str(OUTPUT_DIR / "raw_1p3a" / ".browser_state_v3")
    THREADS_DIR.mkdir(parents=True, exist_ok=True)

    all_backgrounds = []
    all_thread_urls = set()

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=state_dir,
            headless=False,
            viewport={"width": 1280, "height": 900},
        )
        page = ctx.new_page()

        # Quick login check
        page.goto("https://offer.1point3acres.com", wait_until="domcontentloaded")
        time.sleep(2)

        for prog_id in PROGRAMS:
            print(f"\n{'='*60}")
            print(f"  Program {prog_id}")
            print(f"{'='*60}")

            try:
                page.goto(
                    f"https://offer.1point3acres.com/program/{prog_id}/results",
                    wait_until="networkidle",
                    timeout=20000,
                )
            except Exception:
                page.goto(
                    f"https://offer.1point3acres.com/program/{prog_id}/results",
                    wait_until="domcontentloaded",
                    timeout=20000,
                )
            time.sleep(3)
            page.wait_for_load_state("networkidle", timeout=10000)

            # Get page count
            max_page = page.evaluate("""() => {
                const nums = [...document.querySelectorAll('li')]
                    .map(l => parseInt(l.textContent))
                    .filter(n => !isNaN(n) && n > 0);
                return Math.max(1, ...nums);
            }""")
            print(f"  {max_page} pages")

            for pg in range(1, min(max_page + 1, 15)):
                if pg > 1:
                    try:
                        page.locator("li").filter(has_text=re.compile(rf"^{pg}$")).first.click()
                        time.sleep(1500 / 1000)
                    except Exception:
                        break

                detail_count = page.locator('a:has-text("详情")').count()
                print(f"  Page {pg}: {detail_count} entries")

                for i in range(detail_count):
                    try:
                        # Click 详情
                        page.locator('a:has-text("详情")').nth(i).click()
                        time.sleep(1)

                        # Read modal content
                        modal_data = page.evaluate("""() => {
                            const body = document.body.innerHTML;
                            const idx = body.indexOf('查看申请背景');
                            if (idx < 0) return {found: false};

                            const hasRequestMsg = body.indexOf('请求汇报者补充背景') > idx;

                            // Thread link
                            const link = document.querySelector('a[href*="/thread/"]');
                            const threadUrl = link ? link.getAttribute('href') : '';

                            // Modal text
                            const chunk = body.slice(idx, idx + 6000);
                            const tmp = document.createElement('div');
                            tmp.innerHTML = chunk;
                            const text = tmp.innerText;

                            const hasData = text.includes('本科学校档次') && !hasRequestMsg;

                            return {found: true, hasData, threadUrl, text: text.slice(0, 2000)};
                        }""")

                        if modal_data.get("found"):
                            row = {
                                "program_id": prog_id,
                                "result_index": (pg - 1) * 15 + i,
                                "thread_url": modal_data.get("threadUrl", ""),
                                "has_data": modal_data.get("hasData", False),
                            }

                            if modal_data["hasData"]:
                                parsed = parse_modal_text(modal_data["text"])
                                row.update(parsed)
                                print(f"    ✓ [{i}] BG: GPA={parsed.get('gpa_raw','?')[:10]} tier={parsed.get('school_tier','?')[:15]}")
                            else:
                                print(f"    - [{i}] No background data")

                            if modal_data.get("threadUrl"):
                                all_thread_urls.add(modal_data["threadUrl"])

                            all_backgrounds.append(row)

                        # Close modal properly
                        closed = False
                        for sel in [".ant-modal-close", '[aria-label="Close"]', "button:has(svg)"]:
                            try:
                                close_btn = page.locator(sel).first
                                if close_btn.count() > 0 and close_btn.is_visible():
                                    close_btn.click()
                                    closed = True
                                    break
                            except Exception:
                                pass

                        if not closed:
                            # Click outside the modal
                            page.mouse.click(10, 10)

                        time.sleep(0.5)

                        # Verify modal is closed
                        page.evaluate("""() => {
                            const modals = document.querySelectorAll('.ant-modal-wrap');
                            modals.forEach(m => m.style.display = 'none');
                        }""")
                        time.sleep(0.3)

                    except Exception as e:
                        print(f"    ✗ [{i}] Error: {e}")

        # --- Phase 2: Visit thread pages ---
        print(f"\n{'='*60}")
        print(f"  Visiting {len(all_thread_urls)} thread pages")
        print(f"{'='*60}")

        for url in sorted(all_thread_urls):
            tid = re.search(r"/thread/(\d+)", url)
            if not tid:
                continue
            tid_str = tid.group(1)
            out_file = THREADS_DIR / f"{tid_str}.txt"
            if out_file.exists():
                print(f"  Skip {tid_str} (cached)")
                continue

            try:
                full_url = url if url.startswith("http") else f"https://offer.1point3acres.com{url}"
                page.goto(full_url, wait_until="networkidle", timeout=15000)
                time.sleep(1.5)

                text = page.evaluate("""() => {
                    // Get the main post content
                    const article = document.querySelector('article, .thread-content, main');
                    return article ? article.innerText : document.body.innerText.slice(0, 10000);
                }""")

                if text and len(text) > 50:
                    out_file.write_text(
                        json.dumps({"url": full_url, "tid": tid_str}, ensure_ascii=False)
                        + "\n---\n"
                        + text,
                        encoding="utf-8",
                    )
                    print(f"  ✓ thread/{tid_str} ({len(text)} chars)")
                else:
                    print(f"  ✗ thread/{tid_str} (empty)")

            except Exception as e:
                print(f"  ✗ thread/{tid_str}: {e}")

            time.sleep(1)

        ctx.close()

    # Save backgrounds CSV
    with BG_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in all_backgrounds:
            writer.writerow(row)

    # Stats
    with_data = sum(1 for r in all_backgrounds if r.get("has_data"))
    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  Total entries: {len(all_backgrounds)}")
    print(f"  With background: {with_data}")
    print(f"  Thread URLs: {len(all_thread_urls)}")
    print(f"  Threads saved: {len(list(THREADS_DIR.glob('*.txt')))}")
    print(f"  CSV: {BG_CSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    scrape_all()
