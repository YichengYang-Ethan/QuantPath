#!/usr/bin/env python3
# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""
QuantPath Data Collector — 从公开来源爬取MFE申请结果数据

数据来源:
  - QuantNet profile evaluation / admission results 帖子 (公开可访问)
  - GradCafe financial engineering 结果页面 (公开可访问)
  - 手动粘贴 (小红书、chasedream、offershow 等任意文本)

输出格式: data/admissions/collected.csv (与 template.csv 格式一致)

使用方法:
    # 爬取 QuantNet profile 帖子
    python tools/collect_data.py --source quantnet --pages 5

    # 爬取 GradCafe
    python tools/collect_data.py --source gradcafe --pages 10

    # 解析任意文本 (小红书帖子、论坛截图文字等)
    python tools/collect_data.py --source paste --input post.txt

    # 全部来源
    python tools/collect_data.py --source all

Requirements:
    pip install anthropic requests beautifulsoup4 pyyaml
    export ANTHROPIC_API_KEY=your_key_here
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# 确保能找到 tools/ 目录下的 parse_profile 里的 _make_client
sys.path.insert(0, str(Path(__file__).parent))
from parse_profile import _make_client  # reuse credential loader

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

OUTPUT_CSV = Path(__file__).parent.parent / "data" / "admissions" / "collected.csv"
SEEN_URLS_FILE = Path(__file__).parent.parent / "data" / "admissions" / ".seen_urls.txt"

CSV_FIELDS = [
    "id", "gender", "bg_type", "nationality", "gpa", "gpa_scale",
    "gre", "toefl", "major", "intern_desc", "has_paper", "has_research",
    "courses_note", "program", "result", "season", "source",
]

# QuantNet program name → our program ID
PROGRAM_MAP = {
    "baruch": "baruch-mfe",
    "baruch mfe": "baruch-mfe",
    "cmu": "cmu-mscf",
    "cmu mscf": "cmu-mscf",
    "carnegie mellon": "cmu-mscf",
    "mscf": "cmu-mscf",
    "princeton": "princeton-mfin",
    "princeton mfin": "princeton-mfin",
    "princeton mfe": "princeton-mfin",
    "berkeley": "berkeley-mfe",
    "berkeley mfe": "berkeley-mfe",
    "uc berkeley": "berkeley-mfe",
    "columbia msfe": "columbia-msfe",
    "columbia mfe": "columbia-msfe",
    "columbia mafn": "columbia-mafn",
    "columbia": "columbia-msfe",
    "mit": "mit-mfin",
    "mit mfin": "mit-mfin",
    "stanford": "stanford-mcf",
    "stanford mcf": "stanford-mcf",
    "uchicago": "uchicago-msfm",
    "chicago": "uchicago-msfm",
    "uchicago msfm": "uchicago-msfm",
    "cornell": "cornell-mfe",
    "cornell mfe": "cornell-mfe",
    "nyu courant": "nyu-courant",
    "nyu mfe": "nyu-tandon-mfe",
    "nyu tandon": "nyu-tandon-mfe",
    "gatech": "gatech-qcf",
    "georgia tech": "gatech-qcf",
    "qcf": "gatech-qcf",
    "rutgers": "rutgers-mqf",
    "ucla": "ucla-mfe",
    "uiuc": "uiuc-msfe",
    "northwestern": "northwestern-mfe",
    "jhu": "jhu-mfm",
    "johns hopkins": "jhu-mfm",
    "ncstate": "ncstate-mfm",
    "nc state": "ncstate-mfm",
    "fordham": "fordham-msqf",
    "uwash": "uwash-cfrm",
    "washington": "uwash-cfrm",
    "stevens": "stevens-mfe",
    "usc": "usc-msmf",
    "bu": "bu-msmf",
    "boston": "bu-msmf",
    "umich": "umich-mfe",
    "michigan": "umich-mfe",
    "umn": "uminn-mfm",
    "minnesota": "uminn-mfm",
    "utoronto": "utoronto-mmf",
    "toronto": "utoronto-mmf",
    "uncc": "uncc-msmf",
}

# QuantNet 搜索 MFE profile evaluation 帖子的关键词
QUANTNET_SEARCH_QUERIES = [
    "MFE profile evaluation",
    "MFE admission results",
    "MFE 2026 profile",
    "MFE 2025 admission",
    "MSCF profile review",
    "MFE accepted rejected",
]

GRADCAFE_QUERIES = [
    "financial+engineering",
    "computational+finance",
    "mathematical+finance",
    "quantitative+finance",
]

# ---------------------------------------------------------------------------
# Claude 解析提示词
# ---------------------------------------------------------------------------

_PARSE_SYSTEM = """你是一个MFE申请结果数据提取专家。
你的任务：从论坛帖子或社交媒体文本中提取结构化的申请结果数据。

输出格式：每个申请人对应一个JSON对象的列表。每个申请人可以有多个项目结果。

输出必须是严格的JSON数组，无任何额外说明，格式如下：
[
  {
    "gender": "M" 或 "F" 或 "unknown",
    "bg_type": "985" / "211" / "双非一本" / "海本(Top10)" / "海本(Top30)" / "海本(Top50)" / "IIT" / "其他",
    "nationality": "中国大陆" / "美籍" / "印度" / "港澳台" / "其他",
    "gpa": 数字 (如 3.85 或 91.5),
    "gpa_scale": 4 或 100,
    "gre": GRE总分数字 (如 331) 或 null,
    "toefl": TOEFL分数数字 或 null,
    "major": "主专业名称(中文或英文)",
    "intern_desc": "实习经历简述",
    "has_paper": "是" / "否" / "不明",
    "has_research": "是" / "否" / "不明",
    "courses_note": "特殊课程备注(随机微积分/实分析/C++等)",
    "season": "26Fall" 或 "25Fall" 或 "27Fall" 等,
    "source": "来源平台",
    "results": [
      {"program_raw": "项目原名", "program_id": "baruch-mfe", "result": "accepted/rejected/waitlisted/interview"},
      ...
    ]
  }
]

program_id 映射规则：
- Baruch MFE → baruch-mfe
- CMU MSCF / Carnegie Mellon → cmu-mscf
- Princeton MFin → princeton-mfin
- Berkeley MFE → berkeley-mfe
- Columbia MSFE/MFE → columbia-msfe, Columbia MAFN → columbia-mafn
- MIT MFin → mit-mfin
- Stanford MCF → stanford-mcf
- UChicago MSFM → uchicago-msfm
- Cornell MFE → cornell-mfe
- NYU Tandon MFE → nyu-tandon-mfe, NYU Courant → nyu-courant
- GaTech QCF → gatech-qcf
- 其他参考完整列表

如果文本中没有申请结果（只有profile展示，无录取/拒绝信息），返回空数组 []。
如果某字段信息缺失，填 null。
"""


def parse_text_with_claude(text: str, source: str = "manual") -> list[dict]:
    """用 Claude 解析任意文本 → 结构化申请结果列表"""
    client = _make_client()
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=_PARSE_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"来源：{source}\n\n以下是论坛帖子内容，请提取所有申请结果：\n\n{text[:6000]}"
        }]
    )
    import json
    raw = msg.content[0].text.strip()
    # 去除可能的 markdown 代码块
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [警告] Claude 返回了无效 JSON，跳过此条", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# 数据写入
# ---------------------------------------------------------------------------

def load_seen_urls() -> set[str]:
    if SEEN_URLS_FILE.exists():
        return set(SEEN_URLS_FILE.read_text().splitlines())
    return set()


def save_seen_url(url: str) -> None:
    SEEN_URLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SEEN_URLS_FILE.open("a") as f:
        f.write(url + "\n")


def get_next_id(output_path: Path) -> int:
    """读取现有 CSV，返回下一个可用 ID"""
    if not output_path.exists():
        return 1
    with output_path.open() as f:
        reader = csv.DictReader(f)
        ids = [int(row["id"]) for row in reader if row.get("id", "").isdigit()]
    return max(ids, default=0) + 1


def write_records(records: list[dict], output_path: Path, source_label: str) -> int:
    """将解析结果写入 CSV，返回写入的行数"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not output_path.exists()
    next_id = get_next_id(output_path)
    written = 0

    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        # 按 applicant 分组，每个 applicant 的每个 program 结果是一行
        for applicant in records:
            results = applicant.pop("results", [])
            if not results:
                continue
            applicant_id = next_id
            next_id += 1
            for r in results:
                program_id = r.get("program_id") or _fuzzy_map_program(r.get("program_raw", ""))
                if not program_id:
                    continue
                row = {
                    "id": applicant_id,
                    "gender": applicant.get("gender", "unknown"),
                    "bg_type": applicant.get("bg_type", ""),
                    "nationality": applicant.get("nationality", ""),
                    "gpa": applicant.get("gpa", ""),
                    "gpa_scale": applicant.get("gpa_scale", 4),
                    "gre": applicant.get("gre", ""),
                    "toefl": applicant.get("toefl", ""),
                    "major": applicant.get("major", ""),
                    "intern_desc": applicant.get("intern_desc", ""),
                    "has_paper": applicant.get("has_paper", "不明"),
                    "has_research": applicant.get("has_research", "不明"),
                    "courses_note": applicant.get("courses_note", ""),
                    "program": program_id,
                    "result": r.get("result", ""),
                    "season": applicant.get("season", ""),
                    "source": source_label,
                }
                writer.writerow(row)
                written += 1

    return written


def _fuzzy_map_program(name: str) -> Optional[str]:
    """模糊匹配项目名称 → program_id"""
    name_lower = name.lower().strip()
    # 精确匹配
    if name_lower in PROGRAM_MAP:
        return PROGRAM_MAP[name_lower]
    # 模糊匹配
    for key, val in PROGRAM_MAP.items():
        if key in name_lower:
            return val
    return None


# ---------------------------------------------------------------------------
# QuantNet 爬虫
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def fetch_quantnet_thread(url: str) -> str:
    """获取 QuantNet 帖子文本内容"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # 提取所有帖子内容
        posts = soup.find_all("article", class_="message")
        texts = []
        for post in posts[:10]:  # 只取前10楼（结果通常在前几楼）
            content = post.find("div", class_="bbWrapper")
            if content:
                texts.append(content.get_text(separator="\n", strip=True))
        return "\n\n---\n\n".join(texts)
    except Exception as e:
        print(f"  [错误] 获取 {url} 失败: {e}", file=sys.stderr)
        return ""


def search_quantnet_threads(query: str, max_pages: int = 3) -> list[str]:
    """搜索 QuantNet 帖子，返回 URL 列表"""
    urls = []
    for page in range(1, max_pages + 1):
        search_url = (
            f"https://quantnet.com/search/?q={query.replace(' ', '+')}"
            f"&t=thread&c[nodes][0]=3&o=date&page={page}"
        )
        try:
            resp = requests.get(search_url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.find_all("a", class_="contentRow-title")
            if not links:
                break
            for link in links:
                href = link.get("href", "")
                if href.startswith("/threads/"):
                    urls.append("https://quantnet.com" + href)
        except Exception as e:
            print(f"  [错误] 搜索页 {page} 失败: {e}", file=sys.stderr)
        time.sleep(1.5)  # 礼貌性延迟
    return list(dict.fromkeys(urls))  # 去重，保持顺序


def run_quantnet_scraper(max_pages: int = 3, output_path: Path = OUTPUT_CSV) -> int:
    """爬取 QuantNet 并写入 CSV，返回新增记录数"""
    seen = load_seen_urls()
    total_written = 0

    for query in QUANTNET_SEARCH_QUERIES:
        print(f"\n[QuantNet] 搜索: {query}")
        urls = search_quantnet_threads(query, max_pages)
        print(f"  找到 {len(urls)} 个帖子")

        for url in urls:
            if url in seen:
                continue

            print(f"  处理: {url}")
            text = fetch_quantnet_thread(url)
            if not text:
                continue

            records = parse_text_with_claude(text, source="quantnet")
            if records:
                n = write_records(records, output_path, source_label="quantnet")
                print(f"    → 写入 {n} 条记录")
                total_written += n

            save_seen_url(url)
            time.sleep(2)  # 避免频繁请求

    return total_written


# ---------------------------------------------------------------------------
# GradCafe 爬虫
# ---------------------------------------------------------------------------

def fetch_gradcafe_page(query: str, page: int = 1) -> str:
    """获取 GradCafe 一页结果的原始文本"""
    url = f"https://www.thegradcafe.com/survey/index.php?q={query}&t=a&o=&p={page}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if not table:
            return ""
        return table.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"  [错误] GradCafe 第 {page} 页失败: {e}", file=sys.stderr)
        return ""


def run_gradcafe_scraper(max_pages: int = 5, output_path: Path = OUTPUT_CSV) -> int:
    """爬取 GradCafe 并写入 CSV，返回新增记录数"""
    total_written = 0

    for query in GRADCAFE_QUERIES:
        print(f"\n[GradCafe] 搜索: {query}")
        for page in range(1, max_pages + 1):
            print(f"  第 {page} 页...")
            text = fetch_gradcafe_page(query, page)
            if not text or len(text) < 100:
                break

            records = parse_text_with_claude(text, source="gradcafe")
            if records:
                n = write_records(records, output_path, source_label="gradcafe")
                print(f"    → 写入 {n} 条记录")
                total_written += n

            time.sleep(1.5)

    return total_written


# ---------------------------------------------------------------------------
# 手动粘贴解析
# ---------------------------------------------------------------------------

def run_paste_parser(input_path: Optional[str], output_path: Path = OUTPUT_CSV) -> int:
    """解析手动输入的文本"""
    if input_path:
        text = Path(input_path).read_text(encoding="utf-8")
    else:
        print("请粘贴帖子内容（按 Ctrl+D 结束）：")
        text = sys.stdin.read()

    if not text.strip():
        print("错误：无输入内容", file=sys.stderr)
        return 0

    records = parse_text_with_claude(text, source="manual")
    if not records:
        print("未找到申请结果数据（文本中可能没有录取/拒绝信息）")
        return 0

    n = write_records(records, output_path, source_label="manual")
    print(f"成功写入 {n} 条记录")
    return n


# ---------------------------------------------------------------------------
# 统计报告
# ---------------------------------------------------------------------------

def print_stats(output_path: Path = OUTPUT_CSV) -> None:
    """打印数据集统计"""
    if not output_path.exists():
        print("数据文件不存在")
        return

    from collections import Counter
    programs: Counter = Counter()
    results: Counter = Counter()
    sources: Counter = Counter()
    applicant_ids: set = set()

    with output_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            programs[row.get("program", "")] += 1
            results[row.get("result", "")] += 1
            sources[row.get("source", "")] += 1
            applicant_ids.add(row.get("id", ""))

    print(f"\n{'='*50}")
    print(f"数据集统计  {output_path.name}")
    print(f"{'='*50}")
    print(f"申请人数:    {len(applicant_ids)}")
    print(f"记录总数:    {sum(programs.values())}")
    print(f"\n录取结果分布:")
    for k, v in results.most_common():
        print(f"  {k:15} {v}")
    print(f"\n前10项目:")
    for k, v in programs.most_common(10):
        print(f"  {k:20} {v}")
    print(f"\n数据来源:")
    for k, v in sources.most_common():
        print(f"  {k:15} {v}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QuantPath 数据收集器 — 爬取MFE申请结果数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", "-s",
        choices=["quantnet", "gradcafe", "paste", "all", "stats"],
        default="stats",
        help="数据来源 (默认: stats 仅显示统计)"
    )
    parser.add_argument("--pages", "-p", type=int, default=3, help="爬取页数 (默认: 3)")
    parser.add_argument("--input", "-i", help="输入文本文件路径 (用于 --source paste)")
    parser.add_argument(
        "--output", "-o",
        default=str(OUTPUT_CSV),
        help=f"输出 CSV 路径 (默认: {OUTPUT_CSV})"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    total = 0

    if args.source in ("quantnet", "all"):
        total += run_quantnet_scraper(args.pages, output_path)

    if args.source in ("gradcafe", "all"):
        total += run_gradcafe_scraper(args.pages, output_path)

    if args.source == "paste":
        total += run_paste_parser(args.input, output_path)

    if total > 0:
        print(f"\n本次共写入 {total} 条记录")

    print_stats(output_path)


if __name__ == "__main__":
    main()
