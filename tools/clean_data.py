#!/usr/bin/env python3
"""
数据清洗脚本：去重 + 修复污染值

用法:
    python tools/clean_data.py
    python tools/clean_data.py --dry-run   # 只预览，不写入
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent / "data" / "admissions"
INPUT_CSV = DATA_DIR / "collected.csv"
OUTPUT_CSV = DATA_DIR / "collected.csv"  # 原地覆盖

VALID_RESULTS = {"accepted", "rejected", "waitlisted", "interview", "pending"}
CSV_FIELDS = [
    "id", "gender", "bg_type", "nationality",
    "gpa", "gpa_scale", "gre", "toefl",
    "major", "intern_desc", "has_paper", "has_research", "courses_note",
    "program", "result", "season", "source",
]


def dedup_key(row: dict, source: str) -> tuple:
    """生成去重键，不同来源策略不同"""
    if source == "gradcafe":
        # GradCafe：同一program+result+gpa+gre视为重复（多个搜索词造成）
        return ("gradcafe", row["program"], row["result"], row["gpa"], row["gre"])
    elif source == "quantnet_tracker":
        # QuantNet：program+result+gpa+gpa_scale+gre+toefl+season
        return (
            "quantnet_tracker",
            row["program"], row["result"],
            row["gpa"], row["gpa_scale"],
            row["gre"], row["toefl"],
            row["season"],
        )
    else:
        # 手动录入/小红书等：保留所有（量少且各不相同）
        return (source, id(row))  # 永不重复


def clean(dry_run: bool = False) -> None:
    if not INPUT_CSV.exists():
        print(f"文件不存在: {INPUT_CSV}")
        return

    with INPUT_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"读入 {len(rows)} 条记录")

    cleaned = []
    seen_keys: set = set()
    stats = Counter()

    for row in rows:
        source = row.get("source", "")
        result = row.get("result", "")

        # 修复污染的result字段（值是program_id）
        if result not in VALID_RESULTS:
            stats["污染值删除"] += 1
            continue

        # 去重
        key = dedup_key(row, source)
        if key in seen_keys:
            stats["重复删除"] += 1
            continue
        seen_keys.add(key)

        cleaned.append(row)
        stats["保留"] += 1

    print(f"\n清洗结果:")
    print(f"  污染值删除: {stats['污染值删除']}")
    print(f"  重复删除:   {stats['重复删除']}")
    print(f"  保留记录:   {stats['保留']}")

    # 按来源统计
    src_counts = Counter(r["source"] for r in cleaned)
    print(f"\n来源分布:")
    for src, cnt in src_counts.most_common():
        print(f"  {src}: {cnt}")

    # 按项目统计
    prog_counts = Counter(r["program"] for r in cleaned)
    print(f"\n各项目记录数(Top15):")
    for prog, cnt in prog_counts.most_common(15):
        print(f"  {prog}: {cnt}")

    # 结果分布
    result_counts = Counter(r["result"] for r in cleaned)
    print(f"\n结果分布:")
    for res, cnt in result_counts.most_common():
        print(f"  {res}: {cnt}")

    if dry_run:
        print("\n[dry-run] 未写入文件")
        return

    # 重新生成连续ID
    id_counter: dict[str, int] = {}
    for row in cleaned:
        src_key = row.get("id") or ""
        if src_key not in id_counter:
            id_counter[src_key] = len(id_counter) + 1
        row["id"] = str(id_counter[src_key])

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in cleaned:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})

    print(f"\n已写入 {len(cleaned)} 条记录 → {OUTPUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MFE申请数据清洗工具")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不写入")
    args = parser.parse_args()
    clean(dry_run=args.dry_run)
