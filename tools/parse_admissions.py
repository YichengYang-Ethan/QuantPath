#!/usr/bin/env python3
"""
MFE 申请结果批量解析器

专门处理中文来源（小红书、chasedream、一亩三分地、offershow）
和英文来源（QuantNet、Reddit）的MFE申请帖子文本。

用法:
    # 解析单个文本文件
    python tools/parse_admissions.py --input post.txt --season 26Fall --source 小红书

    # 从 stdin 粘贴
    python tools/parse_admissions.py --season 26Fall --source chasedream

    # 解析整个目录的 .txt 文件
    python tools/parse_admissions.py --dir posts/ --season 26Fall --source 小红书

    # 仅验证已有 CSV 数据
    python tools/parse_admissions.py --validate
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_profile import _make_client
from collect_data import (
    CSV_FIELDS, OUTPUT_CSV, write_records, print_stats, PROGRAM_MAP
)

# ---------------------------------------------------------------------------
# 解析提示词 — 针对中文论坛优化
# ---------------------------------------------------------------------------

_SYSTEM_ZH = """你是MFE（金融工程硕士）申请结果数据提取专家，擅长从中文论坛帖子中提取结构化数据。

## 任务
从输入文本中识别并提取所有MFE申请结果。

## 输出格式
严格输出JSON数组，每个元素代表一位申请人：

```json
[
  {
    "gender": "M/F/unknown",
    "bg_type": "985/211/双非一本/海本(Top10)/海本(Top30)/海本(Top50)/IIT/其他",
    "nationality": "中国大陆/美籍/印度/港澳台/其他",
    "gpa": 3.85,
    "gpa_scale": 4,
    "gre": 331,
    "toefl": 110,
    "major": "金工",
    "intern_desc": "2段量化私募+1段投行",
    "has_paper": "是/否/不明",
    "has_research": "是/否/不明",
    "courses_note": "随机微积分+实分析+C++",
    "season": "26Fall",
    "results": [
      {"program_raw": "Baruch MFE", "program_id": "baruch-mfe", "result": "accepted"},
      {"program_raw": "CMU MSCF", "program_id": "cmu-mscf", "result": "rejected"}
    ]
  }
]
```

## 字段说明
- bg_type: 学校背景类型。985/211用于国内院校；海本(Top10/30/50)用于海外院校
- gpa_scale: 4分制填4，百分制填100，印度10分制填10
- gre: 填GRE总分(Quant+Verbal合计，如331)，若只有Quant填Quant分数(如170)
- result 只能是: accepted / rejected / waitlisted / interview / pending
- program_id 映射：Baruch→baruch-mfe, CMU/MSCF→cmu-mscf, Princeton→princeton-mfin,
  Berkeley→berkeley-mfe, Columbia MSFE→columbia-msfe, Columbia MAFN→columbia-mafn,
  MIT→mit-mfin, Stanford→stanford-mcf, UChicago/芝大→uchicago-msfm,
  Cornell→cornell-mfe, NYU Tandon→nyu-tandon-mfe, NYU Courant→nyu-courant,
  GaTech/Georgia Tech→gatech-qcf, Rutgers→rutgers-mqf, UCLA→ucla-mfe,
  UIUC→uiuc-msfe

## 重要规则
1. 如果帖子只是问"我能申请哪些学校？"而没有实际录取结果，返回 []
2. 一个帖子可能包含多位申请人的结果（如汇总帖），全部提取
3. 缺失字段填 null，不要猜测
4. 中文常见缩写：硬背/硬件=GPA+GRE等硬性条件；软背=SOP/推荐信；拿到=accepted；被拒=rejected；WL=waitlisted
5. 国内院校分类：C9+部分985顶尖院校→985；其余211→211；其他普通本科→双非一本
"""


def parse_post(text: str, season: str, source: str) -> list[dict]:
    """解析单个帖子文本"""
    client = _make_client()
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=_SYSTEM_ZH,
        messages=[{
            "role": "user",
            "content": (
                f"来源平台：{source}\n"
                f"申请季：{season}\n\n"
                f"帖子内容：\n{text[:6000]}"
            )
        }]
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        records = json.loads(raw)
        # 注入 season
        for r in records:
            if not r.get("season"):
                r["season"] = season
        return records
    except json.JSONDecodeError:
        print(f"[警告] JSON解析失败，跳过", file=sys.stderr)
        return []


def validate_csv(csv_path: Path = OUTPUT_CSV) -> None:
    """验证 CSV 数据质量"""
    if not csv_path.exists():
        print(f"文件不存在: {csv_path}")
        return

    issues = []
    valid_results = {"accepted", "rejected", "waitlisted", "interview", "pending"}
    valid_programs = set(PROGRAM_MAP.values())

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            if row.get("program") not in valid_programs:
                issues.append(f"行{i}: 未知项目 '{row.get('program')}'")
            if row.get("result") not in valid_results:
                issues.append(f"行{i}: 未知结果 '{row.get('result')}'")
            try:
                gpa = float(row.get("gpa", 0))
                scale = float(row.get("gpa_scale", 4))
                if scale == 4 and not (0 <= gpa <= 4.3):
                    issues.append(f"行{i}: GPA异常 {gpa} (4分制)")
                if scale == 100 and not (0 <= gpa <= 100):
                    issues.append(f"行{i}: GPA异常 {gpa} (百分制)")
            except (ValueError, TypeError):
                pass

    if issues:
        print(f"\n发现 {len(issues)} 个数据问题:")
        for issue in issues[:20]:
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... 还有 {len(issues)-20} 个问题")
    else:
        print("✓ 数据验证通过，无问题")


def merge_with_existing(output_path: Path = OUTPUT_CSV) -> None:
    """将 sample.csv 的数据合并进 collected.csv（如果还没合并过）"""
    sample_path = output_path.parent / "sample.csv"
    if not sample_path.exists():
        return
    if output_path.exists():
        # 检查是否已经合并
        with output_path.open() as f:
            content = f.read()
        if "小红书" in content:
            return  # 已合并

    print(f"合并 {sample_path.name} → {output_path.name} ...")
    rows = []
    with sample_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return

    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    print(f"  合并了 {len(rows)} 条历史数据")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MFE 申请结果批量解析器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", help="输入文本文件路径")
    parser.add_argument("--dir", "-d", help="包含多个 .txt 文件的目录")
    parser.add_argument("--season", default="26Fall", help="申请季，如 26Fall / 25Fall")
    parser.add_argument("--source", default="manual", help="来源平台名称")
    parser.add_argument("--output", "-o", default=str(OUTPUT_CSV), help="输出 CSV 路径")
    parser.add_argument("--validate", action="store_true", help="验证现有 CSV 数据质量")
    parser.add_argument("--merge-sample", action="store_true", help="将 sample.csv 合并进 collected.csv")
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.merge_sample:
        merge_with_existing(output_path)
        print_stats(output_path)
        return

    if args.validate:
        validate_csv(output_path)
        print_stats(output_path)
        return

    texts: list[tuple[str, str]] = []  # (text, filename)

    if args.dir:
        dir_path = Path(args.dir)
        for txt_file in sorted(dir_path.glob("*.txt")):
            texts.append((txt_file.read_text(encoding="utf-8"), txt_file.stem))

    elif args.input:
        texts.append((Path(args.input).read_text(encoding="utf-8"), args.input))

    else:
        print("请粘贴帖子内容（按 Ctrl+D 结束）：")
        text = sys.stdin.read()
        if text.strip():
            texts.append((text, "stdin"))

    if not texts:
        print("错误：无输入内容", file=sys.stderr)
        sys.exit(1)

    total = 0
    for text, fname in texts:
        print(f"\n解析: {fname}")
        records = parse_post(text, args.season, args.source)
        if not records:
            print("  未提取到结果")
            continue
        n = write_records(records, output_path, source_label=args.source)
        print(f"  写入 {n} 条记录")
        total += n

    print(f"\n本次共写入 {total} 条记录")
    print_stats(output_path)


if __name__ == "__main__":
    main()
