#!/usr/bin/env python3
# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""
用历史数据训练每个MFE项目的录取预测模型（逻辑回归）
输出到 data/models/ 目录（YAML格式，轻量，不依赖pickle）

用法:
    python tools/train_model.py
    python tools/train_model.py --min-samples 30  # 最少样本量（默认30）
    python tools/train_model.py --eval           # 同时输出评估报告
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
CSV_PATH = DATA_DIR / "admissions" / "collected.csv"
MODEL_DIR = DATA_DIR / "models"
PROGRAMS_DIR = DATA_DIR / "programs"


def load_real_accept_rates() -> dict[str, float]:
    """Load real (official) acceptance rates from program YAML files."""
    import yaml  # optional dep

    rates: dict[str, float] = {}
    if not PROGRAMS_DIR.exists():
        return rates
    for yml_path in PROGRAMS_DIR.glob("*.yaml"):
        try:
            with yml_path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
            prog_id = data.get("id") or yml_path.stem
            admissions = data.get("admissions") or {}
            rate = admissions.get("acceptance_rate")
            if rate is not None:
                rates[prog_id] = float(rate)
        except Exception:
            pass
    return rates


def load_data(csv_path: Path) -> dict[str, list[dict]]:
    """按项目分组读取数据"""
    programs: dict[str, list[dict]] = defaultdict(list)
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prog = row.get("program", "").strip()
            result = row.get("result", "").strip()
            if not prog or result not in {"accepted", "rejected"}:
                continue

            # 解析GPA（标准化到4分制）
            gpa = _parse_float(row.get("gpa"))
            gpa_scale = _parse_float(row.get("gpa_scale")) or 4.0
            if gpa is not None and gpa_scale:
                if gpa_scale == 100:
                    gpa = gpa / 100 * 4.0
                elif gpa_scale == 10:
                    gpa = gpa / 10 * 4.0

            # 解析GRE（Quant分）
            gre = _parse_float(row.get("gre"))
            if gre and gre > 170:
                gre = None  # 总分制 → 无法区分Quant，丢弃

            toefl = _parse_float(row.get("toefl"))

            record = {
                "gpa": gpa,
                "gre": gre,
                "toefl": toefl,
                "label": 1 if result == "accepted" else 0,
            }
            programs[prog].append(record)

    return dict(programs)


def _parse_float(val) -> float | None:
    try:
        v = float(str(val).strip())
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def build_feature_matrix(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """将记录列表转成特征矩阵，只用gpa/gre两列（覆盖面最广）"""
    X_rows = []
    y = []
    for r in records:
        gpa = r.get("gpa")
        gre = r.get("gre")
        if gpa is None and gre is None:
            continue  # 两个都没有，跳过
        # 缺失值用中位数填充（在下面处理）
        X_rows.append([gpa, gre])
        y.append(r["label"])

    if not X_rows:
        return np.array([]), np.array([])

    X = np.array(X_rows, dtype=float)
    y = np.array(y)

    # 列均值填充缺失
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_means[j]

    return X, y


def train_program(records: list[dict], min_samples: int = 30) -> dict | None:
    """训练单个项目模型，返回参数字典（YAML序列化）"""
    X, y = build_feature_matrix(records)
    if len(y) < min_samples:
        return None

    n_pos = int(y.sum())
    n_neg = int(len(y) - y.sum())
    if n_pos < 5 or n_neg < 5:
        return None

    # 用标准差统计（用于推理时标准化）
    col_stds = np.nanstd(X, axis=0)
    col_stds[col_stds == 0] = 1.0

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
    ])
    pipe.fit(X, y)

    # 5折交叉验证 AUC
    cv_scores = cross_val_score(pipe, X, y, cv=min(5, n_pos, n_neg), scoring="roc_auc")
    auc = float(np.mean(cv_scores))

    # 提取系数用于后续推理（线性模型直接保存）
    scaler = pipe.named_steps["scaler"]
    lr = pipe.named_steps["lr"]

    coef = lr.coef_[0]  # [gpa_coef, gre_coef]
    intercept = float(lr.intercept_[0])

    # 标准化参数（推理时：z = (x - mean) / std; logit = coef @ z + intercept）
    return {
        "n_total": len(y),
        "n_accepted": n_pos,
        "n_rejected": n_neg,
        "accept_rate": round(n_pos / len(y), 4),
        "auc": round(auc, 4),
        "features": ["gpa_4scale", "gre_quant"],
        "means": [round(float(scaler.mean_[0]), 4), round(float(scaler.mean_[1]), 4)],
        "stds": [round(float(scaler.scale_[0]), 4), round(float(scaler.scale_[1]), 4)],
        "coef": [round(float(coef[0]), 6), round(float(coef[1]), 6)],
        "intercept": round(intercept, 6),
        # 便捷分位数（用于展示"申请者平均水平"）
        "gpa_p25": round(float(np.nanpercentile(X[:, 0], 25)), 2),
        "gpa_p50": round(float(np.nanpercentile(X[:, 0], 50)), 2),
        "gpa_p75": round(float(np.nanpercentile(X[:, 0], 75)), 2),
        "gre_p25": round(float(np.nanpercentile(X[:, 1], 25)), 0),
        "gre_p50": round(float(np.nanpercentile(X[:, 1], 50)), 0),
        "gre_p75": round(float(np.nanpercentile(X[:, 1], 75)), 0),
    }


def predict_admission(program_id: str, gpa: float, gre: float, models: dict) -> float | None:
    """用保存的模型参数计算P(录取)"""
    m = models.get(program_id)
    if not m:
        return None
    means = m["means"]
    stds = m["stds"]
    coef = m["coef"]
    intercept = m["intercept"]

    # 标准化
    z_gpa = (gpa - means[0]) / stds[0]
    z_gre = (gre - means[1]) / stds[1]

    logit = coef[0] * z_gpa + coef[1] * z_gre + intercept
    prob = 1.0 / (1.0 + np.exp(-logit))
    return round(float(prob), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练MFE录取预测模型")
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--eval", action="store_true", help="输出详细评估报告")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"读取数据: {CSV_PATH}")
    program_data = load_data(CSV_PATH)
    print(f"发现 {len(program_data)} 个项目")

    models: dict[str, dict] = {}
    skipped = []

    for prog, records in sorted(program_data.items()):
        result = train_program(records, min_samples=args.min_samples)
        if result is None:
            skipped.append((prog, len(records)))
            continue
        models[prog] = result

    # 注入真实接受率（用于预测时的偏差校正）
    real_rates = load_real_accept_rates()
    for prog, m in models.items():
        if prog in real_rates:
            m["real_accept_rate"] = round(real_rates[prog], 4)

    # 保存为JSON（所有项目）
    output_path = MODEL_DIR / "admission_models.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(models, f, indent=2, ensure_ascii=False)

    print(f"\n训练完成: {len(models)} 个项目 → {output_path}")
    print(f"跳过(样本不足): {len(skipped)} 个")

    # 打印训练结果摘要（含偏差校正列）
    print(f"\n{'项目':<22} {'样本':>6} {'数据率':>8} {'真实率':>8} {'AUC':>7} {'GPA中位数':>10}")
    print("-" * 72)
    for prog, m in sorted(models.items(), key=lambda x: -x[1]["n_total"]):
        real = m.get("real_accept_rate")
        real_str = f"{real:.1%}" if real is not None else "  N/A"
        print(
            f"{prog:<22} {m['n_total']:>6} {m['accept_rate']:>8.1%} "
            f"{real_str:>8} {m['auc']:>7.3f} {m['gpa_p50']:>10.2f}"
        )

    if skipped:
        print("\n跳过项目:")
        for prog, n in skipped:
            print(f"  {prog}: {n}条")

    if args.eval:
        print("\n评估：Ethan Yang的预测概率（GPA=4.0, GRE=170）")
        print(f"{'项目':<22} {'P(录取)':>10}")
        print("-" * 35)
        for prog in sorted(models.keys()):
            prob = predict_admission(prog, gpa=4.0, gre=170, models=models)
            print(f"{prog:<22} {prob:>10.1%}")


if __name__ == "__main__":
    main()
