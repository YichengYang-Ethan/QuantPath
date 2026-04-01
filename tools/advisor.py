#!/usr/bin/env python3
# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""AI-powered MFE application strategy advisor.

Runs the complete QuantPath evaluation pipeline, assembles a rich context,
then streams an expert advisory report from Claude covering:
  - Competitive assessment
  - Critical profile gaps
  - Recommended school list with rationale
  - Immediate action plan
  - SOP strategy and differentiators

Usage:
    python tools/advisor.py --profile my_profile.yaml
    python tools/advisor.py --profile my_profile.yaml --save report.md
    python tools/advisor.py --profile my_profile.yaml --focus gaps,sop

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=your_key_here
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402


def _make_client() -> anthropic.Anthropic:
    """Return an Anthropic client, auto-detecting API key or OAuth token.

    Search order:
    1. ANTHROPIC_API_KEY env var
    2. ANTHROPIC_OAUTH_TOKEN env var
    3. ~/.clawdbot/.env file (Clawdbot installation)
    """
    def _load_env_file(path: str) -> dict[str, str]:
        vals: dict[str, str] = {}
        try:
            for line in Path(path).read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    vals[k.strip()] = v.strip()
        except FileNotFoundError:
            pass
        return vals

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return anthropic.Anthropic(api_key=api_key)

    oauth_token = os.environ.get("ANTHROPIC_OAUTH_TOKEN")
    if not oauth_token:
        env = _load_env_file(os.path.expanduser("~/.clawdbot/.env"))
        api_key = env.get("ANTHROPIC_API_KEY")
        oauth_token = env.get("ANTHROPIC_OAUTH_TOKEN")

    if api_key:
        return anthropic.Anthropic(api_key=api_key)
    if oauth_token:
        return anthropic.Anthropic(auth_token=oauth_token)

    raise RuntimeError(
        "No Anthropic credentials found. Set ANTHROPIC_API_KEY or ANTHROPIC_OAUTH_TOKEN."
    )

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a senior MFE admissions advisor with 15+ years of experience at top programs (Baruch, Princeton, CMU, Columbia, MIT, Berkeley). You have reviewed thousands of applications and know exactly what separates admits from rejects.

Your advisory style:
- Honest and direct — you do not sugarcoat weaknesses
- Data-driven — you reference specific scores and statistics
- Strategic — every observation connects to an action
- Personalized — you speak to this applicant's specific situation, not generic advice

You will receive a complete QuantPath quantitative analysis of an applicant. Use this data to produce a comprehensive strategy report.

IMPORTANT CONTEXT ABOUT QUANTPATH SCORES:
- Scores are on a 0-10 scale
- Score >= 9.0 = exceptional (top-tier competitive)
- Score 7.0-8.9 = strong (competitive at most programs)
- Score 5.0-6.9 = adequate (borderline for top programs, fine for mid-tier)
- Score < 5.0 = gap (will likely be flagged by admissions committees)
- Overall score >= 8.0 = competitive at top-15 programs
- Overall score 6.5-7.9 = competitive at mid-tier programs
- Overall score < 6.5 = significant work needed

IMPORTANT CONTEXT ABOUT INTERNATIONAL APPLICANTS:
- F1 visa holders face higher bars at some programs (MIT, Princeton are known to be balanced; CMU, Columbia tend international-friendly)
- US work experience (even internship) dramatically helps F1 applicants
- STEM designation matters for OPT extension (3 years vs 1 year) — critical for F1 job search

Format your response in clean markdown with these sections:

## Executive Summary
2-3 sentences: overall competitiveness and the single most important thing this applicant needs to know.

## Profile Scorecard
A brief analysis of each dimension score — what it means for their applications.

## Top 3 Strengths
What will make admissions committees say yes. Be specific.

## Critical Gaps (Priority-Ordered)
What could get them rejected. Prioritize ruthlessly — no more than 4-5 items. For each gap, state: what it is, why it matters, and exactly how to fix it.

## Recommended School List
Based on the data, your recommended list with:
- Reach (2-3 schools): stretch but achievable
- Target (3-4 schools): strong match
- Safety (2 schools): high probability of admission
For each school, give a 1-sentence rationale.

## Immediate Action Plan (Next 90 Days)
Numbered, prioritized list. Be specific — not "improve CS skills" but "take CMU's free Computational Finance MOOC on Coursera and add it to your resume."

## SOP Strategy
3-4 key themes to build the Statement of Purpose around, based on their specific profile. Include what narrative arc will be most compelling given their background.

## Red Flags to Address
Anything in their profile that needs proactive explanation in the application (gaps, inconsistencies, weak scores in specific areas).
"""

# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def _build_context(profile_path: str) -> str:
    """Run the full QuantPath pipeline and return a structured context string."""
    from core.course_optimizer import optimize_courses
    from core.data_loader import load_all_programs, load_profile
    from core.gap_advisor import analyze_gaps
    from core.list_builder import build_school_list
    from core.prerequisite_matcher import match_prerequisites
    from core.profile_evaluator import evaluate as evaluate_profile
    from core.roi_calculator import calculate_roi
    from core.timeline_generator import generate_timeline

    profile = load_profile(profile_path)
    programs = load_all_programs()

    # Core pipeline
    result = evaluate_profile(profile)
    gaps = analyze_gaps(result.gaps) if result.gaps else []
    course_recs = optimize_courses(profile, programs, max_courses=6)
    school_list = build_school_list(profile, programs, result)
    roi_results = calculate_roi(programs)
    timeline_events = generate_timeline(programs)

    # Prerequisite match for top programs
    top_program_ids = ["baruch-mfe", "cmu-mscf", "columbia-msfe", "princeton-mfin", "mit-mfin",
                       "berkeley-mfe", "uchicago-msfm", "cornell-mfe", "nyu-courant"]
    prereq_matches = {}
    for pid in top_program_ids:
        prog = next((p for p in programs if p.id == pid), None)
        if prog:
            match = match_prerequisites(profile, prog)
            prereq_matches[pid] = match

    lines: list[str] = []

    # ---- Applicant profile ----
    lines.append("=== APPLICANT PROFILE ===")
    lines.append(f"Name        : {profile.name}")
    lines.append(f"University  : {profile.university}")
    lines.append(f"Majors      : {', '.join(profile.majors)}")
    lines.append(f"Overall GPA : {profile.gpa}")
    if profile.gpa_quant:
        lines.append(f"Quant GPA   : {profile.gpa_quant}")
    lines.append(f"International: {'Yes (F1/student visa)' if profile.is_international else 'No (domestic)'}")
    lines.append(f"Years at US institution: {profile.years_at_us_institution}")

    ts = profile.test_scores
    if ts.gre_quant:
        lines.append(f"GRE Quant   : {ts.gre_quant}/170")
    if ts.gre_verbal:
        lines.append(f"GRE Verbal  : {ts.gre_verbal}/170")
    if ts.toefl:
        lines.append(f"TOEFL iBT   : {ts.toefl}")

    if profile.coursework:
        lines.append(f"\nCoursework ({len(profile.coursework)} courses):")
        for c in sorted(profile.coursework, key=lambda x: x.level, reverse=True):
            lines.append(f"  {c.code:12s} {c.name:40s} [{c.category:22s}] {c.grade} (level {c.level})")

    if profile.work_experience:
        lines.append(f"\nWork Experience ({len(profile.work_experience)} entries):")
        for exp in profile.work_experience:
            lines.append(
                f"  [{exp.get('type','internship').upper()}] {exp.get('title','')} "
                f"at {exp.get('company','')} — {exp.get('duration_months','?')} months"
            )
            if exp.get("description"):
                lines.append(f"    {exp['description']}")

    if profile.projects:
        lines.append(f"\nProjects ({len(profile.projects)}):")
        for proj in profile.projects:
            lines.append(f"  {proj.get('name','')}: {proj.get('description','')}")
            for h in proj.get("highlights", []):
                lines.append(f"    - {h}")

    # ---- Evaluation scores ----
    lines.append("\n=== QUANTPATH EVALUATION SCORES ===")
    lines.append(f"Overall Score: {result.overall_score:.2f} / 10.0")
    lines.append("")
    dim_labels = {
        "math": "Mathematics    (30% weight)",
        "statistics": "Statistics     (20% weight)",
        "cs": "Computer Sci   (20% weight)",
        "finance": "Finance/Econ   (15% weight)",
        "gpa": "GPA            (15% weight)",
    }
    for dim, score in result.dimension_scores.items():
        label = dim_labels.get(dim, dim)
        bar = "█" * round(score) + "░" * (10 - round(score))
        lines.append(f"  {label}: {score:.2f}/10  {bar}")

    if result.strengths:
        lines.append("\nStrengths (score >= 9.0):")
        for s in result.strengths:
            lines.append(f"  + {s['factor']:30s} ({s['dimension']}) = {s['score']:.1f}")

    # ---- Gaps ----
    if gaps:
        lines.append(f"\n=== IDENTIFIED GAPS ({len(gaps)} total) ===")
        for g in gaps:
            priority_marker = {"High": "!!!", "Medium": "!! ", "Low": "!  "}.get(g.priority, "   ")
            lines.append(f"  [{priority_marker}] {g.factor:30s} ({g.dimension:12s}) score={g.score:.1f}")
            lines.append(f"       Action: {g.action}")

    # ---- Course recommendations ----
    if course_recs:
        lines.append("\n=== COURSE OPTIMIZATION RECOMMENDATIONS ===")
        for i, rec in enumerate(course_recs, 1):
            lines.append(
                f"  {i}. {rec.category:25s} impact={rec.impact_score:.3f}"
                + (f", prereq_coverage={rec.prereq_coverage}" if rec.prereq_coverage else "")
            )
            lines.append(f"     {rec.reason}")

    # ---- Prerequisite matches for top programs ----
    if prereq_matches:
        lines.append("\n=== PREREQUISITE MATCH — TOP PROGRAMS ===")
        for pid, match in sorted(prereq_matches.items(), key=lambda x: -x[1].match_score):
            prog_name = next((p.name for p in programs if p.id == pid), pid)
            bar = "█" * round(match.match_score * 10) + "░" * (10 - round(match.match_score * 10))
            lines.append(f"  {prog_name:20s}: {match.match_score*100:.0f}%  {bar}")
            if match.missing:
                missing_cats = [m.get("category", "") for m in match.missing]
                lines.append(f"    Missing: {', '.join(missing_cats)}")
            if match.warnings:
                for w in match.warnings[:2]:
                    lines.append(f"    Warning: {w}")

    # ---- School list ----
    lines.append("\n=== RECOMMENDED SCHOOL LIST (QuantPath) ===")
    for tier in ["reach", "target", "safety"]:
        entries = getattr(school_list, tier, [])
        if entries:
            lines.append(f"\n{tier.upper()} ({len(entries)} schools):")
            for entry in entries:
                fit = getattr(entry, "fit_score", 0) or 0
                reason = getattr(entry, "reason", "")
                prob = getattr(entry, "admission_prob", None)
                prob_low = getattr(entry, "prob_low", None)
                prob_high = getattr(entry, "prob_high", None)

                if prob is not None:
                    ci_str = (
                        f" [{prob_low:.0%}–{prob_high:.0%}]"
                        if prob_low is not None and prob_high is not None
                        else ""
                    )
                    prob_str = f"  P(admit)={prob:.0%}{ci_str}"
                else:
                    prob_str = "  P(admit)=N/A"

                lines.append(f"  {entry.name:25s}  fit={fit:.0f}/100{prob_str}")
                if reason:
                    lines.append(f"    → {reason}")

    # ---- ROI ----
    if roi_results:
        lines.append("\n=== ROI ANALYSIS — TOP 8 BY NPV ===")
        lines.append(f"  {'Program':25s} {'Salary':>10s} {'Tuition':>10s} {'NPV 5yr':>12s} {'Payback':>8s}")
        for r in roi_results[:8]:
            lines.append(
                f"  {r.program_name:25s} ${r.avg_salary:>8,}  ${r.tuition:>8,}  ${r.npv_5yr:>10,.0f}  {r.payback_years:.1f}yr"
            )

    # ---- Timeline (upcoming) ----
    if timeline_events:
        upcoming = [e for e in timeline_events if e.get("priority") in ("critical", "high")][:12]
        if upcoming:
            lines.append("\n=== CRITICAL / HIGH PRIORITY TIMELINE ITEMS ===")
            for e in upcoming:
                lines.append(
                    f"  [{e.get('priority','').upper():8s}] {e.get('date',''):12s} "
                    f"[{e.get('category',''):15s}] {e.get('action','')}"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_advisor(profile_path: str, save_path: str | None = None) -> None:
    """Run the full analysis and stream Claude's advisory report."""
    print(f"Running QuantPath pipeline on: {profile_path}")
    print("─" * 60)

    try:
        context = _build_context(profile_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        raise

    print("Generating AI advisory report...\n")
    print("═" * 60)

    client = _make_client()
    full_response: list[str] = []

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Here is the complete QuantPath analysis. Please provide your expert advisory report.\n\n"
                    f"{context}"
                ),
            }
        ],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response.append(text)

    print("\n" + "═" * 60)

    if save_path:
        report_text = "".join(full_response)
        Path(save_path).write_text(report_text, encoding="utf-8")
        print(f"\nReport saved to: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an AI-powered MFE application strategy report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/advisor.py --profile my_profile.yaml
  python tools/advisor.py --profile my_profile.yaml --save strategy_report.md
        """,
    )
    parser.add_argument("--profile", "-p", required=True, help="Path to profile YAML file")
    parser.add_argument("--save", "-s", help="Save report to a markdown file")
    args = parser.parse_args()

    run_advisor(args.profile, save_path=args.save)


if __name__ == "__main__":
    main()
