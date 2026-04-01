#!/usr/bin/env python3
# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""Parse unstructured resume / transcript text into a QuantPath profile YAML.

Uses Claude API to intelligently extract academic and professional information
and map it to the QuantPath data model.

Usage:
    # From a text file
    python tools/parse_profile.py --input resume.txt --output my_profile.yaml

    # From stdin (paste text, press Ctrl+D when done)
    python tools/parse_profile.py --output my_profile.yaml

    # Combine resume + transcript
    cat resume.txt transcript.txt | python tools/parse_profile.py --output my_profile.yaml

Requirements:
    pip install anthropic pyyaml
    export ANTHROPIC_API_KEY=your_key_here
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import os

import anthropic
import yaml


def _make_client() -> anthropic.Anthropic:
    """Return an Anthropic client, auto-detecting API key or OAuth token."""
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
# Course category reference — mirrors core/models.py CourseCategory enum
# ---------------------------------------------------------------------------

_COURSE_CATEGORIES = """
MATHEMATICS:
  calculus            - Calculus I/II/III, Multivariable Calculus, Vector Calculus
  linear_algebra      - Linear Algebra, Matrix Theory, Matrix Computations
  probability         - Probability Theory, Calculus-based Probability, Stochastic Modeling
  ode                 - Ordinary Differential Equations, Differential Equations
  pde                 - Partial Differential Equations
  real_analysis       - Real Analysis, Advanced Calculus, Measure Theory
  numerical_analysis  - Numerical Methods, Numerical Analysis, Scientific Computing
  stochastic_processes - Stochastic Processes, Markov Chains, Brownian Motion
  stochastic_calculus - Ito Calculus, Stochastic Calculus, SDEs, Stochastic Differential Equations
  optimization        - Convex Optimization, Linear Programming, Nonlinear Optimization

STATISTICS & DATA SCIENCE:
  statistics          - Mathematical Statistics, Statistical Inference, Probability & Statistics
  regression          - Regression Analysis, Applied Statistics, Linear Models
  econometrics        - Econometrics, Applied Econometrics, Causal Inference
  time_series         - Time Series Analysis, Forecasting, ARIMA
  stat_computing      - Statistical Computing, R Programming for Stats, Data Analysis in Python/R
  stat_learning       - Statistical Learning, High-Dimensional Statistics, ML from stats perspective
  bayesian            - Bayesian Statistics, Bayesian Inference, MCMC

COMPUTER SCIENCE:
  programming_cpp     - C/C++ Programming, Systems Programming
  programming_python  - Python Programming, Scientific Python
  programming_r       - R Programming (standalone course)
  data_structures     - Data Structures, Data Structures & Algorithms
  algorithms          - Algorithms, Algorithm Design & Analysis
  machine_learning    - Machine Learning, Deep Learning, Neural Networks, AI
  database            - Database Systems, SQL, Data Engineering
  software_engineering - Software Engineering, OOP, Design Patterns, Systems Design

FINANCE & ECONOMICS:
  finance             - Corporate Finance, Investments, Financial Markets, Capital Markets
  derivatives         - Derivatives, Options Pricing, Futures & Forwards, Financial Engineering
  fixed_income        - Fixed Income Securities, Bond Markets, Interest Rate Models
  portfolio_theory    - Portfolio Theory, Asset Pricing, Modern Portfolio Theory, Factor Models
  microeconomics      - Microeconomics, Intermediate Microeconomics, Price Theory
  macroeconomics      - Macroeconomics, Intermediate Macroeconomics, Monetary Economics
  game_theory         - Game Theory, Strategic Decision Making
  risk_management     - Risk Management, Financial Risk, VaR, Credit Risk
  financial_econometrics - Financial Econometrics, Time Series in Finance
  accounting          - Financial Accounting, Managerial Accounting

OTHER:
  other               - Any course that does not fit the above (writing, PE, history, language, etc.)
"""

_SYSTEM_PROMPT = f"""You are a precise academic profile parser for MFE (Master of Financial Engineering) applications.

Your task: extract structured academic and professional information from unstructured resume/transcript text and output a valid QuantPath profile YAML.

=== COURSE CATEGORY MAPPING ===
Map every course to EXACTLY one of these category values (use the exact snake_case string):
{_COURSE_CATEGORIES}

=== GRADE CONVERSION RULES ===
- US letter grades: output as-is (A+, A, A-, B+, B, B-, C+, C, C-, D, F)
- Numeric GPA (0-4.0 scale): extract as float for the gpa field
- Percentage grades: 95-100%→A+, 90-94%→A, 85-89%→A-, 80-84%→B+, 75-79%→B, 70-74%→B-, 65-69%→C+, 60-64%→C
- Chinese letter grades: 优秀(Excellent)→A, 良好(Good)→B+, 中等(Average)→B, 及格(Pass)→C
- Indian CGPA (1-10 scale): divide by 2.5 to convert to 4.0 scale for gpa field; for individual courses, map 9-10→A, 7-8.9→B+, 6-6.9→B, 5-5.9→C
- UK classification: First→A, Upper Second (2:1)→B+, Lower Second (2:2)→B, Third→C
- If grade format is unclear, use "A" as default and note the uncertainty in the course name

=== COURSE LEVEL RULES ===
- 100: Introductory / Freshman year
- 200: Sophomore year / Foundation
- 300: Junior year / Intermediate
- 400: Senior year / Advanced undergraduate
- 500: Graduate level

=== IMPORTANT RULES ===
1. Include ALL courses from transcript — even non-quantitative ones (category: other)
2. If a course spans multiple semesters (e.g., "Calculus I, II, III"), create SEPARATE entries for each
3. For majors: list all majors and minors as separate strings
4. For is_international: true if student is not a US citizen/permanent resident
5. years_at_us_institution: number of years studied at a US university (0 if all abroad)
6. Omit test_scores section entirely if no test scores are mentioned
7. For experience: capture ALL internships, part-time jobs, research positions
8. For projects: include course projects, personal projects, research projects

=== OUTPUT FORMAT ===
Output ONLY valid YAML with NO explanation, NO markdown fences, NO commentary.
Use this exact structure:

personal:
  name: "Full Name"
  university: "University Name"
  majors: ["Major 1", "Major 2"]
  gpa: 3.85
  is_international: true
  years_at_us_institution: 4

test_scores:
  gre_quant: 170
  gre_verbal: 160
  toefl_ibt: 108

courses:
  - code: "MATH 241"
    name: "Calculus III"
    category: calculus
    grade: "A"
    level: 200

  - code: "STAT 400"
    name: "Statistics and Probability I"
    category: statistics
    grade: "A"
    level: 400

experience:
  - type: internship
    title: "Quantitative Research Intern"
    company: "Company Name"
    description: "Brief description of quantitative work performed"
    duration_months: 3

projects:
  - name: "Project Name"
    description: "One-line technical description"
    highlights:
      - "Key quantitative technique or result"
"""


def parse_profile_from_text(text: str, output_path: str) -> dict:
    """Use Claude to parse free-form text into a QuantPath profile YAML.

    Parameters
    ----------
    text:
        Raw resume/transcript content.
    output_path:
        Where to write the resulting YAML file.

    Returns
    -------
    dict
        The parsed profile as a Python dict (also written to output_path).
    """
    client = _make_client()

    print("Sending to Claude for extraction...", flush=True)

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Parse the following academic/professional text into a QuantPath profile YAML.\n"
                    "Extract every course, every internship, and every project you can find.\n\n"
                    f"{text}"
                ),
            }
        ],
    )

    yaml_text = message.content[0].text.strip()

    # Strip markdown code fences if Claude added them
    if yaml_text.startswith("```"):
        lines = yaml_text.split("\n")
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        yaml_text = "\n".join(lines[start:end])

    # Validate it parses as YAML
    try:
        profile_dict = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        print(f"Warning: Claude output was not valid YAML: {exc}", file=sys.stderr)
        print("Raw output saved anyway — please fix manually.", file=sys.stderr)
        profile_dict = {}

    # Write to output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml_text, encoding="utf-8")

    return profile_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse resume/transcript text into a QuantPath profile YAML using Claude AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/parse_profile.py --input resume.txt --output my_profile.yaml
  cat resume.txt transcript.txt | python tools/parse_profile.py --output my_profile.yaml
  python tools/parse_profile.py --output my_profile.yaml   # paste from stdin
        """,
    )
    parser.add_argument("--output", "-o", required=True, help="Output YAML file path")
    parser.add_argument("--input", "-i", help="Input text file (default: read from stdin)")
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        text = input_path.read_text(encoding="utf-8")
    else:
        if sys.stdin.isatty():
            print(
                "Paste your resume/transcript text below.\n"
                "Press Ctrl+D (macOS/Linux) or Ctrl+Z+Enter (Windows) when done:\n"
            )
        text = sys.stdin.read()

    if not text.strip():
        print("Error: No input text provided.", file=sys.stderr)
        sys.exit(1)

    profile = parse_profile_from_text(text, args.output)

    if profile:
        personal = profile.get("personal", {})
        n_courses = len(profile.get("courses", []))
        n_exp = len(profile.get("experience", []))
        n_proj = len(profile.get("projects", []))

        print(f"\nExtracted profile:")
        print(f"  Name       : {personal.get('name', 'N/A')}")
        print(f"  University : {personal.get('university', 'N/A')}")
        print(f"  GPA        : {personal.get('gpa', 'N/A')}")
        print(f"  Majors     : {', '.join(personal.get('majors', []))}")
        print(f"  Courses    : {n_courses}")
        print(f"  Experience : {n_exp} entries")
        print(f"  Projects   : {n_proj} entries")
        print(f"\nProfile saved to: {args.output}")
        print(f"\nNext steps:")
        print(f"  quantpath evaluate --profile {args.output}")
        print(f"  quantpath gaps     --profile {args.output}")
        print(f"  quantpath list     --profile {args.output}")
        print(f"  python tools/advisor.py --profile {args.output}")


if __name__ == "__main__":
    main()
