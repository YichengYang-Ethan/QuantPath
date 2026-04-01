# QuantPath

**Open-source MFE application toolkit** — AI-powered profile evaluation, prerequisite matching, and school selection for top financial engineering programs.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
[![CI](https://github.com/MasterAgentAI/QuantPath/actions/workflows/ci.yml/badge.svg)](https://github.com/MasterAgentAI/QuantPath/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Claude AI](https://img.shields.io/badge/Claude-AI%20Powered-orange.svg)
![Data](https://img.shields.io/badge/Data-8%2C597%2B%20records-blue.svg)

---

## The Problem

Every year, 15,000+ students apply to Master of Financial Engineering (MFE) programs. They all face the same questions:

- **"Am I competitive for [program]?"** — No standardized way to evaluate profiles
- **"Do my courses satisfy prerequisites?"** — Manual cross-referencing across 10+ program websites
- **"Which schools should I apply to?"** — No data-driven reach/target/safety classification
- **"When are the deadlines?"** — Scattered across program websites, different round structures

Consulting services charge **$4,500-10,000 per school**. QuantNet profile evaluations rely on volunteer responses with multi-day turnaround.

QuantPath changes that — and now with **Claude AI integration**, you get a full expert advisory report in minutes.

## Data

| Dataset | Records | Source | Fields |
|---------|---------|--------|--------|
| Admission records | **6,984** | GradCafe, QuantNet | GPA, GRE, bg_type, result, program |
| Offer results | **646** | offer.1point3acres.com | AD/Reject, dates, 12 MFE programs |
| LinkedIn alumni | **930** | Brave Search | employer, undergrad school, 20 programs |
| Program database | **28** | QuantNet 2026 Rankings | prerequisites, deadlines, salaries |
| Trained LR models | **21** | Per-program logistic regression | bias-corrected P(admit) |

**Methodology**: per-program logistic regression on GPA + GRE Quant with bias correction. Raw training data has survivor bias (self-reported), so the model replaces the biased intercept with logit(*r*) where *r* is the official acceptance rate, preserving learned feature slopes.

## Features

### Profile Evaluator
5-dimension scoring system (Math, Statistics, CS, Finance/Econ, GPA) with weighted sub-factors calibrated against admission data from top programs.

```
$ quantpath evaluate --profile my_profile.yaml

┌─ QuantPath Profile Evaluation ──────────────────────┐
│ Wei Zhang | Zhejiang University | GPA 3.74 | Intl    │
└──────────────────────────────────────────────────────┘
  Math            7.3/10  ███████░░░
  Statistics      7.6/10  ███████▌░░
  CS              8.2/10  ████████░░
  Finance/Econ    5.8/10  █████▊░░░░
  GPA             8.4/10  ████████░░

  OVERALL:        7.5/10  ███████▌░░  Competitive

  🎯 Reach:   Baruch MFE, Princeton MFin, CMU MSCF
  🎯 Target:  Columbia MSFE, Cornell MFE, Berkeley MFE
  🎯 Safety:  GaTech QCF, UIUC MSFE, Rutgers MQF

  ⚠️  Gaps Found:
     - Stochastic Calculus: Missing (Critical)
     - ODE: Missing (High)
     - Stochastic Processes: Missing (High)
     - Time Series Analysis: Missing (High)

  ✅ Strengths:
     - Calculus A+ (exceptional foundation)
     - Real Analysis A- (strong mathematical maturity signal)
     - C++ A (essential for Baruch, CMU, Berkeley)
     - Two quant internships
```

### School Ranker
Data-driven reach/target/safety classification using logistic regression trained on 6,984 historical admission records. Shows admission probability for your specific GPA and GRE profile.

```
$ quantpath list --profile my_profile.yaml

  Category  Program             P(Admit)   Fit Score   Rate
  ─────────────────────────────────────────────────────────
  Reach     Baruch MFE          24%        64.1        4%
  Reach     Princeton MFin      16%        59.3        5%
  Reach     CMU MSCF            37%        68.7        17%
  Target    Columbia MSFE       53%        74.2        13%
  Target    Cornell MFE         61%        76.8        21%
  Target    Berkeley MFE        49%        72.1        17%
  Safety    GaTech QCF          74%        81.5        30%
  Safety    UIUC MSFE           81%        84.3        51%
  Safety    Rutgers MQF         88%        87.9        86%
```

### Prerequisite Matcher
Automatically maps your coursework against each program's requirements. Identifies gaps, warnings (e.g., grade below threshold), and match scores.

```
$ quantpath match --profile my_profile.yaml --program baruch-mfe

  Baruch MFE (Baruch College, CUNY)
  Match: 71%   P(Admit): 24%
  Missing: Stochastic Calculus (required)
  ⚠️  Linear Algebra: B+ (borderline — Baruch prefers A-/A)
```

### Test Requirements
One command to check GRE/TOEFL requirements across all programs.

```
$ quantpath tests --profile my_profile.yaml

  Program          GRE        TOEFL
  CMU MSCF         REQUIRED   Waived
  Princeton MFin   Optional   Waived
  Baruch MFE       N/A        Waived
  Berkeley MFE     Exempt     Waived
  Stanford MCF     N/A        Waived
  ...
```

### Application Timeline
Auto-generated month-by-month action plan with all deadlines.

```
$ quantpath timeline

  ── September 2026 ──
  ⚪ Sep 01  Application portals open — create accounts
  🟡 Sep 15  Finalize essays — have them reviewed

  ── November 2026 ──
  🔴 Nov 01  Baruch MFE Round 1 deadline

  ── December 2026 ──
  🔴 Dec 01  CMU MSCF Round 1 / Cornell MFE deadline
  🔴 Dec 15  Princeton MFin deadline
  ...
```

### Programs Database
Structured YAML database of 10 top MFE programs with prerequisites, deadlines, admission stats, essay requirements, and interview formats.

```
$ quantpath programs

  Program          University       Class  Rate  GPA   GRE
  CMU MSCF         Carnegie Mellon  108    10%   3.86  Required
  Princeton MFin   Princeton        35     5%    N/A   Optional
  Baruch MFE       Baruch/CUNY      30     8%    3.84  N/A
  ...
```

### Program Comparison
Compare programs side-by-side on key metrics. Add `--profile` to see your personalized P(Admit) for each.

```
$ quantpath compare --programs cmu-mscf,baruch-mfe,berkeley-mfe --profile my_profile.yaml

  Attribute          CMU MSCF        Baruch MFE       Berkeley MFE
  University         Carnegie Mellon  Baruch/CUNY      UC Berkeley
  Class Size         108              30               80
  Acceptance Rate    10%              8%               12%
  Avg GPA            3.86             3.84             3.80
  GRE Required       Yes              No               Yes
  Application Fee    $125             $75              $275
  ...
  P(Admit) *         21% [14%-30%]   5% [1%-25%]      19% [6%-45%]
  * P(Admit) for Wei Zhang (GPA 3.74)
```

### Gap Analysis
Identify weaknesses and get prioritized action recommendations.

```
$ quantpath gaps --profile my_profile.yaml

  Factor                Dimension   Score    Priority  Recommended Action
  Stochastic Calculus   math        Missing  Critical  Self-study Shreve Vol I-II; list on SOP
  ODE                   math        Missing  High      Take before applying (prerequisite for stoch calc)
  Stochastic Processes  math        Missing  High      Take STAT 433 or equivalent
  Time Series           statistics  Missing  High      Take STAT 429 (ARIMA, GARCH models)
  Finance/Econ          finance     5.8/10   Medium    Add Derivatives or Risk Management course

  Summary:  1 Critical  3 High  1 Medium  (5 total gaps)
```

### Portfolio Optimizer
Greedy marginal-value selection maximizing expected admissions under budget and school-count constraints.

```
$ quantpath portfolio --profile my_profile.yaml --n-schools 8 --budget 1500

  Category  Program          P(Admit)  Fit   Fee     Exp. Contrib.
  ──────────────────────────────────────────────────────────────────
  Reach     CMU MSCF         21%       75.0  $125    +0.21
  Reach     Berkeley MFE     19%       78.0  $275    +0.19
  Target    UIUC MSFE        57%       87.0  $70     +0.57
  Target    JHU MFM          55%       86.0  $80     +0.55
  Target    UWash CFRM       65%       88.0  $85     +0.65
  Safety    Rutgers MQF      85%       92.0  $70     +0.85
  Safety    BU MSMF          82%       92.0  $80     +0.82

  Expected admits: 3.84 schools
  Total fees: $785
```

### What-If Analysis
See how P(admit) changes across all programs under hypothetical GPA/GRE improvements.

```
$ quantpath whatif --profile my_profile.yaml --gpa 3.95 --gre 170

  Program          P(now)  P(hyp)  Delta  Tier Change
  ─────────────────────────────────────────────────────
  NC State MFM     38%     53%     +15%   reach → target
  USC MSMF         29%     40%     +11%   reach → target
  UWash CFRM       65%     76%     +10%   target → safety
  CMU MSCF         21%     29%      +8%
  Baruch MFE        5%      6%      +1%
  ...

  3 program(s) would change tier.
```

## Quick Start

### Option A — AI-Powered (Recommended)

Drop your resume or transcript as plain text. Claude parses it automatically and generates a full strategy report.

```bash
git clone https://github.com/MasterAgentAI/QuantPath.git
cd QuantPath
pip install -e . anthropic

export ANTHROPIC_API_KEY=your_key_here   # get at console.anthropic.com

# Parse your resume/transcript → profile YAML
python tools/parse_profile.py --output my_profile.yaml
# (paste your resume text, press Ctrl+D when done)

# Generate full AI advisory report
python tools/advisor.py --profile my_profile.yaml --save report.md
```

The advisor report covers:
- Competitive assessment across all 5 dimensions
- Priority-ranked gaps with specific fix actions
- Personalized reach/target/safety school list with rationale
- 90-day action plan
- SOP strategy and differentiators

### Option B — Claude Code Integration

If you use [Claude Code](https://claude.ai/code), this repo ships with a `CLAUDE.md` that turns Claude into a full MFE advisor. Just open this repo in Claude Code and say:

> "Here is my resume: [paste]. Analyze my profile and tell me which MFE programs I should apply to."

Claude will automatically parse your profile, run all evaluations, and give personalized advice — no YAML editing required.

### Option C — Manual CLI

```bash
git clone https://github.com/MasterAgentAI/QuantPath.git
cd QuantPath && pip install -e .

# Create your profile
cp examples/sample_profile.yaml my_profile.yaml
# Edit my_profile.yaml with your info

# Run the full pipeline
quantpath evaluate --profile my_profile.yaml
quantpath gaps     --profile my_profile.yaml
quantpath list     --profile my_profile.yaml
quantpath roi
quantpath timeline
quantpath programs
```

## Profile YAML Format

```yaml
personal:
  name: "Your Name"
  university: "Your University"
  majors: ["Major 1", "Major 2"]
  gpa: 3.85
  is_international: true
  years_at_us_institution: 4

courses:
  - code: "MATH 241"
    name: "Calculus III"
    category: calculus        # see categories below
    grade: "A"                # letter or numeric (0-100)
    level: 200                # course level

  - code: "CS 225"
    name: "Data Structures"
    category: data_structures
    grade: "A"
    level: 200

# ... more courses

experience:
  - type: internship
    title: "Quant Research Intern"
    description: "Built trading strategies..."
    duration_months: 3

projects:
  - name: "Project Name"
    description: "What it does"
    highlights:
      - "Key achievement 1"
      - "Key achievement 2"
```

### Course Categories

Categories are organized by discipline and aligned with MFE program prerequisites.

#### Mathematics (Dimension Weight: 30%)

The most critical dimension. Top programs (Baruch, Princeton, CMU) require strong math preparation.

| Category | Examples | Importance |
|----------|---------|------------|
| `calculus` | Calculus I-III, Multivariable Calculus | **Required** by all programs |
| `linear_algebra` | Linear Algebra, Matrix Theory, Eigenvalues | **Required** by all programs |
| `probability` | Probability Theory, Calc-based Probability | **Required** by all programs |
| `ode` | Ordinary Differential Equations | **Required** by most programs |
| `pde` | Partial Differential Equations | Required by Berkeley, Stanford; recommended by top-5 |
| `real_analysis` | Real Analysis, Real Variables, Measure Theory | Required by Princeton, NYU Courant; strongly recommended |
| `numerical_analysis` | Numerical Methods, Numerical Analysis | Required by Berkeley; recommended |
| `stochastic_processes` | Stochastic Processes, Markov Chains | Strongly recommended for top programs |
| `stochastic_calculus` | Stochastic Calculus, Itô Calculus, SDEs | Top differentiator for competitive applicants |
| `optimization` | Convex Optimization, Linear/Nonlinear Programming | Valued by Cornell ORIE, GaTech, CMU |

#### Statistics & Data Science (Dimension Weight: 20%)

| Category | Examples | Importance |
|----------|---------|------------|
| `statistics` | Mathematical Statistics, Statistical Inference | **Required** by all programs |
| `regression` | Regression Analysis, Applied Statistics | Strongly recommended |
| `econometrics` | Econometrics, Causal Inference | Recommended; especially for applied programs |
| `time_series` | Time Series Analysis, ARIMA, GARCH | Strongly recommended for financial modeling |
| `stat_computing` | Statistical Computing (R/Python), Simulation | Recommended |
| `stat_learning` | Statistical Learning, ML from Stats perspective | Recommended |
| `bayesian` | Bayesian Statistics, Bayesian Inference | Nice to have for research-oriented programs |

#### Computer Science (Dimension Weight: 20%)

| Category | Examples | Importance |
|----------|---------|------------|
| `programming_cpp` | C/C++ Programming | **Essential** for Baruch, CMU, Berkeley |
| `programming_python` | Python Programming, NumPy/pandas | **Required** proficiency by all programs |
| `programming_r` | R Programming | Useful for stats-heavy programs |
| `data_structures` | Data Structures | Strongly recommended |
| `algorithms` | Algorithms, Algorithm Design | Strongly recommended |
| `machine_learning` | Machine Learning, Deep Learning | Recommended; increasingly valued |
| `reinforcement_learning` | Reinforcement Learning, MDP, Q-learning | Valued for quant trading (optimal execution, deep hedging) |
| `database` | Database Systems, SQL | Nice to have |
| `software_engineering` | Software Engineering, OOP, Design Patterns | Valued by industry-focused programs |

#### Finance & Economics (Dimension Weight: 15%)

| Category | Examples | Importance |
|----------|---------|------------|
| `finance` | Corporate Finance, Investments, CAPM | Recommended |
| `derivatives` | Derivatives, Options Pricing, Black-Scholes | Strongly recommended |
| `fixed_income` | Fixed Income Securities, Bond Pricing | Nice to have |
| `portfolio_theory` | Portfolio Theory, Asset Pricing, Factor Models | Recommended |
| `microeconomics` | Intermediate Microeconomics | Recommended |
| `macroeconomics` | Intermediate Macroeconomics | Recommended |
| `game_theory` | Game Theory | Lower priority |
| `risk_management` | Risk Management, VaR, Credit Risk | Recommended |
| `financial_econometrics` | Financial Econometrics | Recommended |
| `accounting` | Financial Accounting | Nice to have |

## Programs Covered

28 programs with data from the **QuantNet 2026 Rankings** and official program websites.

| Rank | Program | University | Class Size | Accept Rate | Avg Salary |
|------|---------|-----------|-----------|-------------|------------|
| #1 | MFE | Baruch College | 20 | 4.0% | $178,824 |
| #2 | MFin | Princeton | 44 | 5.4% | $160,217 |
| #3 | MSCF | Carnegie Mellon | 108 | 17.2% | $134,259 |
| #4 | MSFE | Columbia (IEOR) | 136 | 13.3% | $138,000 |
| #5 | MFin | MIT Sloan | 126 | 8.3% | $140,140 |
| #6 | MFE | UC Berkeley | 76 | 17.4% | $154,383 |
| #7 | MSFM | UChicago | 118 | 22.0% | $123,867 |
| #8 | QCF | Georgia Tech | 99 | 30.2% | $114,684 |
| #9 | MAFN | Columbia (Math) | 101 | 22.3% | $122,692 |
| #10 | MFM | NC State | 64 | 16.7% | $105,350 |
| #11 | MFE | Cornell | 53 | 20.7% | $115,000 |
| #12 | MS Math Finance | NYU Courant | 37 | 22.5% | $126,000 |
| #13 | MFE | NYU Tandon | 154 | 38.1% | $107,478 |
| #14 | MFE | UCLA | 79 | 36.0% | $118,270 |
| #15 | MSQF | Fordham | 61 | 59.4% | $135,500 |
| #16 | CFRM | U Washington | 41 | 53.9% | $118,375 |
| #17 | MSFE | UIUC | 30 | 50.7% | $105,200 |
| #18 | MSMF | UNC Charlotte | 19 | 88.2% | $104,375 |
| #19 | MQF | Rutgers | 48 | 86.4% | $111,110 |
| #20 | MFE | Stevens | 59 | 68.0% | $110,982 |
| #21 | MFM | U Minnesota | 15 | 80.7% | $106,163 |
| #22 | MSMF | Boston U | 56 | 80.6% | $84,171 |
| #23 | MFM | Johns Hopkins | 29 | 50.7% | - |
| - | MCF | Stanford | ~10 | ~5% | - |
| - | MFE | U Michigan | 40 | - | - |
| - | MFE | Northwestern | 40 | - | - |
| - | MSMF | USC | 25 | - | - |
| - | MMF | U Toronto | 30 | - | - |

## AI Tools

### `tools/parse_profile.py` — Resume/Transcript Parser

Converts unstructured text (resume paste, transcript copy) into a valid QuantPath profile YAML using Claude.

```bash
# From a file
python tools/parse_profile.py --input resume.txt --output my_profile.yaml

# From stdin (paste, then Ctrl+D)
python tools/parse_profile.py --output my_profile.yaml

# Combine resume + transcript
cat resume.txt transcript.txt | python tools/parse_profile.py --output my_profile.yaml
```

Supports all grade formats: US letter (A/B+), percentages (90%), Chinese (优/良), Indian CGPA (8.5/10), UK classifications (First/2:1).

### `tools/advisor.py` — AI Strategy Advisor

Runs the complete QuantPath pipeline and streams an expert advisory report from Claude.

```bash
python tools/advisor.py --profile my_profile.yaml
python tools/advisor.py --profile my_profile.yaml --save strategy_report.md
```

**Report sections:**
- Executive Summary
- Profile Scorecard (dimension-by-dimension analysis)
- Top 3 Strengths
- Critical Gaps (priority-ordered with specific fixes)
- Recommended School List (Reach/Target/Safety with rationale)
- Immediate Action Plan (next 90 days)
- SOP Strategy (themes, narrative arc, differentiators)
- Red Flags to Address

### Claude Code Integration (`CLAUDE.md`)

Open this repo in Claude Code. The `CLAUDE.md` file gives Claude a complete understanding of the tool, all commands, score interpretation, program database, and advisory patterns.

You can then have a natural conversation:

> "Analyze my profile" → Claude runs the full pipeline and explains results
> "What courses should I take this semester?" → Claude checks gaps + optimizer
> "Compare Baruch and CMU for me" → Claude runs comparison with context
> "Help me write my SOP" → Claude coaches based on your specific profile

---

## Contributing

We welcome contributions! The most impactful ways to help:

1. **Add/update program data** — Submit PRs to `data/programs/` with updated deadlines, stats, or new programs
2. **Add university course mappings** — Help us map courses from more universities to our category system
3. **Report data errors** — Open issues if you spot outdated or incorrect program information
4. **Feature requests** — Tell us what would help your application process

## Roadmap

- [x] Core profile evaluation engine
- [x] Prerequisite matching
- [x] School ranking (reach/target/safety)
- [x] GRE/TOEFL requirement checker
- [x] Application timeline generator
- [x] CLI interface
- [x] Program comparison (`compare`)
- [x] Gap analysis with recommendations (`gaps`)
- [x] Unit test suite (465 tests)
- [x] GitHub Actions CI
- [x] Interview question bank (52 questions, 7 categories)
- [x] Streamlit web dashboard
- [x] 28 programs database (QuantNet 2026 rankings)
- [x] **AI resume/transcript parser** (`tools/parse_profile.py`)
- [x] **AI strategy advisor** (`tools/advisor.py`)
- [x] **Claude Code integration** (`CLAUDE.md`)
- [x] **LR admission model** — bias-corrected P(admit) per program with 95% CI
- [x] **Portfolio optimizer** — maximize expected admissions under budget constraints
- [x] **What-if analysis** — see how GPA/GRE changes affect P(admit) across all programs
- [x] **Profile-aware adjustments** — international status and internship experience factor into predictions
- [x] **Reinforcement learning** course category — RL for quant trading (optimal execution, deep hedging)
- [x] **Data collection pipeline** — 1point3acres scraper + regex parser (`tools/scrape_1p3a.py`)
- [x] **930 LinkedIn alumni profiles** — employer + undergrad data across 20 MFE programs
- [x] **646 offer results** from offer.1point3acres.com with dates + program IDs
- [ ] Multi-feature admission model (GPA + bg_tier + internships + research)
- [ ] University course databases (auto-map courses from 50+ schools)
- [ ] AI essay coach with SOP draft generation
- [ ] Web frontend with PDF upload

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

QuantPath is a tool to assist with MFE application planning. All program data is sourced from official websites and may become outdated. Always verify information on official program websites before making application decisions. This tool does not guarantee admission to any program.
