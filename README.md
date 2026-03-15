# QuantPath

**Open-source MFE application toolkit** — profile evaluation, prerequisite matching, and school selection for top financial engineering programs.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## The Problem

Every year, 15,000+ students apply to Master of Financial Engineering (MFE) programs. They all face the same questions:

- **"Am I competitive for [program]?"** — No standardized way to evaluate profiles
- **"Do my courses satisfy prerequisites?"** — Manual cross-referencing across 10+ program websites
- **"Which schools should I apply to?"** — No data-driven reach/target/safety classification
- **"When are the deadlines?"** — Scattered across program websites, different round structures

Consulting services charge **$4,500-10,000 per school**. QuantNet profile evaluations rely on volunteer responses with multi-day turnaround. **No AI or automated tool exists for MFE applicants.**

QuantPath changes that.

## Features

### Profile Evaluator
5-dimension scoring system (Math, Statistics, CS, Finance/Econ, GPA) with weighted sub-factors calibrated against admission data from top programs.

```
$ quantpath evaluate --profile my_profile.yaml

┌─ QuantPath Profile Evaluation ──────────────────┐
│ Alex Chen | State University | GPA 3.78 | Intl    │
└──────────────────────────────────────────────────┘
  Math            8.2/10  ████████░░
  Statistics      7.5/10  ███████▌░░
  CS              7.0/10  ███████░░░
  Finance/Econ    8.0/10  ████████░░
  GPA             8.5/10  ████████▌░

  OVERALL:        7.8/10  ████████░░  Competitive

  🎯 Reach:   Princeton MFin, Stanford MCF, Baruch MFE
  🎯 Target:  CMU MSCF, Berkeley MFE, MIT MFin
  🎯 Safety:  UCLA MFE, GaTech QCF, Rutgers MQF

  ⚠️  Gaps Found:
     - Stochastic Processes: Missing
     - PDE: Missing

  ✅ Strengths:
     - Real Analysis A-
     - Strong economics coursework
     - Two quant internships
```

### Prerequisite Matcher
Automatically maps your coursework against each program's requirements. Identifies gaps, warnings (e.g., grade below threshold), and match scores.

```
$ quantpath match --profile my_profile.yaml --program baruch-mfe

  Baruch MFE (Baruch College, CUNY)
  Match: 85%
  Missing: None
  ⚠️  Linear Algebra grade 76 below B+ requirement
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
Compare programs side-by-side on key metrics.

```
$ quantpath compare --programs cmu-mscf,baruch-mfe,berkeley-mfe

  Attribute          CMU MSCF        Baruch MFE       Berkeley MFE
  University         Carnegie Mellon  Baruch/CUNY      UC Berkeley
  Class Size         108              30               80
  Acceptance Rate    10%              8%               12%
  Avg GPA            3.86             3.84             3.80
  GRE Required       Yes              No               Yes
  Application Fee    $125             $75              $275
  ...
```

### Gap Analysis
Identify weaknesses and get prioritized action recommendations.

```
$ quantpath gaps --profile my_profile.yaml

  Factor              Dimension   Score    Priority  Recommended Action
  Real Analysis       math        Missing  High      Take Real Analysis (required by top-5)
  Stochastic Proc.    math        Missing  High      Take Stochastic Calculus
  Time Series         statistics  Missing  High      Take Time Series Analysis
  ...

  Summary:  8 High  2 Medium  1 Low  (11 total gaps)
```

## Quick Start

```bash
# Install from source
git clone https://github.com/MasterAgentAI/QuantPath.git
cd QuantPath
pip install -e .

# Create your profile (copy and edit the example)
cp examples/sample_profile.yaml my_profile.yaml

# Run evaluation
quantpath evaluate --profile my_profile.yaml

# Compare programs
quantpath compare --programs cmu-mscf,baruch-mfe,berkeley-mfe

# Analyze gaps
quantpath gaps --profile my_profile.yaml

# Check all programs
quantpath programs

# Match prerequisites
quantpath match --profile my_profile.yaml

# Check test requirements
quantpath tests --profile my_profile.yaml

# Generate timeline
quantpath timeline
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
- [x] Unit test suite (173 tests)
- [x] GitHub Actions CI
- [x] Interview question bank (52 questions, 7 categories)
- [x] Streamlit web dashboard
- [x] 28 programs database (QuantNet 2026 rankings)
- [ ] Course optimizer (fill gaps optimally)
- [ ] AI essay coach (Claude API)
- [ ] University course databases (auto-map courses)

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

QuantPath is a tool to assist with MFE application planning. All program data is sourced from official websites and may become outdated. Always verify information on official program websites before making application decisions. This tool does not guarantee admission to any program.
