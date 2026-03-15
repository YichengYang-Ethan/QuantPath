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
│ Yicheng Yang | UIUC | GPA 4.0 | International   │
└──────────────────────────────────────────────────┘
  Math            9.5/10  ████████▌░
  Statistics     10.0/10  ██████████
  CS             10.0/10  ██████████
  Finance/Econ    9.0/10  █████████░
  GPA             9.5/10  ████████▌░

  OVERALL:        9.6/10  ██████████  Top 3-5 MFE Level

  🎯 Reach:   Princeton MFin, Stanford MCF, Baruch MFE
  🎯 Target:  CMU MSCF, Columbia MSFE, MIT MFin
  🎯 Safety:  NYU Tandon, UChicago MSFM, Cornell MFE

  ⚠️  Gaps Found:
     - Linear Algebra: 76/100 (below Baruch B+ threshold)

  ✅ Strengths:
     - CS major (only 18% of CMU class)
     - 3x C++ courses all A
     - 7x 400-level Statistics courses
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

| Category | Examples |
|----------|---------|
| `calculus` | Calc I-III, Multivariable Calculus |
| `linear_algebra` | Linear Algebra, Matrix Theory |
| `probability` | Probability Theory, Calc-based Probability |
| `statistics` | Mathematical Statistics, Inference |
| `ode` | Ordinary Differential Equations |
| `pde` | Partial Differential Equations |
| `real_analysis` | Real Analysis, Real Variables |
| `numerical_analysis` | Numerical Methods, Numerical Analysis |
| `stochastic_processes` | Stochastic Processes, Random Processes |
| `econometrics` | Econometrics, Economic Statistics |
| `time_series` | Time Series Analysis |
| `stat_computing` | Statistical Computing, Computational Stats |
| `stat_learning` | Statistical Learning, ML/Stats |
| `programming_cpp` | C/C++ Programming |
| `programming_python` | Python Programming |
| `data_structures` | Data Structures |
| `algorithms` | Algorithms |
| `machine_learning` | Machine Learning, Deep Learning |
| `database` | Database Systems |
| `finance` | Corporate Finance, Investments |
| `microeconomics` | Micro, Intermediate Micro |
| `macroeconomics` | Macro, Intermediate Macro |
| `game_theory` | Game Theory |
| `risk_management` | Risk Management, Economics of Risk |
| `financial_econometrics` | Financial Econometrics |

## Programs Covered

| Program | University | QuantNet Rank |
|---------|-----------|---------------|
| MSCF | Carnegie Mellon | #1 |
| MFin | Princeton | #2 |
| MFE | Baruch College | #3 |
| MFE | UC Berkeley | #4 |
| MSFE | Columbia | #5 |
| MFE | NYU Tandon | #6 |
| MFin | MIT | #7 |
| MCF | Stanford | #8 |
| MSFM | UChicago | #9 |
| MFE | Cornell | #10 |

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
- [x] Unit test suite (120 tests)
- [x] GitHub Actions CI
- [ ] Web dashboard (Next.js)
- [ ] Course optimizer (fill gaps optimally)
- [ ] AI essay coach
- [ ] Interview question database
- [ ] More programs (20+)
- [ ] University course databases (auto-map courses)

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

QuantPath is a tool to assist with MFE application planning. All program data is sourced from official websites and may become outdated. Always verify information on official program websites before making application decisions. This tool does not guarantee admission to any program.
