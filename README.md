# QuantPath

**Open-source MFE application toolkit** — profile evaluation, prerequisite matching, admission prediction, and school selection for Master of Financial Engineering programs.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
[![CI](https://github.com/MasterAgentAI/QuantPath/actions/workflows/ci.yml/badge.svg)](https://github.com/MasterAgentAI/QuantPath/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Tests](https://img.shields.io/badge/tests-465%20passed-brightgreen.svg)
![Data](https://img.shields.io/badge/data-8%2C500%2B%20records-blue.svg)
![Programs](https://img.shields.io/badge/programs-28%20MFE-orange.svg)

---

## Why QuantPath

Consulting services charge **$4,500–10,000 per school**. QuantPath provides the same analysis for free — 5-dimension profile scoring, data-driven reach/target/safety classification, and personalized gap analysis across 28 MFE programs.

```
$ quantpath evaluate --profile my_profile.yaml

  Math              8.4/10  ████████░░
  Statistics        9.0/10  █████████░
  CS                9.8/10  ██████████
  Finance/Econ      9.4/10  █████████░
  GPA               10.0/10 ██████████

  OVERALL:  9.2/10  Top 5-10 MFE Level

  Reach:   CMU MSCF, Columbia MSFE, Berkeley MFE, Princeton MFin, Baruch MFE
  Target:  NYU Tandon MFE, UIUC MSFE, MIT MFin
  Safety:  Columbia MAFN, NYU Courant, Rutgers MQF
```

## Data

| Dataset | Records | Source |
|---------|---------|--------|
| Admission records | **6,984** | GradCafe, QuantNet (accepted / rejected / waitlisted) |
| Offer results | **646** | offer.1point3acres.com (AD/Reject with dates) |
| LinkedIn alumni | **930** | 20 MFE programs (employer, undergrad school) |
| Program database | **28** | QuantNet 2026 Rankings (prerequisites, deadlines, salaries) |
| Trained models | **21** | Per-program logistic regression with bias correction |

## Methodology

**Admission prediction**: per-program logistic regression on GPA + GRE Quant with bias correction. Training data has survivor bias (self-reported accept rate ≠ real rate), so the model replaces the biased intercept with logit(*r*) where *r* is the official acceptance rate, preserving learned feature slopes.

**Profile evaluation**: 5-dimension weighted scoring across 37 sub-factors:

| Dimension | Weight | Key Factors |
|-----------|--------|------------|
| Mathematics | 30% | Stochastic calculus, real analysis, numerical methods, optimization |
| Statistics | 20% | Mathematical statistics, time series, econometrics, statistical learning |
| Computer Science | 20% | C++, algorithms, machine learning, reinforcement learning |
| Finance/Econ | 15% | Derivatives, portfolio theory, risk management, financial econometrics |
| GPA | 15% | Cumulative, quantitative, trend |

**School ranking**: reach / target / safety classification using P(admit) thresholds — reach < 40%, target 40–70%, safety ≥ 70% — with a 100-point composite fit score.

## Features

| Command | Description |
|---------|-------------|
| `quantpath evaluate` | 5-dimension profile score with gaps and strengths |
| `quantpath list` | Personalized reach/target/safety school list with P(admit) |
| `quantpath match --program cmu-mscf` | Prerequisite match for a specific program |
| `quantpath gaps` | Priority-ranked gaps with recommended actions |
| `quantpath optimize` | Top courses to take for maximum profile improvement |
| `quantpath compare --programs cmu-mscf,baruch-mfe` | Side-by-side program comparison |
| `quantpath roi` | Tuition, salary, NPV, payback period per program |
| `quantpath timeline` | Month-by-month application action plan |
| `quantpath portfolio --n-schools 8 --budget 1500` | Optimize school list under budget constraints |
| `quantpath whatif --gpa 3.95 --gre 170` | See how improvements change P(admit) |
| `quantpath tests` | GRE/TOEFL requirements across all programs |
| `quantpath programs` | Full program database with rankings and stats |
| `quantpath interview` | Practice questions by category and difficulty |
| `quantpath stats` | Admission statistics from real applicant data |

### AI Tools

| Tool | Description |
|------|-------------|
| `python tools/advisor.py --profile X.yaml` | Full AI advisory report (Claude-powered) |
| `python tools/parse_profile.py --input resume.txt` | Resume/transcript → profile YAML |
| `python tools/scrape_1p3a.py manual` | Parse Chinese forum posts into structured data |
| `CLAUDE.md` | Open in Claude Code for interactive MFE advising |

## Quick Start

```bash
git clone https://github.com/MasterAgentAI/QuantPath.git
cd QuantPath && pip install -e .

# Create profile from the example template
cp examples/sample_profile.yaml my_profile.yaml
# Edit with your courses, GPA, experience

# Run the full pipeline
quantpath evaluate --profile my_profile.yaml
quantpath gaps     --profile my_profile.yaml
quantpath list     --profile my_profile.yaml
```

For AI-powered analysis (requires Anthropic API key):
```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key

# Auto-parse resume → YAML
python tools/parse_profile.py --output my_profile.yaml

# Generate expert advisory report
python tools/advisor.py --profile my_profile.yaml --save report.md
```

## Programs (Top 15)

| # | Program | University | Class | Rate | Avg GPA | Salary |
|---|---------|-----------|-------|------|---------|--------|
| 1 | MFE | Baruch College | 20 | 4% | 3.84 | $179K |
| 2 | MFin | Princeton | 44 | 5% | — | $160K |
| 3 | MSCF | Carnegie Mellon | 108 | 17% | 3.86 | $134K |
| 4 | MSFE | Columbia (IEOR) | 136 | 13% | 3.90 | $138K |
| 5 | MFin | MIT Sloan | 126 | 8% | 3.80 | $140K |
| 6 | MFE | UC Berkeley | 76 | 17% | 3.80 | $154K |
| 7 | MSFM | UChicago | 118 | 22% | — | $124K |
| 8 | QCF | Georgia Tech | 99 | 30% | — | $115K |
| 9 | MAFN | Columbia (Math) | 101 | 22% | — | $123K |
| 10 | MFM | NC State | 64 | 17% | — | $105K |
| 11 | MFE | Cornell | 53 | 21% | — | $115K |
| 12 | MathFin | NYU Courant | 37 | 23% | — | $126K |
| 13 | MFE | NYU Tandon | 154 | 38% | 3.83 | $107K |
| 14 | MFE | UCLA | 79 | 36% | — | $118K |
| 15 | MSQF | Fordham | 61 | 59% | — | $136K |

Full 28-program database with deadlines, prerequisites, essay requirements, and interview formats in `data/programs/`.

## Course Categories

37 categories across 4 academic dimensions, aligned with MFE program prerequisites. Each course in your profile maps to a category that feeds into the scoring engine.

<details>
<summary>Full category reference (click to expand)</summary>

### Mathematics (30%)
`calculus` · `linear_algebra` · `probability` · `ode` · `pde` · `real_analysis` · `numerical_analysis` · `stochastic_processes` · `stochastic_calculus` · `optimization`

### Statistics (20%)
`statistics` · `regression` · `econometrics` · `time_series` · `stat_computing` · `stat_learning` · `bayesian`

### Computer Science (20%)
`programming_cpp` · `programming_python` · `programming_r` · `data_structures` · `algorithms` · `machine_learning` · `reinforcement_learning` · `database` · `software_engineering`

### Finance & Economics (15%)
`finance` · `derivatives` · `fixed_income` · `portfolio_theory` · `microeconomics` · `macroeconomics` · `game_theory` · `risk_management` · `financial_econometrics` · `accounting`

</details>

## Architecture

```
QuantPath/
├── core/                  # Evaluation engine (pure Python + PyYAML)
│   ├── profile_evaluator  # 5-dimension scoring (37 sub-factors)
│   ├── school_ranker      # Reach/target/safety classification
│   ├── lr_predictor       # Logistic regression P(admit) with bias correction
│   ├── list_builder       # Portfolio optimization with geographic diversity
│   ├── gap_advisor        # Gap analysis with action recommendations
│   ├── course_optimizer   # Course impact optimization
│   ├── roi_calculator     # Financial ROI analysis
│   └── calibrator         # Model calibration from real admission data
├── cli/main.py            # 15+ CLI commands
├── web/app.py             # Streamlit dashboard (6 pages)
├── tools/                 # AI-powered utilities
│   ├── advisor.py         # Claude-powered advisory report
│   ├── parse_profile.py   # Resume/transcript → YAML parser
│   ├── scrape_1p3a.py     # 1point3acres forum scraper + regex parser
│   └── train_model.py     # sklearn model training pipeline
├── data/
│   ├── programs/          # 28 program YAML files
│   ├── admissions/        # 8,500+ admission records (CSV + JSON)
│   └── models/            # 21 pre-trained LR models (JSON, no pickle)
└── tests/                 # 465 tests, 0.5s runtime
```

## Contributing

1. **Add/update program data** — PRs to `data/programs/` with updated deadlines or new programs
2. **Add course mappings** — Help map courses from more universities to the category system
3. **Report data errors** — Open issues for outdated program information
4. **Feature requests** — Tell us what would help your application process

## License

MIT — see [LICENSE](LICENSE).

---

*QuantPath is a tool to assist with MFE application planning. All program data is sourced from official websites and QuantNet rankings. Always verify on official program websites before applying.*
