# QuantPath

**MFE admission prediction engine** — GPBoost mixed-effects model on 12,800+ records across 34 programs. Profile scoring, prerequisite matching, and data-driven school ranking.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
[![CI](https://github.com/MasterAgentAI/QuantPath/actions/workflows/ci.yml/badge.svg)](https://github.com/MasterAgentAI/QuantPath/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Tests](https://img.shields.io/badge/tests-465%20passed-brightgreen.svg)
![Data](https://img.shields.io/badge/data-12%2C800%2B%20records-blue.svg)
![Programs](https://img.shields.io/badge/programs-34%20MFE-orange.svg)
![Model](https://img.shields.io/badge/model-AUC%200.716-brightgreen.svg)

---

## Why QuantPath

Consulting services charge **$4,500–10,000 per school**. QuantPath provides the same analysis for free — multi-dimensional profile scoring, admission probability with confidence intervals, and personalized gap analysis across 34 MFE programs.

```
$ quantpath evaluate --profile my_profile.yaml

  Math              8.4/10  ████████░░
  Statistics        9.0/10  █████████░
  CS                9.8/10  ██████████
  Finance/Econ      9.4/10  █████████░
  GPA               10.0/10 ██████████

  OVERALL:  9.2/10  Top 5-10 MFE Level

  Reach:   UChicago MSFM, Cornell MFE, Berkeley MFE, Princeton MFin, Baruch MFE
  Target:  UIUC MSFE, UCLA MFE, NYU Tandon MFE, GaTech QCF
  Safety:  CMU MSCF, MIT MFin, Columbia MAFN, NYU Courant, Rutgers MQF
```

## Data

| Dataset | Records | Source |
|---------|---------|--------|
| Admission records | **12,800+** | GradCafe, QuantNet, Reddit, 1point3acres (accepted/rejected/waitlisted) |
| LinkedIn alumni | **930** | 20 MFE programs (employer, undergrad school, graduation year) |
| Program database | **28** | QuantNet 2026 Rankings (prerequisites, deadlines, salaries, class profiles) |

## Model

**v2 (current)**: GPBoost — LightGBM gradient boosting with per-program random intercepts. Trained on 10,525 labeled records, 13 features, 34 programs. AUC 0.716, Brier 0.203 (5-fold CV).

Feature importance (data-driven):

| Rank | Feature | Importance | What It Captures |
|------|---------|-----------|-----------------|
| 1 | Major relevance | 4,642 | Math/Stats/CS major vs unrelated |
| 2 | Undergrad tier | 1,638 | T10/C9/985/211 vs other |
| 3 | GPA | 927 | Normalized to 4.0 scale |
| 4 | Intern score | 563 | US top quant → China finance → none |
| 5 | Research | 410 | Published → significant → none |
| 6 | GRE Quant | 352 | Low importance (threshold filter) |
| 7 | International | 342 | Nationality effect |

**v1 (fallback)**: Per-program logistic regression on GPA + GRE Quant with bias correction. 21 trained models. Used when GPBoost is not installed.

**Bias correction**: Self-reported data has survivor bias (65% accept rate in data vs 4-30% real). The model replaces the biased intercept with logit(*r*) where *r* is the official acceptance rate, preserving learned feature slopes ([King & Zeng 2001](https://gking.harvard.edu/files/abs/0s-abs.shtml)).

## Features

| Command | Description |
|---------|-------------|
| `quantpath evaluate` | 5-dimension profile score (37 sub-factors) with gaps and strengths |
| `quantpath list` | Personalized reach/target/safety school list with P(admit) + CI |
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

### AI Tools

| Tool | Description |
|------|-------------|
| `python tools/advisor.py --profile X.yaml` | Full AI advisory report (Claude-powered) |
| `python tools/parse_profile.py --input resume.txt` | Resume/transcript → profile YAML |
| `python tools/train_model_v2.py` | Train GPBoost model from admission data |
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

For the v2 model (GPBoost, recommended):
```bash
pip install gpboost numpy
# v2 model is automatically used when gpboost is installed
```

For AI-powered analysis:
```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key
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

37 categories across 4 academic dimensions, aligned with MFE program prerequisites.

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
├── core/                    # Evaluation engine (pure Python + PyYAML)
│   ├── profile_evaluator    # 5-dimension scoring (37 sub-factors)
│   ├── school_ranker        # Reach/target/safety + GPBoost v2 integration
│   ├── lr_predictor         # v1 LR + v2 GPBoost inference (graceful fallback)
│   ├── list_builder         # Portfolio optimization with geographic diversity
│   ├── gap_advisor          # Gap analysis with action recommendations
│   ├── course_optimizer     # Course impact optimization
│   ├── roi_calculator       # Financial ROI analysis
│   └── calibrator           # Model calibration from real admission data
├── cli/main.py              # 15+ CLI commands
├── web/app.py               # Streamlit dashboard (6 pages)
├── tools/
│   ├── train_model_v2.py    # GPBoost training pipeline
│   ├── train_model.py       # v1 sklearn LR training
│   ├── advisor.py           # Claude-powered advisory report
│   ├── parse_profile.py     # Resume/transcript → YAML parser
│   ├── scrape_1p3a.py       # 1point3acres scraper + regex parser
│   └── collect_data.py      # GradCafe + QuantNet data collection
├── scripts/
│   ├── prepare_training_data.py  # Data cleaning + feature matrix generation
│   └── collect_multidim.py       # Multi-source data collection pipeline
├── data/
│   ├── programs/            # 28 program YAML files
│   ├── admissions/          # 12,800+ records (CSV + JSON)
│   └── models/              # GPBoost v2 (.bin + .json) + 21 LR models
└── tests/                   # 465 tests, <1s runtime
```

## Contributing

1. **Add/update program data** — PRs to `data/programs/` with updated deadlines or new programs
2. **Submit admission results** — Add your own data to improve the model
3. **Report data errors** — Open issues for outdated program information
4. **Feature requests** — Tell us what would help your application process

## License

MIT — see [LICENSE](LICENSE).

---

*QuantPath is a tool to assist with MFE application planning. All program data is sourced from official websites and QuantNet rankings. Always verify on official program websites before applying.*
