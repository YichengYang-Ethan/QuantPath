# QuantPath

**MFE admission prediction engine** — GPBoost mixed-effects model on 13,100+ records across 31 programs. Profile scoring, prerequisite matching, and data-driven school ranking.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
[![CI](https://github.com/MasterAgentAI/QuantPath/actions/workflows/ci.yml/badge.svg)](https://github.com/MasterAgentAI/QuantPath/actions)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE-AGPL-3.0)
[![Data: CC BY-NC-SA 4.0](https://img.shields.io/badge/Data-CC_BY--NC--SA_4.0-lightgrey.svg)](data/LICENSE)
![Tests](https://img.shields.io/badge/tests-465%20passed-brightgreen.svg)
![Data](https://img.shields.io/badge/data-13%2C100%2B%20records-blue.svg)
![Programs](https://img.shields.io/badge/programs-15%20focused%20(31%20total)-orange.svg)
![Model](https://img.shields.io/badge/model-AUC%200.723-brightgreen.svg)

---

## Why QuantPath

Consulting services charge **$4,500--10,000 per school**. QuantPath provides the same analysis for free -- multi-dimensional profile scoring, admission probability with confidence intervals, and personalized gap analysis across 15 focused MFE programs (31 total in database).

```
$ quantpath predict --profile my_profile.yaml

  QuantPath Admission Prediction (v2 Model)
  Sample Applicant | Top 30 University | GPA 4.0 | International

  Program                P(admit)   Category
  ─────────────────────────────────────────────
  Princeton MFin           18%      reach
  Baruch MFE               22%      reach
  Berkeley MFE             35%      reach
  CMU MSCF                 38%      reach
  MIT MFin                 25%      reach
  Columbia FE              42%      target
  Yale AM                  15%      reach
  Stanford MCF             20%      reach
  UChicago MSFM            55%      target
  NYU Courant              52%      target
  Columbia MSFE            48%      target
  Cornell MFE              58%      target
  Columbia MAFN            62%      target
  NYU Tandon MFE           72%      safety
  GaTech QCF               75%      safety

  15 programs evaluated (Tier 0 + Tier 1)
```

## Data

| Dataset | Records | Source |
|---------|---------|--------|
| Admission records | **13,100+** | GradCafe, QuantNet, Reddit, 1point3acres, ChaseDream (accepted/rejected/waitlisted) |
| LinkedIn alumni | **930** | 20 MFE programs (employer, undergrad school, graduation year) |
| Program database | **31** | QuantNet 2026 Rankings + official sites (prerequisites, deadlines, salaries) |

## Model

**v1 (primary)**: Per-program logistic regression on GPA + GRE Quant with bias correction and profile adjustments for undergrad tier, internship quality, research, and major relevance. 27 trained models covering all 15 focused programs.

**v2 (fallback)**: GPBoost -- LightGBM gradient boosting with per-program random intercepts. Trained on 11,100+ labeled records, 13 features, 31 programs. AUC 0.723, Brier 0.206 (5-fold CV). Used for programs without a v1 model.

**Bias correction**: Self-reported data has survivor bias (65% accept rate in data vs 4-30% real). The model replaces the biased intercept with logit(*r*) where *r* is the official acceptance rate, preserving learned feature slopes ([King & Zeng 2001](https://gking.harvard.edu/files/abs/0s-abs.shtml)).

## Features

| Command | Description |
|---------|-------------|
| **`quantpath predict`** | **Primary entry point** -- P(admit) for 15 focused programs, reach/target/safety (no transcript needed) |
| `quantpath evaluate` | Profile assessment -- 5-dimension score (37 sub-factors) with gaps and strengths |
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
| `quantpath contribute-upload` | Upload locally saved contribution data to GitHub |

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

# Predict admission probability across 15 focused programs (primary entry point)
quantpath predict --profile my_profile.yaml

# Detailed course evaluation (needs full transcript)
quantpath evaluate --profile my_profile.yaml
quantpath gaps     --profile my_profile.yaml
```

For AI-powered analysis:
```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key
python tools/advisor.py --profile my_profile.yaml --save report.md
```

## Programs (15 Focused — Tier 0 + Tier 1)

**Tier 0** (elite, <10% acceptance or unique positioning):

| # | Program | University | Class | Rate | Avg GPA | Salary |
|---|---------|-----------|-------|------|---------|--------|
| 1 | MFin | Princeton | 44 | 5% | 3.90 | $160K |
| 2 | MFE | Baruch College | 20 | 4% | 3.84 | $179K |
| 3 | MFE | UC Berkeley | 76 | 17% | 3.80 | $154K |
| 4 | MSCF | Carnegie Mellon | 108 | 17% | 3.86 | $134K |
| 5 | MFin | MIT Sloan | 126 | 8% | 3.80 | $140K |
| 6 | MSFE | Columbia FE (Econ) | 25 | 5% | 3.90 | $150K |
| 7 | AM | Yale | 3 | 5% | 3.90 | $145K |
| 8 | MCF | Stanford | 10 | 5% | 3.90 | -- |

**Tier 1** (highly competitive, strong placement):

| # | Program | University | Class | Rate | Avg GPA | Salary |
|---|---------|-----------|-------|------|---------|--------|
| 9 | MSFM | UChicago | 118 | 22% | 3.80 | $124K |
| 10 | MathFin | NYU Courant | 37 | 23% | 3.85 | $126K |
| 11 | MSFE | Columbia (IEOR) | 136 | 13% | 3.90 | $138K |
| 12 | MFE | Cornell | 53 | 21% | 3.80 | $115K |
| 13 | MAFN | Columbia (Math) | 101 | 22% | 3.80 | $123K |
| 14 | MFE | NYU Tandon | 154 | 38% | 3.83 | $107K |
| 15 | QCF | Georgia Tech | 99 | 30% | 3.75 | $115K |

Full 31-program database with deadlines, prerequisites, essay requirements, and interview formats in `data/programs/`.

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
│   ├── programs/            # 31 program YAML files
│   ├── admissions/          # 13,100+ records (CSV + JSON)
│   └── models/              # 27 LR models (primary) + GPBoost v2 (.bin + .json)
└── tests/                   # 465 tests, <1s runtime
```

## Contributing

1. **Add/update program data** — PRs to `data/programs/` with updated deadlines or new programs
2. **Submit admission results** — see [Contributing Your Data](#contributing-your-data) below
3. **Report data errors** — Open issues for outdated program information
4. **Feature requests** — Tell us what would help your application process

### Contributing Your Data

After running `quantpath predict`, you'll be asked whether to contribute your anonymized data. This is the #1 way to improve prediction accuracy for everyone.

**Option A — Automatic (recommended):**
```bash
quantpath predict --profile my_profile.yaml
# Answer "y" when prompted → auto-submits to GitHub
```

**Option B — Browser (no GitHub CLI needed):**

If you don't have `gh` installed, the CLI will save your data locally and offer to open a pre-filled GitHub issue page in your browser. Just click "Submit new issue".

**Option C — Upload later:**
```bash
# If data was saved locally (e.g. gh was not available), submit it later:
quantpath contribute-upload
```

**Privacy:** All data is anonymized before submission — university names are replaced with tiers (e.g. "US T30"), company names with categories (e.g. "top quant"). You choose per-field whether to share the original value or the anonymized version.

## License

This project uses a multi-license structure to balance openness with protection:

| Component | License | Commercial Use |
|-----------|---------|----------------|
| **Source code** | [AGPL-3.0](LICENSE-AGPL-3.0) | Allowed if you release your source code (including SaaS) |
| **Admission data & models** | [CC BY-NC-SA 4.0](data/LICENSE) | Not permitted without written agreement |
| **Program database** | CC BY-SA 4.0 | Allowed with attribution |

**In plain language:**
- **Personal use, academic research, and learning** — fully permitted, no restrictions.
- **Contributing and forking** — welcome under the same license terms.
- **Building a commercial product or SaaS** — the AGPL requires you to open-source your entire codebase. The data and models cannot be used commercially without a separate license.
- **Redistribution of data** — allowed for non-commercial purposes with attribution and ShareAlike.

For commercial licensing inquiries, please [open an issue](https://github.com/MasterAgentAI/QuantPath/issues) or contact MasterAgentAI.

See [LICENSE](LICENSE) for the complete terms and [NOTICE](NOTICE) for attribution details.

---

*QuantPath is a tool to assist with MFE application planning. All program data is sourced from official websites and QuantNet rankings. Always verify on official program websites before applying.*
