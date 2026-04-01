# QuantPath — Product Vision & Architecture

## What QuantPath Is

A data-driven graduate program application decision engine. Applicants input their profile, the system outputs competitiveness percentile per program, most-similar admitted applicant profiles, and gap diagnosis. Replaces traditional consultants (¥25,000–¥500,000+) with structured data and matching algorithms.

**Core insight**: consultants sell pattern recognition. QuantPath replaces personal experience with full-population data.

## Phased Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1 (MVP)** | MFE / MQF / MSFM programs | **In progress — core engine complete** |
| Phase 2 | Data Science / Statistics / OR masters | Planned |
| Phase 3 | CS masters (200+ programs, largest TAM) | Planned |

## Founder

Ethan Yang — UIUC undergraduate (CS + Economics + Statistics triple major, GPA 4.0, Class of 2027). Quantitative Researcher at Square Kettle LLC. Solo-authored prediction market pricing paper (Wang Transform, submitted to FMA 2026). Applying Fall 2027 MFE.

Ethan's own application process is the first use case and best content marketing asset.

Open-source track record: Open-source track record: oracle3 (126+ stars), clawdfolio (PyPI), Coinjure (12,000+ lines contributed)..

---

## Current State (as of March 2026)

### What's Built

**Core Engine** (15,000+ lines Python, 465 tests, all passing):
- `profile_evaluator` — 5-dimension scoring across 37 sub-factors (Math 30%, Stats 20%, CS 20%, Finance 15%, GPA 15%)
- `school_ranker` — Reach/target/safety classification using P(admit) thresholds
- `lr_predictor` — Per-program logistic regression with bias correction (21 trained models)
- `list_builder` — Portfolio optimization with geographic diversity enforcement
- `gap_advisor` — Gap analysis with priority-ranked action recommendations
- `course_optimizer` — Course impact optimization (37 categories including reinforcement learning)
- `prerequisite_matcher` — Maps coursework against each program's requirements
- `roi_calculator` — Tuition vs salary NPV analysis
- `timeline_generator` — Month-by-month application action plan
- `calibrator` — Model calibration from real admission data
- `interview_prep` — Question bank by category and difficulty

**Surfaces**:
- CLI with 15+ commands (`quantpath evaluate`, `list`, `match`, `gaps`, `compare`, `roi`, `whatif`, `portfolio`, etc.)
- Streamlit web dashboard (6 pages with Plotly charts)
- PDF report generation
- Claude AI advisory report (`tools/advisor.py`)

**Data Assets**:

| Dataset | Records | Source | Notes |
|---------|---------|--------|-------|
| Admission records | 6,984 | GradCafe, QuantNet | Accepted/rejected with GPA, GRE, bg_type |
| Offer results | 646 | offer.1point3acres.com API | AD/Reject with exact dates, 12 programs |
| LinkedIn alumni | 930 | Brave Search (Google-indexed profiles) | Employer, undergrad school, 20 programs |
| Offer backgrounds | 15 | offer.1point3acres.com modals | Structured GPA, school tier, research, internships |
| Parsed threads | 22 | 1point3acres forum posts | Regex-parsed admission results |
| BBS threads | 24 | 1point3acres BBS (logged in) | Forum narrative posts |
| Program database | 28 | QuantNet 2026 Rankings + official sites | Prerequisites, deadlines, salaries, class profiles |
| Trained LR models | 21 | Per-program from admission records | Bias-corrected P(admit) with confidence intervals |

**Tools**:
- `tools/scrape_1p3a.py` — 1point3acres forum scraper + regex parser (handles Chinese + English post formats)
- `tools/scrape_offer_backgrounds.py` — Playwright-based offer platform scraper
- `tools/collect_data.py` — QuantNet + GradCafe scraper (requests + BeautifulSoup + Claude parsing)
- `tools/scrape_gradcafe.py` — Regex-based GradCafe/QuantNet scraper (no AI dependency)
- `tools/parse_admissions.py` — Chinese forum post parser (regex, no API key needed)
- `tools/parse_profile.py` — Resume/transcript → YAML parser (Claude-powered)
- `tools/advisor.py` — Full AI advisory report generator
- `tools/train_model.py` — sklearn logistic regression training pipeline
- `tools/clean_data.py` — Data deduplication and cleanup

### What's NOT Built Yet

- Web frontend (Next.js or similar) — currently CLI + Streamlit only
- User accounts / data submission portal
- Database backend (PostgreSQL/Supabase) — currently file-based (YAML + CSV + JSON)
- Multi-feature admission model (planned: GPA + bg_tier + internships + research)
- Community features (discussion, alumni directory)
- B2B analytics dashboard
- Premium subscription system

---

## Product Architecture (Three Layers)

### Layer 1: Program Database

28 MFE programs with standardized fields. Stored as YAML in `data/programs/`.

```yaml
# Example: data/programs/cmu-mscf.yaml
id: cmu-mscf
name: MSCF
full_name: MS in Computational Finance
university: Carnegie Mellon University
department: Tepper School of Business
quantnet_ranking: 3
acceptance_rate: 0.172
class_size: 108
avg_gpa: 3.86
avg_base_salary: 134259
tuition_total: 120000
prerequisites:
  required:
    - {category: calculus, level: "I-III"}
    - {category: probability, level: calculus-based}
    - {category: programming, level: introductory}
  recommended:
    - {category: real_analysis}
    - {category: stochastic_calculus}
    - {category: numerical_methods}
  languages: [cpp, python, r]
deadline_rounds:
  - {round: 1, date: "2026-12-01"}
  - {round: 2, date: "2027-01-15"}
```

### Layer 2: Applicant Profile

Structured YAML profiles. Current schema in `core/models.py`:

```yaml
personal:
  name: "Applicant Name"
  university: "University"
  majors: ["CS", "Statistics"]
  gpa: 4.0
  is_international: true

courses:
  - {code: "STAT433", name: "Stochastic Processes", category: "stochastic_processes", grade: "A", level: 400}
  # 37 course categories across Math/Stats/CS/Finance

experience:
  - {type: internship, company: "Firm", title: "Quant Research Intern", duration_months: 3}

projects:
  - {name: "Project", description: "...", has_paper: true}
```

### Layer 3: Matching Engine

Currently implemented as:
1. **Profile scoring**: 5-dimension weighted evaluation → overall 0-10 score
2. **Admission prediction**: Per-program logistic regression → P(admit) with CI
3. **Prerequisite matching**: Course-by-course comparison → match percentage
4. **Gap analysis**: Identify missing prerequisites + prioritized recommendations
5. **School ranking**: Reach/target/safety classification based on P(admit) thresholds
6. **Portfolio optimization**: Greedy selection maximizing expected admits under budget

**Planned upgrade** (see `plans/` directory):
- Two-stage hierarchical model: per-program GPA base + global soft-feature adjustments (bg_tier, internship, research)
- LinkedIn data as bg_tier priors per program
- GRE demoted to threshold gate (not linear feature)

---

## Data Acquisition Strategy

All acquisition methods are authorized. Current pipelines:

| Method | Source | Records | Tools |
|--------|--------|---------|-------|
| GradCafe scraper | thegradcafe.com | 6,700+ | `tools/scrape_gradcafe.py` (regex) |
| QuantNet scraper | quantnet.com | 250+ | `tools/collect_data.py` (requests + Claude) |
| 1point3acres API | offer.1point3acres.com | 646 | XHR interception via Playwright |
| 1point3acres forum | bbs.1point3acres.com | 100+ threads | `tools/scrape_1p3a.py` (Playwright + regex) |
| LinkedIn search | Google-indexed profiles | 930 | Brave Search API, 7 rounds of targeted queries |
| Offer modal scraper | offer.1point3acres.com | 15 | Playwright click-through with member login |
| Chinese forum parser | 小红书, ChaseDream, OfferShow | 7 | `tools/parse_admissions.py` (Claude) |
| Official program sites | University websites | 28 programs | Manual + structured YAML |

**Cold-start executed**: Started from 0, now at 8,500+ records across 5 sources.

**Next data targets**:
- ChaseDream MFE 录取汇报 (structured template format, high quality)
- 小红书 MFE admission posts (public, rich background data)
- Reddit r/QuantFinance and r/gradadmissions
- More LinkedIn rounds with new query strategies
- User-submitted data once web frontend launches

---

## Target Tech Stack (Web Product)

**Current** (CLI/Local):
```
Python 3.10+ → CLI (argparse + rich) + Streamlit dashboard
Data: YAML + CSV + JSON files
Models: sklearn → JSON (no pickle)
Tests: pytest (465 tests, <1s)
CI: GitHub Actions (Python 3.11 + 3.12)
```

**Planned** (Web SaaS):
```
Frontend:    Next.js 14+ / App Router + TypeScript + Tailwind + shadcn/ui
Backend:     Python FastAPI (ML pipeline) + Next.js API Routes (CRUD)
Database:    PostgreSQL (Supabase) with Row Level Security
Auth:        Supabase Auth (email + Google + LinkedIn OAuth)
ML:          Current sklearn pipeline → FastAPI endpoint
Hosting:     Vercel (frontend) + Railway/Fly.io (FastAPI)
Analytics:   PostHog (product) + Sentry (errors)
```

Why this stack: Next.js for SEO (applicants search "CMU MSCF admission stats"), Supabase for rapid auth/DB, FastAPI for Python ML pipeline compatibility, separate frontend and ML backend for independent iteration.

---

## Monetization

```
Free:     Browse programs, basic matching (GPA + GRE), 3 queries/month
Premium:  $15-30/month — unlimited matching, multi-dimensional scoring,
          similar applicant profiles, gap diagnosis, deadline reminders
B2B:      $50K-500K/year — enrollment analytics for MFE programs
          (yield prediction, applicant pipeline insights)
```

Priority: user growth first, monetization later. Free tier must be genuinely useful.

---

## Competitive Landscape

| Competitor | Strength | QuantPath Advantage |
|-----------|----------|-------------------|
| QuantNet rankings | Brand, community | Personalized matching (not just rankings) |
| GradCafe | 840K+ data points | Structured data (vs free text), course matching |
| 一亩三分地 | Chinese user base | English-first, global programs, no paywall |
| Admissionado / consulting | Personal guidance | 100x cheaper, data-driven, instant results |
| ChaseDream | Chinese MFE community | Automated analysis vs manual forum posts |

**Moat**: Data network effects — more users submit data → better matching → more users. Course-level prerequisite matching is unique (no competitor does this).

---

## Code Conventions

- Python: snake_case, type annotations, numpy-style docstrings
- TypeScript (future): camelCase, functional components, Zod validation
- Git: conventional commits (feat/fix/docs/refactor)
- Testing: all core logic must have unit tests (pytest)
- Privacy: user data anonymized by default, PII only visible to owner
- Models: JSON format (no pickle), portable across platforms
- Dependencies: minimize core deps (PyYAML + rich only for engine)
