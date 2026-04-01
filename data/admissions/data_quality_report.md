# MFE Admission Data Quality Report

**Generated**: 2026-04-01 02:28:35
**Total Records**: 12868

## Records by Source

| Source | Count | % |
|--------|------:|--:|
| github-gc | 4303 | 33.4% |
| reddit | 3393 | 26.4% |
| quantnet_tracker | 3042 | 23.6% |
| gradcafe | 1740 | 13.5% |
| 1p3a-offer | 143 | 1.1% |
| linkedin | 114 | 0.9% |
| 1p3a-bbs | 43 | 0.3% |
| 1p3a-raw | 27 | 0.2% |
| 1p3a-thread | 22 | 0.2% |
| 1p3a-bg | 14 | 0.1% |
| chasedream | 7 | 0.1% |
| 小红书 | 7 | 0.1% |
| quantnet | 6 | 0.0% |
| offershow | 4 | 0.0% |
| 1p3a-member | 3 | 0.0% |

## Quality Tier Distribution

| Tier | Description | Count | % |
|------|-------------|------:|--:|
| A (Gold) | 4+ rich fields + result | 764 | 5.9% |
| B (Silver) | 2-3 rich fields + result | 2470 | 19.2% |
| C (Bronze) | GPA/GRE + result | 3312 | 25.7% |
| D (Basic) | Result only or no result | 6322 | 49.1% |

## Tier Distribution by Source

| Source | A | B | C | D | Total |
|--------|--:|--:|--:|--:|------:|
| github-gc | 0 | 448 | 221 | 3634 | 4303 |
| reddit | 715 | 1606 | 1 | 1071 | 3393 |
| quantnet_tracker | 0 | 0 | 2832 | 210 | 3042 |
| gradcafe | 0 | 306 | 258 | 1176 | 1740 |
| 1p3a-offer | 0 | 0 | 0 | 143 | 143 |
| linkedin | 14 | 68 | 0 | 32 | 114 |
| 1p3a-bbs | 6 | 25 | 0 | 12 | 43 |
| 1p3a-raw | 0 | 9 | 0 | 18 | 27 |
| 1p3a-thread | 5 | 8 | 0 | 9 | 22 |
| 1p3a-bg | 0 | 0 | 0 | 14 | 14 |
| chasedream | 7 | 0 | 0 | 0 | 7 |
| 小红书 | 7 | 0 | 0 | 0 | 7 |
| quantnet | 6 | 0 | 0 | 0 | 6 |
| offershow | 4 | 0 | 0 | 0 | 4 |
| 1p3a-member | 0 | 0 | 0 | 3 | 3 |

## Field Coverage

| Field | Records with data | Coverage % |
|-------|------------------:|-----------:|
| program | 12301 | 95.6% |
| result | 11012 | 85.6% |
| season | 8751 | 68.0% |
| gpa | 4120 | 32.0% |
| gpa_scale | 4413 | 34.3% |
| gre_quant | 997 | 7.7% |
| gre_verbal | 1009 | 7.8% |
| toefl | 474 | 3.7% |
| undergrad_school | 2607 | 20.3% |
| undergrad_tier | 1834 | 14.3% |
| undergrad_country | 1859 | 14.4% |
| major | 3201 | 24.9% |
| major_relevance | 3201 | 24.9% |
| intern_count | 1129 | 8.8% |
| intern_level | 1429 | 11.1% |
| intern_relevance | 1429 | 11.1% |
| has_paper | 418 | 3.2% |
| has_research | 1094 | 8.5% |
| research_level | 12868 | 100.0% |
| gender | 440 | 3.4% |
| nationality | 5121 | 39.8% |

## Top 25 Programs by Record Count

| Program | Count |
|---------|------:|
| finance-unknown | 2105 |
| bu-msmf | 1306 |
| mit-mfin | 1140 |
| cmu-mscf | 1030 |
| columbia-msfe | 968 |
| uchicago-msfm | 545 |
| berkeley-mfe | 518 |
| cornell-mfe | 509 |
| baruch-mfe | 397 |
| gatech-qcf | 374 |
| princeton-mfin | 374 |
| uiuc-msfe | 337 |
| nyu-tandon-mfe | 270 |
| or-unknown | 219 |
| ucla-mfe | 213 |
| stanford-mcf | 212 |
| toronto-mmf | 208 |
| mfe-unknown | 203 |
| uwash-cfrm | 187 |
| rutgers-mqf | 181 |
| usc-msmf | 169 |
| ncstate-mfm | 155 |
| michigan-qfr | 141 |
| jhu-mfm | 106 |
| nyu-mfe | 104 |

## Result Distribution

| Result | Count | % |
|--------|------:|--:|
| accepted | 6576 | 59.7% |
| rejected | 3949 | 35.9% |
| waitlisted | 487 | 4.4% |

## Top Seasons

| Season | Count |
|--------|------:|
| 17Fall | 969 |
| 18Fall | 966 |
| 16Fall | 820 |
| 19Fall | 682 |
| 15Fall | 603 |
| 14Fall | 562 |
| 11Fall | 496 |
| 12Fall | 478 |
| 13Fall | 460 |
| 25Fall | 393 |

## Model-Readiness Summary

- **Tier A+B (model-ready with rich features)**: 3234 records (25.1%)
- **Tier C (basic features)**: 3312 records (25.7%)
- **Tier D (needs enrichment)**: 6322 records (49.1%)
- **Records with accept/reject/waitlist label**: 11012 (85.6%)

## Training Data Files

| File | Records | Description |
|------|--------:|-------------|
| training_data_full.csv | 12868 | All cleaned records |
| training_data_model.csv | 6432 | Tier A+B+C with accept/reject |
| training_data_rich.csv | 3234 | Tier A+B only (multi-feature) |
| feature_matrix.csv | 10525 | Numeric feature matrix for sklearn |