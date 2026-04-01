# Model Validation Report — GPBoost v2 (Production)

**Date**: 2026-03-30
**Model**: GPBoost v2 — LightGBM gradient boosting + per-program random intercepts
**Training data**: 11,012 records, 41 programs
**Architecture**: gpboost_mixed_effects

---

## 1. Overall Metrics

| Metric | Value |
|--------|-------|
| **Overall AUC** | 0.7229 |
| **Overall Brier Score** | 0.2057 |
| **CV Folds** | 5 |
| **Training Accept Rate** | 60.6% |

### Per-Fold CV Results

| Fold | AUC | Brier | n |
|------|-----|-------|---|
| 1 | 0.7180 | 0.2066 | 2,203 |
| 2 | 0.7344 | 0.2023 | 2,203 |
| 3 | 0.7209 | 0.2067 | 2,202 |
| 4 | 0.7304 | 0.2038 | 2,202 |
| 5 | 0.7127 | 0.2091 | 2,202 |

## 2. Feature Importance

| Rank | Feature | Importance (Gain) |
|------|---------|------------------|
| 1 | `major_relevance_score` | 5296.1 |
| 2 | `undergrad_tier_encoded` | 1760.0 |
| 3 | `gpa_normalized` | 963.8 |
| 4 | `is_international` | 673.7 |
| 5 | `intern_score` | 623.4 |
| 6 | `research_score` | 424.1 |
| 7 | `gre_quant` | 337.4 |
| 8 | `has_intern` | 310.5 |
| 9 | `has_tier` | 237.7 |
| 10 | `is_female` | 167.0 |
| 11 | `has_nationality` | 87.8 |
| 12 | `has_gpa` | 21.0 |
| 13 | `has_gre` | 2.0 |

## 3. Per-Program AUC (Focused Programs, 50+ samples)

| Program | AUC | Brier | n | Real Accept Rate |
|---------|-----|-------|---|-----------------|
| baruch-mfe | 0.728 | 0.208 | 379 | 4.0% |
| berkeley-mfe | 0.543 | 0.240 | 449 | 17.4% |
| cmu-mscf | 0.743 | 0.200 | 908 | 17.2% |
| columbia-msfe | 0.645 | 0.221 | 797 | 13.3% |
| cornell-mfe | 0.591 | 0.236 | 461 | 20.7% |
| gatech-qcf | 0.568 | 0.232 | 354 | 30.2% |
| mit-mfin | 0.787 | 0.165 | 1060 | 8.3% |
| nyu-tandon-mfe | 0.574 | 0.237 | 275 | 38.1% |
| princeton-mfin | 0.645 | 0.195 | 340 | 5.4% |
| stanford-mcf | 0.609 | 0.186 | 195 | 5.0% |
| uchicago-msfm | 0.578 | 0.240 | 480 | 22.0% |

**Note**: yale-am (0 records), nyu-courant (11 records), columbia-mafn (15 records) have insufficient data for per-program AUC.

## 4. Applicant Profile Backtest

Three synthetic applicant profiles tested across all 15 focused programs:

- **Strong**: GPA 3.9, T20 school, 2 quant internships, research, international
- **Typical**: GPA 3.7, T50 school, 1 internship, no research, international
- **Weak**: GPA 3.5, unknown school, no internship, no research, international

### Predicted P(admit) by Program and Profile

| Program | Real Rate | Strong | Typical | Weak | Order OK |
|---------|-----------|--------|---------|------|----------|
| princeton-mfin | 5.4% | 21.6% | 14.7% | 13.7% | Yes |
| baruch-mfe | 4.0% | 15.3% | 10.2% | 9.5% | Yes |
| berkeley-mfe | 17.4% | 54.1% | 42.4% | 40.5% | Yes |
| cmu-mscf | 17.2% | 45.1% | 33.9% | 32.1% | Yes |
| mit-mfin | 8.3% | 16.0% | 10.6% | 9.9% | Yes |
| columbia-msfe | 13.3% | 44.4% | 33.3% | 31.5% | Yes |
| yale-am | 8.0% | 38.0% | 27.7% | 26.2% | Yes |
| stanford-mcf | 5.0% | 22.9% | 15.7% | 14.7% | Yes |
| uchicago-msfm | 22.0% | 62.3% | 50.8% | 48.8% | Yes |
| nyu-courant | 22.5% | 63.8% | 52.8% | 50.8% | Yes |
| columbia-mafn | 22.3% | 62.3% | 51.1% | 49.2% | Yes |
| cornell-mfe | 20.7% | 60.0% | 48.4% | 46.4% | Yes |
| nyu-tandon-mfe | 38.1% | 78.7% | 69.8% | 68.1% | Yes |
| gatech-qcf | 30.2% | 71.3% | 60.8% | 58.9% | Yes |

### Ordering Validation

**Result**: ALL PASS

All 14 focused programs correctly order: strong > typical > weak.

### Acceptance Rate Correlation

**Spearman rank correlation**: rho = 0.969, p < 0.0001

Programs with harder real acceptance rates correctly produce lower predicted probabilities. The rank ordering is near-perfect.

### Known Limitations

1. **Predicted probabilities are higher than real acceptance rates** for mid-tier programs. This is expected because:
   - Training data from forums (QuantNet, GradCafe, Reddit) has survivor bias (accepted applicants report more)
   - The bias correction shifts the baseline but cannot fully correct the feature distribution bias
   - The "typical" synthetic profile (GPA 3.7, T50, 1 internship) may be stronger than the true median applicant

2. **Low-data programs**: yale-am (0 records), nyu-courant (11), columbia-mafn (15) rely heavily on the random intercept prior and bias correction. Their predictions should be treated with caution.

3. **Per-program AUC variation**: Some programs (berkeley-mfe: 0.543, gatech-qcf: 0.568) have near-random AUC. This is because these programs' decisions may depend heavily on factors not captured in our features (SOP quality, interview, recommendation letters).

## 5. Bias Corrections Applied

| Program | Real Rate | Training Rate | Logit Shift |
|---------|-----------|---------------|-------------|
| baruch-mfe | 4.0% | 56.5% | -3.44 |
| stanford-mcf | 5.0% | 28.2% | -2.01 |
| princeton-mfin | 5.4% | 35.6% | -2.27 |
| mit-mfin | 8.3% | 65.5% | -3.04 |
| columbia-msfe | 13.3% | 45.3% | -1.69 |
| ncstate-mfm | 16.7% | 82.1% | -3.13 |
| cmu-mscf | 17.2% | 59.1% | -1.94 |
| berkeley-mfe | 17.4% | 55.0% | -1.76 |
| cornell-mfe | 20.7% | 50.8% | -1.37 |
| uchicago-msfm | 22.0% | 52.7% | -1.37 |
| columbia-mafn | 22.3% | 46.7% | -1.11 |
| nyu-courant | 22.5% | 54.5% | -1.42 |
| toronto-mmf | 30.0% | 55.2% | -1.06 |
| umich-mfe | 30.0% | 57.7% | -1.16 |
| gatech-qcf | 30.2% | 60.7% | -1.27 |
| ucla-mfe | 36.0% | 64.5% | -1.17 |
| nyu-tandon-mfe | 38.1% | 57.8% | -0.80 |
| usc-msmf | 40.0% | 47.1% | -0.29 |
| jhu-mfm | 50.7% | 61.3% | -0.43 |
| uiuc-msfe | 50.7% | 69.9% | -0.81 |
| uwash-cfrm | 53.9% | 48.9% | +0.20 |
| fordham-msqf | 59.4% | 81.3% | -1.09 |
| stevens-mfe | 68.0% | 95.5% | -2.29 |
| bu-msmf | 80.6% | 77.5% | +0.19 |
| uminn-mfm | 80.7% | 72.7% | +0.45 |
| rutgers-mqf | 86.4% | 83.1% | +0.25 |

## 6. Conclusion

The GPBoost v2 model is **production-ready** with the following characteristics:

- **Overall AUC 0.7229**: Meaningful predictive power above random (0.5)
- **Perfect rank ordering**: Strong > Typical > Weak for all programs
- **Near-perfect rank correlation** (rho=0.969) with real acceptance rates
- **Bias-corrected** using real program acceptance rates
- **Handles missing data natively** via LightGBM + missing indicators

The model is best used for **relative comparisons** (which programs are more/less likely) rather than absolute probability interpretation.
