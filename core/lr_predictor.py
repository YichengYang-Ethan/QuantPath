# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""Logistic regression admission probability predictor.

Loads pre-trained per-program models from data/models/admission_models.json
and provides P(admission) predictions given GPA, GRE Quant, and optional
profile signals.

Key design decisions
--------------------
Bias correction
    Training data (GradCafe / QuantNet) is self-reported and has severe
    survivor bias: Baruch's observed accept_rate in training = 34.8% but
    the real rate is 4%.  We correct this by replacing the biased intercept
    with logit(real_accept_rate), anchoring the average-applicant prediction
    to the true baseline while preserving the relative GPA/GRE slope.

    Formula:  logit = coef[0]*z_gpa + coef[1]*z_gre + logit(real_accept_rate)

Profile adjustments (applied in logit space)
    is_international (Chinese / Asian national): -0.25  (~-5% at p=0.3)
    1 quant internship:                          +0.10
    2+ quant internships:                        +0.20

Confidence interval
    Approximated from sample size (n) and AUC using:
        SE_logit ≈ 1.96 / sqrt(n_eff)
        n_eff    = n * (2*AUC - 1)^2  (Bamber's index)
    Bounds are clamped to [0, 1].
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .models import UserProfile

_MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "admission_models.json"
_V2_MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "admission_model_v2.json"
_V2_BIN_PATH = Path(__file__).parent.parent / "data" / "models" / "admission_model_v2.bin"
_models: dict | None = None
_v2_model: Any = None
_v2_meta: dict | None = None

# Profile-signal adjustments (logit space)
_ADJ_INTERNATIONAL = -0.25


@dataclass
class AdmitPrediction:
    """Full admission probability prediction with uncertainty bounds."""

    prob: float          # bias-corrected, profile-adjusted P(admit)
    prob_low: float      # lower bound of ~90% approximate CI
    prob_high: float     # upper bound
    is_bias_corrected: bool  # True when real_accept_rate was available


def _load_models() -> dict:
    global _models
    if _models is None:
        if _MODEL_PATH.exists():
            with _MODEL_PATH.open(encoding="utf-8") as f:
                _models = json.load(f)
        else:
            _models = {}
    return _models


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def _ci_half_width(n: int, auc: float, p: float) -> float:
    """Approximate 90% CI half-width in logit space.

    Uses Bamber's effective-N: n_eff = n * (2*AUC - 1)^2
    SE_logit ≈ 1.645 / sqrt(n_eff * p * (1-p))
    """
    gain = max(0.0, 2 * auc - 1)
    n_eff = max(5.0, n * gain ** 2)
    variance = p * (1 - p)
    if variance <= 0:
        return 2.0
    se_logit = 1.645 / math.sqrt(n_eff * variance)
    return min(se_logit, 3.0)   # cap extreme widths


def _compute_logit(
    m: dict,
    gpa: Optional[float],
    gre: Optional[float],
) -> tuple[float, bool]:
    """Compute raw logit and whether bias correction was applied."""
    means = m["means"]
    stds = m["stds"]
    coef = m["coef"]

    gpa_val = gpa if gpa is not None else means[0]
    gre_val = gre if gre is not None else means[1]

    z_gpa = (gpa_val - means[0]) / stds[0]
    z_gre = (gre_val - means[1]) / stds[1]

    feature_logit = coef[0] * z_gpa + coef[1] * z_gre

    real_rate = m.get("real_accept_rate")
    if real_rate is not None and 0 < real_rate < 1:
        # Bias-corrected intercept: anchor baseline to real accept rate
        intercept = _logit(real_rate)
        corrected = True
    else:
        intercept = m["intercept"]
        corrected = False

    return feature_logit + intercept, corrected


def _profile_adjustment(profile: "UserProfile") -> float:
    """Compute logit adjustment from profile signals.

    Uses the same tiered scoring as v2 feature extraction but returns a
    single logit-space adjustment.  Calibrated against actual admission
    outcomes for GPA 3.9+ international applicants in training data
    (n=115, survivor-bias-corrected at 0.55x).
    """
    adj = 0.0

    # --- Nationality ---
    if getattr(profile, "is_international", False):
        adj += _ADJ_INTERNATIONAL

    # --- Undergrad tier (+0 to +0.40) ---
    # US schools carry significantly more weight than Chinese schools
    # in US MFE admissions (familiarity, grading system, network).
    uni = getattr(profile, "university", "").lower()
    if uni:
        _T10 = ["mit", "stanford", "caltech", "princeton", "harvard",
                 "chicago", "penn", "columbia", "berkeley", "yale"]
        _T20 = ["cornell", "carnegie mellon", "cmu", "duke", "northwestern",
                 "johns hopkins", "rice", "vanderbilt", "ucla", "michigan"]
        _T30 = ["nyu", "uiuc", "illinois", "georgia tech", "gatech",
                 "wisconsin", "purdue", "unc"]
        _TOP_INTL = ["tsinghua", "peking", "imperial", "lse", "oxford",
                      "cambridge", "eth", "epfl"]
        _C9 = ["zhejiang", "fudan", "sjtu", "shanghai jiao tong",
                "ustc", "nanjing"]
        _985 = ["wuhan", "sun yat-sen", "huazhong", "sichuan", "tianjin",
                 "southeast", "dalian"]
        if any(s in uni for s in _T10):
            adj += 0.40
        elif any(s in uni for s in _T20):
            adj += 0.33
        elif any(s in uni for s in _TOP_INTL):
            adj += 0.25
        elif any(s in uni for s in _T30):
            adj += 0.20
        elif any(s in uni for s in _C9):
            adj += 0.18
        elif any(s in uni for s in _985):
            adj += 0.08

    # --- Internship quality (+0 to +0.55) ---
    work_exps = getattr(profile, "work_experience", [])
    internships = [
        e for e in work_exps
        if isinstance(e, dict) and e.get("type") == "internship"
    ]
    if internships:
        _US_TOP_QUANT = [
            "citadel", "jane street", "two sigma", "de shaw", "hrt",
            "hudson river", "jump", "imc", "optiver", "sig",
            "susquehanna", "five rings", "tower research",
        ]
        _US_QUANT = [
            "aqr", "point72", "millennium", "bridgewater", "worldquant",
            "man group", "balyasny", "cubist", "drw", "squarepoint",
        ]
        _US_BB = [
            "goldman", "morgan stanley", "jpmorgan", "jp morgan",
            "bank of america", "citi", "barclays", "ubs", "deutsche",
        ]
        best = 0.0
        for exp in internships:
            combined = (
                str(exp.get("company", "")) + " "
                + str(exp.get("description", "")) + " "
                + str(exp.get("title", ""))
            ).lower()
            score = 0.08  # generic internship
            if any(f in combined for f in _US_TOP_QUANT):
                score = 0.40
            elif any(f in combined for f in _US_QUANT):
                score = 0.30
            elif any(f in combined for f in _US_BB):
                score = 0.23
            elif "quant" in combined or "trading" in combined:
                score = 0.20
            best = max(best, score)
        adj += best + min(len(internships) - 1, 3) * 0.08
    elif any(isinstance(e, dict) and e.get("type") == "research"
             for e in work_exps):
        adj += 0.05

    # --- Research & publications (+0 to +0.40) ---
    projects = getattr(profile, "projects", [])
    has_paper = any(
        isinstance(p, dict) and p.get("has_paper")
        for p in projects
    )
    has_research = len(projects) > 0 or any(
        isinstance(e, dict) and e.get("type") == "research"
        for e in work_exps
    )
    if has_paper:
        adj += 0.20
    elif has_research:
        adj += 0.10

    # --- Major relevance (+0 to +0.15) ---
    majors = getattr(profile, "majors", [])
    if majors:
        _QUANT_MAJORS = [
            "math", "mathematics", "applied math",
            "statistics", "stats", "stat",
            "physics",
            "computer science", "computer", "computing", "cs",
        ]
        _ECON_MAJORS = ["econ", "economics", "finance", "financial"]
        n_quant = sum(
            1 for m in majors
            if any(kw == m.lower() or kw in m.lower().split() for kw in _QUANT_MAJORS)
        )
        if n_quant >= 2:
            adj += 0.15
        elif n_quant >= 1:
            adj += 0.08
        elif any(
            any(kw in m.lower() for kw in _ECON_MAJORS)
            for m in majors
        ):
            adj += 0.05

    return adj


def predict_prob_full(
    program_id: str,
    gpa: Optional[float],
    gre: Optional[float],
    profile: Optional["UserProfile"] = None,
) -> Optional[AdmitPrediction]:
    """Return a full AdmitPrediction with CI bounds and profile adjustments.

    Returns None if the program has no trained model or both GPA and GRE
    are missing.
    """
    models = _load_models()
    m = models.get(program_id)
    if not m:
        return None
    if gpa is None and gre is None:
        return None

    raw_logit, corrected = _compute_logit(m, gpa, gre)

    # Profile signal adjustment
    if profile is not None:
        raw_logit += _profile_adjustment(profile)

    prob = round(_sigmoid(raw_logit), 4)

    # Confidence interval in logit space → probability space
    hw = _ci_half_width(m["n_total"], m["auc"], prob)
    prob_low  = round(max(0.0, _sigmoid(raw_logit - hw)), 4)
    prob_high = round(min(1.0, _sigmoid(raw_logit + hw)), 4)

    return AdmitPrediction(
        prob=prob,
        prob_low=prob_low,
        prob_high=prob_high,
        is_bias_corrected=corrected,
    )


def predict_prob(
    program_id: str,
    gpa: Optional[float],
    gre: Optional[float],
    profile: Optional["UserProfile"] = None,
) -> Optional[float]:
    """Return bias-corrected P(admission). Backward-compatible scalar form.

    Returns None if the program has no trained model or inputs are missing.
    """
    result = predict_prob_full(program_id, gpa, gre, profile)
    return result.prob if result is not None else None


def get_model_stats(program_id: str) -> Optional[dict]:
    """Return model stats (n, accept_rate, AUC, GPA/GRE percentiles) for a program."""
    return _load_models().get(program_id)


def has_model(program_id: str) -> bool:
    return program_id in _load_models()


# ===================================================================
# V2 ENGINE — GPBoost mixed-effects model
# ===================================================================

def _load_v2() -> tuple[Any, dict]:
    """Load GPBoost v2 model (binary + metadata).

    Returns (None, None) if gpboost or numpy are not installed.
    """
    global _v2_model, _v2_meta
    if _v2_model is None and _V2_BIN_PATH.exists():
        try:
            import gpboost as gpb
            _v2_model = gpb.Booster(model_file=str(_V2_BIN_PATH))
            with _V2_MODEL_PATH.open(encoding="utf-8") as f:
                _v2_meta = json.load(f)
        except (ImportError, OSError):
            _v2_model = None
            _v2_meta = None
    return _v2_model, _v2_meta


def _extract_v2_features(
    program_id: str,
    gpa: Optional[float],
    gre: Optional[float],
    profile: Optional["UserProfile"],
    meta: dict,
) -> Optional[Any]:
    """Build the 13-feature vector for GPBoost v2 prediction."""
    import numpy as np

    # Map undergrad tier from profile
    undergrad_tier = np.nan
    intern_score = np.nan
    research_score = np.nan
    is_international = np.nan
    is_female = np.nan
    major_relevance = np.nan

    if profile is not None:
        # International
        if hasattr(profile, "is_international"):
            is_international = 1.0 if profile.is_international else 0.0

        # Undergrad tier from university name
        uni = getattr(profile, "university", "").lower()
        if uni:
            # T10
            _T10 = ["mit", "stanford", "caltech", "princeton", "harvard",
                     "chicago", "penn", "columbia", "berkeley", "yale"]
            # T20
            _T20 = ["cornell", "carnegie mellon", "duke", "northwestern",
                     "johns hopkins", "rice", "vanderbilt", "wash u",
                     "ucla", "michigan", "cmu"]
            # T30
            _T30 = ["nyu", "unc", "illinois", "uiuc", "georgia tech",
                     "wisconsin", "boston u", "tufts", "ohio state",
                     "purdue", "uw-madison", "urbana"]
            # C9
            _C9 = ["tsinghua", "peking", "zhejiang", "fudan", "sjtu",
                    "shanghai jiao tong", "ustc", "nanjing", "harbin", "xian jiaotong"]
            # 985
            _985 = ["wuhan", "sun yat-sen", "huazhong", "sichuan", "tianjin",
                     "southeast", "dalian", "chongqing"]
            # Top international
            _TOP_INTL = ["iit", "nus", "ntu", "hku", "hkust", "imperial",
                          "lse", "oxford", "cambridge", "eth", "epfl", "bocconi"]

            if any(s in uni for s in _T10):
                undergrad_tier = 1.0
            elif any(s in uni for s in _C9):
                undergrad_tier = 1.0
            elif any(s in uni for s in _T20):
                undergrad_tier = 2.0
            elif any(s in uni for s in _985):
                undergrad_tier = 2.0
            elif any(s in uni for s in _T30):
                undergrad_tier = 3.0
            elif any(s in uni for s in _TOP_INTL):
                undergrad_tier = 2.0
            else:
                undergrad_tier = 4.0  # known school but not top tier

        # Internship score — tiered by company, country, and role
        work_exps = getattr(profile, "work_experience", [])
        internship_exps = [
            exp for exp in work_exps
            if isinstance(exp, dict) and exp.get("type") == "internship"
        ]

        if internship_exps:
            best_score = 0.0
            for exp in internship_exps:
                combined = (
                    str(exp.get("company", "")) + " "
                    + str(exp.get("description", "")) + " "
                    + str(exp.get("title", ""))
                ).lower()

                # US top quant firms
                _US_TOP_QUANT = [
                    "citadel", "jane street", "two sigma", "de shaw",
                    "hrt", "hudson river", "jump", "imc", "optiver",
                    "sig", "susquehanna", "five rings", "tower research",
                ]
                # US quant funds
                _US_QUANT = [
                    "aqr", "point72", "millennium", "bridgewater",
                    "worldquant", "man group", "balyasny", "cubist",
                    "schonfeld", "verition", "squarepoint", "drw",
                ]
                # US bulge bracket banks
                _US_BB = [
                    "goldman", "morgan stanley", "jpmorgan", "jp morgan",
                    "bank of america", "citi", "barclays", "ubs",
                    "deutsche bank", "bnp", "credit suisse",
                ]
                # China top quant / IB
                _CN_TOP = [
                    "幻方", "九坤", "明汯", "灵均", "锐天", "衍复",
                    "三中一华", "中信", "中金", "华泰", "海通",
                    "citadel cn", "magic formula", "ubiquant",
                    "high-flyer", "jiukun", "mingshi",
                ]
                # US tech
                _US_TECH = [
                    "google", "meta", "amazon", "apple", "microsoft",
                    "nvidia", "tesla",
                ]

                score = 2.0  # default: generic internship
                if any(f in combined for f in _US_TOP_QUANT):
                    score = 10.0
                elif any(f in combined for f in _US_QUANT):
                    score = 8.0
                elif any(f in combined for f in _US_BB):
                    score = 7.0
                elif any(f in combined for f in _CN_TOP):
                    score = 6.0
                elif any(f in combined for f in _US_TECH):
                    score = 5.0
                elif "china" in combined or "cn" in combined:
                    score = 4.0  # China finance (not top tier)

                # Role boost: quant roles get +1
                if any(kw in combined for kw in [
                    "quant", "quantitative", "trading", "alpha",
                    "strategy", "research", "pricing", "derivatives",
                ]):
                    score = min(score + 1.0, 10.0)

                best_score = max(best_score, score)

            # Combine: best single internship + count bonus
            intern_score = min(
                best_score + (len(internship_exps) - 1) * 0.5,
                10.0,
            )
        elif any(
            isinstance(exp, dict) and exp.get("type") == "research"
            for exp in work_exps
        ):
            intern_score = 2.0

        # Research
        has_paper = any(
            isinstance(p, dict) and p.get("has_paper")
            for p in getattr(profile, "projects", [])
        )
        has_research = len(getattr(profile, "projects", [])) > 0 or any(
            isinstance(exp, dict) and exp.get("type") == "research"
            for exp in work_exps
        )
        if has_paper:
            research_score = 3.0
        elif has_research:
            research_score = 2.0

        # Major relevance — pick the BEST major
        majors = getattr(profile, "majors", [])
        if majors:
            major_lower = " ".join(majors).lower()
            if any(kw in major_lower for kw in ["math", "stat", "financial eng"]):
                major_relevance = 1.0
            elif any(kw in major_lower for kw in ["physics", "computer", "ee", "ece"]):
                major_relevance = 0.8
            elif any(kw in major_lower for kw in ["econ", "finance", "actuar"]):
                major_relevance = 0.6
            elif any(kw in major_lower for kw in ["engineer", "data sci"]):
                major_relevance = 0.4
            else:
                major_relevance = 0.2
            # Boost for multiple quant majors
            quant_majors = sum(1 for m in majors if any(
                kw in m.lower() for kw in ["math", "stat", "computer", "econ", "physics"]
            ))
            if quant_majors >= 3:
                major_relevance = 1.0
            elif quant_majors >= 2 and major_relevance < 1.0:
                major_relevance = min(major_relevance + 0.2, 1.0)

    # Missing indicators
    has_gpa = 0.0 if gpa is None else 1.0
    has_gre = 0.0 if gre is None else 1.0
    has_tier = 0.0 if np.isnan(undergrad_tier) else 1.0
    has_intern = 0.0 if np.isnan(intern_score) else 1.0
    has_nat = 0.0 if np.isnan(is_international) else 1.0

    # Feature vector (must match training order)
    features = np.array([[
        gpa if gpa is not None else np.nan,       # gpa_normalized
        gre if gre is not None else np.nan,        # gre_quant
        undergrad_tier,                             # undergrad_tier_encoded
        intern_score,                               # intern_score
        research_score,                             # research_score
        is_international,                           # is_international
        is_female,                                  # is_female
        major_relevance,                            # major_relevance_score
        has_gpa,                                    # has_gpa
        has_gre,                                    # has_gre
        has_tier,                                   # has_tier
        has_intern,                                 # has_intern
        has_nat,                                    # has_nationality
    ]])
    return features


def predict_prob_v2(
    program_id: str,
    gpa: Optional[float] = None,
    gre: Optional[float] = None,
    profile: Optional["UserProfile"] = None,
) -> Optional[AdmitPrediction]:
    """Predict admission probability using GPBoost v2 model.

    Falls back to v1 if GPBoost is not installed or v2 model not available.
    """
    bst, meta = _load_v2()
    if bst is None or meta is None:
        # Fallback to v1
        return predict_prob_full(program_id, gpa, gre, profile)

    import numpy as np

    # Get program numeric ID
    pid_map = meta.get("program_id_map", {})
    pid = pid_map.get(program_id, -1)

    # Build feature vector
    features = _extract_v2_features(program_id, gpa, gre, profile, meta)
    if features is None:
        return predict_prob_full(program_id, gpa, gre, profile)

    group_data = np.array([[pid]])

    try:
        pred = bst.predict(
            data=features,
            group_data_pred=group_data,
            predict_var=False,
            pred_latent=False,
        )
        if isinstance(pred, dict):
            prob = float(pred["response_mean"][0])
        else:
            prob = float(pred[0])
    except Exception:
        return predict_prob_full(program_id, gpa, gre, profile)

    # Apply bias correction
    corrections = meta.get("bias_corrections", {})
    real_rates = meta.get("real_accept_rates", {})
    corr = corrections.get(program_id, {})

    # Fallback: if program not in corrections but has a known real rate,
    # compute correction using global average training accept rate
    if not corr and program_id in real_rates:
        real_rate = real_rates[program_id]
        # Use actual global training accept rate (computed from training data)
        global_train_rate = meta.get("training_stats", {}).get("accept_rate", 0.625)
        corr = {"correction": _logit(real_rate) - _logit(global_train_rate), "real_rate": real_rate}

    # Also check alternate name mappings (e.g., utoronto-mmf → toronto-mmf)
    _NAME_ALIASES = {"utoronto-mmf": "toronto-mmf", "northwestern-mfe": "northwestern-mfe"}
    if not corr:
        alias = _NAME_ALIASES.get(program_id)
        if alias and alias in corrections:
            corr = corrections[alias]

    correction_shift = corr.get("correction", 0.0)
    is_corrected = bool(corr)

    if correction_shift != 0 and math.isfinite(correction_shift):
        logit_prob = _logit(max(0.001, min(0.999, prob)))
        logit_prob += correction_shift
        prob = _sigmoid(logit_prob)

    # Per-program quality gate: if v2 AUC for this program is < 0.50,
    # the model is worse than random — fall back to v1 if available
    _WEAK_V2_PROGRAMS = {
        "stevens-mfe", "michigan-qfr", "ncstate-mfm", "toronto-mmf",
        "fordham-msqf", "rutgers-mqf", "uiuc-msfe", "ucla-mfe",
        "finance-unknown", "or-unknown",
    }
    if program_id in _WEAK_V2_PROGRAMS:
        v1_result = predict_prob_full(program_id, gpa, gre, profile)
        if v1_result is not None:
            return v1_result

    prob = round(max(0.001, min(0.999, prob)), 4)

    # CI from v2 model AUC
    cv_metrics = meta.get("cv_metrics", {})
    auc = cv_metrics.get("auc", 0.7)
    n_total = meta.get("training_stats", {}).get("n_samples", 10000)
    hw = _ci_half_width(n_total, auc, prob)
    logit_p = _logit(prob)
    prob_low = round(max(0.0, _sigmoid(logit_p - hw)), 4)
    prob_high = round(min(1.0, _sigmoid(logit_p + hw)), 4)

    return AdmitPrediction(
        prob=prob,
        prob_low=prob_low,
        prob_high=prob_high,
        is_bias_corrected=is_corrected,
    )


# ===================================================================
# V2 RAW — GPBoost raw prediction without bias correction
# ===================================================================

def _get_v2_raw(
    program_id: str,
    gpa: Optional[float],
    gre: Optional[float],
    profile: Optional["UserProfile"],
) -> Optional[float]:
    """Return v2 RAW probability (before bias correction).

    Used by the ensemble to extract v2's relative feature signal.
    Returns None if GPBoost is unavailable or prediction fails.
    """
    bst, meta = _load_v2()
    if bst is None or meta is None:
        return None

    import numpy as np

    pid_map = meta.get("program_id_map", {})
    pid = pid_map.get(program_id, -1)

    features = _extract_v2_features(program_id, gpa, gre, profile, meta)
    if features is None:
        return None

    try:
        pred = bst.predict(
            data=features,
            group_data_pred=np.array([[pid]]),
            predict_var=False,
            pred_latent=False,
        )
        if isinstance(pred, dict):
            return float(pred["response_mean"][0])
        return float(pred[0])
    except Exception:
        return None


# ===================================================================
# ENSEMBLE — v1 base + v2 residual signal
# ===================================================================

# How much to amplify v2's residual signal.  v2's raw spread is only
# ~0.15 logit between strong and weak applicants, so we scale it up
# to give it meaningful influence on top of v1.
_V2_SIGNAL_WEIGHT = 1.0

# Cache for per-program v2 baselines (computed lazily).
_v2_centers: dict[str, float] = {}


def _get_v2_center(program_id: str) -> Optional[float]:
    """Return the v2 raw prediction for a neutral/average applicant.

    This per-program baseline lets the ensemble measure how much v2
    thinks a *specific* applicant deviates from average at that program,
    rather than relying on a single global constant.
    """
    if program_id in _v2_centers:
        return _v2_centers[program_id]

    # Neutral profile: international, unknown school, no experience
    from .models import UserProfile
    neutral = UserProfile(
        is_international=True,
        university="unknown",
        majors=["finance"],
    )
    raw = _get_v2_raw(program_id, 3.70, None, neutral)
    if raw is not None:
        _v2_centers[program_id] = raw
    return raw


def predict_ensemble(
    program_id: str,
    gpa: Optional[float] = None,
    gre: Optional[float] = None,
    profile: Optional["UserProfile"] = None,
) -> Optional[AdmitPrediction]:
    """Ensemble prediction: v1 base + amplified v2 residual signal.

    Strategy
    --------
    1. v1 provides the well-calibrated base with strong GPA/GRE
       discrimination and profile adjustments.
    2. v2's RAW prediction (before bias correction) captures richer
       features (13-dim) that v1's profile adjustments approximate
       only coarsely.
    3. We extract v2's *residual* — how much v2 thinks this applicant
       deviates from average *at this specific program* — and add it
       to v1's logit.

    Formula:  logit_final = logit(v1) + β × (logit(v2_raw) - logit(v2_center))

    Falls back to v1-only if v2 is unavailable.
    """
    v1 = predict_prob_full(program_id, gpa, gre, profile)
    if v1 is None:
        return predict_prob_v2(program_id, gpa, gre, profile)

    v2_raw = _get_v2_raw(program_id, gpa, gre, profile)
    v2_center = _get_v2_center(program_id)
    if v2_raw is None or v2_center is None:
        return v1

    # v2 residual: how much better/worse than average at THIS program
    v2_residual = (
        _logit(max(0.01, min(0.99, v2_raw)))
        - _logit(max(0.01, min(0.99, v2_center)))
    )

    # Blend
    v1_logit = _logit(max(0.001, min(0.999, v1.prob)))
    final_logit = v1_logit + _V2_SIGNAL_WEIGHT * v2_residual
    prob = round(max(0.001, min(0.999, _sigmoid(final_logit))), 4)

    # CI: preserve v1's CI width
    hw = (
        abs(_logit(v1.prob_high) - _logit(v1.prob))
        if v1.prob_high > v1.prob else 0.5
    )
    prob_low = round(max(0.0, _sigmoid(final_logit - hw)), 4)
    prob_high = round(min(1.0, _sigmoid(final_logit + hw)), 4)

    return AdmitPrediction(
        prob=prob,
        prob_low=prob_low,
        prob_high=prob_high,
        is_bias_corrected=v1.is_bias_corrected,
    )
