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

import numpy as np

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
_ADJ_INTERNSHIP_1  =  0.10
_ADJ_INTERNSHIP_2  =  0.20   # replaces _ADJ_INTERNSHIP_1 for 2+


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
    """Compute logit adjustment from profile signals."""
    adj = 0.0

    if getattr(profile, "is_international", False):
        adj += _ADJ_INTERNATIONAL

    n_internships = sum(
        1 for exp in getattr(profile, "work_experience", [])
        if isinstance(exp, dict) and exp.get("type") == "internship"
    )
    if n_internships >= 2:
        adj += _ADJ_INTERNSHIP_2
    elif n_internships >= 1:
        adj += _ADJ_INTERNSHIP_1

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
    """Load GPBoost v2 model (binary + metadata)."""
    global _v2_model, _v2_meta
    if _v2_model is None and _V2_BIN_PATH.exists():
        try:
            import gpboost as gpb
            _v2_model = gpb.Booster(model_file=str(_V2_BIN_PATH))
            with _V2_MODEL_PATH.open(encoding="utf-8") as f:
                _v2_meta = json.load(f)
        except ImportError:
            _v2_model = None
            _v2_meta = None
    return _v2_model, _v2_meta


def _extract_v2_features(
    program_id: str,
    gpa: Optional[float],
    gre: Optional[float],
    profile: Optional["UserProfile"],
    meta: dict,
) -> Optional[np.ndarray]:
    """Build the 13-feature vector for GPBoost v2 prediction."""
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

        # Internship score
        n_internships = sum(
            1 for exp in getattr(profile, "work_experience", [])
            if isinstance(exp, dict) and exp.get("type") == "internship"
        )
        if n_internships > 0:
            intern_score = min(n_internships * 3.0, 10.0)

        # Research
        has_paper = any(
            isinstance(p, dict) and p.get("has_paper")
            for p in getattr(profile, "projects", [])
        )
        has_research = len(getattr(profile, "projects", [])) > 0
        if has_paper:
            research_score = 3.0
        elif has_research:
            research_score = 2.0

        # Major relevance
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
    corr = corrections.get(program_id, {})
    correction_shift = corr.get("correction", 0.0)
    is_corrected = bool(corr)

    if correction_shift != 0 and math.isfinite(correction_shift):
        logit_prob = _logit(max(0.001, min(0.999, prob)))
        logit_prob += correction_shift
        prob = _sigmoid(logit_prob)

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
