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
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .models import UserProfile

_MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "admission_models.json"
_models: dict | None = None

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
