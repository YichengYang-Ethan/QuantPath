"""Tests for core.lr_predictor."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from core.lr_predictor import (
    AdmitPrediction,
    _logit,
    _sigmoid,
    _ci_half_width,
    _compute_logit,
    predict_prob,
    predict_prob_full,
    has_model,
    get_model_stats,
)
from core.models import TestScores, UserProfile


# ---------------------------------------------------------------------------
# Minimal fake model fixture (does not require admission_models.json)
# ---------------------------------------------------------------------------

_FAKE_MODEL = {
    "baruch-mfe": {
        "n_total": 555,
        "n_accepted": 193,
        "n_rejected": 362,
        "accept_rate": 0.3477,
        "auc": 0.586,
        "features": ["gpa_4scale", "gre_quant"],
        "means": [3.63, 167.5],
        "stds": [0.29, 4.08],
        "coef": [0.314, 0.256],
        "intercept": -0.660,
        "real_accept_rate": 0.04,  # real Baruch rate
        "gpa_p25": 3.5, "gpa_p50": 3.7, "gpa_p75": 3.83,
        "gre_p25": 167.0, "gre_p50": 168.0, "gre_p75": 170.0,
    },
    "rutgers-mqf": {
        "n_total": 200,
        "n_accepted": 165,
        "n_rejected": 35,
        "accept_rate": 0.825,
        "auc": 0.57,
        "features": ["gpa_4scale", "gre_quant"],
        "means": [3.50, 163.0],
        "stds": [0.30, 5.0],
        "coef": [0.25, 0.15],
        "intercept": 1.50,
        "real_accept_rate": 0.864,  # real Rutgers rate (high)
        "gpa_p25": 3.3, "gpa_p50": 3.55, "gpa_p75": 3.75,
        "gre_p25": 160.0, "gre_p50": 163.0, "gre_p75": 167.0,
    },
}


@pytest.fixture(autouse=True)
def mock_models():
    """Patch _load_models to return controlled fake data for all tests."""
    with patch("core.lr_predictor._load_models", return_value=_FAKE_MODEL):
        # Also reset the module-level cache so the mock takes effect
        import core.lr_predictor as lr_mod
        original = lr_mod._models
        lr_mod._models = None
        with patch("core.lr_predictor._models", None):
            yield
        lr_mod._models = original


# ---------------------------------------------------------------------------
# Internal math helpers
# ---------------------------------------------------------------------------


class TestMathHelpers:
    def test_sigmoid_zero(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_large_positive(self):
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_sigmoid_large_negative(self):
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)

    def test_logit_inverse_of_sigmoid(self):
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert _sigmoid(_logit(p)) == pytest.approx(p, abs=1e-6)

    def test_ci_half_width_decreases_with_n(self):
        hw_small = _ci_half_width(n=50, auc=0.65, p=0.3)
        hw_large = _ci_half_width(n=500, auc=0.65, p=0.3)
        assert hw_small > hw_large

    def test_ci_half_width_decreases_with_higher_auc(self):
        hw_low = _ci_half_width(n=300, auc=0.55, p=0.3)
        hw_high = _ci_half_width(n=300, auc=0.75, p=0.3)
        assert hw_low > hw_high

    def test_ci_half_width_random_auc_gives_wide_interval(self):
        # AUC = 0.5 → no signal → n_eff floored to 5, giving wide CI
        hw_random = _ci_half_width(n=500, auc=0.5, p=0.3)
        hw_good = _ci_half_width(n=500, auc=0.75, p=0.3)
        # Random AUC should produce much wider CI than high AUC
        assert hw_random > hw_good * 2


# ---------------------------------------------------------------------------
# predict_prob — backward-compatible scalar API
# ---------------------------------------------------------------------------


class TestPredictProb:
    def test_returns_none_for_unknown_program(self):
        assert predict_prob("nonexistent-prog", gpa=3.8, gre=168) is None

    def test_returns_float_for_known_program(self):
        result = predict_prob("baruch-mfe", gpa=3.8, gre=168)
        assert result is not None
        assert isinstance(result, float)

    def test_result_in_unit_interval(self):
        for gpa in [2.5, 3.0, 3.5, 4.0]:
            p = predict_prob("baruch-mfe", gpa=gpa, gre=168)
            assert p is not None
            assert 0.0 <= p <= 1.0

    def test_higher_gpa_gives_higher_prob(self):
        p_low = predict_prob("baruch-mfe", gpa=3.2, gre=165)
        p_high = predict_prob("baruch-mfe", gpa=3.9, gre=170)
        assert p_high > p_low

    def test_returns_none_when_both_inputs_missing(self):
        assert predict_prob("baruch-mfe", gpa=None, gre=None) is None

    def test_uses_mean_when_gpa_missing(self):
        # Should not raise; uses training mean for missing GPA
        result = predict_prob("baruch-mfe", gpa=None, gre=168)
        assert result is not None

    def test_uses_mean_when_gre_missing(self):
        result = predict_prob("baruch-mfe", gpa=3.8, gre=None)
        assert result is not None


# ---------------------------------------------------------------------------
# Bias correction
# ---------------------------------------------------------------------------


class TestBiasCorrection:
    def test_baruch_bias_corrected_prob_is_low(self):
        """With real_accept_rate=4%, Baruch at avg applicant should give low P."""
        # Average Baruch applicant (z_gpa=0, z_gre=0) → logit = logit(0.04) ≈ -3.18
        p = predict_prob("baruch-mfe", gpa=3.63, gre=167.5)
        assert p is not None
        # Should be in plausible range for an avg applicant at 4% school
        assert p < 0.30

    def test_rutgers_bias_corrected_prob_is_high(self):
        """With real_accept_rate=86.4%, Rutgers at avg applicant should give high P."""
        p = predict_prob("rutgers-mqf", gpa=3.50, gre=163.0)
        assert p is not None
        assert p > 0.60

    def test_bias_correction_uses_real_rate_not_training_rate(self):
        """Baruch training accept_rate=34.8% vs real=4%. Correction must use 4%."""
        # Without correction, avg applicant at Baruch would get sigmoid(-0.66) ≈ 34%
        # With correction, it should be much lower
        p = predict_prob("baruch-mfe", gpa=3.63, gre=167.5)
        assert p is not None
        assert p < 0.20  # would be ~34% without correction

    def test_relative_ordering_preserved_after_correction(self):
        """Higher GPA still → higher P after bias correction."""
        p_low = predict_prob("baruch-mfe", gpa=3.3, gre=165)
        p_high = predict_prob("baruch-mfe", gpa=4.0, gre=170)
        assert p_high > p_low


# ---------------------------------------------------------------------------
# predict_prob_full — AdmitPrediction with CI
# ---------------------------------------------------------------------------


class TestPredictProbFull:
    def test_returns_none_for_unknown_program(self):
        assert predict_prob_full("nonexistent", gpa=3.8, gre=168) is None

    def test_returns_admit_prediction_dataclass(self):
        result = predict_prob_full("baruch-mfe", gpa=3.8, gre=168)
        assert isinstance(result, AdmitPrediction)

    def test_ci_bounds_ordered(self):
        result = predict_prob_full("baruch-mfe", gpa=3.8, gre=168)
        assert result is not None
        assert result.prob_low <= result.prob <= result.prob_high

    def test_ci_bounds_in_unit_interval(self):
        result = predict_prob_full("baruch-mfe", gpa=3.5, gre=165)
        assert result is not None
        assert 0.0 <= result.prob_low
        assert result.prob_high <= 1.0

    def test_is_bias_corrected_true_when_real_rate_present(self):
        result = predict_prob_full("baruch-mfe", gpa=3.8, gre=168)
        assert result is not None
        assert result.is_bias_corrected is True

    def test_ci_width_nonzero(self):
        result = predict_prob_full("baruch-mfe", gpa=3.8, gre=168)
        assert result is not None
        assert result.prob_high - result.prob_low > 0.01


# ---------------------------------------------------------------------------
# Profile-aware adjustments
# ---------------------------------------------------------------------------


def _make_profile(
    gpa: float = 3.8,
    gre: int = 168,
    is_international: bool = False,
    n_internships: int = 0,
) -> UserProfile:
    experience = [
        {"type": "internship", "title": "Quant Intern", "description": "Trading"}
        for _ in range(n_internships)
    ]
    return UserProfile(
        name="Test",
        gpa=gpa,
        gpa_quant=gpa,
        test_scores=TestScores(gre_quant=gre),
        is_international=is_international,
        work_experience=experience,
    )


class TestProfileAdjustments:
    def test_international_lowers_prob(self):
        domestic = _make_profile(is_international=False)
        intl = _make_profile(is_international=True)
        p_dom = predict_prob("baruch-mfe", gpa=3.8, gre=168, profile=domestic)
        p_int = predict_prob("baruch-mfe", gpa=3.8, gre=168, profile=intl)
        assert p_dom > p_int

    def test_internships_raise_prob(self):
        no_intern = _make_profile(n_internships=0)
        with_intern = _make_profile(n_internships=2)
        p_no = predict_prob("baruch-mfe", gpa=3.8, gre=168, profile=no_intern)
        p_yes = predict_prob("baruch-mfe", gpa=3.8, gre=168, profile=with_intern)
        assert p_yes > p_no

    def test_two_internships_better_than_one(self):
        one = _make_profile(n_internships=1)
        two = _make_profile(n_internships=2)
        p_one = predict_prob("baruch-mfe", gpa=3.8, gre=168, profile=one)
        p_two = predict_prob("baruch-mfe", gpa=3.8, gre=168, profile=two)
        assert p_two > p_one

    def test_no_profile_same_as_none_profile(self):
        """predict_prob with no profile should match predict_prob_full with profile=None."""
        p_scalar = predict_prob("baruch-mfe", gpa=3.8, gre=168)
        full = predict_prob_full("baruch-mfe", gpa=3.8, gre=168, profile=None)
        assert p_scalar == pytest.approx(full.prob)


# ---------------------------------------------------------------------------
# has_model / get_model_stats
# ---------------------------------------------------------------------------


class TestModelHelpers:
    def test_has_model_true_for_known(self):
        assert has_model("baruch-mfe") is True

    def test_has_model_false_for_unknown(self):
        assert has_model("fake-prog") is False

    def test_get_model_stats_returns_dict(self):
        stats = get_model_stats("baruch-mfe")
        assert isinstance(stats, dict)
        assert "auc" in stats
        assert "n_total" in stats

    def test_get_model_stats_none_for_unknown(self):
        assert get_model_stats("fake-prog") is None
