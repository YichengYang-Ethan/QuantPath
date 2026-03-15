"""Tests for core.calibrator — model calibration and accuracy evaluation."""

from __future__ import annotations

import pytest

from core.admission_data import AdmissionRecord
from core.calibrator import (
    CalibrationResult,
    ProgramThreshold,
    calibrate_all,
    calibrate_program,
    generate_ranker_overrides,
    predict_outcome,
)
from core.admission_data import compute_program_stats


# ===================================================================
# Test fixtures
# ===================================================================


def _make_baruch_records() -> list[AdmissionRecord]:
    """Simulate Baruch MFE admission data."""
    return [
        # Strong accepted applicants
        AdmissionRecord(
            id="1", program="baruch-mfe", result="accepted",
            gpa_normalized=3.9, bg_tier=1, intern_score=8.0,
            gre=335, has_paper=True, has_research=True,
        ),
        AdmissionRecord(
            id="2", program="baruch-mfe", result="accepted",
            gpa_normalized=3.8, bg_tier=2, intern_score=7.0,
            gre=332, has_paper=True, has_research=True,
        ),
        AdmissionRecord(
            id="3", program="baruch-mfe", result="accepted",
            gpa_normalized=3.85, bg_tier=2, intern_score=6.5,
            gre=330, has_paper=False, has_research=True,
        ),
        # Rejected applicants
        AdmissionRecord(
            id="4", program="baruch-mfe", result="rejected",
            gpa_normalized=3.5, bg_tier=4, intern_score=2.0,
            gre=325, has_paper=False, has_research=False,
        ),
        AdmissionRecord(
            id="5", program="baruch-mfe", result="rejected",
            gpa_normalized=3.6, bg_tier=3, intern_score=3.0,
            gre=328, has_paper=False, has_research=False,
        ),
    ]


def _make_mixed_records() -> list[AdmissionRecord]:
    """Multiple programs for full calibration."""
    records = _make_baruch_records()
    records.extend([
        AdmissionRecord(
            id="6", program="cmu-mscf", result="accepted",
            gpa_normalized=3.85, bg_tier=2, intern_score=7.0,
            gre=333, has_paper=True, has_research=True,
        ),
        AdmissionRecord(
            id="7", program="cmu-mscf", result="rejected",
            gpa_normalized=3.4, bg_tier=4, intern_score=1.0,
            gre=320, has_paper=False, has_research=False,
        ),
        AdmissionRecord(
            id="8", program="gatech-qcf", result="accepted",
            gpa_normalized=3.5, bg_tier=3, intern_score=4.0,
            gre=325, has_paper=False, has_research=False,
        ),
        AdmissionRecord(
            id="9", program="gatech-qcf", result="accepted",
            gpa_normalized=3.6, bg_tier=2, intern_score=5.0,
            gre=328, has_paper=False, has_research=True,
        ),
    ])
    return records


# ===================================================================
# calibrate_program
# ===================================================================


class TestCalibrateProgram:
    """Tests for calibrate_program()."""

    def test_gpa_thresholds(self):
        records = _make_baruch_records()
        stats = compute_program_stats(records, "baruch-mfe")
        threshold = calibrate_program(stats, records)

        # GPA floor should be min GPA of accepted (3.8)
        assert threshold.gpa_floor == pytest.approx(3.8, rel=0.01)
        # GPA target should be average of accepted
        assert 3.8 <= threshold.gpa_target <= 3.9
        # Safe threshold should be high
        assert threshold.gpa_safe >= 3.85

    def test_acceptance_rate(self):
        records = _make_baruch_records()
        stats = compute_program_stats(records, "baruch-mfe")
        threshold = calibrate_program(stats, records)
        assert threshold.observed_acceptance_rate == pytest.approx(3 / 5, rel=0.01)

    def test_confidence_level(self):
        records = _make_baruch_records()
        stats = compute_program_stats(records, "baruch-mfe")
        threshold = calibrate_program(stats, records)
        assert threshold.confidence == "low"  # only 5 records

    def test_bg_tier(self):
        records = _make_baruch_records()
        stats = compute_program_stats(records, "baruch-mfe")
        threshold = calibrate_program(stats, records)
        assert threshold.max_bg_tier_accepted == 2

    def test_empty_program(self):
        records = _make_baruch_records()
        stats = compute_program_stats(records, "nonexistent")
        threshold = calibrate_program(stats, records)
        assert threshold.sample_size == 0


# ===================================================================
# calibrate_all
# ===================================================================


class TestCalibrateAll:
    """Tests for calibrate_all()."""

    def test_all_programs_calibrated(self):
        records = _make_mixed_records()
        result = calibrate_all(records)

        assert isinstance(result, CalibrationResult)
        assert "baruch-mfe" in result.program_thresholds
        assert "cmu-mscf" in result.program_thresholds
        assert "gatech-qcf" in result.program_thresholds

    def test_accuracy_report_present(self):
        records = _make_mixed_records()
        result = calibrate_all(records)

        assert "total_predictions" in result.accuracy_report
        assert result.accuracy_report["total_predictions"] > 0

    def test_recommendations_generated(self):
        records = _make_mixed_records()
        result = calibrate_all(records)
        # Should have at least a data quantity recommendation
        assert len(result.recommendations) > 0


# ===================================================================
# predict_outcome
# ===================================================================


class TestPredictOutcome:
    """Tests for predict_outcome()."""

    def test_strong_applicant_accepted(self):
        threshold = ProgramThreshold(
            program_id="baruch-mfe",
            gpa_target=3.85,
            max_bg_tier_accepted=2,
            observed_acceptance_rate=0.60,
        )
        record = AdmissionRecord(
            gpa_normalized=3.9, bg_tier=1, intern_score=8.0,
            has_paper=True, has_research=True,
        )
        result = predict_outcome(record, threshold)
        assert result == "accepted"

    def test_weak_applicant_rejected(self):
        threshold = ProgramThreshold(
            program_id="baruch-mfe",
            gpa_target=3.85,
            max_bg_tier_accepted=2,
            observed_acceptance_rate=0.10,
        )
        record = AdmissionRecord(
            gpa_normalized=3.2, bg_tier=5, intern_score=0.0,
            has_paper=False, has_research=False,
        )
        result = predict_outcome(record, threshold)
        assert result == "rejected"


# ===================================================================
# generate_ranker_overrides
# ===================================================================


class TestGenerateRankerOverrides:
    """Tests for generate_ranker_overrides()."""

    def test_generates_overrides(self):
        records = _make_mixed_records()
        result = calibrate_all(records)
        overrides = generate_ranker_overrides(result)

        # Should have entries for programs with enough data
        assert isinstance(overrides, dict)
        for pid, ov in overrides.items():
            assert "reach_gpa_threshold" in ov
            assert "safety_gpa_threshold" in ov
            assert "observed_acceptance_rate" in ov
            assert "confidence" in ov

    def test_override_values_reasonable(self):
        records = _make_mixed_records()
        result = calibrate_all(records)
        overrides = generate_ranker_overrides(result)

        for pid, ov in overrides.items():
            assert 0 <= ov["reach_gpa_threshold"] <= 4.0
            assert 0 <= ov["safety_gpa_threshold"] <= 4.0
            assert 0 <= ov["observed_acceptance_rate"] <= 1.0
