# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for core.school_ranker."""

import pytest

from core.models import (
    Course,
    EvaluationResult,
    Prerequisite,
    ProgramData,
    UserProfile,
)
from core.school_ranker import _classify, _compute_fit_score, rank_schools

# ===================================================================
# _classify
# ===================================================================

class TestClassify:
    """Test reach / target / safety classification logic."""

    def test_low_acceptance_rate_is_reach(self) -> None:
        """acceptance_rate < 0.08 -> reach regardless of GPA."""
        assert _classify(user_gpa=4.0, program_avg_gpa=3.5, acceptance_rate=0.05) == "reach"
        assert _classify(user_gpa=4.0, program_avg_gpa=3.5, acceptance_rate=0.07) == "reach"

    def test_gpa_below_avg_is_reach(self) -> None:
        """user_gpa < program_avg_gpa -> reach regardless of acceptance rate."""
        assert _classify(user_gpa=3.5, program_avg_gpa=3.8, acceptance_rate=0.50) == "reach"

    def test_high_acceptance_and_high_gpa_is_safety(self) -> None:
        """acceptance_rate > 0.15 AND user_gpa >= program_avg + 0.1 -> safety."""
        assert _classify(user_gpa=3.9, program_avg_gpa=3.7, acceptance_rate=0.20) == "safety"

    def test_target_when_neither_reach_nor_safety(self) -> None:
        """Moderate conditions -> target."""
        # user_gpa >= program_avg (not reach), acceptance_rate <= 0.15 (not safety)
        assert _classify(user_gpa=3.8, program_avg_gpa=3.8, acceptance_rate=0.12) == "target"

    def test_exact_boundary_acceptance_008(self) -> None:
        """acceptance_rate == 0.08 should NOT trigger reach (strict < 0.08)."""
        assert _classify(user_gpa=4.0, program_avg_gpa=3.5, acceptance_rate=0.08) != "reach"

    def test_exact_boundary_acceptance_015(self) -> None:
        """acceptance_rate == 0.15 should NOT trigger safety (strict > 0.15)."""
        assert _classify(user_gpa=4.0, program_avg_gpa=3.5, acceptance_rate=0.15) != "safety"

    def test_gpa_exactly_equal_to_avg(self) -> None:
        """user_gpa == program_avg_gpa -> not reach (requires strictly less)."""
        result = _classify(user_gpa=3.8, program_avg_gpa=3.8, acceptance_rate=0.20)
        # Not reach because user_gpa is not < program_avg.
        # acceptance > 0.15 but user_gpa (3.8) < program_avg + 0.1 (3.9) -> target
        assert result == "target"

    def test_safety_requires_gpa_margin(self) -> None:
        """Safety needs user_gpa >= program_avg + 0.1. Exactly +0.1 should qualify."""
        assert _classify(user_gpa=3.9, program_avg_gpa=3.8, acceptance_rate=0.20) == "safety"

    def test_zero_acceptance_defaults_to_015(self) -> None:
        """acceptance_rate=0 should default to 0.15 (via `or 0.15`)."""
        # 0.15 is not < 0.08 and not > 0.15, so target if GPA matches.
        assert _classify(user_gpa=3.9, program_avg_gpa=3.8, acceptance_rate=0.0) == "target"

    def test_zero_program_gpa_defaults_to_380(self) -> None:
        """program_avg_gpa=0 defaults to 3.80. user_gpa=3.7 < 3.8 -> reach."""
        assert _classify(user_gpa=3.7, program_avg_gpa=0.0, acceptance_rate=0.20) == "reach"


# ===================================================================
# _compute_fit_score
# ===================================================================

class TestComputeFitScore:
    """Test the composite fit score calculation (0-100)."""

    def test_perfect_fit(self) -> None:
        """High GPA, full prereq match, high acceptance, perfect eval -> near 100."""
        score = _compute_fit_score(
            user_gpa=4.0,
            program_avg_gpa=3.5,
            acceptance_rate=0.25,
            prereq_match_score=1.0,
            overall_eval_score=10.0,
        )
        assert score == pytest.approx(100.0)

    def test_zero_everything(self) -> None:
        """Minimal scores should produce a low fit score."""
        score = _compute_fit_score(
            user_gpa=0.0,
            program_avg_gpa=0.0,   # defaults to 3.80
            acceptance_rate=0.0,    # defaults to 0.15
            prereq_match_score=0.0,
            overall_eval_score=0.0,
        )
        # GPA: user 0.0 vs default 3.80 -> gpa_diff = -3.80 -> gpa_pts = max(0, 25 + (-3.8)*60) = 0
        # Prereq: 0 * 30 = 0
        # Accept: 0.15 -> between 0.03 and 0.20 -> 2 + (0.15-0.03)/(0.20-0.03)*18 = ~14.7
        # Profile: 0
        assert score > 0.0
        assert score < 25.0

    def test_gpa_above_avg_gets_full_25(self) -> None:
        """When user_gpa >= program_avg, GPA component should be 25.0."""
        score_high = _compute_fit_score(4.0, 3.5, 0.15, 0.0, 0.0)
        score_exact = _compute_fit_score(3.5, 3.5, 0.15, 0.0, 0.0)
        # Both should have same GPA component (25.0)
        # Only difference is acceptance component (0.15 is not > 0.15 for the boundary)
        assert score_high == score_exact

    def test_gpa_below_avg_reduces_score(self) -> None:
        """Lower GPA reduces the GPA component."""
        score_above = _compute_fit_score(3.8, 3.5, 0.15, 1.0, 5.0)
        score_below = _compute_fit_score(3.3, 3.5, 0.15, 1.0, 5.0)
        assert score_above > score_below

    def test_prereq_match_linear(self) -> None:
        """Prerequisite match contributes linearly (0-30 pts)."""
        s0 = _compute_fit_score(3.8, 3.5, 0.15, 0.0, 5.0)
        s50 = _compute_fit_score(3.8, 3.5, 0.15, 0.5, 5.0)
        s100 = _compute_fit_score(3.8, 3.5, 0.15, 1.0, 5.0)
        assert s50 - s0 == pytest.approx(15.0)
        assert s100 - s0 == pytest.approx(30.0)

    def test_acceptance_rate_boundaries(self) -> None:
        """acceptance >= 0.20 -> full 20 pts; <= 0.03 -> 2 pts."""
        s_high = _compute_fit_score(3.8, 3.5, 0.30, 0.5, 5.0)
        s_low = _compute_fit_score(3.8, 3.5, 0.02, 0.5, 5.0)
        # Both should have same GPA, prereq, profile pts.
        # Difference is accept component: 20.0 vs 2.0
        assert s_high - s_low == pytest.approx(18.0)

    def test_profile_score_linear(self) -> None:
        """Profile contributes (eval_score / 10) * 25 pts."""
        s0 = _compute_fit_score(3.8, 3.5, 0.15, 0.5, 0.0)
        s10 = _compute_fit_score(3.8, 3.5, 0.15, 0.5, 10.0)
        assert s10 - s0 == pytest.approx(25.0)

    def test_score_range(self) -> None:
        """Fit score should always be in [0, 100]."""
        for user_gpa in [0.0, 2.0, 3.5, 4.0]:
            for ar in [0.01, 0.10, 0.25]:
                for pm in [0.0, 0.5, 1.0]:
                    for es in [0.0, 5.0, 10.0]:
                        score = _compute_fit_score(user_gpa, 3.5, ar, pm, es)
                        assert 0.0 <= score <= 100.0


# ===================================================================
# rank_schools
# ===================================================================

class TestRankSchools:
    """Test the full rank_schools pipeline with mock data."""

    def _make_profile(self) -> UserProfile:
        return UserProfile(
            name="Test",
            gpa=3.8,
            gpa_quant=3.9,
            majors=["Mathematics"],
            coursework=[
                Course("Calc", "MATH101", "calculus", "A", level=200),
                Course("LinAlg", "MATH200", "linear_algebra", "A", level=200),
                Course("Prob", "STAT400", "probability", "A-", level=400),
                Course("Stats", "STAT410", "statistics", "A", level=400),
                Course("C++", "CS101", "programming_cpp", "A-", level=200),
            ],
        )

    def _make_programs(self) -> list[ProgramData]:
        # Use synthetic IDs so LR model fallback (heuristic) is always used.
        # Reach: low acceptance, high avg GPA
        reach = ProgramData(
            id="test-reach-prog",
            name="CMU MSCF",
            university="Carnegie Mellon",
            acceptance_rate=0.05,
            avg_gpa=3.9,
            prerequisites_required=[
                Prerequisite(category="calculus"),
                Prerequisite(category="linear_algebra"),
                Prerequisite(category="probability"),
                Prerequisite(category="programming_cpp"),
            ],
        )
        # Target: moderate acceptance, matched GPA
        target = ProgramData(
            id="test-target-prog",
            name="BU MSMF",
            university="Boston University",
            acceptance_rate=0.12,
            avg_gpa=3.7,
            prerequisites_required=[
                Prerequisite(category="calculus"),
                Prerequisite(category="statistics"),
            ],
        )
        # Safety: high acceptance, lower avg GPA
        safety = ProgramData(
            id="test-safety-prog",
            name="Rutgers MQF",
            university="Rutgers",
            acceptance_rate=0.25,
            avg_gpa=3.5,
            prerequisites_required=[
                Prerequisite(category="calculus"),
            ],
        )
        return [reach, target, safety]

    def test_rank_schools_returns_all_buckets(self) -> None:
        profile = self._make_profile()
        programs = self._make_programs()
        evaluation = EvaluationResult(overall_score=7.0)
        result = rank_schools(profile, programs, evaluation)
        assert "reach" in result
        assert "target" in result
        assert "safety" in result
        assert "all" in result

    def test_all_programs_appear_in_all(self) -> None:
        profile = self._make_profile()
        programs = self._make_programs()
        evaluation = EvaluationResult(overall_score=7.0)
        result = rank_schools(profile, programs, evaluation)
        assert len(result["all"]) == 3

    def test_all_sorted_by_fit_score_desc(self) -> None:
        profile = self._make_profile()
        programs = self._make_programs()
        evaluation = EvaluationResult(overall_score=7.0)
        result = rank_schools(profile, programs, evaluation)
        scores = [r["fit_score"] for r in result["all"]]
        assert scores == sorted(scores, reverse=True)

    def test_each_result_has_required_keys(self) -> None:
        profile = self._make_profile()
        programs = self._make_programs()
        evaluation = EvaluationResult(overall_score=7.0)
        result = rank_schools(profile, programs, evaluation)
        required_keys = {
            "program_id", "name", "university", "category",
            "fit_score", "prereq_match_score", "acceptance_rate", "avg_gpa",
        }
        for entry in result["all"]:
            assert required_keys.issubset(set(entry.keys()))

    def test_classification_matches_category(self) -> None:
        """Programs in each bucket should have matching category values."""
        profile = self._make_profile()
        programs = self._make_programs()
        evaluation = EvaluationResult(overall_score=7.0)
        result = rank_schools(profile, programs, evaluation)
        for entry in result["reach"]:
            assert entry["category"] == "reach"
        for entry in result["target"]:
            assert entry["category"] == "target"
        for entry in result["safety"]:
            assert entry["category"] == "safety"

    def test_reach_program_classified_correctly(self) -> None:
        """Program with 5% acceptance and avg GPA > user GPA should be reach."""
        profile = self._make_profile()
        programs = self._make_programs()
        evaluation = EvaluationResult(overall_score=7.0)
        result = rank_schools(profile, programs, evaluation)
        reach = next(r for r in result["all"] if r["program_id"] == "test-reach-prog")
        assert reach["category"] == "reach"

    def test_safety_program_classified_correctly(self) -> None:
        """Program with 25% acceptance and avg GPA below user -> safety."""
        profile = self._make_profile()
        programs = self._make_programs()
        evaluation = EvaluationResult(overall_score=7.0)
        result = rank_schools(profile, programs, evaluation)
        safety = next(r for r in result["all"] if r["program_id"] == "test-safety-prog")
        assert safety["category"] == "safety"

    def test_empty_programs_list(self) -> None:
        profile = self._make_profile()
        evaluation = EvaluationResult(overall_score=7.0)
        result = rank_schools(profile, [], evaluation)
        assert result["reach"] == []
        assert result["target"] == []
        assert result["safety"] == []
        assert result["all"] == []

    def test_prereq_match_score_reflected(self) -> None:
        """Programs with all prereqs met should have higher match scores."""
        profile = self._make_profile()
        programs = self._make_programs()
        evaluation = EvaluationResult(overall_score=7.0)
        result = rank_schools(profile, programs, evaluation)
        # test-safety-prog only requires calculus, which profile has -> match_score = 1.0
        safety = next(r for r in result["all"] if r["program_id"] == "test-safety-prog")
        assert safety["prereq_match_score"] == 1.0
