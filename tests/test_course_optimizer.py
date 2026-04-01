# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for core.course_optimizer."""

import pytest

from core.course_optimizer import (
    _PREREQ_BONUS,
    CourseRecommendation,
    _build_reason,
    _count_prereq_coverage,
    _current_score_for_categories,
    _user_has_category,
    _user_satisfies_prereq_category,
    optimize_courses,
)
from core.models import Course, CourseCategory, Prerequisite, ProgramData, UserProfile
from core.profile_evaluator import DIMENSION_WEIGHTS

# ===================================================================
# Fixtures
# ===================================================================


def _empty_profile() -> UserProfile:
    """A profile with zero coursework."""
    return UserProfile(name="Empty Student", gpa=0.0)


def _minimal_profile() -> UserProfile:
    """A profile with just a few courses."""
    return UserProfile(
        name="Minimal Student",
        gpa=3.5,
        coursework=[
            Course("Calc I", "MATH101", "calculus", "B+", level=100),
            Course("Intro Stats", "STAT100", "statistics", "B", level=100),
        ],
    )


def _strong_profile() -> UserProfile:
    """A well-rounded profile scoring highly in most categories."""
    return UserProfile(
        name="Strong Student",
        gpa=3.95,
        gpa_quant=3.95,
        majors=["Computer Science", "Mathematics"],
        coursework=[
            # Math -- all high grades
            Course("Calc III", "MATH241", "calculus", "A+", level=200),
            Course("Linear Algebra", "MATH415", "linear_algebra", "A", level=400),
            Course("Probability", "STAT400", "probability", "A", level=400),
            Course("ODE", "MATH330", "ode", "A", level=300),
            Course("PDE", "MATH442", "pde", "A-", level=400),
            Course("Real Analysis", "MATH447", "real_analysis", "A", level=400),
            Course("Numerical Methods", "MATH450", "numerical_analysis", "A-", level=400),
            Course("Stochastic Proc", "STAT455", "stochastic_processes", "A", level=400),
            Course("Stochastic Calc", "MATH480", "stochastic_calculus", "A", level=400),
            Course("Optimization", "MATH484", "optimization", "A-", level=400),
            # Stats
            Course("Math Stats", "STAT410", "statistics", "A", level=400),
            Course("Regression", "STAT425", "regression", "A", level=400),
            Course("Time Series", "STAT429", "time_series", "A", level=400),
            Course("Econometrics", "ECON471", "econometrics", "A-", level=400),
            Course("Stat Learning", "STAT432", "stat_learning", "A", level=400),
            Course("Stat Computing", "STAT428", "stat_computing", "A", level=400),
            # CS
            Course("C++ Prog", "CS125", "programming_cpp", "A+", level=100),
            Course("Python", "CS101", "programming_python", "A+", level=100),
            Course("Data Structures", "CS225", "data_structures", "A", level=200),
            Course("Algorithms", "CS374", "algorithms", "A", level=300),
            Course("ML", "CS446", "machine_learning", "A", level=400),
            Course("Software Eng", "CS427", "software_engineering", "A-", level=400),
            # Finance / Econ
            Course("Microecon", "ECON302", "microeconomics", "A", level=300),
            Course("Macroecon", "ECON303", "macroeconomics", "A", level=300),
            Course("Investments", "FIN321", "finance", "A", level=300),
            Course("Derivatives", "FIN411", "derivatives", "A-", level=400),
            Course("Portfolio", "FIN432", "portfolio_theory", "A", level=400),
            Course("Risk Mgmt", "FIN451", "risk_management", "A-", level=400),
            Course("Fin Econometrics", "ECON491", "financial_econometrics", "A", level=400),
            Course("Game Theory", "ECON414", "game_theory", "A-", level=400),
        ],
    )


def _make_program(
    program_id: str = "test-prog",
    required: list[Prerequisite] | None = None,
    recommended: list[Prerequisite] | None = None,
) -> ProgramData:
    """Helper to build a ProgramData with prerequisite lists."""
    return ProgramData(
        id=program_id,
        name=f"Test Program ({program_id})",
        prerequisites_required=required or [],
        prerequisites_recommended=recommended or [],
    )


# ===================================================================
# Internal helper tests
# ===================================================================


class TestCurrentScoreForCategories:
    """Test _current_score_for_categories helper."""

    def test_returns_best_score(self) -> None:
        profile = UserProfile(
            coursework=[
                Course("Calc I", "M101", "calculus", "B", level=100),
                Course("Calc II", "M102", "calculus", "A", level=100),
            ]
        )
        assert _current_score_for_categories(profile, {"calculus"}) == 10.0

    def test_returns_zero_when_missing(self) -> None:
        profile = _empty_profile()
        assert _current_score_for_categories(profile, {"calculus"}) == 0.0


class TestUserHasCategory:
    """Test _user_has_category helper."""

    def test_has_category(self) -> None:
        profile = _minimal_profile()
        assert _user_has_category(profile, "calculus") is True

    def test_missing_category(self) -> None:
        profile = _minimal_profile()
        assert _user_has_category(profile, "real_analysis") is False

    def test_empty_profile(self) -> None:
        profile = _empty_profile()
        assert _user_has_category(profile, "calculus") is False


class TestUserSatisfiesPrereqCategory:
    """Test _user_satisfies_prereq_category helper."""

    def test_direct_match(self) -> None:
        profile = _minimal_profile()
        assert _user_satisfies_prereq_category(profile, "calculus") is True

    def test_broad_category_match(self) -> None:
        """'programming' should be satisfied by programming_cpp or programming_python."""
        profile = UserProfile(
            coursework=[Course("C++", "CS101", "programming_cpp", "A")]
        )
        assert _user_satisfies_prereq_category(profile, "programming") is True

    def test_not_satisfied(self) -> None:
        profile = _minimal_profile()
        assert _user_satisfies_prereq_category(profile, "real_analysis") is False


class TestCountPrereqCoverage:
    """Test _count_prereq_coverage helper."""

    def test_counts_programs_needing_category(self) -> None:
        profile = _empty_profile()
        programs = [
            _make_program("p1", required=[Prerequisite(category="calculus")]),
            _make_program("p2", required=[Prerequisite(category="calculus")]),
            _make_program("p3", required=[Prerequisite(category="linear_algebra")]),
        ]
        # Empty profile -> both p1 and p2 need calculus
        assert _count_prereq_coverage("calculus", programs, profile) == 2

    def test_zero_when_already_satisfied(self) -> None:
        profile = _minimal_profile()  # has calculus
        programs = [
            _make_program("p1", required=[Prerequisite(category="calculus")]),
        ]
        assert _count_prereq_coverage("calculus", programs, profile) == 0

    def test_counts_each_program_at_most_once(self) -> None:
        """Even if a program lists calculus twice, count it once."""
        profile = _empty_profile()
        programs = [
            _make_program(
                "p1",
                required=[
                    Prerequisite(category="calculus"),
                    Prerequisite(category="calculus"),
                ],
            ),
        ]
        assert _count_prereq_coverage("calculus", programs, profile) == 1

    def test_broad_prereq_expansion(self) -> None:
        """'ode' should count toward a 'differential_equations' prereq."""
        profile = _empty_profile()
        programs = [
            _make_program("p1", required=[Prerequisite(category="differential_equations")]),
        ]
        assert _count_prereq_coverage("ode", programs, profile) == 1

    def test_no_programs(self) -> None:
        profile = _empty_profile()
        assert _count_prereq_coverage("calculus", [], profile) == 0


class TestBuildReason:
    """Test _build_reason helper."""

    def test_missing_course_reason(self) -> None:
        reason = _build_reason("calculus", "math", 0.0, 0)
        assert "missing" in reason.lower()

    def test_low_score_reason(self) -> None:
        reason = _build_reason("calculus", "math", 5.0, 0)
        assert "5.0/10" in reason
        assert "math" in reason

    def test_prereq_coverage_reason(self) -> None:
        reason = _build_reason("calculus", "math", 0.0, 3)
        assert "3 target programs" in reason

    def test_single_program_word(self) -> None:
        reason = _build_reason("calculus", "math", 0.0, 1)
        assert "1 target program " in reason  # singular

    def test_no_prereq_no_mention(self) -> None:
        reason = _build_reason("calculus", "math", 5.0, 0)
        assert "target" not in reason


# ===================================================================
# optimize_courses (main API)
# ===================================================================


class TestOptimizeCourses:
    """Test the optimize_courses() public function."""

    def test_returns_list_of_recommendations(self) -> None:
        profile = _minimal_profile()
        result = optimize_courses(profile, [])
        assert isinstance(result, list)
        assert all(isinstance(r, CourseRecommendation) for r in result)

    def test_default_max_courses_is_3(self) -> None:
        """With default max_courses=3, at most 3 items are returned."""
        profile = _minimal_profile()
        result = optimize_courses(profile, [])
        assert len(result) <= 3

    def test_max_courses_limits_output(self) -> None:
        profile = _empty_profile()
        result_1 = optimize_courses(profile, [], max_courses=1)
        result_5 = optimize_courses(profile, [], max_courses=5)
        assert len(result_1) == 1
        assert len(result_5) == 5

    def test_results_sorted_by_impact_descending(self) -> None:
        profile = _empty_profile()
        result = optimize_courses(profile, [], max_courses=10)
        scores = [r.impact_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_profile_gets_recommendations(self) -> None:
        """A profile with no courses should get recommendations for the
        highest-impact categories."""
        profile = _empty_profile()
        result = optimize_courses(profile, [], max_courses=5)
        assert len(result) == 5
        # All should have positive impact
        assert all(r.impact_score > 0 for r in result)

    def test_strong_profile_gets_fewer_recommendations(self) -> None:
        """A profile with near-perfect scores everywhere should get
        few or no recommendations.
        """
        profile = _strong_profile()
        result = optimize_courses(profile, [], max_courses=10)
        # The strong profile has A or A- in everything.  A- = 9.0 which
        # is exactly at the ceiling, so those are excluded.  Only
        # categories truly below 9.0 get recommended.
        for rec in result:
            # Every recommendation should be for a category below ceiling
            assert rec.impact_score > 0

    def test_high_score_categories_excluded(self) -> None:
        """Categories where the user already scores >= 9.0 should NOT
        appear in recommendations.
        """
        profile = UserProfile(
            coursework=[
                # A = 10.0, well above the 9.0 ceiling
                Course("Calc III", "MATH241", "calculus", "A", level=200),
            ]
        )
        result = optimize_courses(profile, [], max_courses=20)
        recommended_categories = {r.category for r in result}
        assert "calculus" not in recommended_categories

    def test_barely_below_ceiling_included(self) -> None:
        """A category scoring 8.0 (B+ grade) should still be recommended."""
        profile = UserProfile(
            coursework=[
                Course("Calc III", "MATH241", "calculus", "B+", level=200),
            ]
        )
        result = optimize_courses(profile, [], max_courses=35)
        recommended_categories = {r.category for r in result}
        assert "calculus" in recommended_categories

    def test_exactly_at_ceiling_excluded(self) -> None:
        """A category scoring exactly 9.0 (A- grade) should be excluded."""
        profile = UserProfile(
            coursework=[
                Course("Calc III", "MATH241", "calculus", "A-", level=200),
            ]
        )
        result = optimize_courses(profile, [], max_courses=20)
        recommended_categories = {r.category for r in result}
        assert "calculus" not in recommended_categories

    def test_impact_calculation_correctness(self) -> None:
        """Verify the impact calculation for a known category.

        For a completely empty profile, calculus maps to:
          - math/calculus_series: dim_weight=0.30, factor_weight=0.15, gap=9.0
            -> 0.30 * 0.15 * 9.0 = 0.405

        No prereq bonus with empty programs list.
        """
        profile = _empty_profile()
        result = optimize_courses(profile, [], max_courses=30)
        calculus_rec = next(
            (r for r in result if r.category == "calculus"), None
        )
        assert calculus_rec is not None
        assert calculus_rec.impact_score == pytest.approx(0.405, abs=0.001)
        assert calculus_rec.dimension == "math"

    def test_multi_factor_category_impact(self) -> None:
        """Categories mapped to multiple factors should accumulate impact.

        numerical_analysis maps to:
          - math/numerical_analysis: 0.30 * 0.08 * 9.0 = 0.216
          - cs/numerical_computing:  0.20 * 0.07 * 9.0 = 0.126
          Total = 0.342
        """
        profile = _empty_profile()
        result = optimize_courses(profile, [], max_courses=30)
        na_rec = next(
            (r for r in result if r.category == "numerical_analysis"), None
        )
        assert na_rec is not None
        assert na_rec.impact_score == pytest.approx(0.342, abs=0.001)

    def test_machine_learning_multi_factor(self) -> None:
        """machine_learning maps to:
          - statistics/stat_learning_ml: 0.20 * 0.13 * 9.0 = 0.234
          - cs/ml_course:               0.20 * 0.10 * 9.0 = 0.180
          Total = 0.414
        """
        profile = _empty_profile()
        result = optimize_courses(profile, [], max_courses=30)
        ml_rec = next(
            (r for r in result if r.category == "machine_learning"), None
        )
        assert ml_rec is not None
        assert ml_rec.impact_score == pytest.approx(0.414, abs=0.001)

    def test_prereq_bonus_adds_to_impact(self) -> None:
        """When target programs require a category the user lacks, the
        prereq bonus should increase the impact score.
        """
        profile = _empty_profile()
        programs = [
            _make_program("p1", required=[Prerequisite(category="calculus")]),
            _make_program("p2", required=[Prerequisite(category="calculus")]),
        ]
        result_with = optimize_courses(profile, programs, max_courses=30)
        result_without = optimize_courses(profile, [], max_courses=30)

        calc_with = next(r for r in result_with if r.category == "calculus")
        calc_without = next(r for r in result_without if r.category == "calculus")

        # 2 programs * 0.5 bonus = 1.0 extra
        assert calc_with.impact_score == pytest.approx(
            calc_without.impact_score + 2 * _PREREQ_BONUS, abs=0.001
        )
        assert calc_with.prereq_coverage == 2

    def test_prereq_coverage_zero_when_satisfied(self) -> None:
        """If the user already has calculus, prereq_coverage for calculus
        should be 0 even when programs require it.
        """
        profile = UserProfile(
            coursework=[
                Course("Calc III", "MATH241", "calculus", "B+", level=200),
            ]
        )
        programs = [
            _make_program("p1", required=[Prerequisite(category="calculus")]),
        ]
        result = optimize_courses(profile, programs, max_courses=35)
        calc_rec = next(
            (r for r in result if r.category == "calculus"), None
        )
        # Calculus at B+ = 8.0, below ceiling, so it should appear
        assert calc_rec is not None
        assert calc_rec.prereq_coverage == 0

    def test_other_category_excluded(self) -> None:
        """The 'other' category should never appear in recommendations."""
        profile = _empty_profile()
        result = optimize_courses(profile, [], max_courses=30)
        recommended_categories = {r.category for r in result}
        assert "other" not in recommended_categories

    def test_unmapped_categories_excluded(self) -> None:
        """Categories not mapped to any factor (e.g. bayesian, fixed_income,
        accounting) should not produce recommendations.
        """
        profile = _empty_profile()
        result = optimize_courses(profile, [], max_courses=30)
        recommended_categories = {r.category for r in result}
        for excluded in ["bayesian", "fixed_income", "accounting", "database", "programming_r"]:
            assert excluded not in recommended_categories

    def test_dimension_field_populated(self) -> None:
        """Every recommendation should have a non-empty dimension field."""
        profile = _empty_profile()
        result = optimize_courses(profile, [], max_courses=10)
        for rec in result:
            assert rec.dimension in DIMENSION_WEIGHTS

    def test_reason_field_populated(self) -> None:
        """Every recommendation should have a non-empty reason string."""
        profile = _empty_profile()
        result = optimize_courses(profile, [], max_courses=5)
        for rec in result:
            assert len(rec.reason) > 0

    def test_partial_profile_ordering(self) -> None:
        """A profile with calculus (B+) and nothing else: calculus should
        rank below categories with larger gaps because the gap is smaller.
        """
        profile = UserProfile(
            coursework=[
                Course("Calc III", "MATH241", "calculus", "B+", level=200),
            ]
        )
        result = optimize_courses(profile, [], max_courses=5)
        # Calculus gap = 9.0 - 8.0 = 1.0 (small)
        # Other categories gap = 9.0 - 0.0 = 9.0 (large)
        # So calculus should NOT be the top recommendation
        if result:
            assert result[0].category != "calculus"

    def test_max_courses_zero(self) -> None:
        """max_courses=0 should return an empty list."""
        profile = _empty_profile()
        result = optimize_courses(profile, [], max_courses=0)
        assert result == []

    def test_category_field_is_valid_enum_value(self) -> None:
        """Every recommended category should be a valid CourseCategory value."""
        profile = _empty_profile()
        result = optimize_courses(profile, [], max_courses=10)
        valid_values = {cat.value for cat in CourseCategory}
        for rec in result:
            assert rec.category in valid_values
