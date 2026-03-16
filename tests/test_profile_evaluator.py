"""Tests for core.profile_evaluator."""

import pytest

from core.models import Course, EvaluationResult, TestScores, UserProfile
from core.profile_evaluator import (
    _GAP_THRESHOLD,
    _STRENGTH_THRESHOLD,
    DIMENSION_WEIGHTS,
    _best_score_for_categories,
    _best_score_for_category,
    _count_courses,
    _gpa_to_score,
    _has_major,
    evaluate,
    grade_to_score,
)

# ===================================================================
# grade_to_score
# ===================================================================

class TestGradeToScore:
    """Test letter-grade, numeric-grade, and edge-case conversions."""

    @pytest.mark.parametrize("grade,expected", [
        ("A+", 10.0),
        ("A",  10.0),
        ("A-", 9.0),
        ("B+", 8.0),
        ("B",  7.0),
        ("B-", 6.0),
        ("C+", 5.0),
        ("C",  4.0),
    ])
    def test_letter_grades(self, grade: str, expected: float) -> None:
        assert grade_to_score(grade) == expected

    @pytest.mark.parametrize("grade,expected", [
        ("95",  10.0),
        ("90",  10.0),
        ("89",  9.0),
        ("85",  9.0),
        ("84",  8.0),
        ("80",  8.0),
        ("79",  7.0),
        ("76",  7.0),
        ("75",  6.0),
        ("70",  6.0),
        ("65",  5.0),
        ("60",  5.0),
        ("59",  4.0),
        ("30",  4.0),
    ])
    def test_numeric_grades(self, grade: str, expected: float) -> None:
        assert grade_to_score(grade) == expected

    def test_cr_transfer_credit(self) -> None:
        assert grade_to_score("CR") == 8.0

    def test_cr_lowercase(self) -> None:
        # CR is checked via .upper(), so lowercase also works
        assert grade_to_score("cr") == 8.0

    def test_empty_string(self) -> None:
        assert grade_to_score("") == 0.0

    def test_none_string(self) -> None:
        assert grade_to_score("None") == 0.0

    def test_none_mixed_case(self) -> None:
        assert grade_to_score("none") == 0.0
        assert grade_to_score("NONE") == 0.0

    def test_whitespace_only(self) -> None:
        assert grade_to_score("   ") == 0.0

    def test_whitespace_around_grade(self) -> None:
        assert grade_to_score("  A+  ") == 10.0

    def test_unrecognised_string(self) -> None:
        assert grade_to_score("Pass") == 0.0
        assert grade_to_score("XYZ") == 0.0

    def test_numeric_boundary_90(self) -> None:
        """90 should map to 10.0 (>= 90 branch)."""
        assert grade_to_score("90") == 10.0

    def test_numeric_float_string(self) -> None:
        """Float numeric strings should work via float() conversion."""
        assert grade_to_score("92.5") == 10.0
        assert grade_to_score("85.0") == 9.0


# ===================================================================
# grade_to_score — international grading systems
# ===================================================================

class TestGradeToScoreInternational:
    """Test Indian CGPA and UK degree classification conversions."""

    @pytest.mark.parametrize("grade,expected", [
        ("9.5",  10.0),   # 9.0+ -> 10.0
        ("9.0",  10.0),   # boundary: exactly 9.0 -> 10.0
        ("8.5",  9.0),    # 8.0-8.9 -> 9.0
        ("7.5",  8.0),    # 7.0-7.9 -> 8.0
        ("6.5",  7.0),    # 6.0-6.9 -> 7.0
        ("5.5",  6.0),    # 5.0-5.9 -> 6.0
        ("4.5",  5.0),    # 4.0-4.9 -> 5.0
        ("3.5",  4.0),    # < 4.0 -> 4.0
    ])
    def test_indian_cgpa(self, grade: str, expected: float) -> None:
        assert grade_to_score(grade) == expected

    @pytest.mark.parametrize("grade,expected", [
        ("First",        10.0),
        ("1st",          10.0),
        ("2:1",          8.5),
        ("Upper Second", 8.5),
        ("2:2",          7.0),
        ("Lower Second", 7.0),
        ("Third",        5.0),
        ("3rd",          5.0),
    ])
    def test_uk_classification(self, grade: str, expected: float) -> None:
        assert grade_to_score(grade) == expected

    def test_indian_cgpa_top_boundary(self) -> None:
        """10.0 is within the Indian CGPA range (1-10) and should map to 10.0."""
        assert grade_to_score("10.0") == 10.0

    def test_boundary_above_indian_scale(self) -> None:
        """11.0 is outside the Indian CGPA range and falls into percentage scale."""
        # 11.0 is < 60 on the percentage scale -> 4.0
        assert grade_to_score("11.0") == 4.0

    def test_indian_cgpa_lower_boundary(self) -> None:
        """1.0 is the lower bound of the Indian CGPA range -> 4.0 (below 4.0 threshold)."""
        assert grade_to_score("1.0") == 4.0


# ===================================================================
# _gpa_to_score
# ===================================================================

class TestGpaToScore:
    """Test the piecewise-linear GPA-to-10-scale mapping."""

    @pytest.mark.parametrize("gpa,expected", [
        (4.0, 10.0),
        (3.9, 9.5),
        (3.8, 9.0),
        (3.7, 8.5),
        (3.5, 7.5),
        (3.3, 6.5),
        (3.0, 5.0),
        (2.5, 3.0),
        (2.0, 1.0),
    ])
    def test_exact_breakpoints(self, gpa: float, expected: float) -> None:
        assert _gpa_to_score(gpa) == pytest.approx(expected)

    def test_above_4_0_caps_at_10(self) -> None:
        assert _gpa_to_score(4.1) == 10.0
        assert _gpa_to_score(5.0) == 10.0

    def test_below_2_0_returns_1(self) -> None:
        # The function returns max(0.0, breakpoints[-1][1]) = max(0.0, 1.0) = 1.0
        # for gpa <= 2.0
        assert _gpa_to_score(1.9) == pytest.approx(1.0)
        assert _gpa_to_score(1.0) == pytest.approx(1.0)

    def test_zero_gpa(self) -> None:
        assert _gpa_to_score(0.0) == 0.0

    def test_negative_gpa(self) -> None:
        assert _gpa_to_score(-1.0) == 0.0

    def test_interpolation_3_95(self) -> None:
        """3.95 is midway between 3.9 (9.5) and 4.0 (10.0) -> 9.75."""
        assert _gpa_to_score(3.95) == pytest.approx(9.75)

    def test_interpolation_3_65(self) -> None:
        """3.65 is midway between 3.5 (7.5) and 3.7 (8.5) -> 8.25."""
        # frac = (3.65 - 3.5) / (3.7 - 3.5) = 0.15/0.20 = 0.75
        # result = 7.5 + 0.75 * (8.5 - 7.5) = 7.5 + 0.75 = 8.25
        assert _gpa_to_score(3.65) == pytest.approx(8.25)

    def test_interpolation_2_75(self) -> None:
        """2.75 is midway between 2.5 (3.0) and 3.0 (5.0) -> 4.0."""
        assert _gpa_to_score(2.75) == pytest.approx(4.0)


# ===================================================================
# Helper functions
# ===================================================================

class TestHelperFunctions:
    """Test _best_score_for_category, _best_score_for_categories, etc."""

    def _make_courses(self) -> list[Course]:
        return [
            Course(name="Calc I", code="MATH101", category="calculus", grade="A"),
            Course(name="Calc II", code="MATH102", category="calculus", grade="B+"),
            Course(name="LinAlg", code="MATH200", category="linear_algebra", grade="A-"),
        ]

    def test_best_score_single_category(self) -> None:
        courses = self._make_courses()
        # Best calculus grade is "A" -> 10.0
        assert _best_score_for_category(courses, "calculus") == 10.0

    def test_best_score_missing_category(self) -> None:
        courses = self._make_courses()
        assert _best_score_for_category(courses, "probability") == 0.0

    def test_best_score_multi_categories(self) -> None:
        courses = self._make_courses()
        # {calculus, probability} -> best is calculus "A" -> 10.0
        assert _best_score_for_categories(courses, {"calculus", "probability"}) == 10.0

    def test_best_score_empty_courses(self) -> None:
        assert _best_score_for_category([], "calculus") == 0.0

    def test_count_courses_all(self) -> None:
        courses = self._make_courses()
        assert _count_courses(courses) == 3

    def test_count_courses_by_category(self) -> None:
        courses = self._make_courses()
        assert _count_courses(courses, {"calculus"}) == 2
        assert _count_courses(courses, {"linear_algebra"}) == 1
        assert _count_courses(courses, {"probability"}) == 0

    def test_count_courses_min_level(self) -> None:
        courses = [
            Course(name="A", code="A", category="statistics", grade="A", level=300),
            Course(name="B", code="B", category="statistics", grade="A", level=400),
            Course(name="C", code="C", category="statistics", grade="A", level=500),
        ]
        assert _count_courses(courses, {"statistics"}, min_level=400) == 2
        assert _count_courses(courses, {"statistics"}, min_level=500) == 1

    def test_has_major_positive(self) -> None:
        profile = UserProfile(majors=["Computer Science", "Mathematics"])
        assert _has_major(profile, {"computer science"}) is True
        assert _has_major(profile, {"math"}) is True  # "math" in "mathematics"

    def test_has_major_negative(self) -> None:
        profile = UserProfile(majors=["Economics"])
        assert _has_major(profile, {"computer science", "computing"}) is False

    def test_has_major_empty(self) -> None:
        profile = UserProfile(majors=[])
        assert _has_major(profile, {"anything"}) is False


# ===================================================================
# evaluate (full integration)
# ===================================================================

class TestEvaluate:
    """Test the full evaluate() pipeline with a mock profile."""

    def _make_strong_profile(self) -> UserProfile:
        """A strong profile with courses across all dimensions."""
        return UserProfile(
            name="Test Student",
            gpa=3.9,
            gpa_quant=3.95,
            majors=["Computer Science", "Statistics"],
            coursework=[
                # Math
                Course("Calc III", "MATH241", "calculus", "A+", level=200),
                Course("Linear Algebra", "MATH415", "linear_algebra", "A", level=400),
                Course("Probability", "STAT400", "probability", "A-", level=400),
                Course("Real Analysis", "MATH447", "real_analysis", "A", level=400),
                Course("Stochastic", "STAT455", "stochastic_processes", "B+", level=400),
                # Stats
                Course("Math Stats", "STAT410", "statistics", "A", level=400),
                Course("Time Series", "STAT429", "time_series", "A-", level=400),
                Course("Econometrics", "ECON471", "econometrics", "A", level=400),
                # CS
                Course("C++ Prog", "CS225", "programming_cpp", "A", level=200),
                Course("Python", "CS101", "programming_python", "A+", level=100),
                Course("Data Structures", "CS225", "data_structures", "A", level=200),
                Course("ML", "CS446", "machine_learning", "A-", level=400),
                # Finance / Econ
                Course("Microecon", "ECON302", "microeconomics", "A", level=300),
                Course("Finance", "FIN321", "finance", "A-", level=300),
            ],
            test_scores=TestScores(gre_quant=170, gre_verbal=160, toefl=110),
        )

    def _make_weak_profile(self) -> UserProfile:
        """A profile with minimal coursework."""
        return UserProfile(
            name="Sparse Student",
            gpa=3.0,
            majors=["Economics"],
            coursework=[
                Course("Calc I", "MATH101", "calculus", "B", level=100),
                Course("Intro Stats", "STAT100", "statistics", "B+", level=100),
            ],
        )

    def test_evaluate_returns_evaluation_result(self) -> None:
        profile = self._make_strong_profile()
        result = evaluate(profile)
        assert isinstance(result, EvaluationResult)

    def test_evaluate_has_all_dimensions(self) -> None:
        profile = self._make_strong_profile()
        result = evaluate(profile)
        for dim in DIMENSION_WEIGHTS:
            assert dim in result.dimension_scores

    def test_evaluate_scores_in_range(self) -> None:
        profile = self._make_strong_profile()
        result = evaluate(profile)
        for dim, score in result.dimension_scores.items():
            assert 0.0 <= score <= 10.0, f"{dim} score {score} out of range"
        assert 0.0 <= result.overall_score <= 10.0

    def test_strong_profile_higher_than_weak(self) -> None:
        strong = evaluate(self._make_strong_profile())
        weak = evaluate(self._make_weak_profile())
        assert strong.overall_score > weak.overall_score

    def test_gaps_below_threshold(self) -> None:
        """All gap entries should have score < _GAP_THRESHOLD or == 0."""
        profile = self._make_weak_profile()
        result = evaluate(profile)
        for gap in result.gaps:
            assert gap["score"] < _GAP_THRESHOLD or gap["score"] == 0

    def test_strengths_above_threshold(self) -> None:
        """All strength entries should have score >= _STRENGTH_THRESHOLD."""
        profile = self._make_strong_profile()
        result = evaluate(profile)
        for s in result.strengths:
            assert s["score"] >= _STRENGTH_THRESHOLD

    def test_gaps_sorted_ascending(self) -> None:
        profile = self._make_weak_profile()
        result = evaluate(profile)
        if len(result.gaps) > 1:
            scores = [g["score"] for g in result.gaps]
            assert scores == sorted(scores)

    def test_strengths_sorted_descending(self) -> None:
        profile = self._make_strong_profile()
        result = evaluate(profile)
        if len(result.strengths) > 1:
            scores = [s["score"] for s in result.strengths]
            assert scores == sorted(scores, reverse=True)

    def test_overall_is_weighted_sum(self) -> None:
        """overall_score should equal the weighted sum of dimension scores."""
        profile = self._make_strong_profile()
        result = evaluate(profile)
        expected = sum(
            DIMENSION_WEIGHTS[dim] * result.dimension_scores[dim]
            for dim in DIMENSION_WEIGHTS
        )
        assert result.overall_score == pytest.approx(round(expected, 2))

    def test_empty_profile(self) -> None:
        """A profile with zero coursework should produce all-zero math/stats/cs/fin scores."""
        profile = UserProfile(name="Empty", gpa=0.0)
        result = evaluate(profile)
        assert result.dimension_scores["math"] == 0.0
        assert result.dimension_scores["statistics"] == 0.0
        assert result.dimension_scores["cs"] == 0.0
        assert result.dimension_scores["finance_econ"] == 0.0

    def test_cs_major_bonus(self) -> None:
        """A CS major should get the is_cs_major bonus in the CS dimension."""
        cs_profile = UserProfile(
            name="CS Major", gpa=3.5, majors=["Computer Science"],
            coursework=[
                Course("C++", "CS101", "programming_cpp", "A", level=100),
            ],
        )
        non_cs_profile = UserProfile(
            name="Non-CS", gpa=3.5, majors=["Mathematics"],
            coursework=[
                Course("C++", "CS101", "programming_cpp", "A", level=100),
            ],
        )
        cs_result = evaluate(cs_profile)
        non_cs_result = evaluate(non_cs_profile)
        assert cs_result.dimension_scores["cs"] > non_cs_result.dimension_scores["cs"]

    def test_gpa_dimension_trend_with_upper_courses(self) -> None:
        """Profile with 400-level courses should use computed trend, not default 7."""
        profile = UserProfile(
            name="Upper",
            gpa=3.5,
            coursework=[
                Course("Adv Stats", "STAT400", "statistics", "A+", level=400),
                Course("Adv Calc", "MATH400", "calculus", "A", level=400),
            ],
        )
        result = evaluate(profile)
        # Trend should be avg of upper-level grades: (10+10)/2 = 10
        # GPA dimension = 0.50 * gpa_score + 0.30 * gpa_score + 0.20 * 10
        assert result.dimension_scores["gpa"] > 0.0

    def test_school_recommendations_empty(self) -> None:
        """evaluate() returns empty school_recommendations (filled later by ranker)."""
        profile = self._make_strong_profile()
        result = evaluate(profile)
        assert result.school_recommendations == {"reach": [], "target": [], "safety": []}
