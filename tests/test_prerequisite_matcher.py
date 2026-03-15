"""Tests for core.prerequisite_matcher."""

import pytest

from core.models import Course, Prerequisite, PrerequisiteMatch, ProgramData, UserProfile
from core.prerequisite_matcher import (
    match_prerequisites,
    _grade_meets_minimum,
    _find_matching_courses,
)


# ===================================================================
# _grade_meets_minimum
# ===================================================================

class TestGradeMeetsMinimum:
    """Test letter-grade and numeric-grade minimum comparisons."""

    def test_same_grade(self) -> None:
        assert _grade_meets_minimum("A", "A") is True
        assert _grade_meets_minimum("B+", "B+") is True

    def test_higher_grade(self) -> None:
        assert _grade_meets_minimum("A", "B+") is True
        assert _grade_meets_minimum("A+", "B") is True

    def test_lower_grade(self) -> None:
        assert _grade_meets_minimum("B", "A") is False
        assert _grade_meets_minimum("C", "B") is False

    def test_a_plus_vs_a(self) -> None:
        assert _grade_meets_minimum("A+", "A") is True

    def test_b_minus_vs_b(self) -> None:
        assert _grade_meets_minimum("B-", "B") is False

    def test_numeric_fallback(self) -> None:
        """When grades are numeric strings, fall back to grade_to_score."""
        assert _grade_meets_minimum("95", "80") is True
        assert _grade_meets_minimum("70", "90") is False


# ===================================================================
# _find_matching_courses
# ===================================================================

class TestFindMatchingCourses:
    """Test course-to-prerequisite category matching."""

    def _make_courses(self) -> list[Course]:
        return [
            Course("ODE", "MATH330", "ode", "A"),
            Course("PDE", "MATH442", "pde", "A-"),
            Course("Calc", "MATH241", "calculus", "A"),
            Course("Python", "CS101", "programming_python", "A"),
            Course("C++", "CS225", "programming_cpp", "B+"),
        ]

    def test_exact_category_match(self) -> None:
        courses = self._make_courses()
        result = _find_matching_courses(courses, "calculus")
        assert len(result) == 1
        assert result[0].name == "Calc"

    def test_broad_category_match(self) -> None:
        """'differential_equations' maps to both ode and pde."""
        courses = self._make_courses()
        result = _find_matching_courses(courses, "differential_equations")
        assert len(result) == 2
        names = {c.name for c in result}
        assert names == {"ODE", "PDE"}

    def test_programming_broad_match(self) -> None:
        """'programming' should match both cpp and python courses."""
        courses = self._make_courses()
        result = _find_matching_courses(courses, "programming")
        assert len(result) == 2

    def test_no_matching_courses(self) -> None:
        courses = self._make_courses()
        result = _find_matching_courses(courses, "real_analysis")
        assert result == []

    def test_unknown_category_uses_literal(self) -> None:
        """If category is not in _CATEGORY_MAP, uses literal match."""
        courses = [Course("Special", "SP100", "special_topic", "A")]
        result = _find_matching_courses(courses, "special_topic")
        assert len(result) == 1


# ===================================================================
# match_prerequisites (full matching)
# ===================================================================

class TestMatchPrerequisites:
    """Test the full prerequisite matching logic."""

    def _make_profile(self, courses: list[Course]) -> UserProfile:
        return UserProfile(name="Test", coursework=courses, gpa=3.8)

    def _make_program(
        self,
        required: list[Prerequisite] | None = None,
        recommended: list[Prerequisite] | None = None,
    ) -> ProgramData:
        return ProgramData(
            id="test-prog",
            name="Test Program",
            prerequisites_required=required or [],
            prerequisites_recommended=recommended or [],
        )

    def test_all_prereqs_matched(self) -> None:
        """When the user has all required courses, match_score = 1.0."""
        courses = [
            Course("Calc III", "MATH241", "calculus", "A"),
            Course("LinAlg", "MATH415", "linear_algebra", "A-"),
            Course("Prob", "STAT400", "probability", "A"),
        ]
        program = self._make_program(required=[
            Prerequisite(category="calculus"),
            Prerequisite(category="linear_algebra"),
            Prerequisite(category="probability"),
        ])
        profile = self._make_profile(courses)
        result = match_prerequisites(profile, program)

        assert isinstance(result, PrerequisiteMatch)
        assert result.match_score == 1.0
        assert len(result.matched) == 3
        assert len(result.missing) == 0

    def test_partial_match(self) -> None:
        """When the user is missing some required courses."""
        courses = [
            Course("Calc III", "MATH241", "calculus", "A"),
        ]
        program = self._make_program(required=[
            Prerequisite(category="calculus"),
            Prerequisite(category="linear_algebra"),
            Prerequisite(category="probability"),
        ])
        profile = self._make_profile(courses)
        result = match_prerequisites(profile, program)

        assert result.match_score == pytest.approx(1 / 3, abs=0.01)
        assert len(result.matched) == 1
        assert len(result.missing) == 2

    def test_no_prereqs_matched(self) -> None:
        """When the user has none of the required courses."""
        courses = [
            Course("Art History", "ART100", "other", "A"),
        ]
        program = self._make_program(required=[
            Prerequisite(category="calculus"),
            Prerequisite(category="linear_algebra"),
        ])
        profile = self._make_profile(courses)
        result = match_prerequisites(profile, program)

        assert result.match_score == 0.0
        assert len(result.matched) == 0
        assert len(result.missing) == 2

    def test_no_required_prereqs(self) -> None:
        """When the program has no required prereqs, match_score should be 1.0."""
        courses = [Course("Calc", "MATH101", "calculus", "A")]
        program = self._make_program(required=[])
        profile = self._make_profile(courses)
        result = match_prerequisites(profile, program)
        assert result.match_score == 1.0

    def test_min_grade_met(self) -> None:
        """When a prereq has min_grade and the student meets it."""
        courses = [
            Course("Calc", "MATH241", "calculus", "A"),
        ]
        program = self._make_program(required=[
            Prerequisite(category="calculus", min_grade="B+"),
        ])
        profile = self._make_profile(courses)
        result = match_prerequisites(profile, program)

        assert result.match_score == 1.0
        assert len(result.matched) == 1

    def test_min_grade_not_met(self) -> None:
        """When a prereq has min_grade and the student does NOT meet it."""
        courses = [
            Course("Calc", "MATH241", "calculus", "C+"),
        ]
        program = self._make_program(required=[
            Prerequisite(category="calculus", min_grade="B"),
        ])
        profile = self._make_profile(courses)
        result = match_prerequisites(profile, program)

        assert result.match_score == 0.0
        assert len(result.missing) == 1
        # Should also generate a warning about the grade
        assert len(result.warnings) >= 1
        assert "does not meet the minimum" in result.warnings[0]

    def test_best_course_picked_for_min_grade(self) -> None:
        """When multiple courses match, the best-graded one is checked."""
        courses = [
            Course("Calc I", "MATH101", "calculus", "C"),
            Course("Calc II", "MATH102", "calculus", "A"),  # best
        ]
        program = self._make_program(required=[
            Prerequisite(category="calculus", min_grade="B+"),
        ])
        profile = self._make_profile(courses)
        result = match_prerequisites(profile, program)

        assert result.match_score == 1.0
        assert result.matched[0]["student_grade"] == "A"

    def test_recommended_missing_generates_warning(self) -> None:
        """Missing recommended prereqs should produce warnings, not affect match_score."""
        courses = [
            Course("Calc", "MATH241", "calculus", "A"),
        ]
        program = self._make_program(
            required=[Prerequisite(category="calculus")],
            recommended=[Prerequisite(category="stochastic_calculus", note="helpful for derivatives")],
        )
        profile = self._make_profile(courses)
        result = match_prerequisites(profile, program)

        assert result.match_score == 1.0  # recommended doesn't affect score
        assert len(result.warnings) >= 1
        assert "Recommended" in result.warnings[0]

    def test_recommended_present_no_warning(self) -> None:
        """When recommended prereqs are satisfied, no warning is generated for them."""
        courses = [
            Course("Calc", "MATH241", "calculus", "A"),
            Course("Stoch", "STAT455", "stochastic_processes", "A"),
        ]
        program = self._make_program(
            required=[Prerequisite(category="calculus")],
            recommended=[Prerequisite(category="stochastic_calculus")],
        )
        profile = self._make_profile(courses)
        result = match_prerequisites(profile, program)

        assert result.match_score == 1.0
        # stochastic_calculus maps to stochastic_processes, so should be found
        assert len(result.warnings) == 0

    def test_program_id_in_result(self) -> None:
        """The result should carry the program_id."""
        program = self._make_program(required=[Prerequisite(category="calculus")])
        profile = self._make_profile([Course("Calc", "M101", "calculus", "A")])
        result = match_prerequisites(profile, program)
        assert result.program_id == "test-prog"

    def test_match_score_precision(self) -> None:
        """match_score is rounded to 3 decimal places."""
        courses = [Course("Calc", "M101", "calculus", "A")]
        program = self._make_program(required=[
            Prerequisite(category="calculus"),
            Prerequisite(category="linear_algebra"),
            Prerequisite(category="probability"),
        ])
        profile = self._make_profile(courses)
        result = match_prerequisites(profile, program)
        # 1/3 -> 0.333
        assert result.match_score == pytest.approx(0.333, abs=0.001)
