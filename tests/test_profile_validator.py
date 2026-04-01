# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for core.profile_validator."""


from core.models import Course, TestScores, UserProfile
from core.profile_validator import ValidationWarning, validate_profile

# ===================================================================
# Helpers
# ===================================================================


def _make_complete_profile() -> UserProfile:
    """A fully valid profile that should produce zero errors/warnings."""
    return UserProfile(
        name="Jane Doe",
        gpa=3.8,
        gpa_quant=3.9,
        university="UIUC",
        majors=["Mathematics", "Computer Science"],
        coursework=[
            Course("Calc III", "MATH241", "calculus", "A", level=200),
            Course("Linear Algebra", "MATH415", "linear_algebra", "A", level=400),
            Course("Probability", "STAT400", "probability", "A-", level=400),
            Course("Real Analysis", "MATH447", "real_analysis", "A", level=400),
            Course("C++ Prog", "CS225", "programming_cpp", "A", level=200),
        ],
        test_scores=TestScores(gre_quant=170, gre_verbal=160, toefl=110),
        is_international=False,
    )


def _warnings_for_field(
    warnings: list[ValidationWarning], field: str
) -> list[ValidationWarning]:
    """Filter warnings to those targeting a specific field."""
    return [w for w in warnings if w.field == field]


def _warnings_for_level(
    warnings: list[ValidationWarning], level: str
) -> list[ValidationWarning]:
    """Filter warnings to a specific level."""
    return [w for w in warnings if w.level == level]


# ===================================================================
# ValidationWarning dataclass
# ===================================================================


class TestValidationWarningDataclass:
    """Basic tests on the ValidationWarning dataclass itself."""

    def test_fields_accessible(self) -> None:
        w = ValidationWarning(level="error", field="gpa", message="bad")
        assert w.level == "error"
        assert w.field == "gpa"
        assert w.message == "bad"

    def test_equality(self) -> None:
        a = ValidationWarning(level="info", field="x", message="m")
        b = ValidationWarning(level="info", field="x", message="m")
        assert a == b


# ===================================================================
# Perfect profile (no warnings)
# ===================================================================


class TestPerfectProfile:
    """A complete, valid profile should produce no errors or warnings."""

    def test_no_warnings(self) -> None:
        profile = _make_complete_profile()
        warnings = validate_profile(profile)
        errors = _warnings_for_level(warnings, "error")
        warns = _warnings_for_level(warnings, "warning")
        assert errors == []
        assert warns == []

    def test_only_infos_allowed(self) -> None:
        """A perfect profile may have at most info-level notes."""
        profile = _make_complete_profile()
        warnings = validate_profile(profile)
        for w in warnings:
            assert w.level == "info"


# ===================================================================
# Missing name
# ===================================================================


class TestMissingName:
    """Check 6: warn when the profile name is empty."""

    def test_empty_name_warns(self) -> None:
        profile = _make_complete_profile()
        profile.name = ""
        warnings = validate_profile(profile)
        name_warnings = _warnings_for_field(warnings, "name")
        assert len(name_warnings) == 1
        assert name_warnings[0].level == "warning"

    def test_whitespace_only_name_warns(self) -> None:
        profile = _make_complete_profile()
        profile.name = "   "
        warnings = validate_profile(profile)
        name_warnings = _warnings_for_field(warnings, "name")
        assert len(name_warnings) == 1

    def test_valid_name_no_warning(self) -> None:
        profile = _make_complete_profile()
        warnings = validate_profile(profile)
        name_warnings = _warnings_for_field(warnings, "name")
        assert name_warnings == []


# ===================================================================
# GPA anomaly
# ===================================================================


class TestGpaAnomaly:
    """Check 2: GPA out of range or suspiciously low."""

    def test_gpa_above_4_is_error(self) -> None:
        profile = _make_complete_profile()
        profile.gpa = 4.5
        warnings = validate_profile(profile)
        gpa_errors = [
            w for w in warnings if w.field == "gpa" and w.level == "error"
        ]
        assert len(gpa_errors) == 1
        assert "outside" in gpa_errors[0].message

    def test_gpa_negative_is_error(self) -> None:
        profile = _make_complete_profile()
        profile.gpa = -0.5
        warnings = validate_profile(profile)
        gpa_errors = [
            w for w in warnings if w.field == "gpa" and w.level == "error"
        ]
        assert len(gpa_errors) == 1

    def test_gpa_below_1_warns_scale(self) -> None:
        profile = _make_complete_profile()
        profile.gpa = 0.5
        warnings = validate_profile(profile)
        gpa_warns = [
            w for w in warnings if w.field == "gpa" and w.level == "warning"
        ]
        assert len(gpa_warns) == 1
        assert "scale" in gpa_warns[0].message.lower()

    def test_gpa_exactly_0_no_scale_warning(self) -> None:
        """GPA of 0.0 is the default; don't flag as wrong scale."""
        profile = _make_complete_profile()
        profile.gpa = 0.0
        warnings = validate_profile(profile)
        gpa_scale = [
            w
            for w in warnings
            if w.field == "gpa" and "scale" in w.message.lower()
        ]
        assert gpa_scale == []

    def test_gpa_exactly_4_no_error(self) -> None:
        profile = _make_complete_profile()
        profile.gpa = 4.0
        warnings = validate_profile(profile)
        gpa_errors = [
            w for w in warnings if w.field == "gpa" and w.level == "error"
        ]
        assert gpa_errors == []

    def test_gpa_exactly_1_no_scale_warning(self) -> None:
        """GPA of exactly 1.0 should not trigger the below-1.0 warning."""
        profile = _make_complete_profile()
        profile.gpa = 1.0
        warnings = validate_profile(profile)
        gpa_scale = [
            w
            for w in warnings
            if w.field == "gpa" and "scale" in w.message.lower()
        ]
        assert gpa_scale == []

    def test_gpa_boundary_just_above_4(self) -> None:
        profile = _make_complete_profile()
        profile.gpa = 4.01
        warnings = validate_profile(profile)
        gpa_errors = [
            w for w in warnings if w.field == "gpa" and w.level == "error"
        ]
        assert len(gpa_errors) == 1


# ===================================================================
# Core prerequisites
# ===================================================================


class TestCorePrerequisites:
    """Check 1: warn when calculus, linear_algebra, or probability is missing."""

    def test_all_present_no_warning(self) -> None:
        profile = _make_complete_profile()
        warnings = validate_profile(profile)
        prereq_warns = [
            w
            for w in warnings
            if w.field == "coursework" and "prerequisite" in w.message.lower()
        ]
        assert prereq_warns == []

    def test_missing_one_category(self) -> None:
        profile = _make_complete_profile()
        # Remove probability
        profile.coursework = [
            c for c in profile.coursework if c.category != "probability"
        ]
        warnings = validate_profile(profile)
        prereq_warns = [
            w
            for w in warnings
            if w.field == "coursework" and "prerequisite" in w.message.lower()
        ]
        assert len(prereq_warns) == 1
        assert "probability" in prereq_warns[0].message

    def test_missing_all_three(self) -> None:
        profile = _make_complete_profile()
        profile.coursework = [
            Course("Art", "ART100", "other", "A"),
            Course("Music", "MUS100", "other", "A"),
            Course("Dance", "DAN100", "other", "A"),
            Course("Film", "FLM100", "other", "A"),
            Course("Theater", "THR100", "other", "A"),
        ]
        warnings = validate_profile(profile)
        prereq_warns = [
            w
            for w in warnings
            if w.field == "coursework" and "prerequisite" in w.message.lower()
        ]
        assert len(prereq_warns) == 1
        msg = prereq_warns[0].message
        assert "calculus" in msg
        assert "linear_algebra" in msg
        assert "probability" in msg

    def test_missing_two_categories(self) -> None:
        profile = _make_complete_profile()
        profile.coursework = [
            c
            for c in profile.coursework
            if c.category not in ("linear_algebra", "probability")
        ]
        warnings = validate_profile(profile)
        prereq_warns = [
            w
            for w in warnings
            if w.field == "coursework" and "prerequisite" in w.message.lower()
        ]
        assert len(prereq_warns) == 1
        msg = prereq_warns[0].message
        assert "linear_algebra" in msg
        assert "probability" in msg
        assert "calculus" not in msg


# ===================================================================
# Insufficient coursework
# ===================================================================


class TestInsufficientCoursework:
    """Check 4: warn when fewer than 5 courses are listed."""

    def test_zero_courses_warns(self) -> None:
        profile = _make_complete_profile()
        profile.coursework = []
        warnings = validate_profile(profile)
        few_warns = [
            w
            for w in warnings
            if w.field == "coursework" and "unreliable" in w.message.lower()
        ]
        assert len(few_warns) == 1

    def test_four_courses_warns(self) -> None:
        profile = _make_complete_profile()
        profile.coursework = profile.coursework[:4]
        warnings = validate_profile(profile)
        few_warns = [
            w
            for w in warnings
            if w.field == "coursework" and "unreliable" in w.message.lower()
        ]
        assert len(few_warns) == 1

    def test_five_courses_no_warning(self) -> None:
        profile = _make_complete_profile()
        assert len(profile.coursework) == 5
        warnings = validate_profile(profile)
        few_warns = [
            w
            for w in warnings
            if w.field == "coursework" and "unreliable" in w.message.lower()
        ]
        assert few_warns == []

    def test_many_courses_no_warning(self) -> None:
        profile = _make_complete_profile()
        profile.coursework.extend([
            Course("ODE", "MATH330", "ode", "A", level=300),
            Course("PDE", "MATH442", "pde", "A-", level=400),
        ])
        warnings = validate_profile(profile)
        few_warns = [
            w
            for w in warnings
            if w.field == "coursework" and "unreliable" in w.message.lower()
        ]
        assert few_warns == []

    def test_count_in_message(self) -> None:
        profile = _make_complete_profile()
        profile.coursework = profile.coursework[:3]
        warnings = validate_profile(profile)
        few_warns = [
            w
            for w in warnings
            if w.field == "coursework" and "unreliable" in w.message.lower()
        ]
        assert "3" in few_warns[0].message


# ===================================================================
# Duplicate course codes
# ===================================================================


class TestDuplicateCourseCodes:
    """Check 7: warn when any course code appears more than once."""

    def test_no_duplicates_no_warning(self) -> None:
        profile = _make_complete_profile()
        warnings = validate_profile(profile)
        dup_warns = [
            w
            for w in warnings
            if w.field == "coursework" and "Duplicate" in w.message
        ]
        assert dup_warns == []

    def test_single_duplicate(self) -> None:
        profile = _make_complete_profile()
        profile.coursework.append(
            Course("Calc III dup", "MATH241", "calculus", "A", level=200)
        )
        warnings = validate_profile(profile)
        dup_warns = [
            w
            for w in warnings
            if w.field == "coursework" and "Duplicate" in w.message
        ]
        assert len(dup_warns) == 1
        assert "MATH241" in dup_warns[0].message

    def test_multiple_duplicates(self) -> None:
        profile = _make_complete_profile()
        profile.coursework.append(
            Course("Calc dup", "MATH241", "calculus", "B+", level=200)
        )
        profile.coursework.append(
            Course("LinAlg dup", "MATH415", "linear_algebra", "B", level=400)
        )
        warnings = validate_profile(profile)
        dup_warns = [
            w
            for w in warnings
            if w.field == "coursework" and "Duplicate" in w.message
        ]
        assert len(dup_warns) == 1  # single warning listing all duplicates
        assert "MATH241" in dup_warns[0].message
        assert "MATH415" in dup_warns[0].message


# ===================================================================
# International TOEFL hint
# ===================================================================


class TestInternationalToefl:
    """Check 3: info hint for recent international students."""

    def test_international_under_2_years(self) -> None:
        profile = _make_complete_profile()
        profile.is_international = True
        profile.years_at_us_institution = 1
        warnings = validate_profile(profile)
        toefl_infos = [
            w
            for w in warnings
            if w.field == "test_scores" and "TOEFL" in w.message
        ]
        assert len(toefl_infos) == 1
        assert toefl_infos[0].level == "info"

    def test_international_zero_years(self) -> None:
        profile = _make_complete_profile()
        profile.is_international = True
        profile.years_at_us_institution = 0
        warnings = validate_profile(profile)
        toefl_infos = [
            w
            for w in warnings
            if w.field == "test_scores" and "TOEFL" in w.message
        ]
        assert len(toefl_infos) == 1

    def test_international_2_years_no_hint(self) -> None:
        profile = _make_complete_profile()
        profile.is_international = True
        profile.years_at_us_institution = 2
        warnings = validate_profile(profile)
        toefl_infos = [
            w
            for w in warnings
            if w.field == "test_scores" and "TOEFL" in w.message
        ]
        assert toefl_infos == []

    def test_international_4_years_no_hint(self) -> None:
        profile = _make_complete_profile()
        profile.is_international = True
        profile.years_at_us_institution = 4
        warnings = validate_profile(profile)
        toefl_infos = [
            w
            for w in warnings
            if w.field == "test_scores" and "TOEFL" in w.message
        ]
        assert toefl_infos == []

    def test_domestic_no_hint(self) -> None:
        profile = _make_complete_profile()
        profile.is_international = False
        profile.years_at_us_institution = 0
        warnings = validate_profile(profile)
        toefl_infos = [
            w
            for w in warnings
            if w.field == "test_scores" and "TOEFL" in w.message
        ]
        assert toefl_infos == []


# ===================================================================
# No test scores
# ===================================================================


class TestNoTestScores:
    """Check 5: info when GRE quant is missing on a non-empty profile."""

    def test_no_gre_on_nonempty_profile(self) -> None:
        profile = _make_complete_profile()
        profile.test_scores = TestScores()
        warnings = validate_profile(profile)
        gre_infos = [
            w
            for w in warnings
            if w.field == "test_scores" and "GRE" in w.message
        ]
        assert len(gre_infos) == 1
        assert gre_infos[0].level == "info"

    def test_gre_present_no_info(self) -> None:
        profile = _make_complete_profile()
        warnings = validate_profile(profile)
        gre_infos = [
            w
            for w in warnings
            if w.field == "test_scores" and "GRE" in w.message
        ]
        assert gre_infos == []

    def test_empty_profile_no_gre_hint(self) -> None:
        """A truly empty profile should not nag about GRE."""
        profile = UserProfile()
        warnings = validate_profile(profile)
        gre_infos = [
            w
            for w in warnings
            if w.field == "test_scores" and "GRE" in w.message
        ]
        assert gre_infos == []


# ===================================================================
# Multiple warnings firing simultaneously
# ===================================================================


class TestMultipleWarnings:
    """Verify that several checks can fire at once."""

    def test_empty_profile_fires_multiple(self) -> None:
        """A default-constructed profile triggers name + coursework warnings."""
        profile = UserProfile()
        warnings = validate_profile(profile)
        fields_hit = {w.field for w in warnings}
        assert "name" in fields_hit
        assert "coursework" in fields_hit

    def test_bad_gpa_and_missing_prereqs(self) -> None:
        profile = UserProfile(
            name="Test",
            gpa=5.0,
            coursework=[
                Course("Art", "ART100", "other", "A"),
                Course("Music", "MUS100", "other", "A"),
                Course("Dance", "DAN100", "other", "A"),
                Course("Film", "FLM100", "other", "A"),
                Course("Theater", "THR100", "other", "A"),
            ],
        )
        warnings = validate_profile(profile)
        levels = {w.level for w in warnings}
        assert "error" in levels  # GPA
        assert "warning" in levels  # missing prereqs

    def test_international_with_no_gre(self) -> None:
        """International + no GRE should produce two info-level notes."""
        profile = _make_complete_profile()
        profile.is_international = True
        profile.years_at_us_institution = 0
        profile.test_scores = TestScores()
        warnings = validate_profile(profile)
        infos = _warnings_for_level(warnings, "info")
        assert len(infos) == 2

    def test_all_possible_warnings_at_once(self) -> None:
        """Construct a profile that triggers every single check."""
        profile = UserProfile(
            name="",
            gpa=-1.0,
            is_international=True,
            years_at_us_institution=0,
            coursework=[
                Course("Art", "ART100", "other", "A"),
                Course("Art dup", "ART100", "other", "B"),
            ],
            test_scores=TestScores(),
        )
        warnings = validate_profile(profile)
        # Should have: name(warning), gpa(error), core prereqs(warning),
        # insufficient coursework(warning), duplicate codes(warning),
        # international toefl(info), no GRE(info) -- but GRE needs non-empty
        # profile; name="" but coursework non-empty so it counts as non-empty.
        assert len(warnings) >= 6
        levels = {w.level for w in warnings}
        assert "error" in levels
        assert "warning" in levels
        assert "info" in levels


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Miscellaneous edge-case coverage."""

    def test_return_type_is_list(self) -> None:
        result = validate_profile(UserProfile())
        assert isinstance(result, list)

    def test_warnings_are_validation_warning_instances(self) -> None:
        result = validate_profile(UserProfile())
        for w in result:
            assert isinstance(w, ValidationWarning)

    def test_gpa_exactly_boundary_0(self) -> None:
        """GPA 0.0 should not trigger error or scale warning."""
        profile = _make_complete_profile()
        profile.gpa = 0.0
        warnings = validate_profile(profile)
        gpa_warns = _warnings_for_field(warnings, "gpa")
        assert gpa_warns == []
