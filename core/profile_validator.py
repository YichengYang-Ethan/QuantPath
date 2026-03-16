"""Profile validation for QuantPath.

Validates a UserProfile and returns a list of warnings/issues that
should be surfaced to the user before running the full evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.models import UserProfile


@dataclass
class ValidationWarning:
    """A single validation finding."""

    level: str  # "error", "warning", "info"
    field: str  # which field has the issue
    message: str  # human-readable description


# Core prerequisite categories that virtually all MFE programs require.
_CORE_PREREQ_CATEGORIES = {"calculus", "linear_algebra", "probability"}


def validate_profile(profile: UserProfile) -> list[ValidationWarning]:
    """Validate a UserProfile and return all warnings/issues found.

    Checks performed (in order):
      1. Missing core prerequisites (calculus, linear_algebra, probability)
      2. GPA anomalies (out of range, possibly wrong scale)
      3. International TOEFL waiver hint
      4. Insufficient coursework for reliable evaluation
      5. Missing GRE scores
      6. Missing name
      7. Duplicate course codes
    """
    warnings: list[ValidationWarning] = []

    _check_missing_name(profile, warnings)
    _check_gpa_anomaly(profile, warnings)
    _check_core_prerequisites(profile, warnings)
    _check_insufficient_coursework(profile, warnings)
    _check_duplicate_course_codes(profile, warnings)
    _check_international_toefl(profile, warnings)
    _check_no_test_scores(profile, warnings)

    return warnings


# -------------------------------------------------------------------
# Individual check helpers
# -------------------------------------------------------------------


def _check_missing_name(
    profile: UserProfile, warnings: list[ValidationWarning]
) -> None:
    """Warn if the profile name is empty or whitespace-only."""
    if not profile.name or not profile.name.strip():
        warnings.append(
            ValidationWarning(
                level="warning",
                field="name",
                message="Profile name is missing.",
            )
        )


def _check_gpa_anomaly(
    profile: UserProfile, warnings: list[ValidationWarning]
) -> None:
    """Error on impossible GPA values; warn on likely scale mismatch."""
    if profile.gpa > 4.0 or profile.gpa < 0.0:
        warnings.append(
            ValidationWarning(
                level="error",
                field="gpa",
                message=(
                    f"GPA value {profile.gpa} is outside the valid 0.0-4.0 "
                    "range."
                ),
            )
        )
    elif 0.0 < profile.gpa < 1.0:
        warnings.append(
            ValidationWarning(
                level="warning",
                field="gpa",
                message=(
                    f"GPA value {profile.gpa} is unusually low. This might "
                    "indicate a different grading scale."
                ),
            )
        )


def _check_core_prerequisites(
    profile: UserProfile, warnings: list[ValidationWarning]
) -> None:
    """Warn about missing core prerequisite categories."""
    present_categories = {c.category for c in profile.coursework}
    missing = _CORE_PREREQ_CATEGORIES - present_categories
    if missing:
        sorted_missing = sorted(missing)
        warnings.append(
            ValidationWarning(
                level="warning",
                field="coursework",
                message=(
                    "Missing core prerequisite courses: "
                    + ", ".join(sorted_missing)
                    + ". These are required by virtually all MFE programs."
                ),
            )
        )


def _check_insufficient_coursework(
    profile: UserProfile, warnings: list[ValidationWarning]
) -> None:
    """Warn when there are too few courses for a reliable evaluation."""
    if len(profile.coursework) < 5:
        warnings.append(
            ValidationWarning(
                level="warning",
                field="coursework",
                message=(
                    f"Only {len(profile.coursework)} course(s) listed. "
                    "With fewer than 5 courses the evaluation may be "
                    "unreliable."
                ),
            )
        )


def _check_duplicate_course_codes(
    profile: UserProfile, warnings: list[ValidationWarning]
) -> None:
    """Warn when the same course code appears more than once."""
    seen: dict[str, int] = {}
    for course in profile.coursework:
        seen[course.code] = seen.get(course.code, 0) + 1

    duplicates = sorted(code for code, count in seen.items() if count > 1)
    if duplicates:
        warnings.append(
            ValidationWarning(
                level="warning",
                field="coursework",
                message=(
                    "Duplicate course codes found: "
                    + ", ".join(duplicates)
                    + "."
                ),
            )
        )


def _check_international_toefl(
    profile: UserProfile, warnings: list[ValidationWarning]
) -> None:
    """Hint that TOEFL may be needed for recent international students."""
    if profile.is_international and profile.years_at_us_institution < 2:
        warnings.append(
            ValidationWarning(
                level="info",
                field="test_scores",
                message=(
                    "As an international student with fewer than 2 years at "
                    "a US institution, most programs will require TOEFL/IELTS "
                    "scores."
                ),
            )
        )


def _check_no_test_scores(
    profile: UserProfile, warnings: list[ValidationWarning]
) -> None:
    """Suggest adding GRE scores when the profile is otherwise non-empty."""
    if profile.test_scores.gre_quant is None and _profile_is_non_empty(
        profile
    ):
        warnings.append(
            ValidationWarning(
                level="info",
                field="test_scores",
                message=(
                    "No GRE quantitative score provided. Adding GRE scores "
                    "can improve the accuracy of program recommendations."
                ),
            )
        )


def _profile_is_non_empty(profile: UserProfile) -> bool:
    """Return True if the profile has at least some meaningful content."""
    return bool(profile.name) or bool(profile.coursework) or profile.gpa > 0.0
