# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""Prerequisite matching engine.

Checks a user's coursework against a programme's required (and
recommended) prerequisites and returns a structured match report.
"""

from __future__ import annotations

from .models import Course, PrerequisiteMatch, ProgramData, UserProfile
from .profile_evaluator import grade_to_score

# ===================================================================
# Category matching map
# ===================================================================

# Programme YAML prerequisite categories sometimes use broader names
# than the user-transcript taxonomy.  This mapping expands a programme
# prereq category to the set of user-course categories that satisfy it.

_CATEGORY_MAP: dict[str, set[str]] = {
    # Math
    "calculus": {"calculus"},
    "linear_algebra": {"linear_algebra"},
    "probability": {"probability"},
    "statistics": {"statistics"},
    "differential_equations": {"ode", "pde"},
    "ode": {"ode"},
    "pde": {"pde"},
    "real_analysis": {"real_analysis"},
    "numerical_methods": {"numerical_analysis"},
    "numerical_analysis": {"numerical_analysis"},
    "stochastic_calculus": {"stochastic_processes"},
    "stochastic_processes": {"stochastic_processes"},
    # Stats
    "econometrics": {"econometrics", "financial_econometrics"},
    "time_series": {"time_series"},
    "stat_computing": {"stat_computing"},
    "stat_learning": {"stat_learning", "machine_learning"},
    # CS / Programming
    "programming": {"programming_cpp", "programming_python"},
    "programming_cpp": {"programming_cpp"},
    "programming_python": {"programming_python"},
    "data_structures": {"data_structures"},
    "algorithms": {"algorithms"},
    "machine_learning": {"machine_learning", "stat_learning"},
    # Finance / Econ
    "finance": {"finance"},
    "microeconomics": {"microeconomics"},
    "macroeconomics": {"macroeconomics"},
    "risk_management": {"risk_management"},
    "financial_econometrics": {"financial_econometrics"},
    "game_theory": {"game_theory"},
    "partial_differential_equations": {"pde"},
}


# Grade comparison (letter grades -> ordinal for min_grade checks).
_GRADE_ORDER: dict[str, int] = {
    "A+": 10,
    "A": 9,
    "A-": 8,
    "B+": 7,
    "B": 6,
    "B-": 5,
    "C+": 4,
    "C": 3,
    "C-": 2,
    "D": 1,
    "F": 0,
}


def _grade_meets_minimum(student_grade: str, min_grade: str) -> bool:
    """Return True if *student_grade* is at or above *min_grade*.

    Handles both letter grades and numeric grades.
    """
    student_grade = student_grade.strip()
    min_grade = min_grade.strip()

    # Letter-grade path.
    s_ord = _GRADE_ORDER.get(student_grade)
    m_ord = _GRADE_ORDER.get(min_grade)
    if s_ord is not None and m_ord is not None:
        return s_ord >= m_ord

    # Fallback to the numeric score comparison.
    return grade_to_score(student_grade) >= grade_to_score(min_grade)


def _find_matching_courses(
    courses: list[Course],
    prereq_category: str,
) -> list[Course]:
    """Find all user courses that can satisfy a prerequisite category."""
    acceptable = _CATEGORY_MAP.get(prereq_category, {prereq_category})
    return [c for c in courses if c.category in acceptable]


# ===================================================================
# Public API
# ===================================================================


def match_prerequisites(
    profile: UserProfile,
    program: ProgramData,
) -> PrerequisiteMatch:
    """Match a user's transcript against a programme's prerequisites.

    For every *required* prerequisite the function checks:
        1. Does the user have at least one course in the matching
           category?
        2. If the prerequisite specifies a ``min_grade``, does the
           best matching course meet it?

    Recommended prerequisites generate *warnings* if missing but do
    not count against ``match_score``.

    Parameters
    ----------
    profile:
        The applicant's profile.
    program:
        A loaded programme with prerequisite data.

    Returns
    -------
    PrerequisiteMatch
        Structured report with matched/missing/warnings lists and
        an aggregate ``match_score`` in [0, 1].
    """
    matched: list[dict] = []
    missing: list[dict] = []
    warnings: list[str] = []

    total_required = len(program.prerequisites_required)

    for prereq in program.prerequisites_required:
        candidates = _find_matching_courses(profile.coursework, prereq.category)
        entry = {
            "category": prereq.category,
            "level": prereq.level,
            "min_grade": prereq.min_grade,
        }

        if not candidates:
            missing.append(entry)
            continue

        # Pick the best course among matches.
        best = max(candidates, key=lambda c: grade_to_score(c.grade))

        # Check minimum grade requirement.
        if prereq.min_grade and not _grade_meets_minimum(best.grade, prereq.min_grade):
            entry["student_grade"] = best.grade
            entry["course"] = best.name
            missing.append(entry)
            warnings.append(
                f"{prereq.category}: your best grade ({best.grade}) does not "
                f"meet the minimum ({prereq.min_grade}) required by {program.name}."
            )
            continue

        matched.append(
            {
                **entry,
                "course": best.name,
                "student_grade": best.grade,
            }
        )

    # Check recommended prerequisites (warnings only).
    for prereq in program.prerequisites_recommended:
        candidates = _find_matching_courses(profile.coursework, prereq.category)
        if not candidates:
            note = prereq.note or prereq.category
            warnings.append(
                f"Recommended: {prereq.category} -- {note} (not found in your coursework)."
            )

    # Compute match score.
    match_score = len(matched) / total_required if total_required > 0 else 1.0

    return PrerequisiteMatch(
        program_id=program.id,
        matched=matched,
        missing=missing,
        warnings=warnings,
        match_score=round(match_score, 3),
    )
