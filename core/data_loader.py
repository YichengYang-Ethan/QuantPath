# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""Data loading utilities for QuantPath.

Reads programme YAML files and user profile YAML files into the domain
models defined in ``core.models``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import (
    Course,
    DeadlineRound,
    Prerequisite,
    ProgramData,
    TestScores,
    UserProfile,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Package root is one level above this file's directory (core/).
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PACKAGE_ROOT / "data"
_PROGRAMS_DIR = _DATA_DIR / "programs"


def _programs_dir() -> Path:
    """Return the resolved path to the programs YAML directory."""
    if not _PROGRAMS_DIR.is_dir():
        raise FileNotFoundError(
            f"Programs directory not found at {_PROGRAMS_DIR}. "
            "Ensure data/programs/ exists relative to the package root."
        )
    return _PROGRAMS_DIR


# ---------------------------------------------------------------------------
# Programme loaders
# ---------------------------------------------------------------------------


def _parse_prerequisite(raw: dict[str, Any]) -> Prerequisite:
    """Convert a raw YAML prerequisite dict into a Prerequisite object."""
    return Prerequisite(
        category=raw.get("category", ""),
        level=raw.get("level", ""),
        min_grade=raw.get("min_grade"),
        note=raw.get("note", ""),
    )


def _parse_program(raw: dict[str, Any]) -> ProgramData:
    """Map a raw YAML dict to a ProgramData dataclass."""
    admissions = raw.get("admissions", {})
    prereqs = raw.get("prerequisites", {})
    tests = raw.get("tests", {})
    gre = tests.get("gre", {})
    toefl = tests.get("toefl", {})
    deadlines = raw.get("deadlines", {})
    application = raw.get("application", {})
    interview = application.get("interview", {})

    outcomes = raw.get("outcomes", {})

    return ProgramData(
        id=raw.get("id", ""),
        name=raw.get("name", ""),
        full_name=raw.get("full_name", ""),
        university=raw.get("university", ""),
        department=raw.get("department", ""),
        website=raw.get("website", ""),
        quantnet_ranking=raw.get("quantnet_ranking"),
        # Admissions
        class_size=admissions.get("class_size", 0),
        acceptance_rate=admissions.get("acceptance_rate", 0.0),
        avg_gpa=admissions.get("avg_gpa", 0.0),
        gre_quant_avg=admissions.get("gre_quant_avg"),
        international_pct=admissions.get("international_pct"),
        # Prerequisites
        prerequisites_required=[_parse_prerequisite(p) for p in prereqs.get("required", [])],
        prerequisites_recommended=[_parse_prerequisite(p) for p in prereqs.get("recommended", [])],
        languages=prereqs.get("languages", []),
        # Tests -- GRE
        gre_required=gre.get("required", False),
        gre_accepts_gmat=gre.get("accepts_gmat", False),
        gre_exemption=gre.get("exemption"),
        gre_code=gre.get("code"),
        # Tests -- TOEFL
        toefl_waiver_conditions=toefl.get("waiver_conditions", []),
        toefl_min_ibt=(toefl.get("min_scores") or {}).get("toefl_ibt"),
        toefl_min_ielts=(toefl.get("min_scores") or {}).get("ielts"),
        # Deadlines
        deadline_cycle=deadlines.get("cycle", ""),
        deadline_note=deadlines.get("note", ""),
        deadline_rounds=[
            DeadlineRound(
                round=r.get("round", 0),
                date=r.get("date", ""),
                decision_by=r.get("decision_by", ""),
            )
            for r in deadlines.get("rounds", [])
        ],
        # Application
        application_fee=application.get("fee", 0),
        essays=application.get("essays", []),
        video=application.get("video"),
        recommendations=application.get("recommendations", 0),
        resume_max_pages=application.get("resume_max_pages"),
        interview_type=interview.get("type", ""),
        interview_format=interview.get("format", ""),
        # Outcomes
        avg_base_salary=outcomes.get("avg_base_salary"),
        employment_rate_3m=outcomes.get("employment_rate_3m"),
        tuition_total=outcomes.get("tuition_total"),
        # Extras
        special=raw.get("special", []),
        tags=raw.get("tags", []),
    )


def load_program(program_id: str) -> ProgramData:
    """Load a single programme by its id (e.g. ``baruch-mfe``).

    Parameters
    ----------
    program_id:
        Filename stem inside ``data/programs/``, e.g. ``"baruch-mfe"``.

    Returns
    -------
    ProgramData
        Fully-populated programme dataclass.

    Raises
    ------
    FileNotFoundError
        If the corresponding YAML file does not exist.
    """
    path = _programs_dir() / f"{program_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Program file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return _parse_program(raw)


def load_all_programs() -> list[ProgramData]:
    """Load every programme YAML in the ``data/programs/`` directory.

    Returns
    -------
    list[ProgramData]
        All programmes found, sorted by ``quantnet_ranking`` (nulls last).
    """
    programs: list[ProgramData] = []
    for path in sorted(_programs_dir().glob("*.yaml")):
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if raw:
            programs.append(_parse_program(raw))

    # Sort by ranking, pushing None to the end.
    programs.sort(key=lambda p: (p.quantnet_ranking is None, p.quantnet_ranking or 0))
    return programs


# ---------------------------------------------------------------------------
# User profile loader
# ---------------------------------------------------------------------------


def _parse_course(raw: dict[str, Any]) -> Course:
    """Convert a raw YAML course dict into a Course object."""
    return Course(
        name=raw.get("name", ""),
        code=raw.get("code", ""),
        category=raw.get("category", "other"),
        grade=str(raw.get("grade", "")),
        credits=float(raw.get("credits", 3.0)),
        level=int(raw.get("level", 300)),
        university=raw.get("university", ""),
    )


def load_profile(path: str) -> UserProfile:
    """Load a user profile from a YAML file.

    Parameters
    ----------
    path:
        Absolute or relative path to the profile YAML.

    Returns
    -------
    UserProfile
        Populated profile dataclass.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the YAML is empty or unparseable.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Profile file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not raw:
        raise ValueError(f"Profile YAML is empty or invalid: {filepath}")

    # Support both flat and nested (personal: {}) profile formats
    personal = raw.get("personal", {})

    # Test scores — support both top-level and nested
    raw_tests = raw.get("test_scores", personal.get("test_scores", {}))
    test_scores = TestScores(
        gre_quant=raw_tests.get("gre_quant"),
        gre_verbal=raw_tests.get("gre_verbal"),
        toefl=raw_tests.get("toefl_ibt", raw_tests.get("toefl")),
    )

    # Courses — support both "courses" and "coursework" keys
    raw_courses = raw.get("courses", raw.get("coursework", []))

    # Planned future courses (for projected evaluation mode)
    raw_planned = raw.get("planned_courses", [])

    return UserProfile(
        name=personal.get("name", raw.get("name", "")),
        coursework=[_parse_course(c) for c in raw_courses],
        planned_coursework=[_parse_course(c) for c in raw_planned],
        gpa=float(personal.get("gpa", raw.get("gpa", 0.0))),
        gpa_quant=float(raw.get("gpa_quant", 0.0)),
        university=personal.get("university", raw.get("university", "")),
        majors=personal.get("majors", raw.get("majors", [])),
        test_scores=test_scores,
        work_experience=raw.get("work_experience", raw.get("experience", [])),
        projects=raw.get("projects", []),
        is_international=personal.get("is_international", raw.get("is_international", False)),
        years_at_us_institution=int(
            personal.get("years_at_us_institution", raw.get("years_at_us_institution", 0))
        ),
    )
