# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""Application timeline generator.

Produces a month-by-month action plan covering test preparation,
application milestones, and programme-specific deadlines.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from .models import ProgramData

# ===================================================================
# Constants
# ===================================================================

# Priority levels for timeline items.
PRIORITY_CRITICAL = "critical"
PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"

# Action categories.
CAT_TEST = "test_prep"
CAT_APPLICATION = "application"
CAT_DEADLINE = "deadline"
CAT_ESSAY = "essay"
CAT_RECOMMENDATION = "recommendation"
CAT_INTERVIEW = "interview"
CAT_DECISION = "decision"
CAT_PREPARATION = "preparation"


# ===================================================================
# Internal helpers
# ===================================================================


def _parse_date(date_str: str) -> date | None:
    """Parse a YYYY-MM-DD string, returning None on failure."""
    try:
        parts = date_str.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return None


def _add_months(d: date, months: int) -> date:
    """Return *d* shifted by *months* calendar months.

    Day is clamped to the last day of the target month.
    """
    month = d.month + months
    year = d.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    # Clamp day.
    import calendar

    max_day = calendar.monthrange(year, month)[1]
    day = min(d.day, max_day)
    return date(year, month, day)


def _first_of_month(d: date) -> date:
    """Return the first day of d's month."""
    return d.replace(day=1)


# ===================================================================
# Preparation milestones (relative to earliest deadline)
# ===================================================================


def _preparation_milestones(
    earliest_deadline: date,
    start: date,
) -> list[dict[str, Any]]:
    """Generate generic preparation milestones leading up to the
    earliest programme deadline.

    The milestones are anchored relative to ``earliest_deadline`` and
    clipped so nothing appears before ``start``.
    """
    milestones: list[dict[str, Any]] = []

    # GRE preparation (5-6 months before earliest deadline).
    gre_start = _add_months(earliest_deadline, -6)
    gre_test = _add_months(earliest_deadline, -3)

    milestones.extend(
        [
            {
                "date": max(start, gre_start).isoformat(),
                "action": "Begin GRE preparation (if required by target programmes)",
                "category": CAT_TEST,
                "priority": PRIORITY_HIGH,
            },
            {
                "date": max(start, _add_months(earliest_deadline, -4)).isoformat(),
                "action": "Take GRE diagnostic / practice test",
                "category": CAT_TEST,
                "priority": PRIORITY_MEDIUM,
            },
            {
                "date": max(start, gre_test).isoformat(),
                "action": "Take GRE (allows time for retake if needed)",
                "category": CAT_TEST,
                "priority": PRIORITY_CRITICAL,
            },
        ]
    )

    # TOEFL (if applicable, 3-4 months before).
    toefl_target = _add_months(earliest_deadline, -3)
    milestones.append(
        {
            "date": max(start, toefl_target).isoformat(),
            "action": "Take TOEFL/IELTS (if required)",
            "category": CAT_TEST,
            "priority": PRIORITY_HIGH,
        }
    )

    # Recommendations (4-5 months before).
    rec_ask = _add_months(earliest_deadline, -5)
    rec_remind = _add_months(earliest_deadline, -2)
    milestones.extend(
        [
            {
                "date": max(start, rec_ask).isoformat(),
                "action": "Identify and approach recommenders",
                "category": CAT_RECOMMENDATION,
                "priority": PRIORITY_HIGH,
            },
            {
                "date": max(start, rec_remind).isoformat(),
                "action": "Send reminder to recommenders; confirm submission status",
                "category": CAT_RECOMMENDATION,
                "priority": PRIORITY_MEDIUM,
            },
        ]
    )

    # Essays (3-4 months before).
    essay_start = _add_months(earliest_deadline, -4)
    essay_review = _add_months(earliest_deadline, -2)
    milestones.extend(
        [
            {
                "date": max(start, essay_start).isoformat(),
                "action": "Begin drafting personal statements and essays",
                "category": CAT_ESSAY,
                "priority": PRIORITY_HIGH,
            },
            {
                "date": max(start, essay_review).isoformat(),
                "action": "Finalise essays; get peer / advisor review",
                "category": CAT_ESSAY,
                "priority": PRIORITY_HIGH,
            },
        ]
    )

    # Resume and general prep.
    milestones.append(
        {
            "date": max(start, _add_months(earliest_deadline, -5)).isoformat(),
            "action": "Update resume; highlight quantitative and programming experience",
            "category": CAT_PREPARATION,
            "priority": PRIORITY_MEDIUM,
        }
    )

    return milestones


# ===================================================================
# Public API
# ===================================================================


def generate_timeline(
    programs: list[ProgramData],
    start_date: date | None = None,
) -> list[dict[str, Any]]:
    """Generate a month-by-month action plan for the given programmes.

    The timeline includes:
        - Generic preparation milestones (GRE, TOEFL, essays, recs).
        - Programme-specific application deadlines.
        - Decision notification dates.
        - Interview preparation reminders.

    Parameters
    ----------
    programs:
        List of programmes the applicant is targeting.
    start_date:
        When the applicant begins preparation.  Defaults to today.

    Returns
    -------
    list[dict]
        Sorted list of ``{date, action, category, priority}`` dicts
        in chronological order.
    """
    if start_date is None:
        start_date = date.today()

    items: list[dict[str, Any]] = []

    # Collect all deadline dates to find the earliest.
    all_deadlines: list[date] = []

    for prog in programs:
        for rd in prog.deadline_rounds:
            dl = _parse_date(rd.date)
            if dl is None:
                continue
            all_deadlines.append(dl)

            # Application deadline item.
            items.append(
                {
                    "date": dl.isoformat(),
                    "action": (f"[{prog.name}] Round {rd.round} application deadline"),
                    "category": CAT_DEADLINE,
                    "priority": PRIORITY_CRITICAL,
                }
            )

            # Submit application reminder (1 week before deadline).
            submit_reminder = dl - timedelta(days=7)
            if submit_reminder >= start_date:
                items.append(
                    {
                        "date": submit_reminder.isoformat(),
                        "action": (
                            f"[{prog.name}] Finalise and submit Round {rd.round} "
                            f"application (deadline: {dl.isoformat()})"
                        ),
                        "category": CAT_APPLICATION,
                        "priority": PRIORITY_HIGH,
                    }
                )

            # Decision date.
            dec = _parse_date(rd.decision_by) if rd.decision_by else None
            if dec:
                items.append(
                    {
                        "date": dec.isoformat(),
                        "action": (
                            f"[{prog.name}] Round {rd.round} decision expected by this date"
                        ),
                        "category": CAT_DECISION,
                        "priority": PRIORITY_MEDIUM,
                    }
                )

        # Interview prep (if programme has interviews).
        if prog.interview_type:
            # Add interview prep milestone 1 month before first deadline.
            first_dl = None
            for rd in prog.deadline_rounds:
                d = _parse_date(rd.date)
                if d and (first_dl is None or d < first_dl):
                    first_dl = d
            if first_dl:
                prep_date = _add_months(first_dl, -1)
                if prep_date >= start_date:
                    fmt_note = f" ({prog.interview_format})" if prog.interview_format else ""
                    items.append(
                        {
                            "date": prep_date.isoformat(),
                            "action": (
                                f"[{prog.name}] Prepare for interview "
                                f"({prog.interview_type}){fmt_note}"
                            ),
                            "category": CAT_INTERVIEW,
                            "priority": PRIORITY_HIGH,
                        }
                    )

    # Add generic preparation milestones based on earliest deadline.
    if all_deadlines:
        earliest = min(all_deadlines)
        items.extend(_preparation_milestones(earliest, start_date))

    # Deduplicate and sort chronologically.
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        key = f"{item['date']}|{item['action']}"
        if key not in seen:
            seen.add(key)
            unique.append(item)

    # Sort by date, then by priority (critical first).
    priority_order = {
        PRIORITY_CRITICAL: 0,
        PRIORITY_HIGH: 1,
        PRIORITY_MEDIUM: 2,
        PRIORITY_LOW: 3,
    }
    unique.sort(key=lambda x: (x["date"], priority_order.get(x["priority"], 9)))

    return unique
