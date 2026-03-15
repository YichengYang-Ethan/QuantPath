"""Tests for core.timeline_generator."""

import pytest
from datetime import date

from core.models import DeadlineRound, ProgramData
from core.timeline_generator import (
    _parse_date,
    _add_months,
    generate_timeline,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    CAT_DEADLINE,
    CAT_APPLICATION,
    CAT_DECISION,
    CAT_TEST,
    CAT_INTERVIEW,
)


# ===================================================================
# _parse_date
# ===================================================================

class TestParseDate:
    """Test YYYY-MM-DD date parsing."""

    def test_valid_date(self) -> None:
        result = _parse_date("2026-01-15")
        assert result == date(2026, 1, 15)

    def test_valid_date_single_digits(self) -> None:
        """Even though format expects padding, int() conversion handles single digits."""
        result = _parse_date("2026-1-5")
        assert result == date(2026, 1, 5)

    def test_invalid_date_bad_month(self) -> None:
        result = _parse_date("2026-13-01")
        assert result is None

    def test_invalid_date_bad_day(self) -> None:
        result = _parse_date("2026-02-30")
        assert result is None

    def test_invalid_format(self) -> None:
        result = _parse_date("not-a-date")
        assert result is None

    def test_empty_string(self) -> None:
        result = _parse_date("")
        assert result is None

    def test_partial_string(self) -> None:
        result = _parse_date("2026-01")
        assert result is None

    def test_leap_year(self) -> None:
        result = _parse_date("2024-02-29")
        assert result == date(2024, 2, 29)

    def test_non_leap_year_feb_29(self) -> None:
        result = _parse_date("2025-02-29")
        assert result is None


# ===================================================================
# _add_months
# ===================================================================

class TestAddMonths:
    """Test calendar-month addition with day clamping."""

    def test_add_one_month(self) -> None:
        result = _add_months(date(2026, 1, 15), 1)
        assert result == date(2026, 2, 15)

    def test_add_negative_months(self) -> None:
        result = _add_months(date(2026, 6, 15), -3)
        assert result == date(2026, 3, 15)

    def test_cross_year_boundary_forward(self) -> None:
        result = _add_months(date(2025, 11, 15), 3)
        assert result == date(2026, 2, 15)

    def test_cross_year_boundary_backward(self) -> None:
        result = _add_months(date(2026, 2, 15), -3)
        assert result == date(2025, 11, 15)

    def test_day_clamping_jan31_plus_one(self) -> None:
        """Jan 31 + 1 month -> Feb 28 (non-leap) due to clamping."""
        result = _add_months(date(2025, 1, 31), 1)
        assert result == date(2025, 2, 28)

    def test_day_clamping_leap_year(self) -> None:
        """Jan 31 + 1 month in a leap year -> Feb 29."""
        result = _add_months(date(2024, 1, 31), 1)
        assert result == date(2024, 2, 29)

    def test_add_twelve_months(self) -> None:
        result = _add_months(date(2025, 6, 15), 12)
        assert result == date(2026, 6, 15)

    def test_add_zero_months(self) -> None:
        d = date(2026, 3, 15)
        result = _add_months(d, 0)
        assert result == d

    def test_day_clamping_march31_minus_one(self) -> None:
        """March 31 - 1 month -> Feb 28 (non-leap year)."""
        result = _add_months(date(2025, 3, 31), -1)
        assert result == date(2025, 2, 28)


# ===================================================================
# generate_timeline
# ===================================================================

class TestGenerateTimeline:
    """Test the full timeline generation pipeline."""

    def _make_program(
        self,
        name: str = "TestProg",
        rounds: list[dict] | None = None,
        interview_type: str = "",
        interview_format: str = "",
    ) -> ProgramData:
        deadline_rounds = []
        if rounds:
            for r in rounds:
                deadline_rounds.append(DeadlineRound(
                    round=r.get("round", 1),
                    date=r.get("date", ""),
                    decision_by=r.get("decision_by", ""),
                ))
        return ProgramData(
            id=name.lower().replace(" ", "-"),
            name=name,
            deadline_rounds=deadline_rounds,
            interview_type=interview_type,
            interview_format=interview_format,
        )

    def test_empty_programs(self) -> None:
        """No programs -> no timeline items."""
        result = generate_timeline([], start_date=date(2026, 1, 1))
        assert result == []

    def test_program_with_no_deadlines(self) -> None:
        """A program with no deadline rounds produces no items."""
        prog = self._make_program("NoDeadlines", rounds=[])
        result = generate_timeline([prog], start_date=date(2026, 1, 1))
        assert result == []

    def test_basic_deadline_appears(self) -> None:
        """A program deadline should appear as a critical deadline item."""
        prog = self._make_program("CMU MSCF", rounds=[
            {"round": 1, "date": "2026-11-01"},
        ])
        result = generate_timeline([prog], start_date=date(2026, 1, 1))

        deadline_items = [
            i for i in result if i["category"] == CAT_DEADLINE
        ]
        assert len(deadline_items) >= 1
        assert any("Round 1 application deadline" in i["action"] for i in deadline_items)

    def test_submit_reminder_one_week_before(self) -> None:
        """A submit reminder should appear ~7 days before the deadline."""
        prog = self._make_program("TestProg", rounds=[
            {"round": 1, "date": "2026-11-15"},
        ])
        result = generate_timeline([prog], start_date=date(2026, 1, 1))

        reminders = [
            i for i in result
            if i["category"] == CAT_APPLICATION and "Finalise and submit" in i["action"]
        ]
        assert len(reminders) >= 1
        # Should be dated 2026-11-08 (7 days before 2026-11-15)
        assert reminders[0]["date"] == "2026-11-08"

    def test_decision_date_appears(self) -> None:
        """If a round has a decision_by date, it should appear in the timeline."""
        prog = self._make_program("TestProg", rounds=[
            {"round": 1, "date": "2026-11-01", "decision_by": "2027-02-01"},
        ])
        result = generate_timeline([prog], start_date=date(2026, 1, 1))

        decisions = [i for i in result if i["category"] == CAT_DECISION]
        assert len(decisions) >= 1
        assert any("decision expected" in i["action"] for i in decisions)

    def test_preparation_milestones_included(self) -> None:
        """GRE prep, TOEFL, essays, recommendations milestones should appear."""
        prog = self._make_program("TestProg", rounds=[
            {"round": 1, "date": "2026-12-01"},
        ])
        result = generate_timeline([prog], start_date=date(2026, 1, 1))

        categories = {i["category"] for i in result}
        assert CAT_TEST in categories  # GRE/TOEFL prep

        actions = [i["action"] for i in result]
        action_text = " ".join(actions)
        assert "GRE" in action_text
        assert "essay" in action_text.lower()
        assert "recommender" in action_text.lower()

    def test_chronological_order(self) -> None:
        """All items should be sorted by date."""
        prog = self._make_program("TestProg", rounds=[
            {"round": 1, "date": "2026-11-01"},
            {"round": 2, "date": "2027-01-15"},
        ])
        result = generate_timeline([prog], start_date=date(2026, 1, 1))

        dates = [i["date"] for i in result]
        assert dates == sorted(dates)

    def test_no_duplicates(self) -> None:
        """Items should be deduplicated by (date, action) key."""
        prog = self._make_program("TestProg", rounds=[
            {"round": 1, "date": "2026-11-01"},
        ])
        result = generate_timeline([prog], start_date=date(2026, 1, 1))

        keys = [(i["date"], i["action"]) for i in result]
        assert len(keys) == len(set(keys))

    def test_items_not_before_start_date(self) -> None:
        """Preparation milestones should not appear before the start_date.

        Note: deadline items themselves can be at any date (they are programme facts),
        but preparation reminders are clipped to start_date.
        """
        prog = self._make_program("TestProg", rounds=[
            {"round": 1, "date": "2026-03-01"},
        ])
        # Start date is very close to the deadline
        result = generate_timeline([prog], start_date=date(2026, 2, 15))

        for item in result:
            # All items should be on or after start_date (except
            # deadline items which are fixed programme dates)
            if item["category"] not in (CAT_DEADLINE,):
                assert item["date"] >= "2026-02-15"

    def test_interview_prep_appears(self) -> None:
        """Programs with interview_type should generate interview prep items."""
        prog = self._make_program(
            "TestProg",
            rounds=[{"round": 1, "date": "2026-12-01"}],
            interview_type="virtual",
            interview_format="30-minute behavioral",
        )
        result = generate_timeline([prog], start_date=date(2026, 1, 1))

        interview_items = [i for i in result if i["category"] == CAT_INTERVIEW]
        assert len(interview_items) >= 1
        assert "virtual" in interview_items[0]["action"]
        assert "30-minute behavioral" in interview_items[0]["action"]

    def test_no_interview_when_type_empty(self) -> None:
        """Programs without interview_type should NOT generate interview items."""
        prog = self._make_program(
            "TestProg",
            rounds=[{"round": 1, "date": "2026-12-01"}],
            interview_type="",
        )
        result = generate_timeline([prog], start_date=date(2026, 1, 1))

        interview_items = [i for i in result if i["category"] == CAT_INTERVIEW]
        assert len(interview_items) == 0

    def test_multiple_programs(self) -> None:
        """Timeline should incorporate deadlines from multiple programs."""
        prog1 = self._make_program("Prog A", rounds=[
            {"round": 1, "date": "2026-11-01"},
        ])
        prog2 = self._make_program("Prog B", rounds=[
            {"round": 1, "date": "2026-12-01"},
        ])
        result = generate_timeline([prog1, prog2], start_date=date(2026, 1, 1))

        deadline_items = [i for i in result if i["category"] == CAT_DEADLINE]
        program_names = {i["action"].split("]")[0].strip("[") for i in deadline_items}
        assert "Prog A" in program_names
        assert "Prog B" in program_names

    def test_output_item_structure(self) -> None:
        """Each item should have date, action, category, priority keys."""
        prog = self._make_program("TestProg", rounds=[
            {"round": 1, "date": "2026-11-01"},
        ])
        result = generate_timeline([prog], start_date=date(2026, 1, 1))

        for item in result:
            assert "date" in item
            assert "action" in item
            assert "category" in item
            assert "priority" in item

    def test_invalid_deadline_date_skipped(self) -> None:
        """Rounds with unparseable dates should be silently skipped."""
        prog = self._make_program("TestProg", rounds=[
            {"round": 1, "date": "invalid-date"},
            {"round": 2, "date": "2026-12-01"},
        ])
        result = generate_timeline([prog], start_date=date(2026, 1, 1))

        # Only round 2 should produce a deadline
        deadline_items = [i for i in result if i["category"] == CAT_DEADLINE]
        assert len(deadline_items) == 1
        assert "Round 2" in deadline_items[0]["action"]

    def test_default_start_date_is_today(self) -> None:
        """When start_date is None, it defaults to today."""
        prog = self._make_program("TestProg", rounds=[
            {"round": 1, "date": "2030-12-01"},
        ])
        # Not passing start_date, so it defaults to date.today()
        result = generate_timeline([prog])
        # Should produce items without error
        assert isinstance(result, list)
        assert len(result) > 0

    def test_priority_ordering_same_date(self) -> None:
        """When items share a date, critical should come before high/medium."""
        prog = self._make_program("TestProg", rounds=[
            {"round": 1, "date": "2026-11-01"},
        ])
        result = generate_timeline([prog], start_date=date(2026, 1, 1))

        # Find items sharing a date
        from collections import Counter
        date_counts = Counter(i["date"] for i in result)
        for d, count in date_counts.items():
            if count > 1:
                items_on_date = [i for i in result if i["date"] == d]
                priority_order = {
                    PRIORITY_CRITICAL: 0,
                    PRIORITY_HIGH: 1,
                    PRIORITY_MEDIUM: 2,
                }
                priorities = [
                    priority_order.get(i["priority"], 9) for i in items_on_date
                ]
                assert priorities == sorted(priorities)
