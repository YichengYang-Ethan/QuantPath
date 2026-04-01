# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for core.list_builder."""


from core.list_builder import (
    SchoolList,
    SchoolListEntry,
    _apply_geo_diversity,
    _city_cluster,
    _generate_reason,
    build_school_list,
)
from core.models import (
    Course,
    EvaluationResult,
    Prerequisite,
    ProgramData,
    UserProfile,
)

# ===================================================================
# Shared fixtures
# ===================================================================


def _make_profile() -> UserProfile:
    """A reusable applicant profile with a reasonable transcript."""
    return UserProfile(
        name="Test Student",
        gpa=3.8,
        gpa_quant=3.9,
        majors=["Mathematics"],
        coursework=[
            Course("Calc III", "MATH241", "calculus", "A", level=200),
            Course("Linear Algebra", "MATH415", "linear_algebra", "A", level=400),
            Course("Probability", "STAT400", "probability", "A-", level=400),
            Course("Statistics", "STAT410", "statistics", "A", level=400),
            Course("C++ Programming", "CS225", "programming_cpp", "A-", level=200),
            Course("Python", "CS101", "programming_python", "A", level=100),
        ],
    )


def _make_programs(count: int = 9) -> list[ProgramData]:
    """Generate a set of programmes spanning all three buckets.

    With the default profile (GPA 3.8) and default EvaluationResult
    (overall_score=7.0), the classification rules produce:
        - acceptance_rate < 0.08  OR  user_gpa < prog_avg_gpa  ->  reach
        - acceptance_rate > 0.15  AND  user_gpa >= prog_avg + 0.1  ->  safety
        - otherwise  ->  target

    We construct programmes to exercise each bucket.
    """
    # Use synthetic IDs so the LR model has no data and falls back to heuristics.
    # Reach programmes: low acceptance rate or high avg GPA.
    reach_1 = ProgramData(
        id="test-reach-1",
        name="CMU MSCF",
        university="Carnegie Mellon University",
        acceptance_rate=0.05,
        avg_gpa=3.9,
        application_fee=125,
        prerequisites_required=[
            Prerequisite(category="calculus"),
            Prerequisite(category="linear_algebra"),
            Prerequisite(category="probability"),
            Prerequisite(category="programming_cpp"),
        ],
    )
    reach_2 = ProgramData(
        id="test-reach-2",
        name="Baruch MFE",
        university="Baruch College, CUNY",
        acceptance_rate=0.04,
        avg_gpa=3.85,
        application_fee=150,
        prerequisites_required=[
            Prerequisite(category="calculus"),
            Prerequisite(category="programming_cpp"),
        ],
    )
    reach_3 = ProgramData(
        id="test-reach-3",
        name="Princeton MFin",
        university="Princeton University",
        acceptance_rate=0.03,
        avg_gpa=3.95,
        application_fee=100,
        prerequisites_required=[
            Prerequisite(category="calculus"),
        ],
    )
    reach_4 = ProgramData(
        id="test-reach-4",
        name="MIT MFin",
        university="MIT",
        acceptance_rate=0.06,
        avg_gpa=3.92,
        application_fee=175,
        prerequisites_required=[
            Prerequisite(category="calculus"),
            Prerequisite(category="statistics"),
        ],
    )

    # Target programmes: moderate acceptance, GPA near user's.
    target_1 = ProgramData(
        id="test-target-1",
        name="BU MSMF",
        university="Boston University",
        acceptance_rate=0.12,
        avg_gpa=3.7,
        application_fee=95,
        prerequisites_required=[
            Prerequisite(category="calculus"),
            Prerequisite(category="statistics"),
        ],
    )
    target_2 = ProgramData(
        id="test-target-2",
        name="NYU MFE",
        university="New York University",
        acceptance_rate=0.10,
        avg_gpa=3.75,
        application_fee=100,
        prerequisites_required=[
            Prerequisite(category="calculus"),
            Prerequisite(category="linear_algebra"),
        ],
    )
    target_3 = ProgramData(
        id="test-target-3",
        name="GaTech QCF",
        university="Georgia Tech",
        acceptance_rate=0.14,
        avg_gpa=3.65,
        application_fee=85,
        prerequisites_required=[
            Prerequisite(category="calculus"),
        ],
    )

    # Safety programmes: high acceptance, avg GPA well below user's.
    safety_1 = ProgramData(
        id="test-safety-1",
        name="Rutgers MQF",
        university="Rutgers University",
        acceptance_rate=0.25,
        avg_gpa=3.5,
        application_fee=70,
        prerequisites_required=[
            Prerequisite(category="calculus"),
        ],
    )
    safety_2 = ProgramData(
        id="test-safety-2",
        name="UConn MSQF",
        university="University of Connecticut",
        acceptance_rate=0.30,
        avg_gpa=3.4,
        application_fee=75,
        prerequisites_required=[
            Prerequisite(category="calculus"),
        ],
    )

    all_progs = [
        reach_1, reach_2, reach_3, reach_4,
        target_1, target_2, target_3,
        safety_1, safety_2,
    ]
    return all_progs[:count]


def _default_evaluation() -> EvaluationResult:
    return EvaluationResult(overall_score=7.0)


# ===================================================================
# Basic list building
# ===================================================================


class TestBuildSchoolList:
    """Core behaviour of build_school_list."""

    def test_returns_school_list_type(self) -> None:
        sl = build_school_list(_make_profile(), _make_programs(), _default_evaluation())
        assert isinstance(sl, SchoolList)

    def test_reach_count_respects_max(self) -> None:
        sl = build_school_list(
            _make_profile(), _make_programs(), _default_evaluation(), max_reach=2,
        )
        assert len(sl.reach) <= 2

    def test_target_count_respects_max(self) -> None:
        sl = build_school_list(
            _make_profile(), _make_programs(), _default_evaluation(), max_target=3,
        )
        assert len(sl.target) <= 3

    def test_safety_count_respects_max(self) -> None:
        sl = build_school_list(
            _make_profile(), _make_programs(), _default_evaluation(), max_safety=1,
        )
        assert len(sl.safety) <= 1

    def test_default_limits_produce_balanced_list(self) -> None:
        """Default max_reach=3, max_target=4, max_safety=2."""
        sl = build_school_list(_make_profile(), _make_programs(), _default_evaluation())
        assert len(sl.reach) <= 3
        assert len(sl.target) <= 4
        assert len(sl.safety) <= 2
        # Total should be non-zero given our 9-programme set.
        total = len(sl.reach) + len(sl.target) + len(sl.safety)
        assert total > 0


# ===================================================================
# Correct counts per bucket
# ===================================================================


class TestBucketCounts:
    """Verify that the right number of schools land in each bucket."""

    def test_all_reach_programmes_classified(self) -> None:
        """4 reach programmes available; max_reach=3 -> exactly 3."""
        sl = build_school_list(
            _make_profile(), _make_programs(), _default_evaluation(), max_reach=3,
        )
        assert len(sl.reach) == 3

    def test_all_target_programmes_classified(self) -> None:
        """3 target programmes available; max_target=4 -> only 3."""
        sl = build_school_list(
            _make_profile(), _make_programs(), _default_evaluation(), max_target=4,
        )
        assert len(sl.target) == 3

    def test_safety_programmes(self) -> None:
        """2 safety programmes available; max_safety=2 -> exactly 2."""
        sl = build_school_list(
            _make_profile(), _make_programs(), _default_evaluation(), max_safety=2,
        )
        assert len(sl.safety) == 2

    def test_fewer_programmes_than_max(self) -> None:
        """When only 1 safety programme exists, we should get 1."""
        progs = _make_programs()
        # Remove the second safety programme (test-safety-2).
        progs = [p for p in progs if p.id != "test-safety-2"]
        sl = build_school_list(
            _make_profile(), progs, _default_evaluation(), max_safety=2,
        )
        assert len(sl.safety) == 1


# ===================================================================
# Total fees calculation
# ===================================================================


class TestTotalFees:
    """Application fee calculation."""

    def test_total_fees_sum(self) -> None:
        """Total fees should equal the sum of fees for all selected programmes."""
        sl = build_school_list(_make_profile(), _make_programs(), _default_evaluation())
        expected = 0
        all_entries = sl.reach + sl.target + sl.safety
        progs = _make_programs()
        fee_map = {p.id: p.application_fee for p in progs}
        for e in all_entries:
            expected += fee_map.get(e.program_id, 0)
        assert sl.total_application_fees == expected

    def test_fees_zero_when_no_programmes(self) -> None:
        sl = build_school_list(_make_profile(), [], _default_evaluation())
        assert sl.total_application_fees == 0

    def test_fees_with_zero_fee_programme(self) -> None:
        """Programmes with application_fee=0 should contribute nothing."""
        prog = ProgramData(
            id="free-prog",
            name="Free Program",
            university="Free University",
            acceptance_rate=0.25,
            avg_gpa=3.5,
            application_fee=0,
            prerequisites_required=[Prerequisite(category="calculus")],
        )
        profile = _make_profile()
        sl = build_school_list(profile, [prog], _default_evaluation())
        assert sl.total_application_fees == 0


# ===================================================================
# Empty / edge cases
# ===================================================================


class TestEdgeCases:
    """Edge and boundary conditions."""

    def test_empty_program_list(self) -> None:
        sl = build_school_list(_make_profile(), [], _default_evaluation())
        assert sl.reach == []
        assert sl.target == []
        assert sl.safety == []
        assert sl.total_application_fees == 0
        assert "0 schools selected" in sl.summary

    def test_single_programme(self) -> None:
        """A single reach programme should appear in reach only."""
        progs = [_make_programs()[0]]  # test-reach-1 = reach
        sl = build_school_list(_make_profile(), progs, _default_evaluation())
        assert len(sl.reach) == 1
        assert len(sl.target) == 0
        assert len(sl.safety) == 0

    def test_max_zero_returns_empty_bucket(self) -> None:
        """Setting max_reach=0 should produce an empty reach list."""
        sl = build_school_list(
            _make_profile(), _make_programs(), _default_evaluation(),
            max_reach=0, max_target=0, max_safety=0,
        )
        assert sl.reach == []
        assert sl.target == []
        assert sl.safety == []


# ===================================================================
# Entry fields
# ===================================================================


class TestEntryFields:
    """Every SchoolListEntry has the expected attributes."""

    def test_reach_entries_have_required_fields(self) -> None:
        sl = build_school_list(_make_profile(), _make_programs(), _default_evaluation())
        for entry in sl.reach:
            assert isinstance(entry, SchoolListEntry)
            assert entry.program_id
            assert entry.name
            assert entry.university
            assert entry.category == "reach"
            assert isinstance(entry.fit_score, float)
            assert isinstance(entry.prereq_match_score, float)
            assert isinstance(entry.reason, str) and entry.reason

    def test_target_entries_have_correct_category(self) -> None:
        sl = build_school_list(_make_profile(), _make_programs(), _default_evaluation())
        for entry in sl.target:
            assert entry.category == "target"

    def test_safety_entries_have_correct_category(self) -> None:
        sl = build_school_list(_make_profile(), _make_programs(), _default_evaluation())
        for entry in sl.safety:
            assert entry.category == "safety"


# ===================================================================
# Fit score ordering
# ===================================================================


class TestFitScoreOrdering:
    """Schools within each bucket are ordered by fit_score descending."""

    def test_reach_ordered_by_fit_score_desc(self) -> None:
        sl = build_school_list(_make_profile(), _make_programs(), _default_evaluation())
        scores = [e.fit_score for e in sl.reach]
        assert scores == sorted(scores, reverse=True)

    def test_target_ordered_by_fit_score_desc(self) -> None:
        sl = build_school_list(_make_profile(), _make_programs(), _default_evaluation())
        scores = [e.fit_score for e in sl.target]
        assert scores == sorted(scores, reverse=True)

    def test_safety_ordered_by_fit_score_desc(self) -> None:
        sl = build_school_list(_make_profile(), _make_programs(), _default_evaluation())
        scores = [e.fit_score for e in sl.safety]
        assert scores == sorted(scores, reverse=True)


# ===================================================================
# Summary string format
# ===================================================================


class TestSummaryString:
    """The summary field follows a consistent format."""

    def test_summary_contains_counts(self) -> None:
        sl = build_school_list(_make_profile(), _make_programs(), _default_evaluation())
        total = len(sl.reach) + len(sl.target) + len(sl.safety)
        assert f"{total} schools selected" in sl.summary
        assert f"{len(sl.reach)} Reach" in sl.summary
        assert f"{len(sl.target)} Target" in sl.summary
        assert f"{len(sl.safety)} Safety" in sl.summary

    def test_summary_contains_total_fees(self) -> None:
        sl = build_school_list(_make_profile(), _make_programs(), _default_evaluation())
        assert "Total application fees: $" in sl.summary

    def test_summary_format_empty_list(self) -> None:
        sl = build_school_list(_make_profile(), [], _default_evaluation())
        assert "0 schools selected" in sl.summary
        assert "0 Reach" in sl.summary
        assert "0 Target" in sl.summary
        assert "0 Safety" in sl.summary


# ===================================================================
# Reason generation
# ===================================================================


class TestReasonGeneration:
    """Reason strings are meaningful and non-empty."""

    def test_reach_reason_mentions_competitive(self) -> None:
        entry = {
            "prereq_match_score": 1.0,
            "fit_score": 75.0,
            "acceptance_rate": 0.05,
        }
        reason = _generate_reason(entry, "reach")
        assert "100%" in reason
        assert len(reason) > 10

    def test_safety_reason_mentions_safety(self) -> None:
        entry = {
            "prereq_match_score": 0.9,
            "fit_score": 60.0,
            "acceptance_rate": 0.25,
        }
        reason = _generate_reason(entry, "safety")
        assert "safety" in reason.lower() or "Safety" in reason

    def test_target_reason_non_empty(self) -> None:
        entry = {
            "prereq_match_score": 0.8,
            "fit_score": 65.0,
            "acceptance_rate": 0.12,
        }
        reason = _generate_reason(entry, "target")
        assert len(reason) > 5

    def test_low_prereq_reach_gets_growth_reason(self) -> None:
        entry = {
            "prereq_match_score": 0.3,
            "fit_score": 40.0,
            "acceptance_rate": 0.05,
        }
        reason = _generate_reason(entry, "reach")
        # Should still produce something meaningful.
        assert reason


# ===================================================================
# Geographic diversity
# ===================================================================


class TestGeoDiversity:
    """Geographic diversity logic works correctly."""

    def test_city_cluster_detection(self) -> None:
        assert _city_cluster("New York University") == "new_york"
        assert _city_cluster("Columbia University") == "new_york"
        assert _city_cluster("Baruch College, CUNY") == "new_york"
        assert _city_cluster("MIT") == "boston"
        assert _city_cluster("Carnegie Mellon University") is None

    def test_no_swap_when_diverse(self) -> None:
        """When selected schools are already geographically diverse, no change."""
        selected = [
            {"program_id": "a", "university": "Carnegie Mellon", "fit_score": 80},
            {"program_id": "b", "university": "New York University", "fit_score": 70},
        ]
        result = _apply_geo_diversity(selected, selected)
        assert len(result) == 2
        assert result[0]["program_id"] == "a"

    def test_swap_when_all_same_city(self) -> None:
        """When all selected share a city cluster, lowest is swapped out."""
        selected = [
            {"program_id": "a", "university": "Columbia University", "fit_score": 80},
            {"program_id": "b", "university": "NYU Tandon", "fit_score": 70},
        ]
        pool = selected + [
            {"program_id": "c", "university": "Carnegie Mellon", "fit_score": 65},
        ]
        result = _apply_geo_diversity(selected, pool)
        ids = {r["program_id"] for r in result}
        # "b" was lowest and should be swapped for "c".
        assert "c" in ids
        assert len(result) == 2

    def test_single_entry_no_swap(self) -> None:
        """A single entry cannot trigger the diversity swap."""
        selected = [
            {"program_id": "a", "university": "Columbia University", "fit_score": 80},
        ]
        result = _apply_geo_diversity(selected, selected)
        assert len(result) == 1
        assert result[0]["program_id"] == "a"

    def test_no_alternative_keeps_original(self) -> None:
        """When no diverse alternative exists in the pool, keep original."""
        selected = [
            {"program_id": "a", "university": "Columbia University", "fit_score": 80},
            {"program_id": "b", "university": "NYU Tandon", "fit_score": 70},
        ]
        # Pool has no non-NYC alternatives.
        result = _apply_geo_diversity(selected, selected)
        assert len(result) == 2
        ids = {r["program_id"] for r in result}
        assert ids == {"a", "b"}
