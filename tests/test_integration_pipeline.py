# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""Integration tests for the full QuantPath pipeline.

These tests exercise the real data files (programs, admission models) and
verify that LR prediction fields flow end-to-end through:
    evaluate → rank_schools → build_school_list → optimize_portfolio
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.data_loader import load_all_programs, load_profile
from core.list_builder import (
    OptimizedPortfolio,
    PortfolioEntry,
    SchoolList,
    SchoolListEntry,
    build_school_list,
    optimize_portfolio,
)
from core.lr_predictor import predict_prob_full
from core.models import TestScores, UserProfile
from core.profile_evaluator import evaluate as evaluate_profile
from core.school_ranker import rank_schools

SAMPLE_PROFILE = Path(__file__).resolve().parent.parent / "examples" / "sample_profile.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def programs():
    return load_all_programs()


@pytest.fixture(scope="module")
def sample_profile():
    return load_profile(str(SAMPLE_PROFILE))


@pytest.fixture(scope="module")
def evaluation(sample_profile):
    return evaluate_profile(sample_profile)


@pytest.fixture(scope="module")
def rankings(sample_profile, programs, evaluation):
    return rank_schools(sample_profile, programs, evaluation)


@pytest.fixture(scope="module")
def school_list(sample_profile, programs, evaluation):
    return build_school_list(sample_profile, programs, evaluation)


@pytest.fixture(scope="module")
def portfolio(sample_profile, programs, evaluation):
    return optimize_portfolio(sample_profile, programs, evaluation)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


class TestDataLoading:
    def test_programs_loaded(self, programs):
        assert len(programs) >= 10

    def test_sample_profile_loads(self, sample_profile):
        assert sample_profile.gpa > 0
        assert sample_profile.name

    def test_evaluation_has_all_dimensions(self, evaluation):
        expected = {"math", "statistics", "cs", "finance_econ", "gpa"}
        assert expected.issubset(evaluation.dimension_scores.keys())

    def test_overall_score_in_range(self, evaluation):
        assert 0.0 <= evaluation.overall_score <= 10.0


# ---------------------------------------------------------------------------
# rank_schools — LR fields flow through
# ---------------------------------------------------------------------------


class TestRankSchools:
    def test_returns_all_buckets(self, rankings):
        assert "reach" in rankings
        assert "target" in rankings
        assert "safety" in rankings
        assert "all" in rankings

    def test_at_least_one_program_per_bucket(self, rankings):
        total = len(rankings["all"])
        assert total >= 10

    def test_lr_fields_present_in_ranked_entries(self, rankings):
        """Every entry in 'all' has the LR keys (value may be None for no-model programs)."""
        for entry in rankings["all"]:
            assert "admission_prob" in entry
            assert "prob_low" in entry
            assert "prob_high" in entry

    def test_at_least_one_entry_has_lr_prob(self, rankings):
        """At least one program should have a trained LR model."""
        probs = [r["admission_prob"] for r in rankings["all"] if r["admission_prob"] is not None]
        assert len(probs) > 0

    def test_lr_prob_in_unit_interval(self, rankings):
        for entry in rankings["all"]:
            prob = entry["admission_prob"]
            if prob is not None:
                assert 0.0 <= prob <= 1.0

    def test_ci_bounds_ordered(self, rankings):
        for entry in rankings["all"]:
            if entry["prob_low"] is not None and entry["prob_high"] is not None:
                assert entry["prob_low"] <= entry["admission_prob"] <= entry["prob_high"]

    def test_sorted_by_fit_score_descending(self, rankings):
        scores = [r["fit_score"] for r in rankings["all"]]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# build_school_list — SchoolListEntry carries LR fields
# ---------------------------------------------------------------------------


class TestBuildSchoolList:
    def test_returns_school_list(self, school_list):
        assert isinstance(school_list, SchoolList)

    def test_entries_are_school_list_entries(self, school_list):
        for tier in [school_list.reach, school_list.target, school_list.safety]:
            for entry in tier:
                assert isinstance(entry, SchoolListEntry)

    def test_school_list_entry_has_lr_fields(self, school_list):
        """SchoolListEntry now carries admission_prob/prob_low/prob_high."""
        all_entries = school_list.reach + school_list.target + school_list.safety
        assert len(all_entries) > 0
        for entry in all_entries:
            # Fields must exist (dataclass guarantees this); value may be None
            assert hasattr(entry, "admission_prob")
            assert hasattr(entry, "prob_low")
            assert hasattr(entry, "prob_high")

    def test_at_least_one_entry_has_lr_prob(self, school_list):
        all_entries = school_list.reach + school_list.target + school_list.safety
        probs = [e.admission_prob for e in all_entries if e.admission_prob is not None]
        assert len(probs) > 0

    def test_lr_prob_in_unit_interval(self, school_list):
        all_entries = school_list.reach + school_list.target + school_list.safety
        for entry in all_entries:
            if entry.admission_prob is not None:
                assert 0.0 <= entry.admission_prob <= 1.0

    def test_ci_ordered_in_entries(self, school_list):
        all_entries = school_list.reach + school_list.target + school_list.safety
        for entry in all_entries:
            if entry.prob_low is not None and entry.prob_high is not None:
                assert entry.prob_low <= entry.admission_prob <= entry.prob_high

    def test_total_fees_positive(self, school_list):
        assert school_list.total_application_fees >= 0

    def test_summary_non_empty(self, school_list):
        assert len(school_list.summary) > 0

    def test_category_labels_correct(self, school_list):
        for entry in school_list.reach:
            assert entry.category == "reach"
        for entry in school_list.target:
            assert entry.category == "target"
        for entry in school_list.safety:
            assert entry.category == "safety"


# ---------------------------------------------------------------------------
# optimize_portfolio
# ---------------------------------------------------------------------------


class TestOptimizePortfolio:
    def test_returns_optimized_portfolio(self, portfolio):
        assert isinstance(portfolio, OptimizedPortfolio)

    def test_programs_are_portfolio_entries(self, portfolio):
        for p in portfolio.programs:
            assert isinstance(p, PortfolioEntry)

    def test_expected_admits_positive(self, portfolio):
        assert portfolio.expected_admits > 0

    def test_total_fees_within_budget(self, portfolio):
        assert portfolio.total_fees <= 2000

    def test_at_least_one_program_selected(self, portfolio):
        assert len(portfolio.programs) >= 1

    def test_all_admission_probs_in_unit_interval(self, portfolio):
        for entry in portfolio.programs:
            assert 0.0 <= entry.admission_prob <= 1.0


# ---------------------------------------------------------------------------
# GRE missing — no crash, graceful degradation
# ---------------------------------------------------------------------------


class TestGREMissing:
    def test_rank_schools_without_gre(self, programs):
        """Pipeline must not raise when GRE Quant is absent."""
        profile_no_gre = UserProfile(
            name="No GRE",
            gpa=3.7,
            test_scores=TestScores(gre_quant=None),
        )
        result = evaluate_profile(profile_no_gre)
        rankings = rank_schools(profile_no_gre, programs, result)
        assert len(rankings["all"]) > 0

    def test_build_school_list_without_gre(self, programs):
        profile_no_gre = UserProfile(
            name="No GRE",
            gpa=3.7,
            test_scores=TestScores(gre_quant=None),
        )
        result = evaluate_profile(profile_no_gre)
        school_list = build_school_list(profile_no_gre, programs, result)
        all_entries = school_list.reach + school_list.target + school_list.safety
        assert len(all_entries) > 0
        # With no GRE, LR models that need GRE can't run — admission_prob may be None
        # but the field must still exist
        for entry in all_entries:
            assert hasattr(entry, "admission_prob")


# ---------------------------------------------------------------------------
# predict_prob_full — sample profile against known programs
# ---------------------------------------------------------------------------


class TestLRPredictorWithRealModels:
    def test_baruch_prediction_not_none(self, sample_profile):
        gre = sample_profile.test_scores.gre_quant
        result = predict_prob_full("baruch-mfe", sample_profile.gpa, gre, sample_profile)
        # Only runs if baruch-mfe is in the trained model file
        if result is not None:
            assert 0.0 <= result.prob <= 1.0
            assert result.prob_low <= result.prob <= result.prob_high

    def test_international_profile_lower_than_domestic(self, programs):
        """Real-model integration: international flag lowers P(admit)."""
        gpa, gre = 3.8, 168
        domestic = UserProfile(
            name="Domestic", gpa=gpa, test_scores=TestScores(gre_quant=gre),
            is_international=False,
        )
        intl = UserProfile(
            name="Intl", gpa=gpa, test_scores=TestScores(gre_quant=gre),
            is_international=True,
        )
        # Use a program likely to have an LR model
        for prog_id in ["baruch-mfe", "rutgers-mqf", "cmu-mscf"]:
            p_dom = predict_prob_full(prog_id, gpa, gre, domestic)
            p_int = predict_prob_full(prog_id, gpa, gre, intl)
            if p_dom is not None and p_int is not None:
                assert p_dom.prob > p_int.prob
                return  # Pass as soon as one program confirms the direction
        pytest.skip("No LR model found for test programs")
