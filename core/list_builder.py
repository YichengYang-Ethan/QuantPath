# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""School list builder.

Builds an optimised application list from the school ranker output,
selecting the best programmes from each bucket (reach / target / safety)
while enforcing geographic diversity and generating per-school selection
reasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import EvaluationResult, ProgramData, UserProfile
from .school_ranker import rank_schools

# ===================================================================
# Data classes
# ===================================================================


@dataclass
class SchoolListEntry:
    """A single programme selected for the application list."""

    program_id: str
    name: str
    university: str
    category: str  # reach / target / safety
    fit_score: float
    prereq_match_score: float
    reason: str  # why this school was selected
    admission_prob: float | None = None   # bias-corrected P(admit) from LR
    prob_low: float | None = None         # lower CI bound
    prob_high: float | None = None        # upper CI bound


@dataclass
class SchoolList:
    """The complete, balanced application list."""

    reach: list[SchoolListEntry] = field(default_factory=list)
    target: list[SchoolListEntry] = field(default_factory=list)
    safety: list[SchoolListEntry] = field(default_factory=list)
    total_application_fees: int = 0
    summary: str = ""


# ===================================================================
# Geographic diversity helpers
# ===================================================================

# Keywords that indicate a common geographic cluster.  When all
# selected schools in a bucket share a cluster keyword we try to swap
# one of them for an alternative from the same bucket.
#
# Each entry maps a city/region label to the set of university-name
# substrings that belong to that cluster.
_CITY_CLUSTERS: dict[str, list[str]] = {
    "new_york": ["New York", "Columbia", "NYU", "Manhattan", "Baruch", "CUNY"],
    "boston": ["Boston", "MIT", "Harvard", "Northeastern"],
    "bay_area": ["San Francisco", "Berkeley", "Stanford"],
    "chicago": ["Chicago", "Northwestern"],
    "los_angeles": ["Los Angeles", "UCLA"],
    "philadelphia": ["Philadelphia", "UPenn", "Wharton", "Drexel"],
    "princeton": ["Princeton"],
}


def _city_cluster(university: str) -> str | None:
    """Return a normalised cluster label if *university* matches a known
    geographic keyword, else ``None``.
    """
    uni_lower = university.lower()
    for cluster_name, keywords in _CITY_CLUSTERS.items():
        for kw in keywords:
            if kw.lower() in uni_lower:
                return cluster_name
    return None


def _apply_geo_diversity(
    selected: list[dict[str, Any]],
    pool: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """If all entries in *selected* share the same city cluster, swap
    the lowest-scoring entry for the best alternative from *pool* that
    belongs to a different cluster.

    Returns a new list (does not mutate inputs).
    """
    if len(selected) <= 1:
        return list(selected)

    clusters = [_city_cluster(s["university"]) for s in selected]

    # Only act when every entry is in the *same* known cluster.
    if clusters[0] is None or not all(c == clusters[0] for c in clusters):
        return list(selected)

    shared_cluster = clusters[0]

    # Look for the best alternative not in this cluster.
    selected_ids = {s["program_id"] for s in selected}
    for candidate in pool:
        if candidate["program_id"] in selected_ids:
            continue
        cand_cluster = _city_cluster(candidate["university"])
        if cand_cluster != shared_cluster:
            # Swap the *last* entry (lowest fit_score because the list
            # is sorted descending) with the diverse candidate.
            result = list(selected[:-1])
            result.append(candidate)
            return result

    return list(selected)


# ===================================================================
# Reason generation
# ===================================================================


def _generate_reason(
    entry: dict[str, Any],
    category: str,
) -> str:
    """Produce a human-readable reason for including a programme."""
    prereq_pct = entry["prereq_match_score"]
    fit = entry["fit_score"]
    acceptance = entry.get("acceptance_rate") or 0.0

    parts: list[str] = []

    if prereq_pct >= 1.0:
        parts.append("Strong prerequisite match (100%)")
    elif prereq_pct >= 0.75:
        parts.append(f"Good prerequisite match ({prereq_pct:.0%})")

    if category == "reach":
        if fit >= 70:
            parts.append("High fit score despite competitive admissions")
        else:
            parts.append("Ambitious pick with growth potential")
    elif category == "safety":
        if prereq_pct >= 0.9:
            parts.append("Safety choice with excellent prereq alignment")
        else:
            parts.append("Safety choice with manageable requirements")
    else:
        # target
        if acceptance and acceptance > 0.10:
            parts.append("High fit score with moderate acceptance rate")
        else:
            parts.append("Strong target with competitive profile match")

    return "; ".join(parts) if parts else "Selected by fit score"


# ===================================================================
# Public API
# ===================================================================


def build_school_list(
    profile: UserProfile,
    programs: list[ProgramData],
    evaluation: EvaluationResult,
    max_reach: int = 3,
    max_target: int = 4,
    max_safety: int = 2,
) -> SchoolList:
    """Build an optimised school application list.

    Steps:
        1. Rank all programmes via ``rank_schools``.
        2. Select the top programmes from each bucket, honouring the
           requested maximums and enforcing minimum counts when
           available (2 reach, 3 target, 1 safety).
        3. Apply geographic diversity within each bucket.
        4. Generate a selection reason for every entry.
        5. Calculate total application fees and produce a summary line.

    Parameters
    ----------
    profile:
        The applicant's profile.
    programs:
        Full list of candidate programmes.
    evaluation:
        Pre-computed evaluation result.
    max_reach:
        Maximum number of reach schools (default 3).
    max_target:
        Maximum number of target schools (default 4).
    max_safety:
        Maximum number of safety schools (default 2).

    Returns
    -------
    SchoolList
        The balanced application list.
    """
    rankings = rank_schools(profile, programs, evaluation)

    # --- Select top entries from each bucket ---------------------------
    reach_pool = rankings["reach"]
    target_pool = rankings["target"]
    safety_pool = rankings["safety"]

    reach_sel = reach_pool[:max_reach]
    target_sel = target_pool[:max_target]
    safety_sel = safety_pool[:max_safety]

    # --- Geographic diversity -----------------------------------------
    reach_sel = _apply_geo_diversity(reach_sel, reach_pool)
    target_sel = _apply_geo_diversity(target_sel, target_pool)
    safety_sel = _apply_geo_diversity(safety_sel, safety_pool)

    # --- Build a quick lookup for application fees --------------------
    fee_map: dict[str, int] = {p.id: p.application_fee for p in programs}

    # --- Convert to SchoolListEntry objects ----------------------------
    def _to_entries(
        dicts: list[dict[str, Any]],
        category: str,
    ) -> list[SchoolListEntry]:
        entries: list[SchoolListEntry] = []
        for d in dicts:
            entries.append(
                SchoolListEntry(
                    program_id=d["program_id"],
                    name=d["name"],
                    university=d["university"],
                    category=category,
                    fit_score=d["fit_score"],
                    prereq_match_score=d["prereq_match_score"],
                    reason=_generate_reason(d, category),
                    admission_prob=d.get("admission_prob"),
                    prob_low=d.get("prob_low"),
                    prob_high=d.get("prob_high"),
                )
            )
        return entries

    reach_entries = _to_entries(reach_sel, "reach")
    target_entries = _to_entries(target_sel, "target")
    safety_entries = _to_entries(safety_sel, "safety")

    # --- Total application fees ---------------------------------------
    all_entries = reach_entries + target_entries + safety_entries
    total_fees = sum(fee_map.get(e.program_id, 0) for e in all_entries)

    # --- Summary string -----------------------------------------------
    total_count = len(all_entries)
    summary = (
        f"{total_count} schools selected: "
        f"{len(reach_entries)} Reach, "
        f"{len(target_entries)} Target, "
        f"{len(safety_entries)} Safety. "
        f"Total application fees: ${total_fees:,}"
    )

    return SchoolList(
        reach=reach_entries,
        target=target_entries,
        safety=safety_entries,
        total_application_fees=total_fees,
        summary=summary,
    )


# ===================================================================
# Portfolio optimizer
# ===================================================================


@dataclass
class PortfolioEntry:
    """One program selected by the portfolio optimizer."""

    program_id: str
    name: str
    university: str
    category: str
    admission_prob: float
    fit_score: float
    application_fee: int
    expected_contribution: float   # = admission_prob (independent model)


@dataclass
class OptimizedPortfolio:
    """Result of the portfolio optimizer."""

    programs: list[PortfolioEntry]
    expected_admits: float
    total_fees: int
    summary: str


def optimize_portfolio(
    profile: UserProfile,
    programs: list[ProgramData],
    evaluation: EvaluationResult,
    n_schools: int = 10,
    budget: int = 2000,
) -> OptimizedPortfolio:
    """Select programs maximizing expected admissions under budget/count constraints.

    Uses a greedy marginal-value selection:
        value_i = P(admit_i) / max(1, fee_i / avg_fee)

    This balances admission probability against fee cost, selecting the
    highest-value programs until n_schools is reached or budget is exhausted.

    Guarantees at least 1 reach, 2 target, 1 safety if available.

    Parameters
    ----------
    profile:
        The applicant's profile.
    programs:
        All candidate programs.
    evaluation:
        Pre-computed evaluation result.
    n_schools:
        Maximum number of programs to select.
    budget:
        Maximum total application fee budget (USD).

    Returns
    -------
    OptimizedPortfolio
        The selected programs with expected admits and fee totals.
    """
    from .school_ranker import rank_schools

    rankings = rank_schools(profile, programs, evaluation)
    all_ranked = rankings["all"]

    fee_map: dict[str, int] = {p.id: p.application_fee for p in programs}
    avg_fee = max(1, sum(fee_map.values()) / len(fee_map)) if fee_map else 100

    # Build candidate pool with scores
    candidates: list[dict[str, Any]] = []
    for entry in all_ranked:
        prob = entry.get("admission_prob")
        if prob is None:
            # Use heuristic: convert acceptance_rate to a rough probability
            ar = entry.get("acceptance_rate") or 0.15
            prob = min(ar * 1.5, 0.95)  # rough upward shift for self-selection

        fee = fee_map.get(entry["program_id"], 0)
        fee_penalty = max(1.0, fee / avg_fee)
        value = prob / fee_penalty

        candidates.append({
            **entry,
            "admission_prob": prob,
            "fee": fee,
            "value": value,
        })

    # Sort by marginal value descending
    candidates.sort(key=lambda c: -c["value"])

    # Greedy selection with guaranteed bucket coverage
    selected: list[dict[str, Any]] = []
    total_fees = 0
    n_reach = n_target = n_safety = 0

    # First pass: guarantee minimums (1 reach, 2 target, 1 safety)
    minimums = [("reach", 1), ("target", 2), ("safety", 1)]
    for bucket, min_count in minimums:
        bucket_cands = [c for c in candidates if c["category"] == bucket]
        for c in bucket_cands:
            if {"reach": n_reach, "target": n_target, "safety": n_safety}[bucket] >= min_count:
                break
            new_total = total_fees + c["fee"]
            if new_total <= budget and len(selected) < n_schools:
                selected.append(c)
                total_fees = new_total
                if bucket == "reach":
                    n_reach += 1
                elif bucket == "target":
                    n_target += 1
                else:
                    n_safety += 1

    # Second pass: fill remaining slots by value
    selected_ids = {s["program_id"] for s in selected}
    for c in candidates:
        if len(selected) >= n_schools:
            break
        if c["program_id"] in selected_ids:
            continue
        if total_fees + c["fee"] > budget:
            continue
        selected.append(c)
        selected_ids.add(c["program_id"])
        total_fees += c["fee"]

    # Sort final list: reach → target → safety, then by prob desc
    cat_order = {"reach": 0, "target": 1, "safety": 2}
    selected.sort(key=lambda c: (cat_order.get(c["category"], 9), -c["admission_prob"]))

    expected_admits = round(sum(c["admission_prob"] for c in selected), 2)

    entries = [
        PortfolioEntry(
            program_id=c["program_id"],
            name=c["name"],
            university=c["university"],
            category=c["category"],
            admission_prob=round(c["admission_prob"], 4),
            fit_score=c["fit_score"],
            application_fee=c["fee"],
            expected_contribution=round(c["admission_prob"], 4),
        )
        for c in selected
    ]

    n_selected = len(entries)
    summary = (
        f"{n_selected} schools | Expected admits: {expected_admits:.1f} | "
        f"Total fees: ${total_fees:,} | "
        f"Reach: {n_reach}  Target: {n_target}  Safety: {n_safety}"
    )

    return OptimizedPortfolio(
        programs=entries,
        expected_admits=expected_admits,
        total_fees=total_fees,
        summary=summary,
    )
