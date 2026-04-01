# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""ROI calculator for MFE programs.

Computes return-on-investment metrics including payback period, 5-year NPV,
and risk-adjusted ROI for each programme, factoring in tuition, living costs,
salary outcomes, and employment rates.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import ProgramData

# ---------------------------------------------------------------------------
# Living cost data
# ---------------------------------------------------------------------------


@dataclass
class CityLivingCost:
    """Monthly living cost estimate for a given city."""

    city: str
    monthly_rent: int
    monthly_other: int  # food, transport, insurance, etc.


@dataclass
class ROIResult:
    """ROI analysis result for a single programme."""

    program_id: str
    program_name: str
    university: str
    tuition: int
    living_cost_total: int  # for program duration (assume 1.5 years)
    total_cost: int  # tuition + living
    avg_salary: int
    salary_premium: int  # avg_salary - opportunity_cost_salary
    employment_rate: float
    payback_years: float  # total_cost / annual_salary_premium
    npv_5yr: float  # 5-year NPV at discount_rate
    risk_adjusted_roi: float  # ROI weighted by employment_rate (%)


# City living cost estimates (monthly, 2026 estimates)
CITY_COSTS: dict[str, CityLivingCost] = {
    "New York": CityLivingCost("New York", 2200, 1500),
    "Pittsburgh": CityLivingCost("Pittsburgh", 1200, 1000),
    "Boston": CityLivingCost("Boston", 1800, 1300),
    "Chicago": CityLivingCost("Chicago", 1500, 1200),
    "San Francisco": CityLivingCost("San Francisco", 2500, 1600),
    "Los Angeles": CityLivingCost("Los Angeles", 2000, 1400),
    "Atlanta": CityLivingCost("Atlanta", 1300, 1100),
    "Princeton": CityLivingCost("Princeton", 1600, 1200),
    "Ithaca": CityLivingCost("Ithaca", 1200, 1000),
    "Evanston": CityLivingCost("Evanston", 1500, 1200),
    "Ann Arbor": CityLivingCost("Ann Arbor", 1300, 1100),
    "Toronto": CityLivingCost("Toronto", 1600, 1200),
    "Minneapolis": CityLivingCost("Minneapolis", 1200, 1000),
    "Charlotte": CityLivingCost("Charlotte", 1200, 1000),
    "Hoboken": CityLivingCost("Hoboken", 2000, 1400),
    "Seattle": CityLivingCost("Seattle", 1800, 1300),
    "Champaign": CityLivingCost("Champaign", 900, 900),
    "Raleigh": CityLivingCost("Raleigh", 1100, 1000),
}

# University -> city mapping (keys match university field in program YAML files)
UNIVERSITY_CITY: dict[str, str] = {
    "Baruch College, City University of New York": "New York",
    "Carnegie Mellon University": "Pittsburgh",
    "Columbia University": "New York",
    "Cornell University": "Ithaca",
    "New York University": "New York",
    "Princeton University": "Princeton",
    "Massachusetts Institute of Technology": "Boston",
    "Stanford University": "San Francisco",
    "University of California, Berkeley": "San Francisco",
    "University of California, Los Angeles": "Los Angeles",
    "University of Chicago": "Chicago",
    "Georgia Institute of Technology": "Atlanta",
    "Boston University": "Boston",
    "University of Michigan": "Ann Arbor",
    "Northwestern University": "Evanston",
    "University of Toronto": "Toronto",
    "University of Minnesota": "Minneapolis",
    "University of North Carolina at Charlotte": "Charlotte",
    "University of Illinois Urbana-Champaign": "Champaign",
    "Stevens Institute of Technology": "Hoboken",
    "Rutgers University": "New York",
    "Fordham University": "New York",
    "Johns Hopkins University": "Boston",
    "University of Southern California": "Los Angeles",
    "University of Washington": "Seattle",
    "North Carolina State University": "Raleigh",
}


# ---------------------------------------------------------------------------
# ROI calculation
# ---------------------------------------------------------------------------


def _get_city_cost(university: str) -> CityLivingCost:
    """Look up the living cost for a university's city.

    Falls back to New York costs if the university is not in the mapping.
    """
    city = UNIVERSITY_CITY.get(university, "New York")
    return CITY_COSTS.get(city, CITY_COSTS["New York"])


def calculate_roi(
    programs: list[ProgramData],
    opportunity_cost_salary: int = 65000,
    discount_rate: float = 0.05,
    program_duration_months: int = 18,
) -> list[ROIResult]:
    """Calculate ROI metrics for each programme that has tuition and salary data.

    Parameters
    ----------
    programs:
        List of programme data objects (loaded from YAML).
    opportunity_cost_salary:
        Annual salary the student foregoes by attending the programme
        (default: $65,000, approximate undergrad quant salary).
    discount_rate:
        Annual discount rate for NPV calculation (default: 5%).
    program_duration_months:
        Programme duration in months (default: 18).

    Returns
    -------
    List of ``ROIResult`` objects sorted by ``npv_5yr`` descending.
    """
    results: list[ROIResult] = []

    for prog in programs:
        # Skip programmes missing required financial data
        if prog.tuition_total is None or prog.avg_base_salary is None:
            continue

        tuition = prog.tuition_total
        avg_salary = prog.avg_base_salary
        employment_rate = prog.employment_rate_3m if prog.employment_rate_3m is not None else 0.0

        # 1. Living cost
        city_cost = _get_city_cost(prog.university)
        monthly = city_cost.monthly_rent + city_cost.monthly_other
        living_cost_total = monthly * program_duration_months

        # 2. Total cost
        total_cost = tuition + living_cost_total

        # 3. Salary premium
        salary_premium = avg_salary - opportunity_cost_salary

        # 4. Payback years
        if salary_premium > 0:
            payback_years = total_cost / salary_premium
        else:
            payback_years = float("inf")

        # 5. NPV over 5 years
        npv_5yr = -total_cost
        for year in range(1, 6):
            npv_5yr += (salary_premium * employment_rate) / (1 + discount_rate) ** year

        # 6. Risk-adjusted ROI (%)
        if total_cost > 0:
            risk_adjusted_roi = (
                (salary_premium * employment_rate * 5 - total_cost) / total_cost * 100
            )
        else:
            risk_adjusted_roi = 0.0

        results.append(
            ROIResult(
                program_id=prog.id,
                program_name=prog.name,
                university=prog.university,
                tuition=tuition,
                living_cost_total=living_cost_total,
                total_cost=total_cost,
                avg_salary=avg_salary,
                salary_premium=salary_premium,
                employment_rate=employment_rate,
                payback_years=payback_years,
                npv_5yr=npv_5yr,
                risk_adjusted_roi=risk_adjusted_roi,
            )
        )

    # Sort by NPV descending
    results.sort(key=lambda r: r.npv_5yr, reverse=True)
    return results
