"""Tests for core.roi_calculator."""


import pytest

from core.models import ProgramData
from core.roi_calculator import (
    CITY_COSTS,
    UNIVERSITY_CITY,
    _get_city_cost,
    calculate_roi,
)

# ===================================================================
# Helpers
# ===================================================================


def _make_program(
    program_id: str = "test-prog",
    name: str = "Test MFE",
    university: str = "Carnegie Mellon University",
    tuition: int | None = 100000,
    salary: int | None = 150000,
    employment_rate: float | None = 0.95,
) -> ProgramData:
    """Create a minimal ProgramData for testing."""
    return ProgramData(
        id=program_id,
        name=name,
        university=university,
        tuition_total=tuition,
        avg_base_salary=salary,
        employment_rate_3m=employment_rate,
    )


# ===================================================================
# City cost lookup
# ===================================================================


class TestCityCostLookup:
    """Test the university -> city -> cost lookup chain."""

    def test_known_university_returns_correct_city_cost(self) -> None:
        """Carnegie Mellon -> Pittsburgh costs."""
        cost = _get_city_cost("Carnegie Mellon University")
        assert cost.city == "Pittsburgh"
        assert cost.monthly_rent == 1200
        assert cost.monthly_other == 1000

    def test_another_known_university(self) -> None:
        """Columbia -> New York costs."""
        cost = _get_city_cost("Columbia University")
        assert cost.city == "New York"
        assert cost.monthly_rent == 2200
        assert cost.monthly_other == 1500

    def test_unknown_university_defaults_to_new_york(self) -> None:
        """Unknown university falls back to New York costs."""
        cost = _get_city_cost("Unknown University")
        assert cost.city == "New York"
        assert cost.monthly_rent == 2200

    def test_all_yaml_universities_have_mapping(self) -> None:
        """Every university in UNIVERSITY_CITY should map to a valid CITY_COSTS city."""
        for university, city in UNIVERSITY_CITY.items():
            assert city in CITY_COSTS, f"{university} maps to {city} which is not in CITY_COSTS"

    def test_champaign_is_cheapest(self) -> None:
        """Champaign should have the lowest total monthly cost."""
        cost = _get_city_cost("University of Illinois Urbana-Champaign")
        assert cost.city == "Champaign"
        assert cost.monthly_rent + cost.monthly_other == 1800  # 900 + 900


# ===================================================================
# Basic ROI calculation
# ===================================================================


class TestBasicROI:
    """Test core ROI calculation logic with known values."""

    def test_basic_calculation(self) -> None:
        """Verify all fields are computed correctly for a simple case."""
        prog = _make_program(
            tuition=100000,
            salary=150000,
            employment_rate=1.0,
            university="Carnegie Mellon University",
        )
        results = calculate_roi([prog], opportunity_cost_salary=65000)
        assert len(results) == 1

        r = results[0]
        # Pittsburgh: rent=1200, other=1000, 18 months
        assert r.living_cost_total == (1200 + 1000) * 18  # 39600
        assert r.total_cost == 100000 + 39600  # 139600
        assert r.salary_premium == 150000 - 65000  # 85000
        assert r.tuition == 100000
        assert r.avg_salary == 150000
        assert r.employment_rate == 1.0

    def test_living_cost_uses_program_duration(self) -> None:
        """Custom program_duration_months is respected."""
        prog = _make_program(university="Carnegie Mellon University")
        results_18 = calculate_roi([prog], program_duration_months=18)
        results_24 = calculate_roi([prog], program_duration_months=24)
        assert results_24[0].living_cost_total > results_18[0].living_cost_total
        # Pittsburgh: (1200 + 1000) * 24 = 52800 vs (1200 + 1000) * 18 = 39600
        assert results_24[0].living_cost_total == (1200 + 1000) * 24

    def test_program_id_and_name_propagated(self) -> None:
        """ROIResult carries forward program metadata."""
        prog = _make_program(
            program_id="baruch-mfe", name="Baruch MFE",
            university="Columbia University",
        )
        results = calculate_roi([prog])
        r = results[0]
        assert r.program_id == "baruch-mfe"
        assert r.program_name == "Baruch MFE"
        assert r.university == "Columbia University"


# ===================================================================
# Payback years
# ===================================================================


class TestPaybackYears:
    """Test payback period calculation."""

    def test_payback_positive_premium(self) -> None:
        """payback_years = total_cost / salary_premium when premium > 0."""
        prog = _make_program(
            tuition=100000,
            salary=165000,  # premium = 100000
            university="Carnegie Mellon University",
        )
        results = calculate_roi([prog], opportunity_cost_salary=65000)
        r = results[0]
        # total_cost = 100000 + 39600 = 139600, premium = 100000
        expected_payback = 139600 / 100000
        assert r.payback_years == pytest.approx(expected_payback)

    def test_payback_zero_premium(self) -> None:
        """salary == opportunity_cost -> salary_premium = 0 -> payback = inf."""
        prog = _make_program(salary=65000)
        results = calculate_roi([prog], opportunity_cost_salary=65000)
        assert results[0].payback_years == float("inf")

    def test_payback_negative_premium(self) -> None:
        """salary < opportunity_cost -> salary_premium < 0 -> payback = inf."""
        prog = _make_program(salary=50000)
        results = calculate_roi([prog], opportunity_cost_salary=65000)
        assert results[0].payback_years == float("inf")


# ===================================================================
# NPV calculation
# ===================================================================


class TestNPV:
    """Test 5-year NPV calculation."""

    def test_npv_formula(self) -> None:
        """Verify NPV matches manual calculation."""
        prog = _make_program(
            tuition=50000,
            salary=115000,
            employment_rate=1.0,
            university="University of Illinois Urbana-Champaign",
        )
        results = calculate_roi(
            [prog], opportunity_cost_salary=65000, discount_rate=0.05
        )
        r = results[0]

        # Champaign: rent=900, other=900 -> 1800 * 18 = 32400
        total_cost = 50000 + 32400  # 82400
        premium = 115000 - 65000  # 50000

        expected_npv = -total_cost
        for year in range(1, 6):
            expected_npv += (premium * 1.0) / (1.05**year)

        assert r.npv_5yr == pytest.approx(expected_npv, rel=1e-6)

    def test_npv_with_partial_employment(self) -> None:
        """Employment rate < 1.0 reduces the annual cash flow in NPV."""
        prog_full = _make_program(employment_rate=1.0)
        prog_half = _make_program(employment_rate=0.5)
        r_full = calculate_roi([prog_full])[0]
        r_half = calculate_roi([prog_half])[0]
        assert r_full.npv_5yr > r_half.npv_5yr

    def test_npv_with_zero_discount_rate(self) -> None:
        """At 0% discount rate, NPV = sum of (premium * emp_rate) * 5 - total_cost."""
        prog = _make_program(
            tuition=50000,
            salary=115000,
            employment_rate=1.0,
            university="University of Illinois Urbana-Champaign",
        )
        results = calculate_roi([prog], discount_rate=0.0)
        r = results[0]

        total_cost = 50000 + 32400
        premium = 115000 - 65000
        expected = premium * 1.0 * 5 - total_cost  # 250000 - 82400 = 167600

        assert r.npv_5yr == pytest.approx(expected, rel=1e-6)

    def test_npv_negative_when_costs_high(self) -> None:
        """Very high tuition with modest salary -> negative NPV."""
        prog = _make_program(
            tuition=200000,
            salary=80000,
            employment_rate=0.5,
            university="Columbia University",  # New York: expensive
        )
        results = calculate_roi([prog])
        assert results[0].npv_5yr < 0


# ===================================================================
# Risk-adjusted ROI
# ===================================================================


class TestRiskAdjustedROI:
    """Test the risk-adjusted ROI percentage calculation."""

    def test_risk_adjusted_roi_formula(self) -> None:
        """Verify: (salary_premium * employment_rate * 5 - total_cost) / total_cost * 100."""
        prog = _make_program(
            tuition=100000,
            salary=150000,
            employment_rate=0.90,
            university="Carnegie Mellon University",
        )
        results = calculate_roi([prog], opportunity_cost_salary=65000)
        r = results[0]

        # Pittsburgh living: 39600
        total_cost = 100000 + 39600
        premium = 85000
        expected_roi = (premium * 0.90 * 5 - total_cost) / total_cost * 100

        assert r.risk_adjusted_roi == pytest.approx(expected_roi, rel=1e-6)

    def test_roi_higher_with_better_employment(self) -> None:
        """Higher employment rate -> higher risk-adjusted ROI."""
        prog_high = _make_program(employment_rate=1.0)
        prog_low = _make_program(employment_rate=0.7)
        r_high = calculate_roi([prog_high])[0]
        r_low = calculate_roi([prog_low])[0]
        assert r_high.risk_adjusted_roi > r_low.risk_adjusted_roi

    def test_roi_negative_for_bad_deal(self) -> None:
        """Expensive program with low salary and poor employment -> negative ROI."""
        prog = _make_program(
            tuition=150000,
            salary=70000,
            employment_rate=0.3,
            university="Stanford University",  # San Francisco: expensive
        )
        results = calculate_roi([prog])
        assert results[0].risk_adjusted_roi < 0


# ===================================================================
# Filtering and sorting
# ===================================================================


class TestFilteringAndSorting:
    """Test that programs are filtered and sorted correctly."""

    def test_programs_missing_tuition_are_skipped(self) -> None:
        """Programs with tuition_total=None are excluded."""
        prog = _make_program(tuition=None, salary=150000)
        results = calculate_roi([prog])
        assert len(results) == 0

    def test_programs_missing_salary_are_skipped(self) -> None:
        """Programs with avg_base_salary=None are excluded."""
        prog = _make_program(tuition=100000, salary=None)
        results = calculate_roi([prog])
        assert len(results) == 0

    def test_programs_missing_both_are_skipped(self) -> None:
        """Programs missing both tuition and salary are excluded."""
        prog = _make_program(tuition=None, salary=None)
        results = calculate_roi([prog])
        assert len(results) == 0

    def test_programs_missing_employment_rate_default_to_zero(self) -> None:
        """employment_rate_3m=None -> treated as 0.0."""
        prog = _make_program(employment_rate=None)
        results = calculate_roi([prog])
        assert len(results) == 1
        assert results[0].employment_rate == 0.0

    def test_empty_programs_list(self) -> None:
        """Empty input -> empty output."""
        results = calculate_roi([])
        assert results == []

    def test_sort_by_npv_descending(self) -> None:
        """Results are sorted by npv_5yr in descending order."""
        # Low tuition + high salary = best NPV
        best = _make_program(
            program_id="best",
            tuition=30000,
            salary=200000,
            employment_rate=1.0,
            university="University of Illinois Urbana-Champaign",
        )
        # High tuition + low salary = worst NPV
        worst = _make_program(
            program_id="worst",
            tuition=200000,
            salary=70000,
            employment_rate=0.5,
            university="Stanford University",
        )
        middle = _make_program(
            program_id="middle",
            tuition=80000,
            salary=130000,
            employment_rate=0.9,
            university="Boston University",
        )

        # Pass in non-sorted order
        results = calculate_roi([worst, middle, best])
        assert results[0].program_id == "best"
        assert results[-1].program_id == "worst"
        # Verify strict descending
        npvs = [r.npv_5yr for r in results]
        assert npvs == sorted(npvs, reverse=True)

    def test_mixed_valid_and_invalid_programs(self) -> None:
        """Only programs with both tuition and salary appear in results."""
        valid = _make_program(program_id="valid", tuition=50000, salary=120000)
        no_tuition = _make_program(program_id="no-tuit", tuition=None, salary=120000)
        no_salary = _make_program(program_id="no-sal", tuition=50000, salary=None)
        results = calculate_roi([valid, no_tuition, no_salary])
        assert len(results) == 1
        assert results[0].program_id == "valid"


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_salary_equals_opportunity_cost(self) -> None:
        """salary_premium = 0 -> payback = inf, NPV = -total_cost."""
        prog = _make_program(
            tuition=50000,
            salary=65000,
            employment_rate=1.0,
            university="University of Illinois Urbana-Champaign",
        )
        results = calculate_roi([prog], opportunity_cost_salary=65000)
        r = results[0]
        assert r.salary_premium == 0
        assert r.payback_years == float("inf")
        assert r.npv_5yr == pytest.approx(-r.total_cost)

    def test_salary_below_opportunity_cost(self) -> None:
        """Negative salary premium -> payback = inf, negative NPV."""
        prog = _make_program(
            tuition=50000,
            salary=40000,
            employment_rate=1.0,
            university="University of Illinois Urbana-Champaign",
        )
        results = calculate_roi([prog], opportunity_cost_salary=65000)
        r = results[0]
        assert r.salary_premium == -25000
        assert r.payback_years == float("inf")
        assert r.npv_5yr < -r.total_cost  # worse than just losing total_cost

    def test_zero_employment_rate(self) -> None:
        """0% employment -> NPV = -total_cost, ROI negative."""
        prog = _make_program(employment_rate=0.0)
        results = calculate_roi([prog])
        r = results[0]
        assert r.npv_5yr == pytest.approx(-r.total_cost)
        assert r.risk_adjusted_roi < 0

    def test_custom_opportunity_cost(self) -> None:
        """Higher opportunity cost reduces salary premium and ROI."""
        prog = _make_program(salary=150000)
        r_low = calculate_roi([prog], opportunity_cost_salary=50000)[0]
        r_high = calculate_roi([prog], opportunity_cost_salary=100000)[0]
        assert r_low.salary_premium > r_high.salary_premium
        assert r_low.npv_5yr > r_high.npv_5yr

    def test_custom_discount_rate(self) -> None:
        """Higher discount rate reduces NPV."""
        prog = _make_program(salary=150000, employment_rate=1.0)
        r_low = calculate_roi([prog], discount_rate=0.02)[0]
        r_high = calculate_roi([prog], discount_rate=0.15)[0]
        assert r_low.npv_5yr > r_high.npv_5yr
