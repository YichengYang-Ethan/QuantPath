"""Tests for core.test_requirements (GRE and TOEFL checks)."""

import pytest

from core.models import ProgramData, TestScores, UserProfile
from core.test_requirements import check_gre, check_toefl


# ===================================================================
# Helpers
# ===================================================================

def _profile(
    gre_quant: int | None = None,
    gre_verbal: int | None = None,
    toefl: int | None = None,
    is_international: bool = False,
    years_at_us: int = 0,
) -> UserProfile:
    """Create a minimal UserProfile for test requirements testing."""
    return UserProfile(
        name="Test",
        test_scores=TestScores(
            gre_quant=gre_quant,
            gre_verbal=gre_verbal,
            toefl=toefl,
        ),
        is_international=is_international,
        years_at_us_institution=years_at_us,
    )


def _program(
    name: str = "TestProg",
    gre_required: bool = False,
    gre_exemption: str | None = None,
    gre_quant_avg: int | None = None,
    gre_code: str | None = None,
    toefl_waiver_conditions: list[str] | None = None,
    toefl_min_ibt: int | None = None,
    toefl_min_ielts: float | None = None,
) -> ProgramData:
    """Create a minimal ProgramData for test requirements testing."""
    return ProgramData(
        id="test-prog",
        name=name,
        gre_required=gre_required,
        gre_exemption=gre_exemption,
        gre_quant_avg=gre_quant_avg,
        gre_code=gre_code,
        toefl_waiver_conditions=toefl_waiver_conditions or [],
        toefl_min_ibt=toefl_min_ibt,
        toefl_min_ielts=toefl_min_ielts,
    )


# ===================================================================
# check_gre
# ===================================================================

class TestCheckGre:
    """Test GRE requirement/exemption logic."""

    def test_gre_not_required(self) -> None:
        """When the program does not require GRE, result should be not required + exempt."""
        profile = _profile()
        program = _program(gre_required=False)
        result = check_gre(profile, program)

        assert result["required"] is False
        assert result["exempt"] is True
        assert "does not require" in result["reason"]

    def test_gre_not_required_with_note(self) -> None:
        """When GRE is not required but has an exemption note, it should appear in reason."""
        profile = _profile()
        program = _program(gre_required=False, gre_exemption="Own placement test used")
        result = check_gre(profile, program)

        assert result["required"] is False
        assert result["exempt"] is True
        assert "Own placement test used" in result["reason"]

    def test_gre_required_with_exemption(self) -> None:
        """When GRE is required but an exemption is available."""
        profile = _profile()
        program = _program(gre_required=True, gre_exemption="Quantitative assessment replaces GRE")
        result = check_gre(profile, program)

        assert result["required"] is True
        assert result["exempt"] is True
        assert "exemption" in result["reason"]
        assert "Quantitative assessment" in result["reason"]

    def test_gre_required_no_exemption(self) -> None:
        """When GRE is strictly required with no exemption."""
        profile = _profile()
        program = _program(name="Baruch MFE", gre_required=True)
        result = check_gre(profile, program)

        assert result["required"] is True
        assert result["exempt"] is False
        assert "requires the GRE" in result["reason"]

    def test_gre_required_with_user_score_above_avg(self) -> None:
        """When user has GRE score above program average."""
        profile = _profile(gre_quant=170)
        program = _program(gre_required=True, gre_quant_avg=168)
        result = check_gre(profile, program)

        assert result["required"] is True
        assert result["exempt"] is False
        assert "170" in result["reason"]
        assert "meets/exceeds" in result["reason"]

    def test_gre_required_with_user_score_below_avg(self) -> None:
        """When user has GRE score below program average."""
        profile = _profile(gre_quant=160)
        program = _program(gre_required=True, gre_quant_avg=168)
        result = check_gre(profile, program)

        assert result["required"] is True
        assert "160" in result["reason"]
        assert "8 points below" in result["reason"]

    def test_gre_required_with_ets_code(self) -> None:
        """ETS code should appear in the reason when GRE is required."""
        profile = _profile()
        program = _program(gre_required=True, gre_code="5678")
        result = check_gre(profile, program)

        assert result["required"] is True
        assert "ETS code: 5678" in result["reason"]

    def test_gre_required_no_user_score_no_code(self) -> None:
        """When GRE is required but user has no score and no code -- clean output."""
        profile = _profile(gre_quant=None)
        program = _program(gre_required=True, gre_quant_avg=None, gre_code=None)
        result = check_gre(profile, program)

        assert result["required"] is True
        assert result["exempt"] is False
        reason = result["reason"]
        assert "requires the GRE" in reason
        # No score info or code info should appear
        assert "Your GRE Quant" not in reason
        assert "ETS code" not in reason


# ===================================================================
# check_toefl
# ===================================================================

class TestCheckToefl:
    """Test TOEFL/IELTS requirement and waiver logic."""

    def test_domestic_applicant_not_required(self) -> None:
        """Domestic applicants do not need TOEFL."""
        profile = _profile(is_international=False)
        program = _program(toefl_min_ibt=100)
        result = check_toefl(profile, program)

        assert result["required"] is False
        assert result["waived"] is True
        assert "domestic" in result["reason"].lower()

    def test_international_no_waiver(self) -> None:
        """International applicant with no waiver conditions -> required, not waived."""
        profile = _profile(is_international=True, years_at_us=0)
        program = _program(toefl_min_ibt=100, toefl_waiver_conditions=[])
        result = check_toefl(profile, program)

        assert result["required"] is True
        assert result["waived"] is False

    def test_international_english_medium_waiver_met(self) -> None:
        """International applicant with enough years at US institution -> waiver."""
        profile = _profile(is_international=True, years_at_us=3)
        program = _program(
            toefl_waiver_conditions=["English-medium institution degree (2+ years)"]
        )
        result = check_toefl(profile, program)

        assert result["required"] is True
        assert result["waived"] is True
        assert "Waiver eligible" in result["reason"]
        assert "3 year(s)" in result["reason"]

    def test_international_english_medium_waiver_not_met(self) -> None:
        """International applicant without enough years -> not waived."""
        profile = _profile(is_international=True, years_at_us=1)
        program = _program(
            toefl_waiver_conditions=["English-medium institution degree (2+ years)"]
        )
        result = check_toefl(profile, program)

        assert result["required"] is True
        assert result["waived"] is False

    def test_international_english_medium_custom_years(self) -> None:
        """Waiver condition specifies 3+ years."""
        profile = _profile(is_international=True, years_at_us=3)
        program = _program(
            toefl_waiver_conditions=["English-medium institution (3+ years required)"]
        )
        result = check_toefl(profile, program)

        assert result["required"] is True
        assert result["waived"] is True

    def test_international_english_medium_custom_years_not_met(self) -> None:
        """Waiver specifies 3+ years but user has only 2."""
        profile = _profile(is_international=True, years_at_us=2)
        program = _program(
            toefl_waiver_conditions=["English-medium institution (3+ years required)"]
        )
        result = check_toefl(profile, program)

        assert result["required"] is True
        assert result["waived"] is False

    def test_toefl_min_score_in_reason(self) -> None:
        """When TOEFL minimums are set, they should appear in the reason."""
        profile = _profile(is_international=True)
        program = _program(toefl_min_ibt=100, toefl_min_ielts=7.0)
        result = check_toefl(profile, program)

        assert result["required"] is True
        assert "TOEFL iBT minimum: 100" in result["reason"]
        assert "IELTS minimum: 7.0" in result["reason"]

    def test_user_toefl_meets_minimum(self) -> None:
        """When user's TOEFL meets the minimum, reason should say so."""
        profile = _profile(is_international=True, toefl=105)
        program = _program(toefl_min_ibt=100)
        result = check_toefl(profile, program)

        assert result["required"] is True
        assert "105" in result["reason"]
        assert "meets the minimum" in result["reason"]

    def test_user_toefl_below_minimum(self) -> None:
        """When user's TOEFL is below minimum, reason should show the gap."""
        profile = _profile(is_international=True, toefl=90)
        program = _program(toefl_min_ibt=100)
        result = check_toefl(profile, program)

        assert result["required"] is True
        assert "10 more points" in result["reason"]

    def test_no_minimums_set(self) -> None:
        """When no TOEFL/IELTS minimums are set, fallback text should appear."""
        profile = _profile(is_international=True)
        program = _program(toefl_min_ibt=None, toefl_min_ielts=None)
        result = check_toefl(profile, program)

        assert result["required"] is True
        assert "See programme website" in result["reason"]

    def test_unrelated_waiver_condition_not_triggered(self) -> None:
        """Waiver conditions that don't mention 'english-medium' should not trigger."""
        profile = _profile(is_international=True, years_at_us=5)
        program = _program(
            toefl_waiver_conditions=["US citizenship or permanent residency"]
        )
        result = check_toefl(profile, program)

        assert result["required"] is True
        assert result["waived"] is False
