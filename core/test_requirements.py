# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""Standardised test requirement checks (GRE and TOEFL/IELTS).

Determines whether an applicant needs to submit GRE or English
proficiency scores based on programme requirements and the
applicant's background.
"""

from __future__ import annotations

from .models import ProgramData, UserProfile

# ===================================================================
# GRE
# ===================================================================


def check_gre(
    profile: UserProfile,
    program: ProgramData,
) -> dict[str, object]:
    """Check whether the GRE is required for this programme/applicant.

    Returns
    -------
    dict
        Keys:
            required (bool):
                Whether the programme lists GRE as required.
            exempt (bool):
                Whether the applicant qualifies for an exemption.
            reason (str):
                Human-readable explanation of the determination.
    """
    # If the programme does not require GRE at all.
    if not program.gre_required:
        reason_parts = [f"{program.name} does not require the GRE."]
        if program.gre_exemption:
            reason_parts.append(f"Note: {program.gre_exemption}")
        return {
            "required": False,
            "exempt": True,
            "reason": " ".join(reason_parts),
        }

    # Programme requires GRE -- check for exemptions.
    if program.gre_exemption:
        # Some programmes grant automatic exemptions (e.g. own
        # quantitative assessment).
        return {
            "required": True,
            "exempt": True,
            "reason": (
                f"{program.name} normally requires the GRE but offers "
                f"an exemption: {program.gre_exemption}"
            ),
        }

    # No exemption available; GRE is mandatory.
    score_note = ""
    if profile.test_scores.gre_quant is not None:
        score_note = f" Your GRE Quant score: {profile.test_scores.gre_quant}."
        if program.gre_quant_avg is not None:
            diff = profile.test_scores.gre_quant - program.gre_quant_avg
            if diff >= 0:
                score_note += (
                    f" This meets/exceeds the programme average ({program.gre_quant_avg})."
                )
            else:
                score_note += (
                    f" The programme average is {program.gre_quant_avg} "
                    f"(you are {abs(diff)} points below)."
                )

    code_note = ""
    if program.gre_code:
        code_note = f" ETS code: {program.gre_code}."

    return {
        "required": True,
        "exempt": False,
        "reason": f"{program.name} requires the GRE.{score_note}{code_note}",
    }


# ===================================================================
# TOEFL / English proficiency
# ===================================================================


def check_toefl(
    profile: UserProfile,
    program: ProgramData,
) -> dict[str, object]:
    """Check whether English proficiency testing is required.

    The check considers:
        - Whether the applicant is international.
        - Whether they have studied at a US institution long enough
          to qualify for a waiver.
        - Programme-specific waiver conditions.

    Returns
    -------
    dict
        Keys:
            required (bool):
                Whether the programme requires English proficiency
                scores from international applicants.
            waived (bool):
                Whether this specific applicant qualifies for a waiver.
            reason (str):
                Human-readable explanation.
    """
    # Domestic applicants generally do not need TOEFL.
    if not profile.is_international:
        return {
            "required": False,
            "waived": True,
            "reason": "TOEFL/IELTS is not required for domestic applicants.",
        }

    # International applicant -- check waiver conditions.
    waiver_conditions = program.toefl_waiver_conditions or []

    # Common waiver: degree from English-medium institution (2+ years).
    waived = False
    waiver_reason = ""

    for condition in waiver_conditions:
        condition_lower = condition.lower()
        if "english-medium" in condition_lower or "english medium" in condition_lower:
            # Check if user has spent enough years at a US institution.
            years_needed = 2  # default
            # Try to parse years from the condition string.
            for token in condition_lower.split():
                if token.endswith("+"):
                    try:
                        years_needed = int(token.replace("+", ""))
                    except ValueError:
                        pass

            if profile.years_at_us_institution >= years_needed:
                waived = True
                waiver_reason = (
                    f"Waiver eligible: you have {profile.years_at_us_institution} "
                    f"year(s) at a US institution (condition: {condition})."
                )
                break

    if waived:
        return {
            "required": True,
            "waived": True,
            "reason": waiver_reason,
        }

    # Not waived -- build guidance.
    score_info: list[str] = []
    if program.toefl_min_ibt:
        score_info.append(f"TOEFL iBT minimum: {program.toefl_min_ibt}")
    if program.toefl_min_ielts:
        score_info.append(f"IELTS minimum: {program.toefl_min_ielts}")

    user_score_note = ""
    if profile.test_scores.toefl is not None:
        user_score_note = f" Your TOEFL score: {profile.test_scores.toefl}."
        if program.toefl_min_ibt and profile.test_scores.toefl >= program.toefl_min_ibt:
            user_score_note += " This meets the minimum."
        elif program.toefl_min_ibt:
            gap = program.toefl_min_ibt - profile.test_scores.toefl
            user_score_note += f" You need {gap} more points to meet the minimum."

    minimums = "; ".join(score_info) if score_info else "See programme website for minimums"

    return {
        "required": True,
        "waived": False,
        "reason": (
            f"{program.name} requires English proficiency scores for "
            f"international applicants. {minimums}.{user_score_note}"
        ),
    }
