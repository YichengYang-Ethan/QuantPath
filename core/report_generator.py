# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""PDF report generator for QuantPath evaluation results.

Uses fpdf2 to produce a professional multi-page report containing the
applicant's profile summary, dimension scores, gap analysis, and school
recommendations.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from .gap_advisor import GapRecommendation
from .models import EvaluationResult, UserProfile

# ===================================================================
# Colour constants (RGB tuples)
# ===================================================================

_HEADER_BLUE = (44, 62, 80)  # #2c3e50
_SECTION_BLUE = (52, 152, 219)  # #3498db
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_LIGHT_GREY = (245, 245, 245)
_ROW_ALT = (235, 243, 255)
_RED = (220, 53, 69)
_ORANGE = (255, 152, 0)
_BLUE = (52, 152, 219)
_GREEN = (40, 167, 69)
_DARK_GREY = (100, 100, 100)


def _safe(value: Any, fallback: str = "N/A") -> str:
    """Return *value* as a string, or *fallback* when None / empty."""
    if value is None or value == "" or value == 0:
        return fallback
    return str(value)


def _latin1_safe(text: str) -> str:
    """Replace characters outside Latin-1 with ASCII equivalents.

    Built-in PDF fonts (Helvetica, Courier) only support Latin-1.  This
    helper swaps common Unicode punctuation to safe alternatives so the
    report renders without errors.
    """
    replacements = {
        "\u2014": "--",   # em dash
        "\u2013": "-",    # en dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2026": "...",  # ellipsis
        "\u2022": "*",    # bullet
        "\u00b7": "*",    # middle dot
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    # Final fallback: encode to latin-1, replacing anything left
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ===================================================================
# PDF builder
# ===================================================================


def generate_report(
    profile: UserProfile,
    evaluation: EvaluationResult,
    rankings: dict,
    gaps: list,
    output_path: str = "quantpath_report.pdf",
) -> str:
    """Generate a professional PDF evaluation report.

    Parameters
    ----------
    profile:
        The applicant's :class:`UserProfile`.
    evaluation:
        The :class:`EvaluationResult` from the profile evaluator.
    rankings:
        Output of :func:`rank_schools` -- a dict with keys ``reach``,
        ``target``, ``safety``, ``all``.
    gaps:
        Output of :func:`analyze_gaps` -- a list of
        :class:`GapRecommendation` objects.
    output_path:
        Destination file path for the PDF.  Defaults to
        ``"quantpath_report.pdf"``.

    Returns
    -------
    str
        The path the PDF was written to (*output_path*).
    """
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise ImportError(
            "fpdf2 is required for PDF generation. "
            "Install it with: pip install 'quantpath[pdf]'"
        ) from exc

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(15, 15, 15)

    # ------------------------------------------------------------------
    # 1. Title page
    # ------------------------------------------------------------------
    pdf.add_page()

    # Top spacing
    pdf.ln(50)

    # Title block
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*_HEADER_BLUE)
    pdf.cell(0, 14, "QuantPath", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 18)
    pdf.set_text_color(*_SECTION_BLUE)
    pdf.cell(
        0,
        12,
        "MFE Application Report",
        new_x="LMARGIN",
        new_y="NEXT",
        align="C",
    )

    pdf.ln(20)

    # Applicant details
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(*_BLACK)
    pdf.cell(
        0,
        10,
        _latin1_safe(f"Applicant: {profile.name or 'N/A'}"),
        new_x="LMARGIN",
        new_y="NEXT",
        align="C",
    )
    pdf.cell(
        0,
        10,
        _latin1_safe(f"University: {profile.university or 'N/A'}"),
        new_x="LMARGIN",
        new_y="NEXT",
        align="C",
    )
    pdf.cell(
        0,
        10,
        f"Generated: {date.today().isoformat()}",
        new_x="LMARGIN",
        new_y="NEXT",
        align="C",
    )

    # Horizontal rule
    pdf.ln(15)
    y = pdf.get_y()
    pdf.set_draw_color(*_SECTION_BLUE)
    pdf.set_line_width(0.5)
    pdf.line(60, y, pdf.w - 60, y)

    # ------------------------------------------------------------------
    # 2. Profile Summary section
    # ------------------------------------------------------------------
    pdf.add_page()
    _section_header(pdf, "Profile Summary")

    majors_str = ", ".join(profile.majors) if profile.majors else "N/A"
    intl_str = "Yes" if profile.is_international else "No"

    gre_q = _safe(profile.test_scores.gre_quant if profile.test_scores else None)
    gre_v = _safe(profile.test_scores.gre_verbal if profile.test_scores else None)
    toefl = _safe(profile.test_scores.toefl if profile.test_scores else None)

    summary_rows = [
        ("Name", profile.name or "N/A"),
        ("University", profile.university or "N/A"),
        ("GPA", f"{profile.gpa:.2f}" if profile.gpa else "N/A"),
        ("Majors", majors_str),
        ("International", intl_str),
        ("GRE Quant", gre_q),
        ("GRE Verbal", gre_v),
        ("TOEFL iBT", toefl),
    ]

    col_w_label = 50
    col_w_value = 120

    for i, (label, value) in enumerate(summary_rows):
        if i % 2 == 0:
            pdf.set_fill_color(*_ROW_ALT)
        else:
            pdf.set_fill_color(*_WHITE)

        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*_HEADER_BLUE)
        pdf.cell(col_w_label, 8, f"  {label}", fill=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(*_BLACK)
        pdf.cell(
            col_w_value,
            8,
            _latin1_safe(f"  {value}"),
            new_x="LMARGIN",
            new_y="NEXT",
            fill=True,
        )

    # ------------------------------------------------------------------
    # 3. Dimension Scores section
    # ------------------------------------------------------------------
    pdf.ln(8)
    _section_header(pdf, "Dimension Scores")

    dim_labels = {
        "math": "Math",
        "statistics": "Statistics",
        "cs": "Computer Science",
        "finance_econ": "Finance / Econ",
        "gpa": "GPA",
    }

    # Table header
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(*_HEADER_BLUE)
    pdf.set_text_color(*_WHITE)
    pdf.cell(55, 8, "  Dimension", fill=True)
    pdf.cell(25, 8, "Score", fill=True, align="C")
    pdf.cell(100, 8, "  Rating", new_x="LMARGIN", new_y="NEXT", fill=True)

    pdf.set_text_color(*_BLACK)

    for i, (dim_id, label) in enumerate(dim_labels.items()):
        score = evaluation.dimension_scores.get(dim_id, 0.0)
        bar_text = _text_bar(score)

        if i % 2 == 0:
            pdf.set_fill_color(*_ROW_ALT)
        else:
            pdf.set_fill_color(*_WHITE)

        pdf.set_font("Helvetica", "", 10)
        pdf.cell(55, 8, f"  {label}", fill=True)
        pdf.cell(25, 8, f"{score:.1f} / 10", fill=True, align="C")
        pdf.set_font("Courier", "", 10)
        pdf.cell(
            100,
            8,
            f"  {bar_text}",
            new_x="LMARGIN",
            new_y="NEXT",
            fill=True,
        )

    # Overall score row
    pdf.set_fill_color(*_SECTION_BLUE)
    pdf.set_text_color(*_WHITE)
    pdf.set_font("Helvetica", "B", 11)
    overall = evaluation.overall_score
    overall_bar = _text_bar(overall)
    pdf.cell(55, 9, "  OVERALL", fill=True)
    pdf.cell(25, 9, f"{overall:.1f} / 10", fill=True, align="C")
    pdf.set_font("Courier", "B", 11)
    pdf.cell(
        100,
        9,
        f"  {overall_bar}",
        new_x="LMARGIN",
        new_y="NEXT",
        fill=True,
    )
    pdf.set_text_color(*_BLACK)

    # ------------------------------------------------------------------
    # 4. Gap Analysis section
    # ------------------------------------------------------------------
    pdf.add_page()
    _section_header(pdf, "Gap Analysis")

    if not gaps:
        pdf.set_font("Helvetica", "I", 11)
        pdf.set_text_color(*_GREEN)
        pdf.cell(
            0,
            10,
            "No gaps found -- your profile is strong across all dimensions!",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_text_color(*_BLACK)
    else:
        # Column widths
        cw_factor = 38
        cw_dim = 28
        cw_score = 18
        cw_pri = 18
        cw_action = 78

        # Header
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(*_HEADER_BLUE)
        pdf.set_text_color(*_WHITE)
        pdf.cell(cw_factor, 7, "  Factor", fill=True)
        pdf.cell(cw_dim, 7, "  Dimension", fill=True)
        pdf.cell(cw_score, 7, " Score", fill=True, align="C")
        pdf.cell(cw_pri, 7, " Priority", fill=True, align="C")
        pdf.cell(
            cw_action,
            7,
            "  Recommended Action",
            new_x="LMARGIN",
            new_y="NEXT",
            fill=True,
        )

        pdf.set_text_color(*_BLACK)

        for i, gap in enumerate(gaps):
            # Coerce to GapRecommendation or dict
            if isinstance(gap, GapRecommendation):
                factor = gap.factor
                dimension = gap.dimension
                score = gap.score
                priority = gap.priority
                action = gap.action
            else:
                factor = gap.get("factor", "")
                dimension = gap.get("dimension", "")
                score = gap.get("score", 0.0)
                priority = gap.get("priority", "Low")
                action = gap.get("action", "")

            factor_label = factor.replace("_", " ").title()
            score_str = "Missing" if score == 0 else f"{score:.1f}/10"
            action = _latin1_safe(action)

            # Alternating row background
            if i % 2 == 0:
                pdf.set_fill_color(*_ROW_ALT)
            else:
                pdf.set_fill_color(*_WHITE)

            # Compute the row height based on action text length
            pdf.set_font("Helvetica", "", 8)
            action_lines = pdf.multi_cell(
                cw_action, 5, f" {action}", dry_run=True, output="LINES"
            )
            row_h = max(7, len(action_lines) * 5)

            # Check if we need a page break
            if pdf.get_y() + row_h > pdf.h - 25:
                pdf.add_page()
                _section_header(pdf, "Gap Analysis (continued)")
                # Re-draw header
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_fill_color(*_HEADER_BLUE)
                pdf.set_text_color(*_WHITE)
                pdf.cell(cw_factor, 7, "  Factor", fill=True)
                pdf.cell(cw_dim, 7, "  Dimension", fill=True)
                pdf.cell(cw_score, 7, " Score", fill=True, align="C")
                pdf.cell(cw_pri, 7, " Priority", fill=True, align="C")
                pdf.cell(
                    cw_action,
                    7,
                    "  Recommended Action",
                    new_x="LMARGIN",
                    new_y="NEXT",
                    fill=True,
                )
                pdf.set_text_color(*_BLACK)
                if i % 2 == 0:
                    pdf.set_fill_color(*_ROW_ALT)
                else:
                    pdf.set_fill_color(*_WHITE)

            y_before = pdf.get_y()

            pdf.set_font("Helvetica", "", 8)
            pdf.cell(cw_factor, row_h, f"  {factor_label}", fill=True)
            pdf.cell(cw_dim, row_h, f"  {dimension}", fill=True)
            pdf.cell(cw_score, row_h, f" {score_str}", fill=True, align="C")

            # Priority with color
            _set_priority_color(pdf, priority)
            pdf.cell(cw_pri, row_h, f" {priority}", fill=True, align="C")
            pdf.set_text_color(*_BLACK)

            # Action as multi-cell
            x_action = pdf.get_x()
            pdf.set_xy(x_action, y_before)
            pdf.multi_cell(cw_action, 5, f" {action}", fill=True)

            # Make sure we end up at the right Y position
            y_after = max(pdf.get_y(), y_before + row_h)
            pdf.set_y(y_after)

    # ------------------------------------------------------------------
    # 5. School Recommendations section
    # ------------------------------------------------------------------
    pdf.add_page()
    _section_header(pdf, "School Recommendations")

    for label, style_color, key in [
        ("Reach Schools", _RED, "reach"),
        ("Target Schools", _ORANGE, "target"),
        ("Safety Schools", _GREEN, "safety"),
    ]:
        entries = rankings.get(key, [])
        if not entries:
            continue

        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*style_color)
        pdf.cell(0, 8, label, new_x="LMARGIN", new_y="NEXT")

        # Sub-table header
        cw_name = 52
        cw_uni = 52
        cw_fit = 25
        cw_prereq = 30

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(*_HEADER_BLUE)
        pdf.set_text_color(*_WHITE)
        pdf.cell(cw_name, 7, "  Program", fill=True)
        pdf.cell(cw_uni, 7, "  University", fill=True)
        pdf.cell(cw_fit, 7, "Fit Score", fill=True, align="C")
        pdf.cell(
            cw_prereq,
            7,
            "Prereq Match",
            new_x="LMARGIN",
            new_y="NEXT",
            fill=True,
            align="C",
        )

        pdf.set_text_color(*_BLACK)

        for j, entry in enumerate(entries):
            name = entry.get("name", "N/A")
            university = entry.get("university", "N/A")
            fit_score = entry.get("fit_score", 0.0)
            prereq = entry.get("prereq_match_score", 0.0)
            prereq_str = f"{prereq:.0%}" if isinstance(prereq, float) else str(prereq)

            if j % 2 == 0:
                pdf.set_fill_color(*_ROW_ALT)
            else:
                pdf.set_fill_color(*_WHITE)

            pdf.set_font("Helvetica", "", 9)
            pdf.cell(cw_name, 7, _latin1_safe(f"  {name}"), fill=True)
            pdf.cell(cw_uni, 7, _latin1_safe(f"  {university}"), fill=True)
            pdf.cell(cw_fit, 7, f"{fit_score:.1f}", fill=True, align="C")
            pdf.cell(
                cw_prereq,
                7,
                prereq_str,
                new_x="LMARGIN",
                new_y="NEXT",
                fill=True,
                align="C",
            )

        pdf.ln(2)

    # If no schools at all
    has_schools = any(rankings.get(k) for k in ("reach", "target", "safety"))
    if not has_schools:
        pdf.set_font("Helvetica", "I", 11)
        pdf.set_text_color(*_DARK_GREY)
        pdf.cell(
            0,
            10,
            "No school recommendations available. Load program data to see rankings.",
            new_x="LMARGIN",
            new_y="NEXT",
        )

    # ------------------------------------------------------------------
    # 6. Footer on every page
    # ------------------------------------------------------------------
    _add_footers(pdf)

    # ------------------------------------------------------------------
    # Write file
    # ------------------------------------------------------------------
    pdf.output(output_path)
    return output_path


# ===================================================================
# Helpers
# ===================================================================


def _section_header(pdf: Any, title: str) -> None:
    """Render a coloured section header."""
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(*_SECTION_BLUE)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")

    # Underline
    y = pdf.get_y()
    pdf.set_draw_color(*_SECTION_BLUE)
    pdf.set_line_width(0.4)
    pdf.line(15, y, pdf.w - 15, y)
    pdf.ln(4)

    pdf.set_text_color(*_BLACK)


def _text_bar(score: float, width: int = 20) -> str:
    """Render a score (0-10) as a text-based bar using block chars."""
    filled = round(score * width / 10)
    filled = max(0, min(width, filled))
    return "#" * filled + "-" * (width - filled)


def _set_priority_color(pdf: Any, priority: str) -> None:
    """Set the text colour for a priority level."""
    if priority == "High":
        pdf.set_text_color(*_RED)
    elif priority == "Medium":
        pdf.set_text_color(*_ORANGE)
    else:
        pdf.set_text_color(*_BLUE)


def _add_footers(pdf: Any) -> None:
    """Add footer text and page numbers to every page."""
    total = pdf.pages_count
    for page_num in range(1, total + 1):
        pdf.page = page_num
        pdf.set_y(-15)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*_DARK_GREY)
        pdf.cell(
            0,
            5,
            "Generated by QuantPath -- Open-source MFE Application Toolkit",
            align="L",
        )
        pdf.set_y(-15)
        pdf.cell(0, 5, f"Page {page_num} / {total}", align="R")
