"""Five-dimension profile evaluation engine.

Scores an applicant's academic profile across five weighted dimensions
(math, statistics, computer science, finance/economics, GPA) and
produces gap/strength analysis.

Dimension weights:
    math            0.30
    statistics      0.20
    cs              0.20
    finance_econ    0.15
    gpa             0.15
"""

from __future__ import annotations

from typing import Any

from .models import Course, EvaluationResult, UserProfile


# ===================================================================
# Grade conversion
# ===================================================================

_LETTER_TO_SCORE: dict[str, float] = {
    "A+": 10.0,
    "A":  10.0,
    "A-": 9.0,
    "B+": 8.0,
    "B":  7.0,
    "B-": 6.0,
    "C+": 5.0,
    "C":  4.0,
}


def grade_to_score(grade: str) -> float:
    """Convert a letter or numeric grade string to a 0-10 score.

    Letter grades
    -------------
    A+/A -> 10, A- -> 9, B+ -> 8, B -> 7, B- -> 6, C+ -> 5, C -> 4.

    Numeric grades
    --------------
    90+ -> 10, 85-89 -> 9, 80-84 -> 8, 76-79 -> 7, 70-75 -> 6,
    60-69 -> 5, <60 -> 4.

    Returns 0.0 if the grade string is empty or unrecognised.
    """
    grade = grade.strip()
    if not grade or grade.lower() == "none":
        return 0.0

    # Transfer credit — assume competent baseline
    if grade.upper() == "CR":
        return 8.0

    # Try letter lookup first.
    if grade in _LETTER_TO_SCORE:
        return _LETTER_TO_SCORE[grade]

    # Try numeric conversion.
    try:
        num = float(grade)
    except ValueError:
        return 0.0

    if num >= 90:
        return 10.0
    elif num >= 85:
        return 9.0
    elif num >= 80:
        return 8.0
    elif num >= 76:
        return 7.0
    elif num >= 70:
        return 6.0
    elif num >= 60:
        return 5.0
    else:
        return 4.0


# ===================================================================
# Factor helpers
# ===================================================================

def _best_score_for_categories(
    courses: list[Course],
    categories: set[str],
) -> float:
    """Return the highest grade score among courses whose category is in
    *categories*.  Returns 0.0 when no matching course exists (a gap).
    """
    scores = [
        grade_to_score(c.grade)
        for c in courses
        if c.category in categories
    ]
    return max(scores) if scores else 0.0


def _best_score_for_category(courses: list[Course], category: str) -> float:
    """Shortcut when only one category string is needed."""
    return _best_score_for_categories(courses, {category})


def _count_courses(
    courses: list[Course],
    categories: set[str] | None = None,
    min_level: int = 0,
) -> int:
    """Count courses matching optional category set and minimum level."""
    return sum(
        1
        for c in courses
        if (categories is None or c.category in categories)
        and c.level >= min_level
    )


def _has_major(profile: UserProfile, keywords: set[str]) -> bool:
    """Check whether any of the user's majors contain one of *keywords*
    (case-insensitive).
    """
    for major in profile.majors:
        lower = major.lower()
        if any(kw in lower for kw in keywords):
            return True
    return False


# ===================================================================
# Dimension scorers
# ===================================================================

# Each scorer returns (dimension_score, factor_details) where
# factor_details is a list of dicts: {factor, score, weight, courses}.

def _score_math(profile: UserProfile) -> tuple[float, list[dict[str, Any]]]:
    """Math dimension (weight 0.30).

    Factors & weights:
        calculus_series        0.20
        linear_algebra         0.20
        probability            0.15
        ode_pde                0.10
        real_analysis          0.15
        numerical_analysis     0.10
        stochastic_processes   0.10
    """
    cw = profile.coursework
    factors = [
        ("calculus_series",      0.20, _best_score_for_category(cw, "calculus")),
        ("linear_algebra",       0.20, _best_score_for_category(cw, "linear_algebra")),
        ("probability",          0.15, _best_score_for_category(cw, "probability")),
        ("ode_pde",              0.10, _best_score_for_categories(cw, {"ode", "pde"})),
        ("real_analysis",        0.15, _best_score_for_category(cw, "real_analysis")),
        ("numerical_analysis",   0.10, _best_score_for_category(cw, "numerical_analysis")),
        ("stochastic_processes", 0.10, _best_score_for_category(cw, "stochastic_processes")),
    ]
    return _weighted_result(factors)


def _score_statistics(profile: UserProfile) -> tuple[float, list[dict[str, Any]]]:
    """Statistics dimension (weight 0.20).

    Factors & weights:
        math_stats                  0.25
        time_series                 0.20
        econometrics                0.15
        stat_learning_ml            0.15
        stat_computing              0.10
        courses_400_level_count     0.15
    """
    cw = profile.coursework

    # For the 400-level count factor, we normalise: 0 courses -> 0,
    # 1 -> 5, 2 -> 7, 3+ -> 10.
    stat_categories = {
        "statistics", "econometrics", "time_series",
        "stat_learning", "stat_computing",
    }
    n400 = _count_courses(cw, stat_categories, min_level=400)
    count_score = min(10.0, {0: 0.0, 1: 5.0, 2: 7.0}.get(n400, 10.0))

    factors = [
        ("math_stats",              0.25, _best_score_for_category(cw, "statistics")),
        ("time_series",             0.20, _best_score_for_category(cw, "time_series")),
        ("econometrics",            0.15, _best_score_for_category(cw, "econometrics")),
        ("stat_learning_ml",        0.15, _best_score_for_categories(cw, {"stat_learning", "machine_learning"})),
        ("stat_computing",          0.10, _best_score_for_category(cw, "stat_computing")),
        ("courses_400_level_count", 0.15, count_score),
    ]
    return _weighted_result(factors)


def _score_cs(profile: UserProfile) -> tuple[float, list[dict[str, Any]]]:
    """CS dimension (weight 0.20).

    Factors & weights:
        cpp_proficiency       0.30
        python_proficiency    0.20
        data_structures_algo  0.20
        ml_course             0.10
        numerical_computing   0.10
        is_cs_major           0.10   (bonus: 10 if CS major, else 0)
    """
    cw = profile.coursework

    cs_major_score = 10.0 if _has_major(profile, {"computer science", "computing"}) else 0.0

    factors = [
        ("cpp_proficiency",      0.30, _best_score_for_category(cw, "programming_cpp")),
        ("python_proficiency",   0.20, _best_score_for_category(cw, "programming_python")),
        ("data_structures_algo", 0.20, _best_score_for_categories(cw, {"data_structures", "algorithms"})),
        ("ml_course",            0.10, _best_score_for_category(cw, "machine_learning")),
        ("numerical_computing",  0.10, _best_score_for_category(cw, "numerical_analysis")),
        ("is_cs_major",          0.10, cs_major_score),
    ]
    return _weighted_result(factors)


def _score_finance_econ(profile: UserProfile) -> tuple[float, list[dict[str, Any]]]:
    """Finance / Economics dimension (weight 0.15).

    Factors & weights:
        micro_macro              0.20
        investments_finance      0.25
        derivatives              0.15
        risk_management          0.15
        financial_econometrics   0.15
        game_theory              0.10
    """
    cw = profile.coursework
    factors = [
        ("micro_macro",            0.20, _best_score_for_categories(cw, {"microeconomics", "macroeconomics"})),
        ("investments_finance",    0.25, _best_score_for_category(cw, "finance")),
        ("derivatives",            0.15, _best_score_for_categories(cw, {"finance", "risk_management"})),
        ("risk_management",        0.15, _best_score_for_category(cw, "risk_management")),
        ("financial_econometrics", 0.15, _best_score_for_category(cw, "financial_econometrics")),
        ("game_theory",            0.10, _best_score_for_category(cw, "game_theory")),
    ]
    return _weighted_result(factors)


def _score_gpa(profile: UserProfile) -> tuple[float, list[dict[str, Any]]]:
    """GPA dimension (weight 0.15).

    Factors & weights:
        cumulative_gpa    0.50
        quant_gpa         0.30
        trend             0.20

    GPA values are mapped to a 0-10 scale:
        4.0 -> 10, 3.9 -> 9.5, 3.8 -> 9, 3.7 -> 8.5, 3.5 -> 7.5,
        3.3 -> 6.5, 3.0 -> 5, <3.0 -> linear down to 0.
    Trend is estimated from course levels; not computable from
    transcript alone, so we default to 7 (neutral).
    """
    cum_score = _gpa_to_score(profile.gpa)
    quant_score = _gpa_to_score(profile.gpa_quant) if profile.gpa_quant else cum_score

    # Trend estimation: a simple proxy is to check whether the user has
    # many 400/500-level courses with good grades (improving trajectory).
    upper_courses = [
        c for c in profile.coursework if c.level >= 400
    ]
    if upper_courses:
        avg_upper = sum(grade_to_score(c.grade) for c in upper_courses) / len(upper_courses)
        trend_score = min(10.0, avg_upper)
    else:
        trend_score = 7.0  # neutral default

    factors = [
        ("cumulative_gpa", 0.50, cum_score),
        ("quant_gpa",      0.30, quant_score),
        ("trend",          0.20, trend_score),
    ]
    return _weighted_result(factors)


def _gpa_to_score(gpa: float) -> float:
    """Map a 0.0-4.0 GPA to a 0-10 scale.

    Piecewise linear:
        4.0  -> 10.0
        3.9  -> 9.5
        3.8  -> 9.0
        3.7  -> 8.5
        3.5  -> 7.5
        3.3  -> 6.5
        3.0  -> 5.0
        2.5  -> 3.0
        2.0  -> 1.0
        <2.0 -> 0.0
    """
    if gpa <= 0:
        return 0.0
    breakpoints = [
        (4.0, 10.0),
        (3.9, 9.5),
        (3.8, 9.0),
        (3.7, 8.5),
        (3.5, 7.5),
        (3.3, 6.5),
        (3.0, 5.0),
        (2.5, 3.0),
        (2.0, 1.0),
    ]
    if gpa >= breakpoints[0][0]:
        return breakpoints[0][1]
    if gpa <= breakpoints[-1][0]:
        return max(0.0, breakpoints[-1][1])

    for i in range(len(breakpoints) - 1):
        high_gpa, high_score = breakpoints[i]
        low_gpa, low_score = breakpoints[i + 1]
        if low_gpa <= gpa <= high_gpa:
            # Linear interpolation within this segment.
            frac = (gpa - low_gpa) / (high_gpa - low_gpa)
            return low_score + frac * (high_score - low_score)

    return 0.0


# ===================================================================
# Weighted-average helper
# ===================================================================

def _weighted_result(
    factors: list[tuple[str, float, float]],
) -> tuple[float, list[dict[str, Any]]]:
    """Compute the weighted average of *(name, weight, score)* triples.

    Returns the dimension score and the per-factor detail dicts.
    """
    total = sum(w * s for _, w, s in factors)
    details = [
        {"factor": name, "weight": weight, "score": score}
        for name, weight, score in factors
    ]
    return total, details


# ===================================================================
# Public API
# ===================================================================

# Dimension weights (must sum to 1.0).
DIMENSION_WEIGHTS: dict[str, float] = {
    "math":         0.30,
    "statistics":   0.20,
    "cs":           0.20,
    "finance_econ": 0.15,
    "gpa":          0.15,
}

# Thresholds
_GAP_THRESHOLD = 6.0    # factors below this are flagged as gaps
_STRENGTH_THRESHOLD = 9.0  # factors at or above this are strengths


def evaluate(profile: UserProfile) -> EvaluationResult:
    """Run the full 5-dimension evaluation on *profile*.

    Steps:
        1. Score every factor within each dimension.
        2. Compute each dimension's weighted score.
        3. Compute the overall score (weighted across dimensions).
        4. Identify gaps (score 0 or below 6).
        5. Identify strengths (score >= 9).

    Parameters
    ----------
    profile:
        A fully-populated :class:`UserProfile`.

    Returns
    -------
    EvaluationResult
        Contains ``dimension_scores``, ``overall_score``, ``gaps``,
        ``strengths``, and an empty ``school_recommendations`` dict
        (filled later by the school ranker).
    """
    scorers: dict[str, Any] = {
        "math":         _score_math,
        "statistics":   _score_statistics,
        "cs":           _score_cs,
        "finance_econ": _score_finance_econ,
        "gpa":          _score_gpa,
    }

    dimension_scores: dict[str, float] = {}
    all_factors: dict[str, list[dict[str, Any]]] = {}
    gaps: list[dict[str, Any]] = []
    strengths: list[dict[str, Any]] = []

    for dim_name, scorer_fn in scorers.items():
        dim_score, factor_details = scorer_fn(profile)
        dimension_scores[dim_name] = round(dim_score, 2)
        all_factors[dim_name] = factor_details

        # Collect gaps and strengths from this dimension.
        for fd in factor_details:
            entry = {
                "dimension": dim_name,
                "factor": fd["factor"],
                "score": fd["score"],
            }
            if fd["score"] == 0 or fd["score"] < _GAP_THRESHOLD:
                gaps.append(entry)
            if fd["score"] >= _STRENGTH_THRESHOLD:
                strengths.append(entry)

    # Overall score.
    overall = sum(
        DIMENSION_WEIGHTS[dim] * dimension_scores[dim]
        for dim in DIMENSION_WEIGHTS
    )

    return EvaluationResult(
        dimension_scores=dimension_scores,
        overall_score=round(overall, 2),
        gaps=sorted(gaps, key=lambda g: g["score"]),
        strengths=sorted(strengths, key=lambda s: -s["score"]),
    )
