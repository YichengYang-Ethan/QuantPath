"""Data models for the QuantPath evaluation engine.

All domain objects are plain dataclasses so they serialise cleanly and
carry no framework dependencies beyond the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Course category taxonomy
# ---------------------------------------------------------------------------


class CourseCategory(str, Enum):
    """Canonical taxonomy of course categories recognised by the scorer.

    Organised by discipline, aligned with MFE program prerequisites.
    """

    # ── Mathematics ──────────────────────────────────────────────────
    CALCULUS = "calculus"  # Calc I-III, Multivariable Calculus
    LINEAR_ALGEBRA = "linear_algebra"  # Linear Algebra, Matrix Theory
    PROBABILITY = "probability"  # Probability Theory, Calc-based Prob
    ODE = "ode"  # Ordinary Differential Equations
    PDE = "pde"  # Partial Differential Equations
    REAL_ANALYSIS = "real_analysis"  # Real Analysis, Measure Theory
    NUMERICAL_ANALYSIS = "numerical_analysis"  # Numerical Methods
    STOCHASTIC_PROCESSES = "stochastic_processes"  # Stochastic Processes
    STOCHASTIC_CALCULUS = "stochastic_calculus"  # Itô Calculus, SDEs
    OPTIMIZATION = "optimization"  # Convex/Linear/Nonlinear Optimization

    # ── Statistics & Data Science ────────────────────────────────────
    STATISTICS = "statistics"  # Mathematical Statistics, Inference
    REGRESSION = "regression"  # Regression Analysis, Applied Stats
    ECONOMETRICS = "econometrics"  # Econometrics
    TIME_SERIES = "time_series"  # Time Series Analysis
    STAT_COMPUTING = "stat_computing"  # Statistical Computing (R/Python)
    STAT_LEARNING = "stat_learning"  # Statistical Learning, ML/Stats
    BAYESIAN = "bayesian"  # Bayesian Statistics

    # ── Computer Science ─────────────────────────────────────────────
    PROGRAMMING_CPP = "programming_cpp"  # C/C++ Programming
    PROGRAMMING_PYTHON = "programming_python"  # Python Programming
    PROGRAMMING_R = "programming_r"  # R Programming
    DATA_STRUCTURES = "data_structures"  # Data Structures
    ALGORITHMS = "algorithms"  # Algorithms, Algorithm Design
    MACHINE_LEARNING = "machine_learning"  # Machine Learning, Deep Learning
    DATABASE = "database"  # Database Systems, SQL
    SOFTWARE_ENGINEERING = "software_engineering"  # OOP, Design Patterns

    # ── Finance & Economics ──────────────────────────────────────────
    FINANCE = "finance"  # Corporate Finance, Investments
    DERIVATIVES = "derivatives"  # Derivatives, Options Pricing
    FIXED_INCOME = "fixed_income"  # Fixed Income Securities
    PORTFOLIO_THEORY = "portfolio_theory"  # Portfolio Theory, Asset Pricing
    MICROECONOMICS = "microeconomics"  # Micro, Intermediate Micro
    MACROECONOMICS = "macroeconomics"  # Macro, Intermediate Macro
    GAME_THEORY = "game_theory"  # Game Theory
    RISK_MANAGEMENT = "risk_management"  # Risk Management
    FINANCIAL_ECONOMETRICS = "financial_econometrics"  # Financial Econometrics
    ACCOUNTING = "accounting"  # Financial Accounting

    OTHER = "other"


# ---------------------------------------------------------------------------
# Course
# ---------------------------------------------------------------------------


@dataclass
class Course:
    """A single course on the user's transcript."""

    name: str
    code: str
    category: str  # value from CourseCategory (kept as str for YAML compat)
    grade: str  # letter (A+/A/A-/B+/B/B-/C+/C) or numeric string
    credits: float = 3.0
    level: int = 300  # 100/200/300/400/500
    university: str = ""


# ---------------------------------------------------------------------------
# User profile
# ---------------------------------------------------------------------------


@dataclass
class TestScores:
    """Standardised test scores."""

    gre_quant: Optional[int] = None
    gre_verbal: Optional[int] = None
    toefl: Optional[int] = None


@dataclass
class UserProfile:
    """Everything we know about an applicant."""

    # Personal
    name: str = ""

    # Academic record
    coursework: list[Course] = field(default_factory=list)
    gpa: float = 0.0
    gpa_quant: float = 0.0
    university: str = ""
    majors: list[str] = field(default_factory=list)

    # Tests
    test_scores: TestScores = field(default_factory=TestScores)

    # Experience
    work_experience: list[dict[str, Any]] = field(default_factory=list)
    projects: list[dict[str, Any]] = field(default_factory=list)

    # International status
    is_international: bool = False
    years_at_us_institution: int = 0

    def to_dict(self) -> dict:
        """Serialise back to the YAML-compatible dict format.

        The returned dict mirrors the structure expected by
        ``data_loader.load_profile`` and the example YAML in
        ``examples/sample_profile.yaml``.
        """
        personal: dict[str, Any] = {
            "name": self.name,
            "university": self.university,
            "majors": list(self.majors),
            "gpa": self.gpa,
            "is_international": self.is_international,
            "years_at_us_institution": self.years_at_us_institution,
        }

        test_scores: dict[str, Any] = {}
        if self.test_scores.gre_quant is not None:
            test_scores["gre_quant"] = self.test_scores.gre_quant
        if self.test_scores.gre_verbal is not None:
            test_scores["gre_verbal"] = self.test_scores.gre_verbal
        if self.test_scores.toefl is not None:
            test_scores["toefl_ibt"] = self.test_scores.toefl

        courses: list[dict[str, Any]] = []
        for c in self.coursework:
            course_dict: dict[str, Any] = {
                "code": c.code,
                "name": c.name,
                "category": c.category,
                "grade": c.grade,
                "level": c.level,
                "credits": c.credits,
            }
            if c.university:
                course_dict["university"] = c.university
            courses.append(course_dict)

        experience: list[dict[str, Any]] = [dict(e) for e in self.work_experience]
        projects: list[dict[str, Any]] = [dict(p) for p in self.projects]

        result: dict[str, Any] = {"personal": personal}
        if test_scores:
            result["test_scores"] = test_scores
        result["courses"] = courses
        result["experience"] = experience
        result["projects"] = projects
        return result


# ---------------------------------------------------------------------------
# Program data (loaded from YAML)
# ---------------------------------------------------------------------------


@dataclass
class Prerequisite:
    """A single prerequisite entry from a program YAML."""

    category: str
    level: str = ""
    min_grade: Optional[str] = None
    note: str = ""


@dataclass
class DeadlineRound:
    """A single application round with its deadline and decision date."""

    round: int
    date: str
    decision_by: str = ""


@dataclass
class ProgramData:
    """Representation of a graduate programme loaded from YAML.

    Fields mirror the YAML schema used in ``data/programs/*.yaml``.
    """

    id: str = ""
    name: str = ""
    full_name: str = ""
    university: str = ""
    department: str = ""
    website: str = ""
    quantnet_ranking: Optional[int] = None

    # Admissions
    class_size: int = 0
    acceptance_rate: float = 0.0
    avg_gpa: float = 0.0
    gre_quant_avg: Optional[int] = None
    international_pct: Optional[float] = None

    # Prerequisites
    prerequisites_required: list[Prerequisite] = field(default_factory=list)
    prerequisites_recommended: list[Prerequisite] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)

    # Tests
    gre_required: bool = False
    gre_accepts_gmat: bool = False
    gre_exemption: Optional[str] = None
    gre_code: Optional[str] = None
    toefl_waiver_conditions: list[str] = field(default_factory=list)
    toefl_min_ibt: Optional[int] = None
    toefl_min_ielts: Optional[float] = None

    # Deadlines
    deadline_cycle: str = ""
    deadline_note: str = ""
    deadline_rounds: list[DeadlineRound] = field(default_factory=list)

    # Application details
    application_fee: int = 0
    essays: list[dict[str, Any]] = field(default_factory=list)
    video: Optional[dict[str, Any]] = None
    recommendations: int = 0
    resume_max_pages: Optional[int] = None
    interview_type: str = ""
    interview_format: str = ""

    # Outcomes (from QuantNet ranking data)
    avg_base_salary: Optional[int] = None
    employment_rate_3m: Optional[float] = None
    tuition_total: Optional[int] = None

    # Extras
    special: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation outputs
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Full output of the 5-dimension profile evaluation."""

    dimension_scores: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    gaps: list[dict[str, Any]] = field(default_factory=list)
    strengths: list[dict[str, Any]] = field(default_factory=list)
    school_recommendations: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {"reach": [], "target": [], "safety": []}
    )


@dataclass
class PrerequisiteMatch:
    """Result of matching a user profile against a single programme."""

    program_id: str
    matched: list[dict[str, Any]] = field(default_factory=list)
    missing: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    match_score: float = 0.0
