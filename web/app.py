"""QuantPath Streamlit Dashboard.

Run with:
    streamlit run web/app.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Ensure the package root is importable regardless of working directory.
# ---------------------------------------------------------------------------
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from core.data_loader import load_all_programs, load_profile  # noqa: E402
from core.gap_advisor import analyze_gaps  # noqa: E402
from core.models import CourseCategory  # noqa: E402
from core.profile_evaluator import evaluate as evaluate_profile  # noqa: E402
from core.school_ranker import rank_schools  # noqa: E402
from core.timeline_generator import generate_timeline  # noqa: E402

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="QuantPath",
    page_icon="\U0001f4ca",  # bar chart emoji
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar branding and navigation
# ---------------------------------------------------------------------------
SAMPLE_PROFILE_PATH = _PACKAGE_ROOT / "examples" / "sample_profile.yaml"

PAGES = [
    "Profile Builder",
    "Profile Evaluation",
    "Program Explorer",
    "Program Comparison",
    "Gap Analysis",
    "Application Timeline",
]

with st.sidebar:
    st.markdown(
        "<h1 style='text-align:center;margin-bottom:0'>QuantPath</h1>"
        "<p style='text-align:center;color:gray;margin-top:0'>"
        "MFE Application Toolkit</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio("Navigation", PAGES, index=0, label_visibility="collapsed")
    st.divider()
    st.caption("Built with Streamlit + Plotly")


# ---------------------------------------------------------------------------
# Helpers: profile loading
# ---------------------------------------------------------------------------

def _load_profile_from_bytes(raw_bytes: bytes) -> None:
    """Write uploaded YAML bytes to a temp file and load into session state."""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".yaml", delete=False, mode="wb",
    )
    tmp.write(raw_bytes)
    tmp.flush()
    tmp.close()
    try:
        profile = load_profile(tmp.name)
        st.session_state["profile"] = profile
        st.session_state["profile_path"] = tmp.name
    except Exception as exc:
        st.error(f"Failed to load profile: {exc}")


def _load_sample_profile() -> None:
    """Load the bundled sample profile."""
    try:
        profile = load_profile(str(SAMPLE_PROFILE_PATH))
        st.session_state["profile"] = profile
        st.session_state["profile_path"] = str(SAMPLE_PROFILE_PATH)
    except Exception as exc:
        st.error(f"Failed to load sample profile: {exc}")


def _get_profile():
    """Return the currently loaded profile or None."""
    return st.session_state.get("profile")


def _require_profile():
    """Show a warning if no profile is loaded and return the profile or None."""
    profile = _get_profile()
    if profile is None:
        st.warning(
            "No profile loaded. Please go to **Profile Evaluation** and "
            "upload a YAML profile or click **Use Sample Profile** first."
        )
    return profile


# ---------------------------------------------------------------------------
# Helpers: data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading programs...")
def _load_programs():
    return load_all_programs()


# ---------------------------------------------------------------------------
# Helpers: formatting
# ---------------------------------------------------------------------------

_DIM_LABELS = {
    "math": "Math",
    "statistics": "Statistics",
    "cs": "Computer Science",
    "finance_econ": "Finance / Econ",
    "gpa": "GPA",
}

_PRIORITY_COLORS = {
    "critical": "#e74c3c",
    "high": "#e67e22",
    "medium": "#f1c40f",
    "low": "#3498db",
    "High": "#e74c3c",
    "Medium": "#e67e22",
    "Low": "#3498db",
}


# ===================================================================
# Page 0: Profile Builder
# ===================================================================

# Category groupings by subject area (values match CourseCategory enum)
_CATEGORY_GROUPS: dict[str, list[str]] = {
    "Math": [
        "calculus", "linear_algebra", "probability", "ode", "pde",
        "real_analysis", "numerical_analysis", "stochastic_processes",
        "stochastic_calculus", "optimization",
    ],
    "Statistics": [
        "statistics", "regression", "econometrics", "time_series",
        "stat_computing", "stat_learning", "bayesian",
    ],
    "Computer Science": [
        "programming_cpp", "programming_python", "programming_r",
        "data_structures", "algorithms", "machine_learning",
        "database", "software_engineering",
    ],
    "Finance & Economics": [
        "finance", "derivatives", "fixed_income", "portfolio_theory",
        "microeconomics", "macroeconomics", "game_theory",
        "risk_management", "financial_econometrics", "accounting",
    ],
}


def page_profile_builder() -> None:
    from core.models import Course, TestScores, UserProfile

    st.header("Profile Builder")
    st.caption(
        "Build your applicant profile interactively. "
        "When finished, click **Build Profile** to load it into the evaluation engine, "
        "or **Download YAML** to save it to a file."
    )

    # ── 1. Personal Information ───────────────────────────────────────
    with st.expander("Personal Information", expanded=True):
        pb_name = st.text_input("Name", key="pb_name")
        pb_university = st.text_input("University", key="pb_university")
        pb_majors_raw = st.text_input(
            "Majors (comma-separated)",
            key="pb_majors",
            placeholder="e.g. Applied Mathematics, Economics",
        )
        pb_gpa = st.number_input(
            "GPA", min_value=0.0, max_value=4.0, value=0.0, step=0.01, key="pb_gpa",
        )
        pb_international = st.checkbox("International student", key="pb_international")
        pb_years_us = 0
        if pb_international:
            pb_years_us = st.number_input(
                "Years at US institution",
                min_value=0, max_value=10, value=0, step=1, key="pb_years_us",
            )

    # ── 2. Test Scores ────────────────────────────────────────────────
    with st.expander("Test Scores"):
        pb_gre_q = st.number_input(
            "GRE Quant (leave 0 if not taken)",
            min_value=0, max_value=170, value=0, step=1, key="pb_gre_q",
        )
        pb_gre_v = st.number_input(
            "GRE Verbal (leave 0 if not taken)",
            min_value=0, max_value=170, value=0, step=1, key="pb_gre_v",
        )
        pb_toefl = st.number_input(
            "TOEFL iBT (leave 0 if not taken)",
            min_value=0, max_value=120, value=0, step=1, key="pb_toefl",
        )

    # ── 3. Coursework ─────────────────────────────────────────────────
    with st.expander("Coursework"):
        # Subject area helper
        subject_area = st.selectbox(
            "Subject Area (reference guide)",
            options=["-- select --"] + list(_CATEGORY_GROUPS.keys()),
            key="pb_subject_area",
        )
        if subject_area != "-- select --":
            cats = _CATEGORY_GROUPS[subject_area]
            st.caption(
                f"Categories for **{subject_area}**: "
                + ", ".join(c.replace("_", " ").title() for c in cats)
            )

        all_category_values = [e.value for e in CourseCategory]
        grade_options = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C"]
        level_options = [100, 200, 300, 400, 500]

        course_data = st.data_editor(
            [
                {
                    "name": "", "code": "", "category": "other",
                    "grade": "A", "level": 300, "credits": 3.0,
                }
                for _ in range(3)
            ],
            column_config={
                "name": st.column_config.TextColumn("Name"),
                "code": st.column_config.TextColumn("Code"),
                "category": st.column_config.SelectboxColumn(
                    "Category", options=all_category_values,
                ),
                "grade": st.column_config.SelectboxColumn(
                    "Grade", options=grade_options,
                ),
                "level": st.column_config.SelectboxColumn(
                    "Level", options=level_options,
                ),
                "credits": st.column_config.NumberColumn(
                    "Credits", min_value=0.0, max_value=6.0, step=0.5, default=3.0,
                ),
            },
            num_rows="dynamic",
            use_container_width=True,
            key="pb_courses",
        )

    # ── 4. Experience & Projects ──────────────────────────────────────
    with st.expander("Experience & Projects"):
        st.markdown("**Work Experience**")
        experience_data = st.data_editor(
            [
                {"type": "", "title": "", "company": "", "description": "", "duration_months": 0}
            ],
            column_config={
                "type": st.column_config.TextColumn("Type"),
                "title": st.column_config.TextColumn("Title"),
                "company": st.column_config.TextColumn("Company"),
                "description": st.column_config.TextColumn("Description"),
                "duration_months": st.column_config.NumberColumn(
                    "Duration (months)", min_value=0, step=1,
                ),
            },
            num_rows="dynamic",
            use_container_width=True,
            key="pb_experience",
        )

        st.markdown("**Projects**")
        project_data = st.data_editor(
            [
                {"name": "", "description": ""}
            ],
            column_config={
                "name": st.column_config.TextColumn("Name"),
                "description": st.column_config.TextColumn("Description"),
            },
            num_rows="dynamic",
            use_container_width=True,
            key="pb_projects",
        )

    # ── Action buttons ────────────────────────────────────────────────
    st.divider()
    col_build, col_download = st.columns(2)

    with col_build:
        if st.button("Build Profile", type="primary", use_container_width=True):
            # Parse majors
            majors = [m.strip() for m in pb_majors_raw.split(",") if m.strip()]

            # Parse test scores (treat 0 as not provided)
            test_scores = TestScores(
                gre_quant=pb_gre_q if pb_gre_q >= 130 else None,
                gre_verbal=pb_gre_v if pb_gre_v >= 130 else None,
                toefl=pb_toefl if pb_toefl > 0 else None,
            )

            # Parse courses (skip empty rows)
            courses: list[Course] = []
            for row in course_data:
                if row.get("name", "").strip() or row.get("code", "").strip():
                    courses.append(Course(
                        name=row.get("name", ""),
                        code=row.get("code", ""),
                        category=row.get("category", "other"),
                        grade=row.get("grade", "A"),
                        credits=float(row.get("credits", 3.0)),
                        level=int(row.get("level", 300)),
                    ))

            # Parse experience (skip empty rows)
            experience = [
                {k: v for k, v in row.items()}
                for row in experience_data
                if row.get("title", "").strip()
            ]

            # Parse projects (skip empty rows)
            projects = [
                {k: v for k, v in row.items()}
                for row in project_data
                if row.get("name", "").strip()
            ]

            profile = UserProfile(
                name=pb_name,
                university=pb_university,
                majors=majors,
                gpa=pb_gpa,
                test_scores=test_scores,
                coursework=courses,
                work_experience=experience,
                projects=projects,
                is_international=pb_international,
                years_at_us_institution=pb_years_us,
            )
            st.session_state["profile"] = profile
            st.success(
                f"Profile built for **{profile.name or 'Unnamed'}** with "
                f"{len(courses)} courses, {len(experience)} experiences, "
                f"and {len(projects)} projects. "
                "Navigate to **Profile Evaluation** to see your results."
            )

    with col_download:
        profile = _get_profile()
        if profile is not None:
            yaml_str = yaml.dump(
                profile.to_dict(), default_flow_style=False, sort_keys=False,
            )
            st.download_button(
                label="Download YAML",
                data=yaml_str,
                file_name="quantpath_profile.yaml",
                mime="text/yaml",
                use_container_width=True,
            )
        else:
            st.button(
                "Download YAML",
                disabled=True,
                use_container_width=True,
                help="Build a profile first before downloading.",
            )


# ===================================================================
# Page 1: Profile Evaluation
# ===================================================================

def page_profile_evaluation() -> None:
    st.header("Profile Evaluation")

    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Upload your profile YAML",
            type=["yaml", "yml"],
            key="profile_uploader",
        )
    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Use Sample Profile", use_container_width=True):
            _load_sample_profile()

    if uploaded is not None:
        _load_profile_from_bytes(uploaded.getvalue())

    profile = _get_profile()
    if profile is None:
        st.info("Upload a profile YAML or click **Use Sample Profile** to get started.")
        return

    programs = _load_programs()
    result = evaluate_profile(profile)
    rankings = rank_schools(profile, programs, result)

    # ----- Profile summary card -----
    st.subheader("Profile Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Name", profile.name or "N/A")
    c2.metric("University", profile.university or "N/A")
    c3.metric("GPA", f"{profile.gpa:.2f}")
    c4.metric("Majors", ", ".join(profile.majors) if profile.majors else "N/A")

    st.divider()

    # ----- Radar chart (plotly) -----
    st.subheader("Dimension Scores")

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        dims = list(_DIM_LABELS.keys())
        labels = [_DIM_LABELS[d] for d in dims]
        scores = [result.dimension_scores.get(d, 0) for d in dims]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.25)",
            line=dict(color="rgb(99, 110, 250)", width=2),
            name="Your Profile",
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10], tickvals=[2, 4, 6, 8, 10]),
            ),
            showlegend=False,
            height=380,
            margin=dict(t=30, b=30, l=60, r=60),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        rows = []
        for d in dims:
            score = result.dimension_scores.get(d, 0)
            rows.append({
                "Dimension": _DIM_LABELS[d],
                "Score": f"{score:.2f} / 10",
                "Rating": (
                    "Excellent" if score >= 9
                    else "Good" if score >= 7
                    else "Needs Work" if score >= 5
                    else "Weak"
                ),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # ----- Overall score -----
    st.divider()
    overall = result.overall_score
    level = (
        "Top 3-5 MFE Level"
        if overall >= 9.5
        else "Top 5-10 MFE Level"
        if overall >= 8.5
        else "Competitive"
        if overall >= 7.5
        else "Needs Improvement"
    )
    m1, m2, m3 = st.columns([1, 2, 1])
    with m2:
        st.metric(
            label="Overall Score",
            value=f"{overall:.2f} / 10",
            delta=level,
        )

    # ----- Gaps and strengths -----
    st.divider()
    col_gaps, col_strengths = st.columns(2)

    with col_gaps:
        st.subheader("Gaps")
        if result.gaps:
            for gap in result.gaps:
                factor = gap["factor"].replace("_", " ").title()
                dim = gap["dimension"]
                score = gap["score"]
                if score == 0:
                    st.markdown(f"- **{factor}** ({dim}): :red[Missing]")
                else:
                    st.markdown(f"- **{factor}** ({dim}): :orange[{score:.1f}/10]")
        else:
            st.success("No gaps found!")

    with col_strengths:
        st.subheader("Strengths")
        if result.strengths:
            for s in result.strengths:
                factor = s["factor"].replace("_", " ").title()
                dim = s["dimension"]
                score = s["score"]
                st.markdown(f"- **{factor}** ({dim}): :green[{score:.1f}/10]")
        else:
            st.info("No outstanding strengths detected.")

    # ----- School recommendations -----
    st.divider()
    st.subheader("School Recommendations")

    for category, color, label in [
        ("reach", "red", "Reach"),
        ("target", "orange", "Target"),
        ("safety", "green", "Safety"),
    ]:
        schools = rankings.get(category, [])
        if not schools:
            continue
        with st.expander(f":{color}[{label}] ({len(schools)} programs)", expanded=True):
            for sch in schools:
                c1, c2, c3 = st.columns([3, 1, 1])
                c1.write(f"**{sch['name']}** -- {sch['university']}")
                c2.write(f"Fit: {sch['fit_score']:.1f}/100")
                c3.write(f"Prereq: {sch['prereq_match_score']:.0%}")


# ===================================================================
# Page 2: Program Explorer
# ===================================================================

def page_program_explorer() -> None:
    st.header("Program Explorer")

    programs = _load_programs()

    if not programs:
        st.warning("No programs found in the database.")
        return

    rows = []
    for p in programs:
        rows.append({
            "Name": p.name,
            "University": p.university,
            "Rank": p.quantnet_ranking if p.quantnet_ranking else None,
            "Class Size": p.class_size if p.class_size else None,
            "Acceptance Rate": f"{p.acceptance_rate:.0%}" if p.acceptance_rate else "N/A",
            "Avg Salary": f"${p.avg_base_salary:,}" if p.avg_base_salary else "N/A",
            "Emp 3m": f"{p.employment_rate_3m:.0%}" if p.employment_rate_3m else "N/A",
            "Tuition": f"${p.tuition_total:,}" if p.tuition_total else "N/A",
            "GRE": "Required" if p.gre_required else "Optional",
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Detail expanders
    st.divider()
    st.subheader("Program Details")

    selected_name = st.selectbox(
        "Select a program for details",
        options=[p.name for p in programs],
    )

    if selected_name:
        prog = next(p for p in programs if p.name == selected_name)

        with st.expander(f"{prog.name} -- {prog.university}", expanded=True):
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Class Size", prog.class_size or "N/A")
            d2.metric(
                "Acceptance Rate",
                f"{prog.acceptance_rate:.0%}" if prog.acceptance_rate else "N/A",
            )
            d3.metric(
                "Avg Salary",
                f"${prog.avg_base_salary:,}" if prog.avg_base_salary else "N/A",
            )
            d4.metric(
                "Employment 3m",
                f"{prog.employment_rate_3m:.0%}" if prog.employment_rate_3m else "N/A",
            )

            st.markdown("**Required Prerequisites:**")
            if prog.prerequisites_required:
                for pr in prog.prerequisites_required:
                    grade_note = f" (min grade: {pr.min_grade})" if pr.min_grade else ""
                    st.markdown(f"- {pr.category.replace('_', ' ').title()}{grade_note}")
            else:
                st.write("None listed.")

            st.markdown("**Recommended Prerequisites:**")
            if prog.prerequisites_recommended:
                for pr in prog.prerequisites_recommended:
                    note = f" -- {pr.note}" if pr.note else ""
                    st.markdown(
                        f"- {pr.category.replace('_', ' ').title()}{note}"
                    )
            else:
                st.write("None listed.")

            if prog.languages:
                st.markdown(f"**Languages:** {', '.join(prog.languages)}")

            if prog.deadline_rounds:
                st.markdown("**Deadlines:**")
                for rd in prog.deadline_rounds:
                    decision = f" (decision by {rd.decision_by})" if rd.decision_by else ""
                    st.markdown(f"- Round {rd.round}: {rd.date}{decision}")

            if prog.website:
                st.markdown(f"[Program Website]({prog.website})")


# ===================================================================
# Page 3: Program Comparison
# ===================================================================

def page_program_comparison() -> None:
    st.header("Program Comparison")

    programs = _load_programs()
    if not programs:
        st.warning("No programs found in the database.")
        return

    prog_names = [p.name for p in programs]
    selected = st.multiselect(
        "Select up to 3 programs to compare",
        options=prog_names,
        max_selections=3,
    )

    if len(selected) < 2:
        st.info("Select at least 2 programs (up to 3) to compare.")
        return

    progs_by_name = {p.name: p for p in programs}
    chosen = [progs_by_name[n] for n in selected]

    # Build comparison columns
    header_cols = st.columns([2] + [1] * len(chosen))
    header_cols[0].markdown("**Attribute**")
    for i, prog in enumerate(chosen):
        header_cols[i + 1].markdown(f"**{prog.name}**")

    st.divider()

    def _comparison_row(label: str, values: list[str]) -> None:
        cols = st.columns([2] + [1] * len(values))
        cols[0].write(label)
        for i, v in enumerate(values):
            cols[i + 1].write(v)

    _comparison_row("University", [p.university for p in chosen])

    _comparison_row(
        "Class Size",
        [str(p.class_size) if p.class_size else "N/A" for p in chosen],
    )

    _comparison_row(
        "Acceptance Rate",
        [f"{p.acceptance_rate:.0%}" if p.acceptance_rate else "N/A" for p in chosen],
    )

    _comparison_row(
        "Avg GPA",
        [f"{p.avg_gpa:.2f}" if p.avg_gpa else "N/A" for p in chosen],
    )

    _comparison_row(
        "Tuition",
        [f"${p.tuition_total:,}" if p.tuition_total else "N/A" for p in chosen],
    )

    _comparison_row(
        "Avg Base Salary",
        [f"${p.avg_base_salary:,}" if p.avg_base_salary else "N/A" for p in chosen],
    )

    _comparison_row(
        "Employment (3m)",
        [f"{p.employment_rate_3m:.0%}" if p.employment_rate_3m else "N/A" for p in chosen],
    )

    _comparison_row(
        "GRE Required",
        ["Yes" if p.gre_required else "No" for p in chosen],
    )

    _comparison_row(
        "TOEFL Min (iBT)",
        [str(p.toefl_min_ibt) if p.toefl_min_ibt else "N/A" for p in chosen],
    )

    _comparison_row(
        "Application Fee",
        [f"${p.application_fee}" if p.application_fee else "N/A" for p in chosen],
    )

    _comparison_row(
        "Recommendations",
        [str(p.recommendations) if p.recommendations else "N/A" for p in chosen],
    )

    def _round_date(prog, round_num):
        for r in prog.deadline_rounds:
            if r.round == round_num:
                return r.date
        return "N/A"

    _comparison_row("Deadline Round 1", [_round_date(p, 1) for p in chosen])
    _comparison_row("Deadline Round 2", [_round_date(p, 2) for p in chosen])

    _comparison_row(
        "Prerequisites (req.)",
        [str(len(p.prerequisites_required)) for p in chosen],
    )

    _comparison_row(
        "Interview",
        [
            p.interview_type.replace("_", " ").title() if p.interview_type else "N/A"
            for p in chosen
        ],
    )


# ===================================================================
# Page 4: Gap Analysis
# ===================================================================

def page_gap_analysis() -> None:
    st.header("Gap Analysis")

    profile = _require_profile()
    if profile is None:
        return

    result = evaluate_profile(profile)

    if not result.gaps:
        st.success(
            "No gaps found -- your profile looks strong across all dimensions!"
        )
        return

    recommendations = analyze_gaps(result.gaps)

    # Summary metrics
    high_count = sum(1 for r in recommendations if r.priority == "High")
    med_count = sum(1 for r in recommendations if r.priority == "Medium")
    low_count = sum(1 for r in recommendations if r.priority == "Low")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Gaps", len(recommendations))
    m2.metric("High Priority", high_count)
    m3.metric("Medium Priority", med_count)
    m4.metric("Low Priority", low_count)

    st.divider()

    # Gaps table with color coding
    rows = []
    for rec in recommendations:
        if rec.score == 0:
            score_display = "Missing"
        else:
            score_display = f"{rec.score:.1f} / 10"

        rows.append({
            "Factor": rec.factor.replace("_", " ").title(),
            "Dimension": rec.dimension.replace("_", " ").title(),
            "Score": score_display,
            "Priority": rec.priority,
            "Recommended Action": rec.action,
        })

    # Use custom styling via column_config
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Factor": st.column_config.TextColumn(width="medium"),
            "Dimension": st.column_config.TextColumn(width="small"),
            "Score": st.column_config.TextColumn(width="small"),
            "Priority": st.column_config.TextColumn(width="small"),
            "Recommended Action": st.column_config.TextColumn(width="large"),
        },
    )

    # Detailed cards by priority
    st.divider()
    st.subheader("Detailed Recommendations")

    for priority_label, color in [("High", "red"), ("Medium", "orange"), ("Low", "blue")]:
        prio_recs = [r for r in recommendations if r.priority == priority_label]
        if not prio_recs:
            continue
        with st.expander(
            f":{color}[{priority_label} Priority] ({len(prio_recs)} items)",
            expanded=(priority_label == "High"),
        ):
            for rec in prio_recs:
                factor_label = rec.factor.replace("_", " ").title()
                score_str = "Missing" if rec.score == 0 else f"{rec.score:.1f}/10"
                st.markdown(
                    f"**{factor_label}** ({rec.dimension}) -- Score: {score_str}"
                )
                st.markdown(f"> {rec.action}")
                st.markdown("---")


# ===================================================================
# Page 5: Application Timeline
# ===================================================================

def page_application_timeline() -> None:
    st.header("Application Timeline")

    programs = _load_programs()
    if not programs:
        st.warning("No programs found in the database.")
        return

    events = generate_timeline(programs, date.today())

    if not events:
        st.info("No timeline events generated. Programs may not have deadline data.")
        return

    # Group events by month
    months: dict[str, list[dict]] = {}
    for event in events:
        d = date.fromisoformat(event["date"])
        month_key = d.strftime("%Y-%m")
        month_label = d.strftime("%B %Y")
        if month_key not in months:
            months[month_key] = []
        event_copy = dict(event)
        event_copy["_date_obj"] = d
        event_copy["_month_label"] = month_label
        months[month_key].append(event_copy)

    priority_icons = {
        "critical": ":red_circle:",
        "high": ":orange_circle:",
        "medium": ":large_yellow_circle:",
        "low": ":large_blue_circle:",
    }

    for month_key in sorted(months.keys()):
        month_events = months[month_key]
        month_label = month_events[0]["_month_label"]
        with st.expander(f"{month_label} ({len(month_events)} events)", expanded=True):
            for event in month_events:
                d = event["_date_obj"]
                icon = priority_icons.get(event["priority"], ":white_circle:")
                priority_display = event["priority"].capitalize()
                category_display = event["category"].replace("_", " ").title()

                st.markdown(
                    f"{icon} **{d.strftime('%b %d')}** -- "
                    f"{event['action']}  \n"
                    f"&nbsp;&nbsp;&nbsp;&nbsp;"
                    f"*{category_display}* | {priority_display}"
                )


# ===================================================================
# Page router
# ===================================================================

if page == "Profile Builder":
    page_profile_builder()
elif page == "Profile Evaluation":
    page_profile_evaluation()
elif page == "Program Explorer":
    page_program_explorer()
elif page == "Program Comparison":
    page_program_comparison()
elif page == "Gap Analysis":
    page_gap_analysis()
elif page == "Application Timeline":
    page_application_timeline()
