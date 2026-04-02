#!/usr/bin/env python3
# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""QuantPath CLI — MFE application toolkit."""

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.admission_data import load_admission_csv, load_all_admission_data, summarize_records
from core.calibrator import calibrate_all, generate_ranker_overrides
from core.course_optimizer import optimize_courses
from core.data_loader import load_all_programs, load_profile
from core.gap_advisor import analyze_gaps, program_gaps
from core.interview_prep import (
    get_questions_by_category,
    get_questions_by_difficulty,
    get_questions_for_program,
    get_random_quiz,
    load_questions,
)
from core.list_builder import build_school_list, optimize_portfolio
from core.prerequisite_matcher import match_prerequisites
from core.profile_evaluator import evaluate as evaluate_profile
from core.roi_calculator import calculate_roi
from core.school_ranker import rank_schools
from core.test_requirements import check_gre, check_toefl
from core.timeline_generator import generate_timeline

console = Console()


def bar(score: float, width: int = 10) -> str:
    """Render a score as a bar chart."""
    filled = round(score * width / 10)
    return "█" * filled + "░" * (width - filled)


def _quick_profile_interactive() -> str:
    """Interactive prompt to build a minimal profile YAML. Returns saved file path."""
    import os

    console.print("\n[bold]Quick Profile Builder[/bold]")
    console.print("Answer a few questions to get your admission predictions.\n")

    name = input("  Your name: ").strip() or "Applicant"

    while True:
        gpa_raw = input("  GPA (e.g. 3.85): ").strip()
        if not gpa_raw:
            gpa = ""
            break
        try:
            gpa_val = float(gpa_raw)
            if not 0.0 <= gpa_val <= 4.0:
                console.print("  [yellow]GPA should be between 0.0 and 4.0.[/yellow]")
                continue
            gpa = gpa_raw
            break
        except ValueError:
            console.print("  [yellow]Please enter a number (e.g. 3.85).[/yellow]")

    university = input("  University (e.g. Peking University, UIUC): ").strip()
    majors_raw = input("  Major(s), comma-separated (e.g. Math, CS): ").strip()
    is_intl = input("  International student? (y/n): ").strip().lower()

    while True:
        gre_raw = input("  GRE Quant score (press Enter to skip): ").strip()
        if not gre_raw:
            gre = ""
            break
        try:
            gre_val = int(gre_raw)
            if not 130 <= gre_val <= 170:
                console.print("  [yellow]GRE Quant is 130–170.[/yellow]")
                continue
            gre = gre_raw
            break
        except ValueError:
            console.print("  [yellow]Please enter a number (e.g. 168).[/yellow]")

    console.print("\n  [dim]Internships (press Enter when done):[/dim]")
    internships = []
    while True:
        company = input("    Company name (or Enter to finish): ").strip()
        if not company:
            break
        country = input("    Country (us/china/other) [us]: ").strip().lower() or "us"
        role = (
            input("    Role (quant/finance/tech/other) [quant]: ").strip().lower()
            or "quant"
        )
        internships.append({"company": company, "country": country, "role": role})

    has_paper = input("  Do you have a published paper? (y/n): ").strip().lower()
    has_research = input("  Any research experience? (y/n): ").strip().lower()

    # Build YAML
    lines = ["personal:"]
    lines.append(f'  name: "{name}"')
    if gpa:
        lines.append(f"  gpa: {gpa}")
    if university:
        lines.append(f'  university: "{university}"')
    if majors_raw:
        majors = [m.strip() for m in majors_raw.split(",")]
        lines.append(f"  majors: {majors}")
    lines.append(f"  is_international: {'true' if is_intl == 'y' else 'false'}")

    if gre:
        lines.append("\ntest_scores:")
        lines.append(f"  gre_quant: {gre}")

    if internships:
        lines.append("\nexperience:")
        for intern in internships:
            c = intern["company"]
            country = intern["country"]
            role = intern["role"]
            desc = f"{role} role, {country}"
            lines.append(f'  - {{type: internship, company: "{c}", description: "{desc}"}}')

    if has_paper == "y" or has_research == "y":
        lines.append("\nprojects:")
        if has_paper == "y":
            lines.append('  - {name: "Research", has_paper: true}')
        elif has_research == "y":
            lines.append('  - {name: "Research", description: "Research experience"}')

    lines.append(
        "\n# --- Add courses below for deeper analysis (evaluate/gaps/optimize) ---"
        "\n# See examples/sample_profile.yaml for the full format."
        "\n# courses:"
        '\n#   - {code: "MATH 310", name: "Linear Algebra",'
        " category: linear_algebra, grade: \"A\", level: 300}"
        '\n#   - {code: "STAT 420", name: "Probability",'
        " category: probability, grade: \"A-\", level: 400}"
    )

    yaml_content = "\n".join(lines)

    profiles_dir = os.path.join(os.path.dirname(__file__), "..", "profiles")
    os.makedirs(profiles_dir, exist_ok=True)

    safe_name = name.lower().replace(" ", "_")[:20] or "my_profile"
    profile_path = os.path.join(profiles_dir, f"{safe_name}.yaml")

    if os.path.exists(profile_path):
        profile_path = os.path.join(
            profiles_dir,
            f"{safe_name}_{int(__import__('time').time())}.yaml",
        )

    with open(profile_path, "w") as f:
        f.write(yaml_content)

    abs_path = os.path.abspath(profile_path)
    console.print(f"\n  [green]Profile saved to {abs_path}[/green]")
    console.print(
        "  [dim]You can now run other commands with this profile:[/dim]"
    )
    console.print(
        f"  [dim]  quantpath evaluate --profile {abs_path}[/dim]"
    )
    console.print(
        f"  [dim]  quantpath gaps     --profile {abs_path}[/dim]\n"
    )
    return abs_path


def cmd_predict(args: argparse.Namespace) -> None:
    """Pure v2 model prediction — reach/target/safety without course evaluation."""
    from core.lr_predictor import predict_ensemble

    if args.profile:
        profile = load_profile(args.profile)
    else:
        # Interactive mode
        tmp_path = _quick_profile_interactive()
        profile = load_profile(tmp_path)
    programs = load_all_programs()

    # Focused program list: Tier 0 + Tier 1 only
    _TIER0 = [
        "princeton-mfin", "baruch-mfe", "berkeley-mfe",
        "cmu-mscf", "mit-mfin", "columbia-msfe-econ",
        "yale-am", "stanford-mcf",
    ]
    _TIER1 = [
        "uchicago-msfm", "nyu-courant", "columbia-msfe",
        "cornell-mfe", "columbia-mafn", "nyu-tandon-mfe", "gatech-qcf",
    ]
    _FOCUSED_PROGRAMS = set(_TIER0 + _TIER1)

    gre_quant = None
    if profile.test_scores:
        gre_quant = getattr(profile.test_scores, "gre_quant", None)

    # Check if GPBoost v2 model is available
    _v2_available = True
    try:
        import gpboost  # noqa: F401
    except ImportError:
        _v2_available = False
        console.print(
            "[yellow]⚠ gpboost not installed — using v1 fallback model "
            "(fewer programs, lower accuracy).\n"
            "  Install with: pip install gpboost numpy[/yellow]\n"
        )

    console.print(
        Panel(
            f"{profile.name} | {profile.university} | GPA {profile.gpa}"
            + (" | International" if profile.is_international else ""),
            title="QuantPath Admission Prediction",
            style="bold",
        )
    )

    # Predict for focused programs only (Tier 0 + Tier 1)
    results = []
    skipped = []
    for prog in programs:
        if prog.id not in _FOCUSED_PROGRAMS:
            continue
        pred = predict_ensemble(prog.id, profile.gpa, gre_quant, profile)
        if pred is None:
            skipped.append(prog.university + " " + (prog.full_name or prog.name))
            continue

        prob = pred.prob
        if prob >= 0.70:
            cat = "safety"
        elif prob >= 0.40:
            cat = "target"
        else:
            cat = "reach"

        results.append({
            "program_id": prog.id,
            "name": prog.full_name or prog.name,
            "university": prog.university,
            "prob": prob,
            "prob_low": pred.prob_low,
            "prob_high": pred.prob_high,
            "category": cat,
            "corrected": pred.is_bias_corrected,
            "acceptance_rate": prog.acceptance_rate,
        })

    results.sort(key=lambda x: -x["prob"])

    # Display by category
    for cat, label, style in [
        ("reach", "Reach", "red"),
        ("target", "Target", "yellow"),
        ("safety", "Safety", "green"),
    ]:
        entries = [r for r in results if r["category"] == cat]
        if not entries:
            continue

        table = Table(title=f"{label} Schools", style=style, show_lines=False)
        table.add_column("Program", min_width=20)
        table.add_column("University", min_width=20)
        table.add_column("P(Admit)", justify="right", min_width=12)
        table.add_column("Real Rate", justify="right", min_width=10)

        for e in entries:
            prob_str = f"{e['prob']:.0%} [{e['prob_low']:.0%}-{e['prob_high']:.0%}]"
            rate_str = f"{e['acceptance_rate']:.0%}" if e["acceptance_rate"] else "—"
            table.add_row(e["name"], e["university"], prob_str, rate_str)

        console.print(table)
        console.print()

    # Summary
    n_reach = sum(1 for r in results if r["category"] == "reach")
    n_target = sum(1 for r in results if r["category"] == "target")
    n_safety = sum(1 for r in results if r["category"] == "safety")
    console.print(
        f"  {n_reach + n_target + n_safety} programs: "
        f"[red]{n_reach} Reach[/red] | "
        f"[yellow]{n_target} Target[/yellow] | "
        f"[green]{n_safety} Safety[/green]"
    )
    if skipped:
        console.print(
            f"  [dim]({len(skipped)} programs skipped — no model data: "
            + ", ".join(skipped) + ")[/dim]"
        )
    console.print()
    model_label = (
        "GPBoost v2 model (AUC 0.723)" if _v2_available
        else "v1 logistic regression fallback"
    )
    console.print(
        f"  [dim]Predictions from {model_label}. "
        "Based on GPA, university tier, internships, research, nationality, major. "
        "No course-level data required.[/dim]"
    )

    # Offer to contribute anonymized data
    console.print()
    try:
        contribute = input(
            "  Help improve predictions — contribute your anonymized data? (y/n): "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        contribute = "n"
    if contribute == "y":
        gh_ok = _check_gh_ready()
        _submit_contribution(profile, results, gh_ready=gh_ok)


# ===================================================================
# ANONYMOUS DATA CONTRIBUTION
# ===================================================================


def _check_gh_ready() -> bool:
    """Pre-flight check: is ``gh`` installed and authenticated?

    Returns True if ready, False otherwise (with user-visible warnings).
    """
    import shutil
    import subprocess

    if not shutil.which("gh"):
        console.print(
            "\n  [yellow]GitHub CLI (gh) not installed — data will be saved locally.[/yellow]"
            "\n  [dim]Install: https://cli.github.com  "
            "(or just submit via browser when prompted)[/dim]"
        )
        return False

    try:
        subprocess.run(
            ["gh", "auth", "status"],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        console.print(
            "\n  [yellow]GitHub CLI not logged in — data will be saved locally.[/yellow]"
            "\n  [dim]You can submit via browser, or log in later: "
            "gh auth login && quantpath contribute-upload[/dim]"
        )
        return False


def _classify_university(uni: str) -> str:
    """Map university name to anonymous tier. No original name is kept."""
    uni = uni.lower()
    _T10 = ["mit", "stanford", "caltech", "princeton", "harvard",
             "chicago", "penn", "columbia", "berkeley", "yale"]
    _T20 = ["cornell", "carnegie mellon", "cmu", "duke", "northwestern",
             "johns hopkins", "rice", "vanderbilt", "ucla", "michigan"]
    _T30 = ["nyu", "uiuc", "illinois", "georgia tech", "gatech",
             "wisconsin", "purdue", "unc"]
    _TOP_INTL = ["tsinghua", "peking", "imperial", "lse", "oxford",
                  "cambridge", "eth", "epfl"]
    _C9 = ["zhejiang", "fudan", "sjtu", "shanghai jiao tong", "ustc", "nanjing"]
    if any(s in uni for s in _T10):
        return "US T10"
    if any(s in uni for s in _T20):
        return "US T20"
    if any(s in uni for s in _T30):
        return "US T30"
    if any(s in uni for s in _TOP_INTL):
        return "Top International"
    if any(s in uni for s in _C9):
        return "China C9"
    return "Other"


def _classify_internships(work_exp: list) -> str:
    """Summarize internships as anonymous categories."""
    internships = [
        e for e in work_exp
        if isinstance(e, dict) and e.get("type") == "internship"
    ]
    if not internships:
        return "None"

    _TOP_QUANT = ["citadel", "jane street", "two sigma", "de shaw", "hrt",
                   "hudson river", "jump", "imc", "optiver", "sig",
                   "susquehanna", "five rings", "tower research"]
    _QUANT = ["aqr", "point72", "millennium", "bridgewater", "worldquant",
               "man group", "balyasny", "cubist", "drw"]
    _BB = ["goldman", "morgan stanley", "jpmorgan", "jp morgan",
            "bank of america", "citi", "barclays", "ubs"]

    categories = []
    for exp in internships:
        combined = " ".join(
            str(exp.get(k, ""))
            for k in ("company", "description", "title", "country", "location")
        ).lower()
        is_china = "china" in combined or "cn" in combined
        is_us = " us " in f" {combined} " or "united states" in combined

        if any(f in combined for f in _TOP_QUANT):
            categories.append("top quant (US)")
        elif any(f in combined for f in _QUANT):
            categories.append("quant fund (US)")
        elif any(f in combined for f in _BB):
            categories.append("bulge bracket (US)")
        elif "quant" in combined or "trading" in combined:
            if is_china:
                categories.append("quant (China)")
            elif is_us:
                categories.append("quant (US)")
            else:
                categories.append("quant (other)")
        elif is_china:
            categories.append("finance (China)")
        elif is_us:
            categories.append("finance (US)")
        else:
            categories.append("other")
    return f"{len(internships)}x: " + ", ".join(categories)


def _classify_major(majors: list) -> str:
    """Map majors to anonymous category."""
    if not majors:
        return "Unknown"
    _QUANT = [
        "math", "mathematics", "applied math",
        "statistics", "stats", "stat",
        "physics",
        "computer science", "computer", "computing", "cs",
    ]
    _ECON = ["econ", "economics", "finance", "financial"]
    n_quant = sum(
        1 for m in majors
        if any(kw == m.lower() or kw in m.lower().split() for kw in _QUANT)
    )
    if n_quant >= 2:
        return "Multi-quant (math/stats/CS)"
    if n_quant >= 1:
        return "Quant-related"
    if any(any(kw in m.lower() for kw in _ECON) for m in majors):
        return "Econ/Finance"
    return "Other"


def _ask_detail(field: str, raw: str, anonymous: str) -> str:
    """Ask user whether to share the original value or the anonymous version."""
    try:
        choice = input(
            f"    {field}: share \"{raw}\" or anonymous \"{anonymous}\"? "
            "(o=original / a=anonymous / s=skip) [o]: "
        ).strip().lower() or "o"
    except (EOFError, KeyboardInterrupt):
        choice = "a"
    if choice == "s":
        return ""
    if choice == "a":
        return anonymous
    return raw


def _build_contribution_body(profile, results: list) -> str | None:
    """Interactive per-field anonymization, returns markdown body or None if cancelled."""

    console.print(
        "\n  [bold]Choose what to share for each field[/bold]"
    )
    console.print(
        "  [dim]o = original (more useful for model), "
        "a = anonymous, s = skip[/dim]\n"
    )

    anon_lines = []

    anon_lines.append(f"- GPA: {profile.gpa}")

    gre = (
        getattr(profile.test_scores, "gre_quant", None)
        if profile.test_scores else None
    )
    if gre:
        anon_lines.append(f"- GRE Quant: {gre}")

    uni_tier = _classify_university(profile.university)
    if profile.university:
        val = _ask_detail("University", profile.university, uni_tier)
        if val:
            anon_lines.append(f"- University: {val}")

    major_cat = _classify_major(profile.majors)
    if profile.majors:
        raw_majors = ", ".join(profile.majors)
        val = _ask_detail("Majors", raw_majors, major_cat)
        if val:
            anon_lines.append(f"- Majors: {val}")

    anon_lines.append(
        f"- International: {'yes' if profile.is_international else 'no'}"
    )

    internships = [
        e for e in profile.work_experience
        if isinstance(e, dict) and e.get("type") == "internship"
    ]
    intern_anon = _classify_internships(profile.work_experience)
    if internships:
        raw_interns = "; ".join(
            str(e.get("company", "")) + " (" + str(e.get("description", "")) + ")"
            for e in internships
        )
        val = _ask_detail("Internships", raw_interns, intern_anon)
        if val:
            anon_lines.append(f"- Internships: {val}")
    else:
        anon_lines.append("- Internships: None")

    has_paper = any(
        isinstance(p, dict) and p.get("has_paper")
        for p in profile.projects
    )
    has_research = len(profile.projects) > 0
    if has_paper:
        anon_lines.append("- Research: Published paper")
    elif has_research:
        anon_lines.append("- Research: Yes (no paper)")
    else:
        anon_lines.append("- Research: No")

    pred_lines = []
    for r in sorted(results, key=lambda x: -x["prob"]):
        pred_lines.append(
            f"- {r['name']} ({r['university']}): "
            f"{r['prob']:.0%} [{r['prob_low']:.0%}-{r['prob_high']:.0%}] "
            f"— {r['category']}"
        )

    outcome_lines = []
    for r in sorted(results, key=lambda x: -x["prob"]):
        outcome_lines.append(f"- [ ] {r['name']} ({r['university']}): pending")

    console.print("\n  [bold]Final data to be shared:[/bold]\n")
    for line in anon_lines:
        console.print(f"    {line}")
    console.print()

    try:
        confirm = input("  Submit this to GitHub? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        confirm = "n"
    if confirm != "y":
        console.print("  [dim]Cancelled.[/dim]")
        return None

    return (
        "## Profile\n\n"
        + "\n".join(anon_lines)
        + "\n\n## Prediction Results\n\n"
        + "\n".join(pred_lines)
        + "\n\n## Actual Outcomes (please update later!)\n\n"
        + "\n".join(outcome_lines)
        + "\n\n---\n*Auto-generated by `quantpath predict`*"
    )


_REPO = "MasterAgentAI/QuantPath"
_ISSUE_TITLE = "[data] Anonymous admission contribution"


def _save_contribution_locally(body: str) -> str:
    """Save contribution markdown to ``data/contributions/`` and return the path."""
    import os
    from datetime import datetime

    contrib_dir = os.path.join(os.path.dirname(__file__), "..", "data", "contributions")
    os.makedirs(contrib_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(contrib_dir, f"contribution_{ts}.md")
    with open(path, "w") as f:
        f.write(body)
    return os.path.abspath(path)


def _submit_contribution(profile, results: list, *, gh_ready: bool = True) -> None:
    """Build contribution with per-field privacy choices and submit."""
    import subprocess

    body = _build_contribution_body(profile, results)
    if body is None:
        return

    if not gh_ready:
        path = _save_contribution_locally(body)
        console.print(f"\n  [green]Saved locally:[/green] {path}")
        _offer_browser_submit(body, path)
        return

    cmd = [
        "gh", "issue", "create",
        "--repo", _REPO,
        "--title", _ISSUE_TITLE,
        "--body", body,
    ]

    label_ok = _check_label_exists("data-contribution")
    if label_ok:
        cmd.extend(["--label", "data-contribution"])

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        console.print("\n  [green]Submitted! Thank you for contributing.[/green]")
        url = result.stdout.strip()
        if url:
            console.print(f"  [dim]Issue URL: {url}[/dim]")
        console.print(
            "  [dim]Please come back and update your actual outcomes "
            "when you receive decisions.[/dim]"
        )
    except subprocess.CalledProcessError as exc:
        path = _save_contribution_locally(body)
        stderr = exc.stderr or ""
        console.print(f"\n  [yellow]GitHub submit failed: {stderr.strip()}[/yellow]")
        console.print(f"  [green]Saved locally:[/green] {path}")
        _offer_browser_submit(body, path)


def _offer_browser_submit(body: str, local_path: str) -> None:
    """Give the user a choice: open browser to submit, or keep local for later."""
    import urllib.parse

    console.print("\n  [bold]How would you like to submit?[/bold]")
    console.print("  [dim]1 = Open browser (no gh needed)  2 = Skip (submit later)[/dim]")

    try:
        choice = input("  Choice [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        choice = "2"

    if choice == "1":
        import webbrowser

        params = urllib.parse.urlencode({
            "title": _ISSUE_TITLE,
            "body": body,
            "labels": "data-contribution",
        })
        url = f"https://github.com/{_REPO}/issues/new?{params}"

        if len(url) > 8000:
            console.print(
                "  [yellow]Data too long for URL. "
                "Please copy the local file contents into the issue manually.[/yellow]"
            )
            issue_url = f"https://github.com/{_REPO}/issues/new"
            console.print(f"  [dim]Opening: {issue_url}[/dim]")
            webbrowser.open(issue_url)
        else:
            webbrowser.open(url)
            console.print("  [green]Opened GitHub issue page in browser.[/green]")

        console.print(f"  [dim]Local backup: {local_path}[/dim]")
    else:
        console.print("\n  [dim]Submit later with:[/dim]")
        console.print("  [bold]quantpath contribute-upload[/bold]")


def cmd_contribute_upload(args: argparse.Namespace) -> None:
    """Upload locally saved contribution files to GitHub."""
    import glob
    import os
    import subprocess

    contrib_dir = os.path.join(os.path.dirname(__file__), "..", "data", "contributions")
    contrib_dir = os.path.abspath(contrib_dir)

    if not os.path.isdir(contrib_dir):
        console.print("  [dim]No pending contributions found.[/dim]")
        return

    files = sorted(glob.glob(os.path.join(contrib_dir, "contribution_*.md")))
    if not files:
        console.print("  [dim]No pending contributions found.[/dim]")
        return

    console.print(f"\n  [bold]Found {len(files)} pending contribution(s):[/bold]\n")
    for i, f in enumerate(files, 1):
        ts = os.path.basename(f).replace("contribution_", "").replace(".md", "")
        console.print(f"  {i}. {ts}  [dim]({f})[/dim]")

    console.print()
    try:
        sel = input(f"  Submit which? (1-{len(files)}, a=all, q=quit) [a]: ").strip().lower() or "a"
    except (EOFError, KeyboardInterrupt):
        return

    if sel == "q":
        return

    if sel == "a":
        to_submit = files
    else:
        try:
            idx = int(sel) - 1
            to_submit = [files[idx]]
        except (ValueError, IndexError):
            console.print("  [yellow]Invalid selection.[/yellow]")
            return

    gh_ok = _check_gh_ready()

    for fpath in to_submit:
        with open(fpath) as f:
            body = f.read()

        if gh_ok:
            cmd = [
                "gh", "issue", "create",
                "--repo", _REPO,
                "--title", _ISSUE_TITLE,
                "--body", body,
            ]
            if _check_label_exists("data-contribution"):
                cmd.extend(["--label", "data-contribution"])
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                url = result.stdout.strip()
                console.print(f"  [green]Submitted![/green] {url}")
                os.remove(fpath)
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr or ""
                console.print(f"  [yellow]Failed: {stderr.strip()}[/yellow]")
                _offer_browser_submit(body, fpath)
        else:
            _offer_browser_submit(body, fpath)


def _check_label_exists(label: str) -> bool:
    """Check if a label exists on the remote repo (best-effort)."""
    import subprocess

    try:
        subprocess.run(
            ["gh", "label", "list", "--repo", _REPO,
             "--search", label, "--limit", "1"],
            check=True,
            capture_output=True,
        )
        return True
    except Exception:
        return False


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a user profile against MFE programs."""
    profile = load_profile(args.profile)
    programs = load_all_programs()
    projected = getattr(args, "projected", False)
    result = evaluate_profile(profile, projected=projected)

    # PDF output path requested
    output_path = getattr(args, "output", None)
    if output_path and output_path.endswith(".pdf"):
        rankings = rank_schools(profile, programs, result, projected=projected, use_v2=True)
        gap_recs = analyze_gaps(result.gaps) if result.gaps else []

        from core.report_generator import generate_report

        path = generate_report(
            profile=profile,
            evaluation=result,
            rankings=rankings,
            gaps=gap_recs,
            output_path=output_path,
        )
        console.print(f"[green]PDF report saved to:[/green] {path}")
        return

    mode_label = (
        " [bold yellow](Projected — including planned courses)[/bold yellow]"
        if projected
        else ""
    )
    # Header
    console.print()
    console.print(
        Panel(
            f"[bold]{profile.name}[/bold] | {profile.university} | "
            f"GPA {profile.gpa} | {'International' if profile.is_international else 'Domestic'}"
            + mode_label,
            title="QuantPath Profile Evaluation",
            border_style="cyan",
        )
    )

    # Dimension scores
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Dimension", style="bold", width=16)
    table.add_column("Score", width=10)
    table.add_column("Bar", width=12)

    dim_labels = {
        "math": "Math",
        "statistics": "Statistics",
        "cs": "CS",
        "finance_econ": "Finance/Econ",
        "gpa": "GPA",
    }

    for dim_id, label in dim_labels.items():
        score = result.dimension_scores.get(dim_id, 0)
        color = "green" if score >= 9 else "yellow" if score >= 7 else "red"
        table.add_row(
            f"  {label}",
            f"[{color}]{score:.1f}/10[/{color}]",
            f"[{color}]{bar(score)}[/{color}]",
        )

    console.print(table)
    console.print()

    # Overall
    overall = result.overall_score
    overall_color = "green" if overall >= 9 else "yellow" if overall >= 7 else "red"
    level = (
        "Top 3-5 MFE Level"
        if overall >= 9.5
        else "Top 5-10 MFE Level"
        if overall >= 8.5
        else "Competitive"
        if overall >= 7.5
        else "Needs Improvement"
    )
    console.print(
        f"  [bold]OVERALL:[/bold]    [{overall_color}]{overall:.1f}/10[/{overall_color}]  "
        f"[{overall_color}]{bar(overall)}[/{overall_color}]  {level}"
    )
    console.print()

    # School recommendations (use projected mode if flag set)
    rankings = rank_schools(profile, programs, result, projected=projected, use_v2=True)

    if rankings.get("reach"):
        reach_names = ", ".join(r["name"] for r in rankings["reach"])
        console.print(f"  Reach:   {reach_names}")
    if rankings.get("target"):
        target_names = ", ".join(r["name"] for r in rankings["target"])
        console.print(f"  Target:  {target_names}")
    if rankings.get("safety"):
        safety_names = ", ".join(r["name"] for r in rankings["safety"])
        console.print(f"  Safety:  {safety_names}")
    console.print()

    # If projected mode, show comparison: current vs projected
    if projected and profile.planned_coursework:
        current_result = evaluate_profile(profile, projected=False)
        curr_overall = current_result.overall_score
        proj_overall = result.overall_score
        diff = proj_overall - curr_overall
        diff_color = "green" if diff > 0 else "red"

        compare_table = Table(
            title="[bold]Current vs Projected Profile[/bold]",
            border_style="yellow",
            show_lines=True,
        )
        compare_table.add_column("Dimension", style="bold", width=16)
        compare_table.add_column("Current", justify="right", width=10)
        compare_table.add_column("Projected", justify="right", width=10)
        compare_table.add_column("Change", justify="right", width=10)

        dim_labels_compare = {
            "math": "Math",
            "statistics": "Statistics",
            "cs": "CS",
            "finance_econ": "Finance/Econ",
            "gpa": "GPA",
        }
        for dim_id, label in dim_labels_compare.items():
            curr_s = current_result.dimension_scores.get(dim_id, 0)
            proj_s = result.dimension_scores.get(dim_id, 0)
            delta = proj_s - curr_s
            d_color = "green" if delta > 0.1 else "dim" if abs(delta) <= 0.1 else "red"
            compare_table.add_row(
                label,
                f"{curr_s:.1f}",
                f"[bold]{proj_s:.1f}[/bold]",
                f"[{d_color}]{delta:+.1f}[/{d_color}]",
            )
        compare_table.add_row(
            "[bold]OVERALL[/bold]",
            f"{curr_overall:.1f}",
            f"[bold]{proj_overall:.1f}[/bold]",
            f"[{diff_color}]{diff:+.1f}[/{diff_color}]",
        )
        console.print(compare_table)
        n_planned = len(profile.planned_coursework)
        console.print(
            f"  [dim]({n_planned} planned courses included: "
            + ", ".join(c.code for c in profile.planned_coursework[:5])
            + ("..." if n_planned > 5 else "") + ")[/dim]"
        )
        console.print()

    # Gaps
    if result.gaps:
        console.print("  [bold red]⚠️  Gaps Found:[/bold red]")
        for gap in result.gaps:
            dim = gap.get("dimension", "")
            factor = gap.get("factor", "").replace("_", " ").title()
            score = gap.get("score", 0)
            if score == 0:
                console.print(f"     - {factor} [dim]({dim})[/dim]: [red]Missing[/red]")
            else:
                console.print(
                    f"     - {factor} [dim]({dim})[/dim]: [yellow]{score:.1f}/10[/yellow]"
                )
        console.print()

    # Strengths
    if result.strengths:
        console.print("  [bold green]✅ Strengths:[/bold green]")
        for strength in result.strengths:
            dim = strength.get("dimension", "")
            factor = strength.get("factor", "").replace("_", " ").title()
            score = strength.get("score", 0)
            console.print(f"     - {factor} [dim]({dim})[/dim]: [green]{score:.1f}/10[/green]")
        console.print()


def cmd_match(args: argparse.Namespace) -> None:
    """Match prerequisites against specific programs."""
    from core.lr_predictor import predict_prob_full

    profile = load_profile(args.profile)
    programs = load_all_programs()

    if args.program:
        programs = [p for p in programs if p.id == args.program]
        if not programs:
            console.print(f"[red]Program '{args.program}' not found.[/red]")
            return

    gre_quant = profile.test_scores.gre_quant

    console.print()
    console.print(Panel("Prerequisite Match Report", border_style="cyan"))

    for program in programs:
        match = match_prerequisites(profile, program)
        color = (
            "green" if match.match_score >= 0.8 else "yellow" if match.match_score >= 0.6 else "red"
        )

        console.print(f"\n  [bold]{program.name}[/bold] ({program.university})")
        console.print(f"  Match: [{color}]{match.match_score:.0%}[/{color}]")

        lr_pred = predict_prob_full(program.id, profile.gpa, gre_quant, profile)
        if lr_pred is not None:
            pcolor = "green" if lr_pred.prob >= 0.6 else "yellow" if lr_pred.prob >= 0.35 else "red"
            ci_str = (
                f" [dim][{lr_pred.prob_low:.0%}–{lr_pred.prob_high:.0%}][/dim]"
                if lr_pred.prob_low is not None and lr_pred.prob_high is not None
                else ""
            )
            bc_flag = " [dim](bias-corrected)[/dim]" if lr_pred.is_bias_corrected else ""
            console.print(f"  P(Admit): [{pcolor}]{lr_pred.prob:.0%}[/{pcolor}]{ci_str}{bc_flag}")

        if match.missing:
            missing_names = [
                m.get("category", m.get("name", str(m))) if isinstance(m, dict) else str(m)
                for m in match.missing
            ]
            console.print(f"  [red]Missing:[/red] {', '.join(missing_names)}")
        if match.warnings:
            for w in match.warnings:
                console.print(f"  [yellow]⚠️  {w}[/yellow]")
    console.print()


def cmd_tests(args: argparse.Namespace) -> None:
    """Check GRE/TOEFL requirements."""
    profile = load_profile(args.profile)
    programs = load_all_programs()

    console.print()
    table = Table(title="Test Requirements", border_style="cyan")
    table.add_column("Program", style="bold")
    table.add_column("GRE")
    table.add_column("TOEFL")

    for program in programs:
        gre = check_gre(profile, program)
        toefl = check_toefl(profile, program)

        gre_str = (
            "[green]Exempt[/green]"
            if gre["exempt"]
            else "[red]REQUIRED[/red]"
            if gre["required"]
            else "[yellow]Optional[/yellow]"
        )
        toefl_str = (
            "[green]Waived[/green]"
            if toefl["waived"]
            else "[red]REQUIRED[/red]"
            if toefl["required"]
            else "[yellow]Check[/yellow]"
        )
        table.add_row(program.name, gre_str, toefl_str)

    console.print(table)
    console.print()


def cmd_timeline(args: argparse.Namespace) -> None:
    """Generate application timeline."""
    programs = load_all_programs()
    from datetime import date

    events = generate_timeline(programs, date.today())

    console.print()
    console.print(Panel("Application Timeline", border_style="cyan"))

    current_month = None
    for event in events:
        # Timeline returns ISO date strings; parse to date objects.
        d = date.fromisoformat(event["date"])
        month = d.strftime("%B %Y")
        if month != current_month:
            console.print(f"\n  [bold cyan]── {month} ──[/bold cyan]")
            current_month = month

        priority_icon = (
            "🔴"
            if event["priority"] == "critical"
            else "🟡"
            if event["priority"] == "high"
            else "⚪"
        )
        console.print(f"  {priority_icon} {d.strftime('%b %d')}  {event['action']}")

    console.print()


def cmd_programs(args: argparse.Namespace) -> None:
    """List all programs in the database."""
    programs = load_all_programs()

    console.print()
    table = Table(title="MFE Programs Database (QuantNet 2026)", border_style="cyan")
    table.add_column("#", justify="right", width=3)
    table.add_column("Program", style="bold")
    table.add_column("University")
    table.add_column("Class", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("Salary", justify="right")
    table.add_column("Emp 3m", justify="right")
    table.add_column("GRE")

    for p in programs:
        rank = str(p.quantnet_ranking) if p.quantnet_ranking else "-"
        rate = f"{p.acceptance_rate:.0%}" if p.acceptance_rate else "N/A"
        size = str(p.class_size) if p.class_size else "N/A"
        salary = f"${p.avg_base_salary:,}" if p.avg_base_salary else "N/A"
        emp = f"{p.employment_rate_3m:.0%}" if p.employment_rate_3m else "N/A"
        gre = "Required" if p.gre_required else "Optional"
        table.add_row(rank, p.name, p.university, size, rate, salary, emp, gre)

    console.print(table)
    console.print()


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare 2-3 programs side-by-side."""
    from core.lr_predictor import predict_prob_full

    program_ids = [pid.strip() for pid in args.programs.split(",")]

    if len(program_ids) < 2 or len(program_ids) > 3:
        console.print("[red]Please specify 2 or 3 program IDs separated by commas.[/red]")
        return

    all_programs = load_all_programs()
    programs_by_id = {p.id: p for p in all_programs}

    selected = []
    for pid in program_ids:
        if pid not in programs_by_id:
            console.print(f"[red]Program '{pid}' not found.[/red]")
            return
        selected.append(programs_by_id[pid])

    # Optional profile-specific P(admit) row
    profile = None
    profile_path = getattr(args, "profile", None)
    if profile_path:
        profile = load_profile(profile_path)

    console.print()
    console.print(
        Panel(
            "Side-by-Side Program Comparison",
            border_style="cyan",
        )
    )

    table = Table(border_style="cyan", show_lines=True)
    table.add_column("Attribute", style="bold", width=22)
    for prog in selected:
        table.add_column(prog.name, min_width=18)

    # University
    table.add_row("University", *[p.university for p in selected])

    # Class Size
    table.add_row(
        "Class Size",
        *[str(p.class_size) if p.class_size else "N/A" for p in selected],
    )

    # Acceptance Rate
    table.add_row(
        "Acceptance Rate",
        *[f"{p.acceptance_rate:.0%}" if p.acceptance_rate else "N/A" for p in selected],
    )

    # Avg GPA
    table.add_row(
        "Avg GPA",
        *[f"{p.avg_gpa:.2f}" if p.avg_gpa else "N/A" for p in selected],
    )

    # GRE Required
    table.add_row(
        "GRE Required",
        *["Yes" if p.gre_required else "No" for p in selected],
    )

    # TOEFL Min
    table.add_row(
        "TOEFL Min",
        *[str(p.toefl_min_ibt) if p.toefl_min_ibt else "N/A" for p in selected],
    )

    # Application Fee
    table.add_row(
        "Application Fee",
        *[f"${p.application_fee}" if p.application_fee else "N/A" for p in selected],
    )

    # Recommendations
    table.add_row(
        "Recommendations",
        *[str(p.recommendations) if p.recommendations else "N/A" for p in selected],
    )

    # Deadline Round 1
    def _round_date(prog, round_num):
        for r in prog.deadline_rounds:
            if r.round == round_num:
                return r.date
        return "N/A"

    table.add_row(
        "Deadline Round 1",
        *[_round_date(p, 1) for p in selected],
    )

    # Deadline Round 2
    table.add_row(
        "Deadline Round 2",
        *[_round_date(p, 2) for p in selected],
    )

    # Prerequisites (required count)
    table.add_row(
        "Prerequisites (req.)",
        *[str(len(p.prerequisites_required)) for p in selected],
    )

    # Tuition
    table.add_row(
        "Tuition",
        *[f"${p.tuition_total:,}" if p.tuition_total else "N/A" for p in selected],
    )

    # Avg Salary
    table.add_row(
        "Avg Base Salary",
        *[f"${p.avg_base_salary:,}" if p.avg_base_salary else "N/A" for p in selected],
    )

    # Employment Rate
    table.add_row(
        "Employment (3m)",
        *[f"{p.employment_rate_3m:.0%}" if p.employment_rate_3m else "N/A" for p in selected],
    )

    # Interview type
    table.add_row(
        "Interview",
        *[
            p.interview_type.replace("_", " ").title() if p.interview_type else "N/A"
            for p in selected
        ],
    )

    # QuantNet Ranking
    table.add_row(
        "QuantNet Rank",
        *[f"#{p.quantnet_ranking}" if p.quantnet_ranking else "N/R" for p in selected],
    )

    # P(Admit) — only when --profile is provided
    if profile is not None:
        gre_quant = profile.test_scores.gre_quant
        prob_cells = []
        for p in selected:
            lr_pred = predict_prob_full(p.id, profile.gpa, gre_quant, profile)
            if lr_pred is not None:
                prob = lr_pred.prob
                pcolor = "green" if prob >= 0.6 else "yellow" if prob >= 0.35 else "red"
                ci = (
                    f" [{lr_pred.prob_low:.0%}-{lr_pred.prob_high:.0%}]"
                    if lr_pred.prob_low is not None and lr_pred.prob_high is not None
                    else ""
                )
                prob_cells.append(f"[{pcolor}]{lr_pred.prob:.0%}[/{pcolor}]{ci}")
            else:
                prob_cells.append("[dim]N/A[/dim]")
        table.add_row("P(Admit) *", *prob_cells)

    console.print(table)
    if profile is not None:
        console.print(f"  [dim]* P(Admit) for {profile.name} (GPA {profile.gpa})[/dim]")
    console.print()


def _difficulty_badge(difficulty: str) -> str:
    """Return a Rich-formatted badge for question difficulty."""
    colors = {"easy": "green", "medium": "yellow", "hard": "red"}
    color = colors.get(difficulty, "white")
    label = difficulty.upper()
    return f"[{color}] {label} [/{color}]"


def cmd_interview(args: argparse.Namespace) -> None:
    """Display interview practice questions as a study guide."""
    categories = load_questions()

    # --list-categories: just print the category table and exit.
    if args.list_categories:
        console.print()
        table = Table(title="Interview Question Categories", border_style="cyan")
        table.add_column("ID", style="bold")
        table.add_column("Category")
        table.add_column("Questions", justify="right")
        table.add_column("Difficulties")

        for cat in categories:
            diffs = sorted({q.difficulty for q in cat.questions})
            diff_str = ", ".join(diffs)
            table.add_row(cat.id, cat.name, str(len(cat.questions)), diff_str)

        total = sum(len(c.questions) for c in categories)
        console.print(table)
        console.print(
            f"\n  [bold]{total}[/bold] questions across [bold]{len(categories)}[/bold] categories\n"
        )
        return

    # Build the question pool based on filters.
    if args.category:
        questions = get_questions_by_category(args.category, categories)
        if not questions:
            console.print(f"[red]Category '{args.category}' not found.[/red]")
            console.print("[dim]Use --list-categories to see available categories.[/dim]")
            return
    elif args.difficulty:
        questions = get_questions_by_difficulty(args.difficulty, categories)
    elif args.program:
        questions = get_questions_for_program(args.program, categories)
        if not questions:
            console.print(f"[red]No questions tagged for program '{args.program}'.[/red]")
            return
    else:
        questions = get_random_quiz(
            n=args.count,
            categories=categories,
        )

    # Apply additional filters when a primary filter was already set.
    if args.category and args.difficulty:
        questions = [q for q in questions if q.difficulty == args.difficulty.lower()]
    if args.program and args.category:
        questions = [q for q in questions if args.program in q.programs]

    # Limit to requested count (unless fewer available).
    if len(questions) > args.count:
        questions = questions[: args.count]

    if not questions:
        console.print("[yellow]No questions match the given filters.[/yellow]")
        return

    # Header
    console.print()
    filter_parts = []
    if args.category:
        filter_parts.append(f"Category: [bold]{args.category}[/bold]")
    if args.difficulty:
        filter_parts.append(f"Difficulty: [bold]{args.difficulty}[/bold]")
    if args.program:
        filter_parts.append(f"Program: [bold]{args.program}[/bold]")
    filter_desc = "  |  ".join(filter_parts) if filter_parts else "All categories"

    console.print(
        Panel(
            f"[bold]Interview Practice Set[/bold]  --  {len(questions)} questions\n{filter_desc}",
            border_style="cyan",
            title="QuantPath Interview Prep",
        )
    )

    # Render each question
    for idx, q in enumerate(questions, 1):
        badge = _difficulty_badge(q.difficulty)
        topics_str = ", ".join(q.topics)

        # Question header line
        header = Text()
        header.append(f"Q{idx}", style="bold cyan")
        header.append(f"  [{q.category_name}]", style="dim")

        # Build panel content
        lines = []
        lines.append(f"{q.question}")
        lines.append("")
        lines.append(f"Difficulty: {badge}    Topics: [cyan]{topics_str}[/cyan]")

        if q.programs:
            prog_str = ", ".join(q.programs)
            lines.append(f"Programs:   [dim]{prog_str}[/dim]")

        lines.append("")
        lines.append(f"[dim italic]Hint: {q.hint}[/dim italic]")
        lines.append("")
        lines.append("[bold]Solution:[/bold]")
        for sol_line in q.solution.strip().splitlines():
            lines.append(f"  {sol_line}")

        console.print(
            Panel(
                "\n".join(lines),
                title=str(header),
                border_style="blue",
                padding=(1, 2),
            )
        )

    # Summary footer
    cats_seen = sorted({q.category_name for q in questions})
    diff_counts = {}
    for q in questions:
        diff_counts[q.difficulty] = diff_counts.get(q.difficulty, 0) + 1
    diff_summary = "  ".join(f"{_difficulty_badge(d)}: {c}" for d, c in sorted(diff_counts.items()))

    console.print(
        Panel(
            f"[bold]{len(questions)}[/bold] questions  |  "
            f"Categories: {', '.join(cats_seen)}\n"
            f"Difficulty breakdown: {diff_summary}",
            title="Practice Set Summary",
            border_style="green",
        )
    )
    console.print()


def _fmt_prob(e: object) -> str:
    """Format admission_prob + CI for display, using pre-computed SchoolListEntry fields."""
    prob = getattr(e, "admission_prob", None)
    if prob is None:
        return "[dim]N/A[/dim]"
    pcolor = "green" if prob >= 0.6 else "yellow" if prob >= 0.35 else "red"
    low = getattr(e, "prob_low", None)
    high = getattr(e, "prob_high", None)
    ci = f" [dim][{low:.0%}–{high:.0%}][/dim]" if low is not None and high is not None else ""
    return f"[{pcolor}]{prob:.0%}[/{pcolor}]{ci}"


def cmd_list(args: argparse.Namespace) -> None:
    """Build and display an optimised school application list."""
    profile = load_profile(args.profile)
    programs = load_all_programs()
    projected = getattr(args, "projected", False)
    evaluation = evaluate_profile(profile, projected=projected)
    school_list = build_school_list(profile, programs, evaluation)

    # Warn if GRE is missing — LR predictions fall back to training mean
    if profile.test_scores.gre_quant is None:
        console.print(
            "  [yellow]Note: GRE Quant not provided — P(Admit) estimates use "
            "program average GRE as a proxy and may be optimistic.[/yellow]"
        )

    console.print()
    console.print(
        Panel(
            f"[bold]Application List[/bold] for {profile.name}",
            border_style="cyan",
        )
    )

    # One table per category — P(Admit) uses pre-computed values from rank_schools
    for label, entries, style in [
        ("Reach", school_list.reach, "red"),
        ("Target", school_list.target, "yellow"),
        ("Safety", school_list.safety, "green"),
    ]:
        if not entries:
            continue
        table = Table(
            title=f"[bold {style}]{label} Schools[/bold {style}]",
            border_style=style,
            show_lines=True,
        )
        table.add_column("Program", style="bold", min_width=20)
        table.add_column("University", min_width=18)
        table.add_column("Fit", justify="right", width=6)
        table.add_column("Prereq", justify="right", width=7)
        table.add_column("P(Admit)  [CI]", justify="right", width=20)
        table.add_column("Reason", min_width=28)

        for e in entries:
            table.add_row(
                e.name,
                e.university,
                f"{e.fit_score:.1f}",
                f"{e.prereq_match_score:.0%}",
                _fmt_prob(e),
                e.reason,
            )
        console.print(table)
        console.print()

    # Summary footer.
    console.print(
        f"  [bold]Total application fees:[/bold] ${school_list.total_application_fees:,}"
    )
    console.print(f"  [dim]{school_list.summary}[/dim]")
    console.print()


def cmd_roi(args: argparse.Namespace) -> None:
    """Display ROI analysis for all MFE programs."""
    programs = load_all_programs()
    results = calculate_roi(
        programs,
        opportunity_cost_salary=args.opportunity_cost,
        discount_rate=args.discount_rate,
    )

    if not results:
        console.print("[yellow]No programs with both tuition and salary data found.[/yellow]")
        return

    console.print()
    table = Table(title="MFE Program ROI Analysis", border_style="cyan", show_lines=True)
    table.add_column("Program", style="bold", min_width=16)
    table.add_column("Tuition", justify="right")
    table.add_column("Living Cost", justify="right")
    table.add_column("Total Cost", justify="right")
    table.add_column("Avg Salary", justify="right")
    table.add_column("Payback (yrs)", justify="right")
    table.add_column("5yr NPV", justify="right")
    table.add_column("Risk-Adj ROI", justify="right")

    for r in results:
        # Color-code NPV
        if r.npv_5yr >= 0:
            npv_str = f"[green]${r.npv_5yr:,.0f}[/green]"
        else:
            npv_str = f"[red]${r.npv_5yr:,.0f}[/red]"

        # Color-code ROI
        if r.risk_adjusted_roi >= 100:
            roi_str = f"[green]{r.risk_adjusted_roi:.0f}%[/green]"
        elif r.risk_adjusted_roi >= 0:
            roi_str = f"[yellow]{r.risk_adjusted_roi:.0f}%[/yellow]"
        else:
            roi_str = f"[red]{r.risk_adjusted_roi:.0f}%[/red]"

        # Payback years
        if r.payback_years == float("inf"):
            payback_str = "[red]N/A[/red]"
        else:
            payback_str = f"{r.payback_years:.1f}"

        table.add_row(
            f"{r.program_name}\n[dim]{r.university}[/dim]",
            f"${r.tuition:,}",
            f"${r.living_cost_total:,}",
            f"${r.total_cost:,}",
            f"${r.avg_salary:,}",
            payback_str,
            npv_str,
            roi_str,
        )

    console.print(table)
    console.print()
    console.print(
        f"  [dim]Assumptions: opportunity cost salary = ${args.opportunity_cost:,}, "
        f"discount rate = {args.discount_rate:.0%}, "
        f"program duration = 18 months[/dim]"
    )
    console.print()


def cmd_optimize(args: argparse.Namespace) -> None:
    """Recommend courses to maximize profile improvement."""
    profile = load_profile(args.profile)
    programs = load_all_programs()
    recommendations = optimize_courses(profile, programs, max_courses=args.max_courses)

    console.print()
    console.print(
        Panel(
            f"[bold]Course Optimizer[/bold] for {profile.name}\n"
            f"Top {args.max_courses} courses to maximize your profile score",
            border_style="cyan",
        )
    )

    if not recommendations:
        console.print(
            "  [bold green]No course recommendations -- your profile"
            " is strong across all dimensions![/bold green]"
        )
        console.print()
        return

    table = Table(border_style="cyan", show_lines=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("Course Category", style="bold", min_width=22)
    table.add_column("Dimension", width=14)
    table.add_column("Impact", justify="right", width=8)
    table.add_column("Prereq", justify="right", width=7)
    table.add_column("Why", min_width=40)

    for idx, rec in enumerate(recommendations, 1):
        # Color-code impact score
        if rec.impact_score >= 1.0:
            impact_str = f"[red]{rec.impact_score:.2f}[/red]"
        elif rec.impact_score >= 0.5:
            impact_str = f"[yellow]{rec.impact_score:.2f}[/yellow]"
        else:
            impact_str = f"[cyan]{rec.impact_score:.2f}[/cyan]"

        prereq_str = (
            f"[red]{rec.prereq_coverage}[/red]"
            if rec.prereq_coverage > 0
            else "[dim]0[/dim]"
        )

        category_label = rec.category.replace("_", " ").title()

        table.add_row(
            str(idx),
            category_label,
            rec.dimension.replace("_", "/").title(),
            impact_str,
            prereq_str,
            rec.reason,
        )

    console.print(table)
    console.print()
    console.print(
        "  [dim]Impact = dimension_weight * factor_weight * gap_to_9.0"
        " + prereq_bonus (0.5 per program)[/dim]"
    )
    console.print()


def cmd_gaps(args: argparse.Namespace) -> None:
    """Analyze profile gaps and suggest improvements."""
    profile = load_profile(args.profile)
    programs = load_all_programs()
    result = evaluate_profile(profile)

    # --- Per-program mode -------------------------------------------
    target_program_id = getattr(args, "program", None)
    if target_program_id:
        matched = [p for p in programs if p.id == target_program_id]
        if not matched:
            console.print(f"[red]Program '{target_program_id}' not found.[/red]")
            return
        prog = matched[0]
        report = program_gaps(profile, prog, result)

        console.print()
        console.print(
            Panel(
                f"[bold]Program Gap Analysis[/bold]\n"
                f"{profile.name}  →  {report.program_name} ({report.university})",
                border_style="cyan",
            )
        )

        # Admission probability
        if report.admission_prob is not None:
            pcolor = (
                "green" if report.admission_prob >= 0.6
                else "yellow" if report.admission_prob >= 0.35
                else "red"
            )
            ci_str = (
                f" [dim][{report.prob_low:.0%}–{report.prob_high:.0%} CI][/dim]"
                if report.prob_low is not None else ""
            )
            console.print(
                f"  P(Admit): [{pcolor}]{report.admission_prob:.0%}[/{pcolor}]{ci_str}"
            )
        console.print(
            f"  Prereq Match: {report.prereq_match_score:.0%}  |  "
            f"GPA Gap: {report.gpa_gap:+.2f}"
        )
        console.print()

        if not report.items:
            console.print(
                "  [bold green]No gaps for this program — "
                "your profile is well aligned.[/bold green]"
            )
            console.print()
            return

        gap_table = Table(border_style="cyan", show_lines=True)
        gap_table.add_column("Severity", width=10, justify="center")
        gap_table.add_column("Gap", style="bold", min_width=28)
        gap_table.add_column("Action", min_width=44)

        sev_colors = {"Critical": "red", "High": "red", "Medium": "yellow", "Low": "cyan"}
        for item in report.items:
            sc = sev_colors.get(item.severity, "white")
            gap_table.add_row(
                f"[{sc}]{item.severity}[/{sc}]",
                item.label,
                item.detail,
            )

        console.print(gap_table)
        console.print(
            f"  [dim]{report.n_critical} Critical  {report.n_high} High  "
            f"{len(report.items) - report.n_critical - report.n_high} Medium/Low[/dim]"
        )
        console.print()
        return

    # --- Profile-level gap mode (default) ---------------------------
    console.print()
    console.print(
        Panel(
            f"[bold]Gap Analysis[/bold] for {profile.name}",
            border_style="cyan",
        )
    )

    if not result.gaps:
        console.print(
            "  [bold green]No gaps found -- your profile looks"
            " strong across all dimensions![/bold green]"
        )
        console.print()
        return

    recommendations = analyze_gaps(result.gaps)

    table = Table(border_style="cyan", show_lines=True)
    table.add_column("Factor", style="bold", width=22)
    table.add_column("Dimension", width=14)
    table.add_column("Score", width=10, justify="center")
    table.add_column("Priority", width=10, justify="center")
    table.add_column("Recommended Action", min_width=40)

    for rec in recommendations:
        # Color-code score
        if rec.score == 0:
            score_str = "[red]Missing[/red]"
        else:
            score_str = f"[yellow]{rec.score:.1f}/10[/yellow]"

        # Color-code priority
        priority_colors = {
            "High": "red",
            "Medium": "yellow",
            "Low": "cyan",
        }
        pcolor = priority_colors.get(rec.priority, "white")
        priority_str = f"[{pcolor}]{rec.priority}[/{pcolor}]"

        factor_label = rec.factor.replace("_", " ").title()

        table.add_row(
            factor_label,
            rec.dimension,
            score_str,
            priority_str,
            rec.action,
        )

    console.print(table)

    # Summary counts
    high_count = sum(1 for r in recommendations if r.priority == "High")
    med_count = sum(1 for r in recommendations if r.priority == "Medium")
    low_count = sum(1 for r in recommendations if r.priority == "Low")

    console.print()
    console.print(
        f"  [bold]Summary:[/bold]  "
        f"[red]{high_count} High[/red]  "
        f"[yellow]{med_count} Medium[/yellow]  "
        f"[cyan]{low_count} Low[/cyan]  "
        f"({len(recommendations)} total gaps)"
    )
    console.print()


def cmd_stats(args: argparse.Namespace) -> None:
    """Show statistics from real admission data."""
    if args.file:
        records = load_admission_csv(args.file)
    else:
        records = load_all_admission_data()

    if not records:
        console.print("[yellow]No admission data found.[/yellow]")
        console.print("[dim]Add CSV files to data/admissions/ or use --file.[/dim]")
        return

    summary = summarize_records(records)

    console.print()
    console.print(
        Panel(
            f"[bold]{summary['total_records']}[/bold] records from "
            f"[bold]{summary['unique_applicants']}[/bold] applicants  |  "
            f"Sources: {', '.join(summary['sources'])}  |  "
            f"Seasons: {', '.join(summary['seasons'])}",
            title="Admission Data Statistics",
            border_style="cyan",
        )
    )

    # Per-program breakdown
    table = Table(border_style="cyan", title="Per-Program Breakdown")
    table.add_column("Program", style="bold")
    table.add_column("Accepted", style="green", justify="right")
    table.add_column("Rejected", style="red", justify="right")
    table.add_column("Waitlisted", style="yellow", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Obs. Rate", justify="right")

    for prog_id, counts in sorted(summary["programs"].items()):
        acc = counts.get("accepted", 0)
        rej = counts.get("rejected", 0)
        wl = counts.get("waitlisted", 0)
        total = acc + rej + wl
        decided = acc + rej
        rate = f"{acc / decided:.0%}" if decided > 0 else "N/A"
        table.add_row(prog_id, str(acc), str(rej), str(wl), str(total), rate)

    console.print(table)

    # Gender & nationality summary
    gender = summary.get("gender_dist", {})
    nat = summary.get("nationality_dist", {})
    if gender or nat:
        gender_str = f"M:{gender.get('M', 0)} F:{gender.get('F', 0)}"
        nat_parts = [f"{k}:{v}" for k, v in sorted(nat.items(), key=lambda x: -x[1])]
        console.print(
            f"  [bold]Demographics:[/bold]  Gender: {gender_str}  |  "
            f"Nationality: {', '.join(nat_parts)}"
        )
    console.print()

    # GPA distribution
    from core.admission_data import compute_all_program_stats

    all_stats = compute_all_program_stats(records)
    if all_stats:
        console.print(Panel("Accepted vs Rejected GPA Comparison", border_style="cyan"))
        gpa_table = Table(border_style="cyan")
        gpa_table.add_column("Program", style="bold")
        gpa_table.add_column("Avg GPA (Acc)", style="green", justify="right")
        gpa_table.add_column("Avg GPA (Rej)", style="red", justify="right")
        gpa_table.add_column("Gap", justify="right")
        gpa_table.add_column("Top Feature", justify="right")

        for pid, stats in sorted(all_stats.items()):
            if stats.accepted == 0:
                continue
            gpa_acc = f"{stats.avg_gpa_accepted:.2f}" if stats.avg_gpa_accepted else "N/A"
            gpa_rej = f"{stats.avg_gpa_rejected:.2f}" if stats.avg_gpa_rejected else "N/A"
            gap = ""
            if stats.avg_gpa_accepted and stats.avg_gpa_rejected:
                diff = stats.avg_gpa_accepted - stats.avg_gpa_rejected
                gap = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"

            top_feat = ""
            if stats.feature_importance:
                top = max(stats.feature_importance, key=lambda k: abs(stats.feature_importance[k]))
                top_feat = f"{top} ({stats.feature_importance[top]:.2f})"

            gpa_table.add_row(pid, gpa_acc, gpa_rej, gap, top_feat)

        console.print(gpa_table)
        console.print()

    # LR model quality summary
    from core.lr_predictor import get_model_stats
    from core.lr_predictor import has_model as _has_model

    model_programs = [pid for pid in summary["programs"] if _has_model(pid)]
    if model_programs:
        console.print(Panel("LR Admission Model Quality", border_style="cyan"))
        lr_table = Table(border_style="cyan")
        lr_table.add_column("Program", style="bold")
        lr_table.add_column("AUC", justify="right")
        lr_table.add_column("Samples", justify="right")
        lr_table.add_column("Bias Corrected")
        lr_table.add_column("Features")

        for pid in sorted(model_programs):
            stats = get_model_stats(pid)
            if stats is None:
                continue
            auc = stats.get("auc", 0)
            auc_color = "green" if auc >= 0.65 else "yellow" if auc >= 0.55 else "red"
            bc = "Yes" if stats.get("real_accept_rate") is not None else "No"
            bc_color = "green" if bc == "Yes" else "dim"
            lr_table.add_row(
                pid,
                f"[{auc_color}]{auc:.3f}[/{auc_color}]",
                str(stats.get("n_total", 0)),
                f"[{bc_color}]{bc}[/{bc_color}]",
                ", ".join(stats.get("features", [])),
            )

        console.print(lr_table)
        no_model = [pid for pid in summary["programs"] if not _has_model(pid)]
        if no_model:
            console.print(f"  [dim]No LR model: {', '.join(sorted(no_model))}[/dim]")
        console.print()


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Calibrate scoring model using real admission data."""
    if args.file:
        records = load_admission_csv(args.file)
    else:
        records = load_all_admission_data()

    if not records:
        console.print("[yellow]No admission data found.[/yellow]")
        console.print("[dim]Add CSV files to data/admissions/ or use --file.[/dim]")
        return

    console.print()
    console.print(Panel("Running Calibration...", border_style="cyan"))

    result = calibrate_all(records)

    # Thresholds table
    table = Table(border_style="cyan", title="Calibrated Program Thresholds")
    table.add_column("Program", style="bold")
    table.add_column("GPA Floor", justify="right")
    table.add_column("GPA Target", justify="right")
    table.add_column("GPA Safe", justify="right")
    table.add_column("Obs. Rate", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Confidence")

    for pid, threshold in sorted(result.program_thresholds.items()):
        conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(
            threshold.confidence, "white"
        )
        table.add_row(
            pid,
            f"{threshold.gpa_floor:.2f}",
            f"{threshold.gpa_target:.2f}",
            f"{threshold.gpa_safe:.2f}",
            f"{threshold.observed_acceptance_rate:.0%}",
            str(threshold.sample_size),
            f"[{conf_color}]{threshold.confidence}[/{conf_color}]",
        )

    console.print(table)

    # Global feature weights
    if result.global_feature_weights:
        console.print()
        console.print(Panel("Global Feature Importance", border_style="cyan"))
        fw_table = Table(border_style="cyan")
        fw_table.add_column("Feature", style="bold")
        fw_table.add_column("Weight", justify="right")
        fw_table.add_column("Bar", width=20)

        for feat, weight in result.global_feature_weights.items():
            bar_len = round(weight * 40)
            fw_table.add_row(feat, f"{weight:.1%}", "█" * bar_len)

        console.print(fw_table)

    # Accuracy report
    acc = result.accuracy_report
    if acc.get("total_predictions", 0) > 0:
        console.print()
        accuracy_pct = acc.get("accuracy", 0)
        acc_color = "green" if accuracy_pct >= 0.7 else "yellow" if accuracy_pct >= 0.5 else "red"
        console.print(
            f"  [bold]Model Accuracy:[/bold] [{acc_color}]{accuracy_pct:.0%}[/{acc_color}]  "
            f"({acc['correct']} correct / {acc['correct'] + acc['incorrect']} decided, "
            f"{acc['borderline']} borderline)"
        )

    # Recommendations
    if result.recommendations:
        console.print()
        console.print(Panel("Recommendations", border_style="yellow"))
        for rec in result.recommendations:
            console.print(f"  - {rec}")

    console.print()

    # If --apply flag, show the overrides that would be applied
    if args.apply:
        overrides = generate_ranker_overrides(result)
        if overrides:
            console.print(Panel("Ranker Overrides (Applied)", border_style="green"))
            for pid, ov in sorted(overrides.items()):
                console.print(
                    f"  {pid}: reach<{ov['reach_gpa_threshold']:.2f} "
                    f"safe>={ov['safety_gpa_threshold']:.2f} "
                    f"[dim](n={ov['sample_size']}, {ov['confidence']})[/dim]"
                )
            console.print()


def cmd_portfolio(args: argparse.Namespace) -> None:
    """Optimize school portfolio to maximize expected admissions."""
    profile = load_profile(args.profile)
    programs = load_all_programs()
    evaluation = evaluate_profile(profile)

    n_schools = getattr(args, "n_schools", 10)
    budget = getattr(args, "budget", 2000)

    portfolio = optimize_portfolio(
        profile, programs, evaluation,
        n_schools=n_schools,
        budget=budget,
    )

    console.print()
    console.print(
        Panel(
            f"[bold]Portfolio Optimizer[/bold] for {profile.name}\n"
            f"Maximizing expected admissions under "
            f"n≤{n_schools} schools and ${budget:,} fee budget",
            border_style="cyan",
        )
    )

    cat_styles = {"reach": "red", "target": "yellow", "safety": "green"}
    table = Table(border_style="cyan", show_lines=True)
    table.add_column("Category", width=8, justify="center")
    table.add_column("Program", style="bold", min_width=20)
    table.add_column("University", min_width=18)
    table.add_column("P(Admit)", justify="right", width=9)
    table.add_column("Fit", justify="right", width=6)
    table.add_column("Fee", justify="right", width=7)
    table.add_column("Exp. Contrib.", justify="right", width=12)

    for e in portfolio.programs:
        cat_color = cat_styles.get(e.category, "white")
        pcolor = (
            "green" if e.admission_prob >= 0.60
            else "yellow" if e.admission_prob >= 0.35
            else "red"
        )
        table.add_row(
            f"[{cat_color}]{e.category.title()}[/{cat_color}]",
            e.name,
            e.university,
            f"[{pcolor}]{e.admission_prob:.0%}[/{pcolor}]",
            f"{e.fit_score:.1f}",
            f"${e.application_fee:,}" if e.application_fee else "N/A",
            f"[green]+{e.expected_contribution:.2f}[/green]",
        )

    console.print(table)
    console.print()
    console.print(
        f"  [bold]Expected admits:[/bold] [green]{portfolio.expected_admits:.2f}[/green] schools\n"
        f"  [bold]Total fees:[/bold] ${portfolio.total_fees:,}\n"
        f"  [dim]{portfolio.summary}[/dim]"
    )
    console.print()


def cmd_whatif(args: argparse.Namespace) -> None:
    """Show how P(admit) changes under hypothetical GPA/GRE improvements."""
    from core.lr_predictor import has_model, predict_prob_full

    profile = load_profile(args.profile)
    gpa_now = profile.gpa
    gre_now = profile.test_scores.gre_quant

    gpa_hyp = getattr(args, "gpa", None) or gpa_now
    gre_hyp = getattr(args, "gre", None) or gre_now

    programs = load_all_programs()
    prog_ids = [p.id for p in programs if has_model(p.id)]

    console.print()
    console.print(
        Panel(
            f"[bold]What-If Analysis[/bold] for {profile.name}\n"
            f"Current: GPA {gpa_now}, GRE {gre_now or 'N/A'}  →  "
            f"Hypothetical: GPA {gpa_hyp}, GRE {gre_hyp or 'N/A'}",
            border_style="cyan",
        )
    )

    table = Table(border_style="cyan", show_lines=True)
    table.add_column("Program", style="bold", min_width=22)
    table.add_column("P(now)", justify="right", width=8)
    table.add_column("P(hyp)", justify="right", width=8)
    table.add_column("Delta", justify="right", width=8)
    table.add_column("Tier Change", width=14)

    def _tier(p: float) -> str:
        if p >= 0.70:
            return "safety"
        if p >= 0.40:
            return "target"
        return "reach"

    rows = []
    for pid in prog_ids:
        p_now = predict_prob_full(pid, gpa_now, gre_now, profile)
        p_hyp = predict_prob_full(pid, gpa_hyp, gre_hyp, profile)
        if p_now is None or p_hyp is None:
            continue
        delta = p_hyp.prob - p_now.prob
        tier_now = _tier(p_now.prob)
        tier_hyp = _tier(p_hyp.prob)
        rows.append((pid, p_now.prob, p_hyp.prob, delta, tier_now, tier_hyp))

    rows.sort(key=lambda r: -r[3])  # sort by delta descending

    for pid, p_now, p_hyp, delta, tier_now, tier_hyp in rows:
        prog = next((p for p in programs if p.id == pid), None)
        name = prog.name if prog else pid
        now_color = "green" if p_now >= 0.6 else "yellow" if p_now >= 0.35 else "red"
        hyp_color = "green" if p_hyp >= 0.6 else "yellow" if p_hyp >= 0.35 else "red"
        d_color = "green" if delta > 0.01 else "dim" if abs(delta) <= 0.01 else "red"
        tier_str = ""
        if tier_now != tier_hyp:
            tier_str = f"[green]{tier_now} → {tier_hyp}[/green]"
        table.add_row(
            name,
            f"[{now_color}]{p_now:.0%}[/{now_color}]",
            f"[{hyp_color}]{p_hyp:.0%}[/{hyp_color}]",
            f"[{d_color}]{delta:+.0%}[/{d_color}]",
            tier_str,
        )

    console.print(table)

    upgrades = sum(1 for *_, tn, th in rows if tn != th)
    if upgrades:
        console.print(f"\n  [green]{upgrades} program(s) would change tier.[/green]")
    console.print()


class _FriendlyParser(argparse.ArgumentParser):
    """Override error() to give actionable hints when --profile is missing."""

    def error(self, message: str) -> None:
        if "--profile" in message:
            console.print(
                f"\n  [yellow]{self.prog}: --profile is required.[/yellow]\n"
            )
            console.print("  [bold]Option A[/bold] — Run predict first (auto-creates a profile):")
            console.print("    quantpath predict")
            console.print(
                "    [dim]# this saves a profile you can reuse"
                " with other commands[/dim]\n"
            )
            console.print("  [bold]Option B[/bold] — Create a profile manually:")
            console.print("    cp examples/sample_profile.yaml profiles/my_profile.yaml")
            console.print("    # edit the file with your courses and GPA")
            console.print(f"    {self.prog} --profile profiles/my_profile.yaml\n")
            raise SystemExit(2)
        super().error(message)


def main() -> None:
    parser = _FriendlyParser(
        prog="quantpath",
        description="QuantPath — Open-source MFE application toolkit",
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", parser_class=_FriendlyParser,
    )

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate your profile")
    p_eval.add_argument("--profile", "-p", required=True, help="Path to profile YAML")
    p_eval.add_argument(
        "--output",
        "-o",
        help="Output file path (use .pdf extension for PDF report)",
    )
    p_eval.add_argument(
        "--projected",
        action="store_true",
        help="Include planned_courses from profile YAML (shows profile at application time)",
    )

    # match
    p_match = subparsers.add_parser("match", help="Match prerequisites")
    p_match.add_argument("--profile", "-p", required=True, help="Path to profile YAML")
    p_match.add_argument("--program", help="Specific program ID (default: all)")

    # tests
    p_tests = subparsers.add_parser("tests", help="Check GRE/TOEFL requirements")
    p_tests.add_argument("--profile", "-p", required=True, help="Path to profile YAML")

    # timeline
    subparsers.add_parser("timeline", help="Generate application timeline")

    # programs
    subparsers.add_parser("programs", help="List all programs")

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare 2-3 programs side-by-side")
    p_compare.add_argument(
        "--programs",
        required=True,
        help="Comma-separated program IDs (e.g., cmu-mscf,baruch-mfe,berkeley-mfe)",
    )
    p_compare.add_argument(
        "--profile", "-p", help="Profile YAML — adds a personalized P(Admit) row"
    )

    # interview
    p_interview = subparsers.add_parser("interview", help="Practice MFE interview questions")
    p_interview.add_argument(
        "--category", "-c", help="Filter by category ID (e.g. probability, finance)"
    )
    p_interview.add_argument(
        "--difficulty",
        "-d",
        choices=["easy", "medium", "hard"],
        help="Filter by difficulty level",
    )
    p_interview.add_argument(
        "--program", help="Show questions for a specific program (e.g. baruch-mfe)"
    )
    p_interview.add_argument(
        "--count",
        "-n",
        type=int,
        default=5,
        help="Number of questions to display (default: 5)",
    )
    p_interview.add_argument(
        "--list-categories",
        action="store_true",
        help="List all available question categories and exit",
    )

    # roi
    p_roi = subparsers.add_parser("roi", help="ROI analysis for MFE programs")
    p_roi.add_argument(
        "--opportunity-cost",
        type=int,
        default=65000,
        help="Annual opportunity cost salary (default: $65,000)",
    )
    p_roi.add_argument(
        "--discount-rate",
        type=float,
        default=0.05,
        help="Discount rate for NPV (default: 0.05)",
    )

    # gaps
    p_gaps = subparsers.add_parser("gaps", help="Analyze profile gaps and suggest improvements")
    p_gaps.add_argument("--profile", "-p", required=True, help="Path to profile YAML")
    p_gaps.add_argument(
        "--program",
        help="Show gaps specific to one program (e.g. baruch-mfe, cmu-mscf)",
    )

    # stats (real data)
    p_stats = subparsers.add_parser("stats", help="Show statistics from real admission data")
    p_stats.add_argument("--file", "-f", help="Path to a specific CSV file (default: all)")

    # calibrate (real data)
    p_cal = subparsers.add_parser("calibrate", help="Calibrate model using real admission data")
    p_cal.add_argument("--file", "-f", help="Path to a specific CSV file (default: all)")
    p_cal.add_argument(
        "--apply", action="store_true", help="Show ranker overrides that would be applied"
    )

    # optimize
    p_optimize = subparsers.add_parser(
        "optimize", help="Recommend courses to maximize profile improvement"
    )
    p_optimize.add_argument("--profile", "-p", required=True, help="Path to profile YAML")
    p_optimize.add_argument(
        "--max-courses",
        type=int,
        default=3,
        help="Maximum number of courses to recommend (default: 3)",
    )

    # list
    p_list = subparsers.add_parser("list", help="Build optimised school application list")
    p_list.add_argument("--profile", "-p", required=True, help="Path to profile YAML")
    p_list.add_argument(
        "--projected",
        action="store_true",
        help="Include planned courses (shows school list at application time)",
    )

    # portfolio
    p_portfolio = subparsers.add_parser(
        "portfolio", help="Optimize school portfolio to maximize expected admissions"
    )
    p_portfolio.add_argument("--profile", "-p", required=True, help="Path to profile YAML")
    p_portfolio.add_argument(
        "--n-schools",
        type=int,
        default=10,
        help="Max number of schools to select (default: 10)",
    )
    p_portfolio.add_argument(
        "--budget",
        type=int,
        default=2000,
        help="Max total application fees in USD (default: $2,000)",
    )

    # whatif
    p_whatif = subparsers.add_parser(
        "whatif", help="Show how P(admit) changes with hypothetical GPA/GRE"
    )
    p_whatif.add_argument("--profile", "-p", required=True, help="Path to profile YAML")
    p_whatif.add_argument(
        "--gpa", type=float, help="Hypothetical GPA (default: current)"
    )
    p_whatif.add_argument(
        "--gre", type=int, help="Hypothetical GRE Quant (default: current)"
    )

    # predict (new — pure v2 model, no course evaluation needed)
    p_predict = subparsers.add_parser(
        "predict",
        help="Predict reach/target/safety using v2 model (no detailed transcript needed)",
    )
    p_predict.add_argument(
        "--profile", "-p", required=False, default=None,
        help="Path to profile YAML (omit for interactive mode)",
    )

    subparsers.add_parser(
        "contribute-upload",
        help="Upload locally saved contribution files to GitHub",
    )

    args = parser.parse_args()

    if args.command is None:
        console.print(
            "\n  [bold]Welcome to QuantPath![/bold] "
            "MFE admission prediction & planning toolkit.\n"
        )
        console.print("  [green]Get started:[/green]")
        console.print("    quantpath predict                 "
                       "[dim]← start here (interactive, no setup needed)[/dim]")
        console.print("    quantpath programs                "
                       "[dim]← browse all 31 MFE programs[/dim]")
        console.print("    quantpath compare --programs X,Y  "
                       "[dim]← side-by-side comparison[/dim]")
        console.print()
        console.print("  [dim]For full command list: quantpath --help[/dim]\n")
        return

    commands = {
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
        "match": cmd_match,
        "tests": cmd_tests,
        "timeline": cmd_timeline,
        "programs": cmd_programs,
        "compare": cmd_compare,
        "interview": cmd_interview,
        "roi": cmd_roi,
        "gaps": cmd_gaps,
        "stats": cmd_stats,
        "calibrate": cmd_calibrate,
        "optimize": cmd_optimize,
        "list": cmd_list,
        "portfolio": cmd_portfolio,
        "whatif": cmd_whatif,
        "contribute-upload": cmd_contribute_upload,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
