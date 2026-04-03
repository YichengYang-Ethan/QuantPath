#!/usr/bin/env python3
"""Batch re-predict all 31 data-contribution issues (#3–#33) on MasterAgentAI/QuantPath.

Steps:
  1. For issues #3–#24 (no YAML): parse the ## Profile section, generate YAML.
  2. Run ensemble predictions for all 31 issues.
  3. Format new ## Prediction Results markdown.
  4. Update each issue body via `gh issue edit`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap — make QuantPath importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import yaml  # noqa: E402

from core.data_loader import load_all_programs, load_profile  # noqa: E402
from core.lr_predictor import predict_ensemble  # noqa: E402

REPO = "MasterAgentAI/QuantPath"
PROFILES_DIR = _ROOT / "profiles"

# ---------------------------------------------------------------------------
# Program list (must match cli/main.py)
# ---------------------------------------------------------------------------
_TIER0 = [
    "princeton-mfin", "baruch-mfe", "berkeley-mfe",
    "cmu-mscf", "mit-mfin", "columbia-msfe-econ",
    "yale-am", "stanford-mcf",
]
_TIER1 = [
    "uchicago-msfm", "nyu-courant", "columbia-msfe",
    "cornell-mfe", "columbia-mafn", "nyu-tandon-mfe", "gatech-qcf",
]
_FOCUSED = set(_TIER0 + _TIER1)

# Program metadata keyed by id
_prog_map: dict[str, dict] = {}


def _build_prog_map() -> None:
    """Cache program full_name + university from YAML files."""
    global _prog_map
    if _prog_map:
        return
    for prog in load_all_programs():
        _prog_map[prog.id] = {
            "full_name": prog.full_name or prog.name,
            "university": prog.university,
        }


# ---------------------------------------------------------------------------
# Issue <-> YAML mapping for #25–#33
# ---------------------------------------------------------------------------
_ISSUE_YAML_MAP: dict[int, str] = {
    25: "data_contrib_uw_amath_ds.yaml",
    26: "data_contrib_26fall_mfin_mix.yaml",
    27: "data_contrib_22fall_am_econ.yaml",
    28: "data_contrib_23fall_joint_fe.yaml",
    29: "data_contrib_near_grand_slam.yaml",
    30: "data_contrib_26fall_985_overseas.yaml",
    31: "data_contrib_mfe_grand_slam_qingbei.yaml",
    32: "data_contrib_22fall_psu_mfe.yaml",
    33: "data_contrib_actuary_uiuc_mfe.yaml",
}


# ---------------------------------------------------------------------------
# Helpers: fetch issue body via gh CLI
# ---------------------------------------------------------------------------

def _gh_issue_body(number: int) -> str:
    """Return the raw body of a GitHub issue."""
    result = subprocess.run(
        ["gh", "issue", "view", str(number), "--repo", REPO, "--json", "body"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)["body"]


def _gh_issue_edit(number: int, new_body: str) -> None:
    """Overwrite the body of a GitHub issue."""
    subprocess.run(
        ["gh", "issue", "edit", str(number), "--repo", REPO, "--body", new_body],
        capture_output=True, text=True, check=True,
    )


# ---------------------------------------------------------------------------
# Profile section parser (issues #3–#24)
# ---------------------------------------------------------------------------

def _extract_profile_section(body: str) -> str:
    """Return the text between ## Profile and the next ##."""
    lines = body.split("\n")
    capture = False
    section: list[str] = []
    for line in lines:
        if line.startswith("## Profile"):
            capture = True
            continue
        if line.startswith("## ") and capture:
            break
        if capture:
            section.append(line)
    return "\n".join(section).strip()


def _parse_gpa(text: str) -> float | None:
    """Extract GPA from profile text."""
    m = re.search(r"GPA[:\s]+([0-9]+\.[0-9]+)", text)
    if m:
        return float(m.group(1))
    return None


def _parse_gre_quant(text: str) -> int | None:
    """Extract GRE Quant score from various formats."""
    # "Q170" / "Quant 170" / "Q170+"
    m = re.search(r"(?:Q|Quant)\s*(\d{3})\+?", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # "334 (Q170 V164)" or "170+170"
    m = re.search(r"(\d{3})\s*\+\s*(\d{3})", text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return max(a, b) if max(a, b) <= 170 else None
    # "GRE Quant: 170"
    m = re.search(r"GRE\s+Quant[:\s]+(\d{3})", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # "GRE: 330+" — can't extract quant alone
    m = re.search(r"GRE[:\s]+(\d{3})\+?\s*$", text, re.MULTILINE)
    if m:
        total = int(m.group(1))
        if total >= 320:
            # Estimate quant as 170 for high total (conservative but common)
            return 170
    return None


def _parse_university(text: str) -> tuple[str, int]:
    """Parse university tier/name. Returns (university_name, years_at_us_institution)."""
    text_lower = text.lower()

    # Specific university names
    specific_names = [
        ("University of Toronto", 0),
        ("University of Waterloo", 0),
        ("University of Michigan", 4),
        ("Zhejiang University", 0),
        ("Pennsylvania State University", 4),
        ("University of Washington", 4),
        ("University of Pennsylvania", 4),
        ("Sun Yat-sen University", 0),
        ("Tsinghua University", 0),
        ("Peking University", 0),
    ]
    for name, yrs in specific_names:
        if name.lower() in text_lower:
            return name, yrs

    # Tier strings
    if "t10" in text_lower or "us t10" in text_lower:
        return "Harvard University", 4
    if "t20" in text_lower or "us t20" in text_lower or "t25" in text_lower or "us t25" in text_lower:
        return "Cornell University", 4
    if "t30" in text_lower or "us t30" in text_lower:
        return "University of Illinois Urbana-Champaign", 4
    if "t50" in text_lower or "us t50" in text_lower:
        return "Ohio State University", 4
    if "c9" in text_lower or "中9" in text_lower:
        return "Peking University", 0
    if "985" in text_lower:
        return "Wuhan University", 0
    if "211" in text_lower or "两财一贸" in text_lower:
        return "Central University of Finance and Economics", 0

    # Hong Kong
    if "hong kong" in text_lower or "hk" in text_lower:
        return "Hong Kong University of Science and Technology", 0

    # "美本" (US undergrad, school not disclosed)
    if "美本" in text or "us bachelor" in text_lower:
        return "University of Illinois Urbana-Champaign", 4  # default US T30

    # Fallback
    return "Unknown University", 0


def _parse_majors(text: str) -> list[str]:
    """Parse majors from profile text."""
    majors: list[str] = []
    text_lower = text.lower()

    # Look for the Majors line
    m = re.search(r"Majors?[:\s]+(.+)", text, re.IGNORECASE)
    raw = m.group(1).strip() if m else text_lower

    # Detect specific keywords
    kw_map = {
        "math": "Mathematics",
        "applied math": "Applied Mathematics",
        "stat": "Statistics",
        "computer science": "Computer Science",
        "cs": "Computer Science",
        "econ": "Economics",
        "finance": "Finance",
        "financial math": "Financial Mathematics",
        "financial eng": "Financial Engineering",
        "data science": "Data Science",
        "physics": "Physics",
        "actuar": "Actuarial Science",
        "quantitative finance": "Quantitative Finance",
        "farm": "Financial Analysis and Risk Management",
    }

    raw_lower = raw.lower()

    # Multi-quant pattern
    if "multi-quant" in raw_lower:
        # Check parenthetical
        paren = re.search(r"\((.+?)\)", raw)
        if paren:
            inner = paren.group(1)
            for token in re.split(r"[/+,&]", inner):
                token = token.strip().lower()
                for kw, name in kw_map.items():
                    if kw in token and name not in majors:
                        majors.append(name)
                        break
        if not majors:
            majors = ["Mathematics", "Statistics", "Computer Science"]
        return majors

    if "quant-related" in raw_lower or "quant related" in raw_lower:
        # Check parenthetical
        paren = re.search(r"\((.+?)\)", raw)
        if paren:
            inner = paren.group(1)
            for token in re.split(r"[/+,&]", inner):
                token = token.strip().lower()
                for kw, name in kw_map.items():
                    if kw in token and name not in majors:
                        majors.append(name)
                        break
        if not majors:
            majors = ["Mathematics", "Statistics"]
        return majors

    # Direct keyword matching
    for kw, name in kw_map.items():
        if kw in raw_lower and name not in majors:
            majors.append(name)

    # Deduplicate: if both Mathematics and Applied Mathematics, keep Applied
    if "Applied Mathematics" in majors and "Mathematics" in majors:
        majors.remove("Mathematics")

    return majors if majors else ["Mathematics"]


def _parse_internships(text: str) -> list[dict]:
    """Parse internship info from profile text into work_experience entries."""
    internships: list[dict] = []

    # "Internships: unknown" / "not disclosed"
    intern_line = ""
    for line in text.split("\n"):
        if re.search(r"internship", line, re.IGNORECASE):
            intern_line = line
            break

    if not intern_line:
        return []

    lower = intern_line.lower()
    if "unknown" in lower or "not disclosed" in lower:
        return []

    # Parse count: "3x:" or "5x" or "3x —"
    count_m = re.search(r"(\d+)x", lower)
    count = int(count_m.group(1)) if count_m else 0

    # If no explicit count, count colons/semicolons/numbered items
    if count == 0:
        # Count entries separated by ; or numbered (1) (2) etc
        entries = re.split(r"[;]|\(\d+\)", intern_line)
        count = max(len([e for e in entries if e.strip()]) - 1, 1)

    # Detect quant keywords
    is_quant = any(kw in lower for kw in [
        "quant", "trading", "buy-side", "hf", "hedge fund",
        "citadel", "jane street", "two sigma", "de shaw",
        "goldman", "morgan stanley", "jpmorgan",
    ])
    is_us = "us" in lower or "united states" in lower
    is_china = "china" in lower or "domestic" in lower or "中" in lower

    # Try to parse individual entries if described
    # Patterns like "1 buy-side quant, 2 non-quant finance"
    parts = re.split(r"[;,]", intern_line.split(":", 1)[-1] if ":" in intern_line else intern_line)

    for i in range(count):
        entry: dict = {"type": "internship", "duration_months": 3}

        if i < len(parts):
            part = parts[i].lower().strip()
            if any(kw in part for kw in ["quant", "buy-side", "hf", "hedge", "qr", "factor", "hft"]):
                entry["title"] = "Quant Research Intern"
                entry["company"] = "Quant firm"
                entry["description"] = "Quantitative research internship"
            elif any(kw in part for kw in ["goldman", "morgan stanley", "jpmorgan", "jp morgan", "bank", "ib", "ibd"]):
                entry["title"] = "Investment Banking Intern"
                entry["company"] = "Investment bank"
                entry["description"] = "Investment banking internship"
            elif any(kw in part for kw in ["tech", "ds", "data", "ml", "ai", "algo"]):
                entry["title"] = "Data Science / Tech Intern"
                entry["company"] = "Tech company"
                entry["description"] = "Technology / data science internship"
            elif any(kw in part for kw in ["pe", "am", "asset"]):
                entry["title"] = "Finance Intern"
                entry["company"] = "PE / Asset Management firm"
                entry["description"] = "Private equity or asset management"
            elif "finance" in part or "金融" in part:
                entry["title"] = "Finance Intern"
                entry["company"] = "Finance company"
                entry["description"] = "Finance internship"
            else:
                entry["title"] = "Intern"
                entry["company"] = "Company"
                entry["description"] = "Internship"

            # Country detection per part
            if "us" in part or "united states" in part:
                entry["country"] = "us"
            elif "china" in part or "domestic" in part or "cn" in part or "hk" in part:
                entry["country"] = "china"
            elif is_us:
                entry["country"] = "us"
            elif is_china:
                entry["country"] = "china"
            else:
                entry["country"] = "china"  # default for international applicants
        else:
            # Fallback generic entry
            if is_quant:
                entry.update({"title": "Quant Intern", "company": "Quant firm",
                              "description": "Quantitative internship",
                              "country": "china"})
            else:
                entry.update({"title": "Finance Intern", "company": "Finance company",
                              "description": "Finance internship",
                              "country": "china"})

        internships.append(entry)

    return internships


def _parse_research(text: str) -> list[dict]:
    """Parse research/project info into projects entries."""
    projects: list[dict] = []
    text_lower = text.lower()

    # Find research line
    research_line = ""
    for line in text.split("\n"):
        if re.search(r"research|paper|publication|科研|项目", line, re.IGNORECASE):
            research_line = line
            break

    if not research_line:
        return []

    lower = research_line.lower()

    # No research
    if any(neg in lower for neg in ["none", "not disclosed", "unknown", "not mentioned",
                                     "no paper", "一点科研经历也没"]):
        # Check if there's still some research despite negative language
        if "no paper" in lower and ("campus" in lower or "research" in lower):
            # Has research but no paper
            projects.append({
                "name": "Research project",
                "description": "Campus research (no publication)",
                "has_paper": False,
            })
            return projects
        if "一点科研经历也没" in lower:
            return []
        if "none" in lower:
            return []
        return []

    # Has publication
    has_paper = any(kw in lower for kw in [
        "published paper", "publication", "paper",
        "first-author", "first author", "journal",
        "conference", "sci", "ei", "abs",
    ])

    # Count research projects
    count_m = re.search(r"(\d+)x", lower)
    count = int(count_m.group(1)) if count_m else 1

    if has_paper:
        projects.append({
            "name": "Research publication",
            "description": "Published research paper",
            "has_paper": True,
        })
        # Add remaining as non-paper research
        for i in range(count - 1):
            projects.append({
                "name": f"Research project {i+2}",
                "description": "Campus research project",
                "has_paper": False,
            })
    else:
        # Research without paper
        for i in range(count):
            projects.append({
                "name": f"Research project {i+1}",
                "description": "Campus research project",
                "has_paper": False,
            })

    return projects


def _parse_is_international(text: str) -> bool:
    """Default True for most applicants."""
    lower = text.lower()
    if "international: no" in lower or "international: false" in lower:
        return False
    return True


def build_profile_yaml(issue_number: int, profile_text: str) -> str:
    """Build a minimal profile YAML from parsed profile text fields."""
    gpa = _parse_gpa(profile_text)
    gre_quant = _parse_gre_quant(profile_text)
    uni_name, years_us = _parse_university(profile_text)
    majors = _parse_majors(profile_text)
    internships = _parse_internships(profile_text)
    projects = _parse_research(profile_text)
    is_intl = _parse_is_international(profile_text)

    profile: dict = {
        "personal": {
            "name": f"Applicant (issue #{issue_number})",
            "university": uni_name,
            "majors": majors,
            "gpa": gpa or 3.5,
            "is_international": is_intl,
            "years_at_us_institution": years_us,
        },
    }

    if gre_quant:
        profile["test_scores"] = {"gre_quant": gre_quant}

    if internships:
        profile["work_experience"] = internships

    if projects:
        profile["projects"] = projects

    return yaml.dump(profile, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Prediction engine
# ---------------------------------------------------------------------------

def run_predictions(profile_path: str) -> list[dict]:
    """Run ensemble predictions for focused programs. Returns sorted results."""
    _build_prog_map()

    profile = load_profile(profile_path)
    gre_quant = None
    if profile.test_scores:
        gre_quant = getattr(profile.test_scores, "gre_quant", None)

    results: list[dict] = []
    for prog in load_all_programs():
        if prog.id not in _FOCUSED:
            continue
        pred = predict_ensemble(prog.id, profile.gpa, gre_quant, profile)
        if pred is None:
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
            "full_name": prog.full_name or prog.name,
            "university": prog.university,
            "prob": prob,
            "prob_low": pred.prob_low,
            "prob_high": pred.prob_high,
            "category": cat,
        })

    results.sort(key=lambda x: -x["prob"])
    return results


def format_prediction_section(results: list[dict], profile_path: str) -> str:
    """Format the ## Prediction Results markdown section."""
    lines: list[str] = ["## Prediction Results", ""]
    for r in results:
        pct = round(r["prob"] * 100)
        lo = round(r["prob_low"] * 100)
        hi = round(r["prob_high"] * 100)
        lines.append(
            f"- {r['full_name']} ({r['university']}): "
            f"{pct}% [{lo}%-{hi}%] — {r['category']}"
        )

    n_reach = sum(1 for r in results if r["category"] == "reach")
    n_target = sum(1 for r in results if r["category"] == "target")
    n_safety = sum(1 for r in results if r["category"] == "safety")
    n_total = len(results)

    # Relative profile path for display
    rel_path = str(Path(profile_path).relative_to(_ROOT))

    lines.append("")
    lines.append(
        f"*Command: `quantpath predict --profile {rel_path}` "
        f"(GPBoost v2 — retrained 2026-04-02). "
        f"{n_total} programs — {n_reach} reach / {n_target} target / {n_safety} safety.*"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Issue body surgery — replace only ## Prediction Results
# ---------------------------------------------------------------------------

def replace_prediction_section(body: str, new_section: str) -> str:
    """Replace the ## Prediction Results block in the issue body.

    Preserves everything before it (## Profile) and everything after it
    (## Actual Outcomes or ## Actual Outcomes (please update later!)).
    """
    lines = body.split("\n")

    # Find section boundaries
    pred_start = None
    pred_end = None

    for i, line in enumerate(lines):
        if line.startswith("## Prediction Results"):
            pred_start = i
        elif pred_start is not None and line.startswith("## Actual Outcomes"):
            pred_end = i
            break

    if pred_start is None:
        # No prediction section found — append before Actual Outcomes
        for i, line in enumerate(lines):
            if line.startswith("## Actual Outcomes"):
                pred_start = i
                pred_end = i
                break
        if pred_start is None:
            # No Actual Outcomes either — append at end
            return body + "\n\n" + new_section

    # Build new body: before + new prediction + after
    before = lines[:pred_start]
    after = lines[pred_end:]

    # Ensure blank line between sections
    new_lines = before
    if new_lines and new_lines[-1].strip():
        new_lines.append("")
    new_lines.append(new_section)
    new_lines.append("")
    new_lines.extend(after)

    return "\n".join(new_lines)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def process_issue(number: int) -> None:
    """Process a single issue: generate YAML if needed, predict, update."""
    print(f"\n{'='*60}")
    print(f"  Processing issue #{number}")
    print(f"{'='*60}")

    # Step 1: Determine / create profile YAML
    if number in _ISSUE_YAML_MAP:
        # Issues #25–#33 have existing YAML files
        yaml_name = _ISSUE_YAML_MAP[number]
        profile_path = str(PROFILES_DIR / yaml_name)
        print(f"  Using existing YAML: {yaml_name}")
    else:
        # Issues #3–#24: parse profile section and generate YAML
        yaml_name = f"data_contrib_issue_{number}.yaml"
        profile_path = str(PROFILES_DIR / yaml_name)

        print(f"  Fetching issue body...")
        body = _gh_issue_body(number)
        profile_text = _extract_profile_section(body)

        if not profile_text:
            print(f"  WARNING: No ## Profile section found in issue #{number}, skipping.")
            return

        print(f"  Parsing profile text...")
        yaml_content = build_profile_yaml(number, profile_text)

        # Write YAML
        with open(profile_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Auto-generated from issue #{number} profile section\n")
            fh.write(yaml_content)
        print(f"  Created: {yaml_name}")

    # Step 2: Run predictions
    print(f"  Running ensemble predictions...")
    try:
        results = run_predictions(profile_path)
    except Exception as exc:
        print(f"  ERROR: Prediction failed for issue #{number}: {exc}")
        return

    if not results:
        print(f"  WARNING: No predictions generated for issue #{number}, skipping.")
        return

    n_reach = sum(1 for r in results if r["category"] == "reach")
    n_target = sum(1 for r in results if r["category"] == "target")
    n_safety = sum(1 for r in results if r["category"] == "safety")
    print(f"  Results: {len(results)} programs — {n_reach} reach / {n_target} target / {n_safety} safety")

    # Step 3: Format new prediction section
    new_pred = format_prediction_section(results, profile_path)

    # Step 4: Update issue body
    print(f"  Fetching current issue body...")
    current_body = _gh_issue_body(number)

    new_body = replace_prediction_section(current_body, new_pred)

    print(f"  Updating issue #{number}...")
    try:
        _gh_issue_edit(number, new_body)
        print(f"  DONE: Issue #{number} updated successfully.")
    except subprocess.CalledProcessError as exc:
        print(f"  ERROR: Failed to update issue #{number}: {exc.stderr}")


def main() -> None:
    print("=" * 60)
    print("  QuantPath Batch Re-Prediction (GPBoost v2 retrained)")
    print("  Issues: #3–#33 on MasterAgentAI/QuantPath")
    print("=" * 60)

    # Verify gh CLI is available
    try:
        subprocess.run(["gh", "auth", "status"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: GitHub CLI (gh) not authenticated. Run 'gh auth login' first.")
        sys.exit(1)

    # Pre-load programs and model
    _build_prog_map()
    print(f"\nLoaded {len(_prog_map)} programs.")
    print(f"Focused programs: {len(_FOCUSED)}")

    # Process all 31 issues
    success = 0
    failed = 0
    for number in range(3, 34):
        try:
            process_issue(number)
            success += 1
        except Exception as exc:
            print(f"  UNHANDLED ERROR on issue #{number}: {exc}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  COMPLETE: {success} updated, {failed} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
