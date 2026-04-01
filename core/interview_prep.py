# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""Interview preparation module for QuantPath.

Loads and queries a structured database of MFE interview questions, with
filtering by category, difficulty, and target program.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PACKAGE_ROOT / "data"
_QUESTIONS_FILE = _DATA_DIR / "interview_questions.yaml"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Question:
    """A single interview question."""

    question: str
    difficulty: str
    topics: list[str]
    hint: str
    solution: str
    programs: list[str]
    category_id: str = ""
    category_name: str = ""


@dataclass
class Category:
    """A category grouping related questions."""

    id: str
    name: str
    questions: list[Question]


@dataclass
class QuizResult:
    """Result of a completed quiz session."""

    total: int
    answered: int
    categories_covered: list[str]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_questions(path: Optional[str | Path] = None) -> list[Category]:
    """Load interview questions from the YAML database.

    Parameters
    ----------
    path:
        Path to the YAML file.  Defaults to ``data/interview_questions.yaml``
        relative to the package root.

    Returns
    -------
    list[Category]
        Parsed categories with their questions.

    Raises
    ------
    FileNotFoundError
        If the questions file does not exist.
    """
    filepath = Path(path) if path else _QUESTIONS_FILE
    if not filepath.exists():
        raise FileNotFoundError(f"Interview questions file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    categories: list[Category] = []
    for cat_raw in raw.get("categories", []):
        cat_id = cat_raw.get("id", "")
        cat_name = cat_raw.get("name", "")
        questions: list[Question] = []

        for q_raw in cat_raw.get("questions", []):
            questions.append(
                Question(
                    question=q_raw.get("question", ""),
                    difficulty=q_raw.get("difficulty", "medium"),
                    topics=q_raw.get("topics", []),
                    hint=q_raw.get("hint", ""),
                    solution=q_raw.get("solution", ""),
                    programs=q_raw.get("programs", []),
                    category_id=cat_id,
                    category_name=cat_name,
                )
            )

        categories.append(Category(id=cat_id, name=cat_name, questions=questions))

    return categories


def _all_questions(categories: list[Category]) -> list[Question]:
    """Flatten all questions from every category."""
    return [q for cat in categories for q in cat.questions]


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_questions_by_category(
    category_id: str,
    categories: Optional[list[Category]] = None,
) -> list[Question]:
    """Return all questions belonging to *category_id*.

    Parameters
    ----------
    category_id:
        The category identifier (e.g. ``"probability"``).
    categories:
        Pre-loaded categories.  If ``None``, loads from disk.
    """
    if categories is None:
        categories = load_questions()

    for cat in categories:
        if cat.id == category_id:
            return list(cat.questions)
    return []


def get_questions_by_difficulty(
    difficulty: str,
    categories: Optional[list[Category]] = None,
) -> list[Question]:
    """Return all questions matching *difficulty* (``easy``, ``medium``, ``hard``).

    Parameters
    ----------
    difficulty:
        One of ``"easy"``, ``"medium"``, ``"hard"``.
    categories:
        Pre-loaded categories.  If ``None``, loads from disk.
    """
    if categories is None:
        categories = load_questions()

    difficulty = difficulty.lower()
    return [q for q in _all_questions(categories) if q.difficulty == difficulty]


def get_questions_for_program(
    program_id: str,
    categories: Optional[list[Category]] = None,
) -> list[Question]:
    """Return all questions tagged for *program_id*.

    Parameters
    ----------
    program_id:
        A programme identifier such as ``"baruch-mfe"`` or ``"cmu-mscf"``.
    categories:
        Pre-loaded categories.  If ``None``, loads from disk.
    """
    if categories is None:
        categories = load_questions()

    return [q for q in _all_questions(categories) if program_id in q.programs]


def get_random_quiz(
    n: int = 10,
    categories_filter: Optional[list[str]] = None,
    difficulty: Optional[str] = None,
    categories: Optional[list[Category]] = None,
) -> list[Question]:
    """Select a random set of questions for a quiz.

    Parameters
    ----------
    n:
        Number of questions to return.
    categories_filter:
        Limit to these category IDs.  ``None`` means all categories.
    difficulty:
        Limit to this difficulty level.  ``None`` means all levels.
    categories:
        Pre-loaded categories.  If ``None``, loads from disk.

    Returns
    -------
    list[Question]
        Up to *n* randomly selected questions.
    """
    if categories is None:
        categories = load_questions()

    pool = _all_questions(categories)

    if categories_filter:
        pool = [q for q in pool if q.category_id in categories_filter]

    if difficulty:
        difficulty = difficulty.lower()
        pool = [q for q in pool if q.difficulty == difficulty]

    if not pool:
        return []

    return random.sample(pool, min(n, len(pool)))


# ---------------------------------------------------------------------------
# QuizSession
# ---------------------------------------------------------------------------


@dataclass
class QuizSession:
    """Tracks a quiz session, presenting questions one at a time.

    Usage
    -----
    >>> session = QuizSession(questions)
    >>> while session.has_next():
    ...     q = session.next_question()
    ...     # display q.question, q.hint, q.solution
    ...     session.mark_answered()
    >>> result = session.finish()
    """

    questions: list[Question]
    _index: int = field(default=0, init=False, repr=False)
    _answered: int = field(default=0, init=False, repr=False)

    def has_next(self) -> bool:
        """Return ``True`` if there are more questions."""
        return self._index < len(self.questions)

    def next_question(self) -> Question:
        """Return the current question and advance the pointer.

        Raises
        ------
        StopIteration
            If all questions have been presented.
        """
        if not self.has_next():
            raise StopIteration("No more questions in this session.")
        q = self.questions[self._index]
        self._index += 1
        return q

    def current_progress(self) -> tuple[int, int]:
        """Return ``(current_number, total)``."""
        return self._index, len(self.questions)

    def mark_answered(self) -> None:
        """Increment the answered counter."""
        self._answered += 1

    def finish(self) -> QuizResult:
        """Finalise the session and return the result summary."""
        cats = sorted({q.category_name for q in self.questions})
        return QuizResult(
            total=len(self.questions),
            answered=self._answered,
            categories_covered=cats,
        )
