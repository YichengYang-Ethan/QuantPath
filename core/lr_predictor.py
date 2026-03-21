"""Logistic regression admission probability predictor.

Loads pre-trained per-program models from data/models/admission_models.json
and provides P(admission) predictions given GPA and GRE Quant.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

_MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "admission_models.json"
_models: dict | None = None


def _load_models() -> dict:
    global _models
    if _models is None:
        if _MODEL_PATH.exists():
            with _MODEL_PATH.open(encoding="utf-8") as f:
                _models = json.load(f)
        else:
            _models = {}
    return _models


def predict_prob(
    program_id: str,
    gpa: Optional[float],
    gre: Optional[float],
) -> Optional[float]:
    """Return P(admission) for a given program, GPA (4-scale), and GRE Quant.

    Returns None if the program has no trained model or inputs are missing.
    """
    models = _load_models()
    m = models.get(program_id)
    if not m:
        return None
    if gpa is None and gre is None:
        return None

    means = m["means"]
    stds = m["stds"]
    coef = m["coef"]
    intercept = m["intercept"]

    # Use model mean as fallback for missing values
    gpa_val = gpa if gpa is not None else means[0]
    gre_val = gre if gre is not None else means[1]

    z_gpa = (gpa_val - means[0]) / stds[0]
    z_gre = (gre_val - means[1]) / stds[1]

    logit = coef[0] * z_gpa + coef[1] * z_gre + intercept
    prob = 1.0 / (1.0 + math.exp(-logit))
    return round(prob, 4)


def get_model_stats(program_id: str) -> Optional[dict]:
    """Return model stats (n, accept_rate, AUC, GPA/GRE percentiles) for a program."""
    return _load_models().get(program_id)


def has_model(program_id: str) -> bool:
    return program_id in _load_models()
