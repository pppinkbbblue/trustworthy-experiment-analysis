"""Validation layer: verbatim numerical grounding.

Given a narrative and the set of numbers the deterministic layer actually
produced, we check that every number appearing in the narrative traces back to
a computed value. Numbers that do not are flagged as ungrounded (hallucinated).
This is what turns "the model shouldn't invent numbers" into an enforceable,
measurable guarantee.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_numbers(text: str) -> List[float]:
    """Pull numeric tokens from text (ignoring years-only tokens is left to caller)."""
    out = []
    for m in _NUMBER_RE.finditer(text):
        try:
            out.append(float(m.group()))
        except ValueError:
            continue
    return out


def _allowed_representations(values: Iterable[float], ndigits: int = 2) -> Set[float]:
    """Build the set of acceptable numeric forms for grounding.

    For each computed value we accept the raw value, its rounded forms, and --
    for values that look like proportions in [0, 1] -- the percentage form
    (x * 100), since narratives often say "23%" for a 0.23 rate.
    """
    allowed: Set[float] = set()
    for v in values:
        if v is None:
            continue
        forms = {v, round(v, ndigits), round(v, 1), round(v, 0)}
        if -1.0 <= v <= 1.0:
            pct = v * 100.0
            forms |= {pct, round(pct, ndigits), round(pct, 1), round(pct, 0)}
        allowed |= forms
    return allowed


@dataclass
class GroundingReport:
    numbers_found: List[float]
    grounded: List[float]
    ungrounded: List[float]

    @property
    def hallucination_rate(self) -> float:
        n = len(self.numbers_found)
        return (len(self.ungrounded) / n) if n else 0.0

    @property
    def traceability(self) -> float:
        n = len(self.numbers_found)
        return (len(self.grounded) / n) if n else 1.0


def validate_grounding(
    narrative: str,
    computed_facts: Dict[str, float],
    rel_tol: float = 0.02,
    abs_tol: float = 0.5,
    ignore_below: float = None,
) -> GroundingReport:
    """Flag numbers in `narrative` not matching any computed fact within tolerance.

    rel_tol/abs_tol absorb rounding in the narrative. `ignore_below` can be used
    by callers to skip small integers if desired (default: check everything).
    """
    allowed = _allowed_representations(computed_facts.values())
    found = extract_numbers(narrative)
    grounded, ungrounded = [], []
    for num in found:
        if ignore_below is not None and abs(num) < ignore_below:
            grounded.append(num)
            continue
        ok = any(
            (abs(num - a) <= abs_tol) or (a != 0 and abs(num - a) / abs(a) <= rel_tol)
            for a in allowed
        )
        (grounded if ok else ungrounded).append(num)
    return GroundingReport(numbers_found=found, grounded=grounded, ungrounded=ungrounded)


def redact_ungrounded(narrative: str, report: GroundingReport) -> str:
    """Optionally mask ungrounded numbers so a flagged narrative fails safe."""
    text = narrative
    for num in report.ungrounded:
        token = f"{num:g}"
        text = text.replace(token, "[UNVERIFIED]")
    return text
