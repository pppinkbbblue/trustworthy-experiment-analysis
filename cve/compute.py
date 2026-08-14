"""Deterministic computation layer.

Every statistic reported by the system is produced here, in code, the same way
every time. The language model is never allowed to compute or alter these
numbers -- it only explains them. This module has no LLM dependency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via the error function (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class Arm:
    """One experiment arm (e.g., control or a treatment)."""
    name: str
    users: int
    conversions: int

    @property
    def rate(self) -> float:
        return self.conversions / self.users if self.users else 0.0


@dataclass(frozen=True)
class MetricResult:
    """Deterministic result comparing a treatment arm against control.

    All fields are computed; none are model-generated. `metric_id` gives every
    number a stable, quotable handle used later for verbatim grounding.
    """
    metric_id: str
    treatment: str
    control: str
    treatment_rate: float
    control_rate: float
    absolute_lift: float          # treatment_rate - control_rate
    relative_lift_pct: float      # 100 * (t - c) / c
    standard_error: float
    z_score: float
    p_value: float                # two-sided, two-proportion z-test
    prob_treatment_better: float  # P(treatment rate > control rate), normal approx
    treatment_users: int
    control_users: int

    def rounded(self, ndigits: int = 4) -> "MetricResult":
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, ndigits)
        return MetricResult(**d)

    def facts(self) -> Dict[str, float]:
        """The quotable numeric facts, keyed for grounding/validation."""
        return {
            "treatment_rate": self.treatment_rate,
            "control_rate": self.control_rate,
            "absolute_lift": self.absolute_lift,
            "relative_lift_pct": self.relative_lift_pct,
            "standard_error": self.standard_error,
            "z_score": self.z_score,
            "p_value": self.p_value,
            "prob_treatment_better": self.prob_treatment_better,
            "treatment_users": float(self.treatment_users),
            "control_users": float(self.control_users),
        }


def compare_arms(control: Arm, treatment: Arm, metric_id: str) -> MetricResult:
    """Two-proportion comparison. Deterministic and pure."""
    pc, pt = control.rate, treatment.rate
    abs_lift = pt - pc
    rel_lift = (abs_lift / pc * 100.0) if pc > 0 else float("nan")

    # Standard error of the difference in proportions.
    se = math.sqrt(
        (pt * (1 - pt) / treatment.users if treatment.users else 0.0)
        + (pc * (1 - pc) / control.users if control.users else 0.0)
    )

    # Pooled two-proportion z-test (two-sided).
    pooled = (control.conversions + treatment.conversions) / (
        control.users + treatment.users
    )
    se_pooled = math.sqrt(
        pooled * (1 - pooled) * (1 / control.users + 1 / treatment.users)
    ) if control.users and treatment.users else 0.0
    z = (abs_lift / se_pooled) if se_pooled > 0 else 0.0
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))

    # Normal-approximation probability that the treatment rate exceeds control.
    prob_better = _normal_cdf(abs_lift / se) if se > 0 else (
        1.0 if abs_lift > 0 else 0.0 if abs_lift < 0 else 0.5
    )

    return MetricResult(
        metric_id=metric_id,
        treatment=treatment.name,
        control=control.name,
        treatment_rate=pt,
        control_rate=pc,
        absolute_lift=abs_lift,
        relative_lift_pct=rel_lift,
        standard_error=se,
        z_score=z,
        p_value=p,
        prob_treatment_better=prob_better,
        treatment_users=treatment.users,
        control_users=control.users,
    ).rounded()


def analyze_experiment(
    control: Arm, treatments: List[Arm], metric_name: str = "conversion"
) -> List[MetricResult]:
    """Compare each treatment arm against the shared control arm."""
    return [
        compare_arms(control, t, metric_id=f"{metric_name}:{t.name}")
        for t in treatments
    ]


def decision(result: MetricResult, threshold: float = 0.95) -> str:
    """A deterministic launch call from the probability the treatment wins."""
    if result.prob_treatment_better >= threshold:
        return "positive"
    if result.prob_treatment_better <= (1 - threshold):
        return "negative"
    return "inconclusive"
