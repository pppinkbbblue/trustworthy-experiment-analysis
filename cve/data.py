"""Synthetic experiment data with known ground truth.

Using synthetic data is deliberate: because we set the true conversion rates
ourselves, we know the exact correct answer for every statistic. That lets us
measure numerical error precisely, which is impossible when ground truth is
unknown. No proprietary or real customer data is used anywhere in this project.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List

from .compute import Arm, analyze_experiment, MetricResult


@dataclass(frozen=True)
class SyntheticExperiment:
    experiment_id: str
    metric_name: str
    control: Arm
    treatments: List[Arm]
    true_control_rate: float
    true_treatment_rates: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "metric_name": self.metric_name,
            "control": asdict(self.control),
            "treatments": [asdict(t) for t in self.treatments],
            "true_control_rate": self.true_control_rate,
            "true_treatment_rates": self.true_treatment_rates,
        }

    def ground_truth(self) -> List[MetricResult]:
        """The correct deterministic results for this experiment."""
        return analyze_experiment(self.control, self.treatments, self.metric_name)


def generate_experiment(
    rng: random.Random,
    experiment_id: str,
    n_treatments: int = 1,
    min_users: int = 2_000,
    max_users: int = 200_000,
    base_rate_range=(0.02, 0.40),
    effect_range=(-0.05, 0.08),
    metric_name: str = "conversion",
) -> SyntheticExperiment:
    """Draw true rates, then sample observed conversions from binomials."""
    control_rate = rng.uniform(*base_rate_range)
    control_users = rng.randint(min_users, max_users)
    control_conv = sum(1 for _ in range(control_users) if rng.random() < control_rate) \
        if control_users <= 5000 else round(
            control_users * control_rate
            + rng.gauss(0, (control_rate * (1 - control_rate) * control_users) ** 0.5)
        )
    control_conv = max(0, min(control_users, int(control_conv)))
    control = Arm(name="control", users=control_users, conversions=control_conv)

    treatments: List[Arm] = []
    true_treatment_rates: Dict[str, float] = {}
    for i in range(n_treatments):
        effect = rng.uniform(*effect_range)
        t_rate = max(0.001, min(0.999, control_rate + effect))
        t_users = rng.randint(min_users, max_users)
        t_conv = round(
            t_users * t_rate
            + rng.gauss(0, (t_rate * (1 - t_rate) * t_users) ** 0.5)
        )
        t_conv = max(0, min(t_users, int(t_conv)))
        name = f"T{i + 1}"
        treatments.append(Arm(name=name, users=t_users, conversions=t_conv))
        true_treatment_rates[name] = t_rate

    return SyntheticExperiment(
        experiment_id=experiment_id,
        metric_name=metric_name,
        control=control,
        treatments=treatments,
        true_control_rate=control_rate,
        true_treatment_rates=true_treatment_rates,
    )


def generate_dataset(
    n: int = 50, seed: int = 7, max_treatments: int = 3
) -> List[SyntheticExperiment]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        k = rng.randint(1, max_treatments)
        out.append(generate_experiment(rng, experiment_id=f"EXP_{i:03d}", n_treatments=k))
    return out


def save_dataset(experiments: List[SyntheticExperiment], path: str) -> None:
    with open(path, "w") as f:
        json.dump([e.to_dict() for e in experiments], f, indent=2)


def load_dataset(path: str) -> List[SyntheticExperiment]:
    with open(path) as f:
        raw = json.load(f)
    out = []
    for e in raw:
        out.append(
            SyntheticExperiment(
                experiment_id=e["experiment_id"],
                metric_name=e["metric_name"],
                control=Arm(**e["control"]),
                treatments=[Arm(**t) for t in e["treatments"]],
                true_control_rate=e["true_control_rate"],
                true_treatment_rates=e["true_treatment_rates"],
            )
        )
    return out
