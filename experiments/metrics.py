"""Evaluation metrics for comparing conditions against known ground truth."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from cve.compute import decision
from cve.data import SyntheticExperiment
from cve.pipeline import ConditionResult

_NUMERIC_FIELDS = [
    "treatment_rate", "control_rate", "relative_lift_pct", "prob_treatment_better",
]


def _close(a: float, b: float, rel_tol: float = 0.02, abs_tol: float = 0.5) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= abs_tol or (b != 0 and abs(a - b) / abs(b) <= rel_tol)


def ground_truth_maps(exp: SyntheticExperiment, threshold: float = 0.95):
    gt_numbers: Dict[str, Dict[str, float]] = {}
    gt_decision: Dict[str, str] = {}
    for r in exp.ground_truth():
        gt_numbers[r.treatment] = {
            "treatment_rate": r.treatment_rate,
            "control_rate": r.control_rate,
            "relative_lift_pct": r.relative_lift_pct,
            "prob_treatment_better": r.prob_treatment_better,
        }
        gt_decision[r.treatment] = decision(r, threshold)
    return gt_numbers, gt_decision


@dataclass
class ExperimentScore:
    experiment_id: str
    condition: str
    numeric_fields_total: int
    numeric_fields_wrong: int
    decisions_total: int
    decisions_wrong: int
    hallucination_rate: float
    traceability: float

    @property
    def numeric_error_rate(self) -> float:
        return self.numeric_fields_wrong / self.numeric_fields_total if self.numeric_fields_total else 0.0

    @property
    def decision_error_rate(self) -> float:
        return self.decisions_wrong / self.decisions_total if self.decisions_total else 0.0


def score(exp: SyntheticExperiment, cond: ConditionResult, threshold: float = 0.95) -> ExperimentScore:
    gt_numbers, gt_decision = ground_truth_maps(exp, threshold)

    n_total = n_wrong = 0
    for tname, gt in gt_numbers.items():
        reported = cond.reported_numbers.get(tname, {})
        for field in _NUMERIC_FIELDS:
            n_total += 1
            if not _close(reported.get(field), gt[field]):
                n_wrong += 1

    d_total = d_wrong = 0
    for tname, gt_dec in gt_decision.items():
        d_total += 1
        if cond.decisions.get(tname, "") != gt_dec:
            d_wrong += 1

    hr = cond.grounding.hallucination_rate if cond.grounding else 0.0
    tr = cond.grounding.traceability if cond.grounding else 1.0

    return ExperimentScore(
        experiment_id=exp.experiment_id,
        condition=cond.condition,
        numeric_fields_total=n_total,
        numeric_fields_wrong=n_wrong,
        decisions_total=d_total,
        decisions_wrong=d_wrong,
        hallucination_rate=hr,
        traceability=tr,
    )


def aggregate(scores: List[ExperimentScore]) -> Dict[str, float]:
    if not scores:
        return {}
    n = len(scores)
    return {
        "n_experiments": n,
        "numeric_error_rate": sum(s.numeric_error_rate for s in scores) / n,
        "decision_error_rate": sum(s.decision_error_rate for s in scores) / n,
        "hallucination_rate": sum(s.hallucination_rate for s in scores) / n,
        "traceability": sum(s.traceability for s in scores) / n,
    }
