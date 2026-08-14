"""Pipelines: the proposed Compute-Validate-Explain flow and the baseline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .compute import MetricResult, decision
from .data import SyntheticExperiment
from .explain import cve_explain, llm_only_analysis
from .llm import LLMClient
from .validate import GroundingReport, validate_grounding, redact_ungrounded


def _all_facts(results: List[MetricResult]) -> Dict[str, float]:
    facts: Dict[str, float] = {}
    for r in results:
        for k, v in r.facts().items():
            facts[f"{r.metric_id}:{k}"] = v
    return facts


@dataclass
class ConditionResult:
    condition: str
    experiment_id: str
    narrative: str
    reported_numbers: Dict[str, Dict[str, float]] = field(default_factory=dict)
    decisions: Dict[str, str] = field(default_factory=dict)
    grounding: Optional[GroundingReport] = None
    computed: List[MetricResult] = field(default_factory=list)


def run_cve(client: LLMClient, exp: SyntheticExperiment, threshold: float = 0.95) -> ConditionResult:
    """Compute -> Validate -> Explain. Numbers come from code; the model only writes prose."""
    results = exp.ground_truth()                    # Compute (deterministic)
    narrative = cve_explain(client, results)        # Explain (LLM, constrained)
    facts = _all_facts(results)
    report = validate_grounding(narrative, facts)   # Validate (grounding)
    safe_narrative = redact_ungrounded(narrative, report)
    decisions = {r.treatment: decision(r, threshold) for r in results}
    reported = {
        r.treatment: {
            "treatment_rate": r.treatment_rate,
            "control_rate": r.control_rate,
            "relative_lift_pct": r.relative_lift_pct,
            "prob_treatment_better": r.prob_treatment_better,
        }
        for r in results
    }
    return ConditionResult(
        condition="cve",
        experiment_id=exp.experiment_id,
        narrative=safe_narrative,
        reported_numbers=reported,
        decisions=decisions,
        grounding=report,
        computed=results,
    )


def run_llm_only(client: LLMClient, exp: SyntheticExperiment) -> ConditionResult:
    """Baseline: the model computes the numbers and explains them."""
    out = llm_only_analysis(client, exp)
    parsed = out["parsed"]
    reported: Dict[str, Dict[str, float]] = {}
    decisions: Dict[str, str] = {}
    for a in parsed.get("analyses", []):
        name = a.get("treatment")
        if not name:
            continue
        reported[name] = {
            "treatment_rate": a.get("treatment_rate"),
            "control_rate": a.get("control_rate"),
            "relative_lift_pct": a.get("relative_lift_pct"),
            "prob_treatment_better": a.get("prob_treatment_better"),
        }
        decisions[name] = a.get("decision", "")
    # Grounding is checked against the TRUE computed facts, so numbers the model
    # invented or miscomputed are flagged.
    facts = _all_facts(exp.ground_truth())
    report = validate_grounding(out["narrative"], facts)
    return ConditionResult(
        condition="llm_only",
        experiment_id=exp.experiment_id,
        narrative=out["narrative"],
        reported_numbers=reported,
        decisions=decisions,
        grounding=report,
        computed=exp.ground_truth(),
    )
