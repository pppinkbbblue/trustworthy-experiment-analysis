"""Explanation layer and the LLM-only baseline.

Two ways to get a written analysis of an experiment:

* `llm_only_analysis` -- the baseline. The model is handed the raw arm counts
  and asked to BOTH compute the statistics and explain them. This is where a
  real model can miscompute or fabricate numbers.

* `cve_explain` -- the proposed approach. The deterministic layer has already
  computed every statistic; the model is handed those results and constrained
  to interpret them only, never to produce a number that is not present.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List

from .compute import MetricResult
from .data import SyntheticExperiment
from .llm import LLMClient


# --- prompts ---------------------------------------------------------------

_LLM_ONLY_SYSTEM = (
    "You are a data analyst. You will be given the raw counts of an A/B "
    "experiment. Compute the statistics yourself and summarize the outcome."
)

_CVE_SYSTEM = (
    "You are an analyst who INTERPRETS already-computed statistics. You must "
    "not compute, estimate, or introduce any number that is not present in the "
    "provided results. Refer to the metric_id when discussing a result. If a "
    "value is not provided, say it is not available rather than guessing."
)


def _llm_only_user(exp: SyntheticExperiment) -> str:
    lines = [
        f"Experiment {exp.experiment_id}, metric '{exp.metric_name}'.",
        f"control: users={exp.control.users}, conversions={exp.control.conversions}",
    ]
    for t in exp.treatments:
        lines.append(f"{t.name}: users={t.users}, conversions={t.conversions}")
    lines.append(
        "\nFor EACH treatment vs control, return the conversion rates, the "
        "relative lift in percent, the probability the treatment is better "
        "(0-1), and a decision in {positive, negative, inconclusive} using a "
        "0.95 threshold. Respond with STRICT JSON of the form:\n"
        '{"analyses":[{"treatment":"T1","treatment_rate":..,"control_rate":..,'
        '"relative_lift_pct":..,"prob_treatment_better":..,"decision":".."}],'
        '"narrative":"..."}'
    )
    return "\n".join(lines)


def _facts_table(results: List[MetricResult]) -> str:
    rows = ["metric_id | treatment | treatment_rate | control_rate | "
            "relative_lift_pct | prob_treatment_better | p_value"]
    for r in results:
        rows.append(
            f"{r.metric_id} | {r.treatment} | {r.treatment_rate} | "
            f"{r.control_rate} | {r.relative_lift_pct} | "
            f"{r.prob_treatment_better} | {r.p_value}"
        )
    return "\n".join(rows)


def _cve_user(results: List[MetricResult]) -> str:
    return (
        "Here are the computed results. Write a short, plain-English summary "
        "for a decision-maker. Use only the numbers below.\n\n"
        + _facts_table(results)
    )


# --- JSON extraction -------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Best-effort parse of a JSON object from model output."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            return {}
    return {}


# --- public API ------------------------------------------------------------

def llm_only_analysis(client: LLMClient, exp: SyntheticExperiment) -> dict:
    """Baseline: the model computes and explains. Returns parsed dict + raw."""
    raw = client.complete(_LLM_ONLY_SYSTEM, _llm_only_user(exp))
    parsed = _extract_json(raw)
    return {"raw": raw, "parsed": parsed, "narrative": parsed.get("narrative", raw)}


def cve_explain(client: LLMClient, results: List[MetricResult]) -> str:
    """Proposed: the model only interprets pre-computed numbers."""
    return client.complete(_CVE_SYSTEM, _cve_user(results))
