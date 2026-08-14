"""Compute-Validate-Explain (CVE): trustworthy LLM-assisted experiment analysis."""

from .compute import Arm, MetricResult, analyze_experiment, compare_arms, decision
from .data import SyntheticExperiment, generate_dataset, load_dataset, save_dataset
from .llm import get_client
from .pipeline import run_cve, run_llm_only, ConditionResult
from .validate import validate_grounding, GroundingReport

__all__ = [
    "Arm", "MetricResult", "analyze_experiment", "compare_arms", "decision",
    "SyntheticExperiment", "generate_dataset", "load_dataset", "save_dataset",
    "get_client", "run_cve", "run_llm_only", "ConditionResult",
    "validate_grounding", "GroundingReport",
]

__version__ = "0.1.0"
