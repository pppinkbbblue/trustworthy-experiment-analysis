"""Run the benchmark: compare LLM-only vs Compute-Validate-Explain.

Usage:
    # plumbing check only (no real results):
    python -m experiments.run_benchmark --llm mock --n 20

    # real results (requires the relevant SDK + API key in the environment):
    OPENAI_API_KEY=... python -m experiments.run_benchmark --llm openai:gpt-4o-mini --n 50
    ANTHROPIC_API_KEY=... python -m experiments.run_benchmark --llm anthropic:claude-3-5-sonnet-latest --n 50

Results are written to results/ as JSON. Numbers produced with --llm mock are
tagged as such and MUST NOT be used in the paper -- the mock cannot exhibit
hallucination and does not represent a real model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cve.data import generate_dataset
from cve.llm import get_client
from cve.pipeline import run_cve, run_llm_only
from experiments.metrics import score, aggregate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="mock", help="mock | openai:<model> | anthropic:<model>")
    ap.add_argument("--n", type=int, default=20, help="number of synthetic experiments")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    client = get_client(args.llm)
    dataset = generate_dataset(n=args.n, seed=args.seed)

    cve_scores, llm_scores = [], []
    per_experiment = []
    for exp in dataset:
        c = run_cve(client, exp)
        l = run_llm_only(client, exp)
        sc, sl = score(exp, c), score(exp, l)
        cve_scores.append(sc)
        llm_scores.append(sl)
        per_experiment.append({
            "experiment_id": exp.experiment_id,
            "cve": vars(sc),
            "llm_only": vars(sl),
        })

    summary = {
        "llm": client.name,
        "is_mock": args.llm == "mock",
        "n": args.n,
        "seed": args.seed,
        "aggregate": {
            "llm_only": aggregate(llm_scores),
            "cve": aggregate(cve_scores),
        },
    }

    os.makedirs(args.out, exist_ok=True)
    tag = client.name.replace(":", "_").replace("/", "_")
    with open(os.path.join(args.out, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, f"per_experiment_{tag}.json"), "w") as f:
        json.dump(per_experiment, f, indent=2)

    print(json.dumps(summary, indent=2))
    if summary["is_mock"]:
        print("\n[warning] mock backend: these numbers are for plumbing only, "
              "not for the paper.", file=sys.stderr)


if __name__ == "__main__":
    main()
