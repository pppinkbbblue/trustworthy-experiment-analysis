"""Fill the paper's [RESULT] placeholders from a real benchmark summary.

Reads a results/summary_*.json produced by run_benchmark.py and writes
paper/paper_filled.md with every placeholder replaced. The template
(paper/paper.md) is left untouched.

Usage:
    python3 -m experiments.fill_paper                     # auto-pick newest non-mock summary
    python3 -m experiments.fill_paper --summary results/summary_anthropic_....json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _auto_summary() -> str:
    candidates = glob.glob(os.path.join(ROOT, "results", "summary_*.json"))
    real = []
    for c in candidates:
        try:
            with open(c) as f:
                if not json.load(f).get("is_mock", False):
                    real.append(c)
        except Exception:
            continue
    if not real:
        raise SystemExit(
            "No non-mock summary found in results/. Run a real model first, e.g.\n"
            "  ANTHROPIC_API_KEY=... python3 -m experiments.run_benchmark "
            "--llm anthropic:claude-3-5-sonnet-latest --n 50"
        )
    return max(real, key=os.path.getmtime)


def build_mapping(summary: dict) -> dict:
    agg = summary["aggregate"]
    m = {
        "n": str(summary.get("n", "")),
        "seed": str(summary.get("seed", "")),
        "llm": summary.get("llm", ""),
    }
    for cond in ("llm_only", "cve"):
        a = agg.get(cond, {})
        for key in ("numeric_error_rate", "decision_error_rate",
                    "hallucination_rate", "traceability"):
            if key in a:
                m[f"{cond}.{key}"] = _pct(a[key])
    return m


def _row(label: str, a: dict) -> str:
    def g(k):
        return _pct(a[k]) if k in a else "n/a"
    return (f"| {label} | {g('numeric_error_rate')} | {g('decision_error_rate')} "
            f"| {g('hallucination_rate')} | {g('traceability')} |")


def fill(text: str, summary: dict) -> str:
    mapping = build_mapping(summary)

    # Named placeholders: [RESULT: key]
    def repl_named(match):
        key = match.group(1).strip()
        if key not in mapping:
            raise SystemExit(f"No value for placeholder [RESULT: {key}]")
        return mapping[key]

    text = re.sub(r"\[RESULT:\s*([^\]]+)\]", repl_named, text)

    # Regenerate the two results-table data rows (they use bare [RESULT] cells).
    agg = summary["aggregate"]
    text = re.sub(r"^\|\s*LLM-only\b.*$", _row("LLM-only", agg.get("llm_only", {})),
                  text, flags=re.MULTILINE)
    text = re.sub(r"^\|\s*CVE\b.*$", _row("CVE", agg.get("cve", {})),
                  text, flags=re.MULTILINE)

    if "[RESULT" in text:
        leftover = re.findall(r"\[RESULT[^\]]*\]?", text)
        raise SystemExit(f"Unfilled placeholders remain: {leftover}")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=None)
    ap.add_argument("--template", default=os.path.join(ROOT, "paper", "paper.md"))
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "paper_filled.md"))
    args = ap.parse_args()

    summary_path = args.summary or _auto_summary()
    with open(summary_path) as f:
        summary = json.load(f)
    if summary.get("is_mock"):
        print("[warning] this summary is from the mock backend; do not use for the paper.",
              file=sys.stderr)

    with open(args.template) as f:
        text = f.read()
    filled = fill(text, summary)
    with open(args.out, "w") as f:
        f.write(filled)
    print(f"Filled paper written to {args.out} (from {os.path.basename(summary_path)}).")


if __name__ == "__main__":
    main()
