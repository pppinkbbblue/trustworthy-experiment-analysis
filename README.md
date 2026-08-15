# Compute–Validate–Explain (CVE)

**Trustworthy, hallucination-resistant LLM-assisted analysis of experiments.**

Large language models are increasingly used to read and summarize quantitative
results. They are also prone to producing fluent, confident, and *wrong*
numbers — which is unacceptable when those numbers drive decisions. This project
implements and benchmarks a simple architecture that removes that failure mode
by construction:

1. **Compute** — a deterministic layer computes every statistic in code.
2. **Validate** — every number in the model's write-up is checked against the
   computed values; ungrounded numbers are flagged (and can be redacted).
3. **Explain** — the language model is restricted to *interpreting* the computed
   results in plain language. It never computes or invents a number.

The claim is not that "LLMs are bad." It is that for high-stakes analysis, the
model should interpret, and code should compute — and that this separation can
be *enforced and measured*, not just hoped for.

## Why it matters

Experiments and evidence-based decisions are central in domains where a wrong
conclusion is costly — clinical research, financial risk, and public-sector
program evaluation among them. As AI is adopted to speed up this analysis, the
reliability of AI output becomes a first-order concern. CVE is a small,
open, reproducible study of one concrete way to make it reliable.

## Install & run

No third-party runtime dependencies (standard library only). Tests use `pytest`.

```bash
python3 -m pytest -q                       # run the test suite
python3 -m experiments.run_benchmark --llm mock --n 20   # plumbing check
```

To produce **real** benchmark numbers, plug in a model (needs the SDK + an API
key in your environment):

```bash
pip install openai
OPENAI_API_KEY=... python3 -m experiments.run_benchmark --llm openai:gpt-4o-mini --n 50

pip install anthropic
ANTHROPIC_API_KEY=... python3 -m experiments.run_benchmark --llm anthropic:claude-3-5-sonnet-latest --n 50
```

> The `mock` backend is for exercising the code only. It cannot exhibit
> hallucination and does **not** represent a real model, so its numbers must not
> be reported as results.

## How it's evaluated

Data is **synthetic with known ground truth**: because we set the true rates, we
know the exact correct answer for every statistic, so numerical error is
measured precisely (impossible with real, unknown-truth data). Two conditions
are compared on the same experiments:

| Condition | Who computes the numbers | Who writes the summary |
|-----------|--------------------------|------------------------|
| `llm_only` (baseline) | the model | the model |
| `cve` (this work) | deterministic code | the model (constrained) |

Metrics: **numeric error rate** (vs. ground truth), **decision error rate**,
**hallucination rate** (numbers in the prose not traceable to a computed value),
and **traceability**.

## Layout

```
cve/         compute.py  data.py  llm.py  explain.py  validate.py  pipeline.py
experiments/ metrics.py  run_benchmark.py
tests/       unit + end-to-end (mock) tests
paper/       preprint draft
```

## Status

Framework and benchmark harness are complete and tested. The accompanying
technical report (`paper/paper.md`) describes the method and releases the
benchmark; it does not include an original empirical study — run the harness
(above) with a model of your choice to reproduce the metrics.

## License

MIT © 2026 Xinyi Shi
