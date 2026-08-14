# Contributing

Contributions and issues are welcome.

## Development

```bash
python3 -m pip install -e ".[test]"   # editable install with test deps
python3 -m pytest -q                   # run the test suite
```

## Guidelines

- The core framework (`cve/`) stays dependency-free (standard library only).
  Model backends are optional extras.
- The deterministic layer (`cve/compute.py`) is the source of truth for every
  statistic. It must never depend on an LLM, and changes to it should come with
  tests in `tests/test_compute.py`.
- Benchmark numbers reported anywhere must come from a real model run, not the
  `mock` backend. The mock exists only to exercise the plumbing.
- Keep changes small and covered by tests.

## Reproducing the benchmark

```bash
ANTHROPIC_API_KEY=... python3 -m experiments.run_benchmark \
  --llm anthropic:claude-3-5-sonnet-latest --n 50
python3 -m experiments.fill_paper
```
