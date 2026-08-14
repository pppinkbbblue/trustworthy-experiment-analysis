# Compute–Validate–Explain: Enforcing and Measuring Numerical Grounding in LLM-Assisted Experiment Analysis

**Xinyi Shi**
Independent researcher · Seattle, WA
[email]

> DRAFT v1 (2026-08). Bracketed RESULT markers are filled only from real model
> runs, via `python3 -m experiments.fill_paper` (which reads
> `results/summary_*.json` and writes `paper/paper_filled.md`). They are
> intentionally left unfilled here rather than invented. Verify all citations
> before posting.

## Abstract

Large language models (LLMs) are increasingly used to interpret quantitative
results, but they can produce fluent yet incorrect numbers — a serious problem
when those numbers inform decisions. We study a simple architecture,
Compute–Validate–Explain (CVE), for LLM-assisted analysis of controlled
experiments (A/B tests): a deterministic layer computes every statistic, a
validation layer checks that each number in the model's write-up is traceable to
a computed value, and the LLM is restricted to interpreting results rather than
computing them. Using synthetic experiments with known ground truth, we compare
CVE against an LLM-only baseline that both computes and explains. We report
numeric error rate, decision error rate, and a numerical-hallucination rate
defined via verbatim grounding. The LLM-only baseline exhibits a numeric error
rate of `[RESULT: llm_only.numeric_error_rate]` and a hallucination rate of
`[RESULT: llm_only.hallucination_rate]`, while CVE reduces numeric error to
`[RESULT: cve.numeric_error_rate]` and hallucination to
`[RESULT: cve.hallucination_rate]` by construction. We release the framework and
benchmark as open source.

## 1. Introduction

Organizations increasingly ask LLMs to read experiment results and explain what
happened. This is attractive because interpreting statistics is tedious and
requires expertise. It is also risky: an LLM asked to compute a lift or a
significance value may return a number that is plausible and wrong. In settings
where experiments drive consequential decisions — clinical research, financial
risk, public-sector program evaluation — such errors are costly and hard to
catch precisely because the output reads authoritatively.

We argue that for high-stakes analysis the roles should be separated: **code
should compute, and the model should interpret**. This is not a new sentiment,
but two things are often missing in practice. First, the separation is usually
*encouraged* (via prompting) rather than *enforced*. Second, the benefit is
rarely *measured* on data where the correct answer is known. This paper
addresses both.

**Contributions.**
1. A concrete architecture, CVE, that (a) computes all statistics
   deterministically, (b) constrains the LLM to interpretation, and (c) adds a
   validation step that flags any number in the narrative not traceable to a
   computed value, making grounding an enforceable and measurable property.
2. An open, reproducible benchmark built on synthetic experiments with known
   ground truth, with metrics for numeric error, decision error, and numerical
   hallucination.
3. An empirical comparison of an LLM-only baseline against CVE.

## 2. Related Work

Our approach draws on a line of work separating reasoning/explanation from
computation. Program-aided and program-of-thought methods delegate calculation
to executed code [Gao et al. 2023; Chen et al. 2023]. Tool-use methods let
models call external functions [Schick et al. 2023], and reasoning-and-acting
frameworks interleave tool calls with reasoning [Yao et al. 2023]. Chain-of-
thought prompting improves reasoning but does not guarantee numerical
correctness [Wei et al. 2022]. Hallucination in generated text is well
documented [Ji et al. 2023]. CVE differs in emphasis: rather than teaching the
model to compute or call tools, it removes computation from the model entirely
for this task, and adds a *post-hoc verbatim-grounding check* that yields a
measurable hallucination rate specific to numerical claims in experiment
analysis. (Citations to be verified before posting.)

## 3. Method

Given an experiment with a control arm and one or more treatment arms:

- **Compute.** A deterministic module computes conversion rates, absolute and
  relative lift, standard error, a two-proportion z-test, and the normal-
  approximation probability that the treatment beats control. Each result
  carries a stable `metric_id`.
- **Explain.** The LLM receives *only* the computed results and a system
  instruction to interpret them without introducing any number not present, and
  to reference `metric_id`s. It produces a plain-language summary.
- **Validate.** We extract numeric tokens from the narrative and check each
  against the set of computed values (allowing rounding and percentage forms
  within tolerance). Ungrounded numbers are flagged and can be redacted, so a
  narrative that violates grounding fails safe.

Decisions (positive / negative / inconclusive) are derived deterministically
from the computed probability against a threshold, never from the model.

## 4. Benchmark Design

**Data.** We generate synthetic experiments: true control and treatment rates
are drawn, and observed conversions are sampled. Because true rates are known,
the correct value of every statistic is known exactly. No real or proprietary
data is used.

**Conditions.** (i) *LLM-only*: the model is given raw arm counts and asked to
compute and explain, returning structured numbers plus a narrative. (ii) *CVE*:
as in Section 3.

**Metrics.** Numeric error rate (fraction of reported numeric fields differing
from ground truth beyond tolerance); decision error rate; hallucination rate
(fraction of narrative numbers not grounded in computed values); traceability.

**Setup.** `[RESULT: n]` experiments, seed `[RESULT: seed]`, model
`[RESULT: llm]`, temperature 0.

## 5. Results

| Condition | Numeric error | Decision error | Hallucination | Traceability |
|-----------|---------------|----------------|---------------|--------------|
| LLM-only  | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |
| CVE       | `[RESULT]` | `[RESULT]` | `[RESULT]` | `[RESULT]` |

(Filled from `results/summary_*.json` after a real run.)

## 6. Discussion and Limitations

CVE reduces numeric error and hallucination to near zero *by construction* for
the quantities the deterministic layer computes; the empirical question is how
large those errors are for the LLM-only baseline and how they scale with the
number of metrics and arms. This is the honest framing of the contribution: the
architecture is simple, and the value is in enforcing and *quantifying* a
property practitioners otherwise assume.

Limitations: (1) synthetic data may not capture the messiness of real
pipelines; (2) the deterministic layer must implement the correct statistics —
CVE moves the trust from the model to that code, which must be reviewed; (3) the
grounding check is numeric and does not catch qualitative misinterpretation;
(4) results depend on the specific model and prompt.

## 7. Conclusion

For high-stakes, LLM-assisted experiment analysis, separating deterministic
computation from model explanation — and *verifying* that separation — is a
simple, measurable way to make output trustworthy. We release the framework and
benchmark to support further study.

## References

- Gao, Madaan, Zhou, Alon, Liu, Yang, Callan, Neubig. *PAL: Program-aided
  Language Models.* arXiv:2211.10435; ICML 2023.
- Chen, Ma, Wang, Cohen. *Program of Thoughts Prompting: Disentangling
  Computation from Reasoning for Numerical Reasoning Tasks.* arXiv:2211.12588;
  TMLR 2023.
- Schick, Dwivedi-Yu, Dessì, et al. *Toolformer: Language Models Can Teach
  Themselves to Use Tools.* arXiv:2302.04761; NeurIPS 2023.
- Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao. *ReAct: Synergizing Reasoning and
  Acting in Language Models.* arXiv:2210.03629; ICLR 2023.
- Wei, Wang, Schuurmans, et al. *Chain-of-Thought Prompting Elicits Reasoning in
  Large Language Models.* arXiv:2201.11903; NeurIPS 2022.
- Ji, Lee, Frieske, et al. *Survey of Hallucination in Natural Language
  Generation.* arXiv:2202.03629; ACM Computing Surveys, 2023.

Note: the closest prior work is Program of Thoughts (Chen et al.), which also
"disentangles computation from reasoning." CVE differs by (a) targeting
experiment/causal analysis rather than arithmetic word problems, (b) removing
computation from the model entirely rather than having it generate code, and
(c) adding a post-hoc verbatim-grounding check that yields a measurable
numerical-hallucination rate.
