# Compute–Validate–Explain: A Verifiable Architecture for LLM-Assisted Experiment Analysis

**Xinyi Shi**
Independent researcher · Seattle, WA
joypinkblue@gmail.com

> Technical report / preprint (not peer-reviewed).
> Companion open-source framework: https://github.com/pppinkbbblue/trustworthy-experiment-analysis
> This report describes the method and releases a reproducible benchmark harness.
> It does not report an original empirical study; the harness is provided so the
> metrics can be reproduced with any chosen model (see §6). The worked example in
> §5.6 is illustrative, not an experimental result.

## Abstract

Large language models (LLMs) are increasingly used to interpret quantitative
results, yet a well-documented failure mode is that they produce fluent but
incorrect numbers even when their reasoning is otherwise sound. When such systems
inform real decisions, this is a serious barrier to safe adoption. This report
presents **Compute–Validate–Explain (CVE)**, an architecture for LLM-assisted
analysis of controlled experiments (A/B tests) that removes numerical
hallucination *by construction*: a deterministic layer computes every statistic,
the language model is restricted to interpreting results it may not generate, and
a validation layer verifies that every number in the model's output is traceable
to a computed value. We describe the architecture in detail, give a worked
illustrative example and the grounding algorithm, define evaluation metrics
(numeric error, decision error, and a verbatim-grounding hallucination rate), and
release an open-source implementation and a reproducible benchmark harness built
on synthetic experiments with known ground truth. We analyze what the design does
and does not guarantee, discuss deployment considerations, and position CVE
relative to program-aided and tool-use methods. The central claim is not that any
particular model is unreliable, but that for high-stakes analysis reliability
should be a property of the *architecture* — enforced and measurable — rather than
a hoped-for property of the model.

## 1. Introduction

Organizations increasingly ask LLMs to read experiment results and explain what
happened, because interpreting statistics is tedious and expertise-dependent. A
marketer, product manager, or clinician who wants to know "did the treatment
work, and by how much?" would prefer a plain-language answer to a table of
numbers and a statistics background. LLMs are attractive precisely because they
can produce that answer.

They are also risky in this role. An LLM asked to compute a lift, a conversion
rate, or a significance value may return a number that is plausible and wrong,
wrapped in confident prose. In settings where experiments drive consequential
decisions — clinical research, financial risk, public-sector program evaluation,
and product decisions at scale — such errors are costly and hard to catch,
precisely because the surrounding narrative reads authoritatively and invites
trust.

We argue that for high-stakes analysis the two roles an LLM is being asked to play
should be separated: **computation** (producing the numbers) and **explanation**
(interpreting them in language). Code should compute; the model should interpret.
This sentiment is not new — it underlies program-aided and tool-use methods (§2) —
but in practice the separation is usually *encouraged* through prompting rather
than *enforced*, and the grounding of the output is rarely *verified*. A system
that "should not" invent numbers is not the same as a system that *cannot* report
an unverified one. This report is about closing that gap for a concrete, common,
and high-stakes task: reading the results of controlled experiments.

**Contributions.**
1. A concrete architecture, CVE, that applies the compute/explain separation to
   experiment (A/B-test) analysis, with the LLM structurally confined to
   interpretation.
2. A validation step, with an accompanying algorithm, that flags any number in the
   narrative not traceable to a computed value — turning numerical grounding into
   an enforceable and measurable property rather than an aspiration.
3. A set of evaluation metrics (numeric error, decision error, verbatim-grounding
   hallucination rate, traceability) defined precisely enough to be reproduced.
4. An open-source framework and a reproducible benchmark harness — synthetic
   experiments with known ground truth — so the approach can be evaluated against
   any model, together with an analysis of what the design guarantees and what it
   does not.

## 2. Background and Related Work

Our approach builds on a line of work that separates reasoning or explanation from
computation. **Program-aided language models** (PAL) [Gao et al. 2023] and
**Program of Thoughts** [Chen et al. 2023] have the model emit code whose
execution produces the answer, on the observation that LLMs decompose problems
well but make arithmetic and logical mistakes in the solution step. **Tool-use**
methods such as Toolformer [Schick et al. 2023] teach models to call external
functions, and **ReAct** [Yao et al. 2023] interleaves tool calls with reasoning.
**Chain-of-thought** prompting [Wei et al. 2022] improves multi-step reasoning but
offers no guarantee of numerical correctness, and hallucination in generated text
is surveyed broadly in [Ji et al. 2023].

CVE is closest in spirit to Program of Thoughts, which explicitly "disentangles
computation from reasoning." It differs in three respects. First, it targets
*experiment and causal analysis* — reading A/B-test results — rather than
arithmetic word problems. Second, it does not have the model generate code to be
executed; it removes computation from the model's remit entirely and hands it
only pre-computed results to interpret, which narrows the model's surface for
error to language rather than arithmetic. Third, and most importantly, it adds a
*post-hoc verbatim-grounding check* that inspects the model's final narrative and
flags any number not traceable to the computed set. This converts "the model
should not invent numbers" into a property that is checked on every output and can
be measured across a benchmark.

## 3. The Problem: Numerical Hallucination in Analysis

It is useful to distinguish two failure modes when an LLM is asked to both compute
and explain:

- **Computation error.** The model attempts the arithmetic or statistics and gets
  it wrong (a miscomputed lift, an incorrect rate, a mis-stated p-value). This is
  well documented even when the problem is correctly decomposed [Gao et al. 2023].
- **Fabrication.** The model states a number that has no basis in the input at all
  — a plausible-sounding figure introduced during generation. This is a form of
  hallucination [Ji et al. 2023].

In casual use, both are a nuisance. In experiment analysis they are dangerous for
the same reason: a single wrong lift, rate, or significance value can flip a
launch decision, and because the narrative around it is fluent and confident, the
error is easy to miss and easy to act upon. The person consuming the summary is
often precisely the person least equipped to independently check the numbers —
that is why they asked for a plain-language answer in the first place.

The goal of CVE is to make it structurally impossible for the reported numbers to
be model-invented for the quantities the system computes, and to make any
violation detectable rather than silent.

## 4. Method: Compute–Validate–Explain

### 4.1 Overview

Given an experiment with a control arm and one or more treatment arms, CVE runs
three stages: a deterministic **Compute** stage, a constrained **Explain** stage,
and a **Validate** stage that gates the output.

### 4.2 Compute

A deterministic module computes, for each treatment against control: the
conversion rates, the absolute lift (difference of rates), the relative lift
(percent), the standard error of the difference, a two-proportion z-test, and the
normal-approximation probability that the treatment rate exceeds the control rate.
Each result is tagged with a stable `metric_id` so that it can be referenced and
audited. No model is involved; the same inputs always yield the same outputs.

### 4.3 Explain

The LLM receives *only* the computed results (as a small table keyed by
`metric_id`) and a system instruction to interpret them in plain language for a
decision-maker, without introducing any number that is not present in the table,
and to say a value is unavailable rather than guess it. It does not see the raw
arm counts and is never asked to compute anything.

### 4.4 Validate

After the model produces its narrative, the validation stage extracts every
numeric token from the text and checks each against the set of computed values,
allowing for rounding and percentage forms within tolerance. Any number that does
not match a computed value is flagged as *ungrounded* and can be redacted so the
output fails safe. In pseudocode:

```
computed  = { all numeric values produced by the Compute stage }
allowed   = computed ∪ { rounded forms } ∪ { percentage forms for rates in [0,1] }
for each number n extracted from the narrative:
    if not any(close(n, a) for a in allowed):   # within rel/abs tolerance
        flag n as UNGROUNDED
hallucination_rate = |ungrounded| / |numbers found|
```

Grounding is therefore not a matter of trusting the prompt; it is checked on every
output and quantified.

### 4.5 Deterministic decisions

The launch decision (positive / negative / inconclusive) is derived
deterministically from the computed probability against a fixed threshold, never
from the model. The model may explain the decision but cannot change it.

### 4.6 Worked example (illustrative)

*The following numbers are illustrative, to show the mechanism; they are not an
experimental result.* Suppose control shows 10,000 conversions among 100,000 users
(10.0%) and a treatment shows 11,000 among 100,000 (11.0%). The Compute stage
produces: control rate 10.0%, treatment rate 11.0%, absolute lift 1.0 point,
relative lift 10.0%, and a high probability the treatment beats control. The
Explain stage might write: "The treatment increased the conversion rate from 10.0%
to 11.0%, a relative lift of about 10%, and is very likely a real improvement."
Every number in that sentence (10.0, 11.0, 10) traces to a computed value, so
validation passes. Had the model instead written "…and revenue rose 4.2%," the
token 4.2 would match no computed value and be flagged as ungrounded — caught, not
shipped.

### 4.7 Design guarantee and its scope

Because every reported statistic originates in the deterministic layer, and the
validation step flags any number not traceable to it, **the system does not
present fabricated numerical values for the quantities it computes.** This is a
property of the architecture, not of the model. Its scope is exactly the set of
quantities the Compute stage produces; it does not cover numbers outside that set
(see §7), and it does not, by itself, guarantee that the *qualitative*
interpretation is sound.

## 5. Metrics

We define the following, all measurable against known ground truth:

- **Numeric error rate** — the fraction of reported numeric fields (e.g., rate,
  lift, probability) that differ from the ground-truth value beyond a set
  tolerance. For the LLM-only baseline the model reports these itself; for CVE
  they come from the Compute stage.
- **Decision error rate** — the fraction of treatments whose reported
  launch decision differs from the deterministic ground-truth decision.
- **Numerical-hallucination rate** — the fraction of numeric tokens in the
  narrative that are not traceable to any computed value (§4.4).
- **Traceability** — the complement: the fraction of narrative numbers that are
  grounded.

For CVE, numeric error and hallucination are zero by construction for computed
quantities; the metrics are most informative when applied to the baseline, and as
a regression check that CVE's own narrative stays grounded.

## 6. Open-Source Framework and Benchmark Harness

We release the full implementation and a benchmark harness
(https://github.com/pppinkbbblue/trustworthy-experiment-analysis):

- a deterministic compute layer; a synthetic-experiment generator with known
  ground truth; a provider-agnostic LLM interface (OpenAI, Anthropic, Amazon
  Bedrock, or a deterministic mock); the validation layer; two pipelines (CVE and
  an LLM-only baseline that both computes and explains); an evaluation module; and
  a unit-test suite.

**Why synthetic data.** Because the true rates are set by the generator, the
correct value of every statistic is known exactly, which allows numerical error to
be measured precisely — impossible when ground truth is unknown. The generator
draws base rates and effects, then samples observed conversions, producing
experiments with one or more treatment arms.

**Reproduction.** The harness runs both conditions on the same experiments and
reports the metrics of §5:

```
python3 -m experiments.run_benchmark --llm <provider:model> --n 50
```

Anyone can run it against a chosen model to obtain concrete numbers. We deliberately
report no model-specific results here; the harness is provided as an open benchmark
so results are reproducible and comparable rather than asserted.

## 7. Threat Model: What the Guarantee Does and Does Not Cover

CVE's guarantee is precise and therefore limited. It is worth stating the
boundaries plainly.

- **Covered:** fabricated or miscomputed values for any statistic the Compute
  stage produces. These cannot reach the user unflagged.
- **Not covered by construction:**
  - *Correctness of the Compute stage itself.* Trust is shifted from the model to
    the deterministic code, which must implement the right statistics and be
    reviewed and tested accordingly. CVE moves the trust boundary; it does not
    remove it.
  - *Qualitative misinterpretation.* The model could ground every number yet still
    describe a flat result as a win. Grounding checks numbers, not judgment.
  - *Numbers outside the computed set.* If the narrative introduces a quantity the
    system never computed (e.g., an unrelated revenue figure), validation flags it
    as ungrounded — which is the safe behavior — but the system cannot supply the
    correct value.
  - *Selection of what to compute.* If the wrong metric or window is analyzed, the
    output can be grounded and still misleading.

These boundaries argue for CVE as one layer in a trustworthy-analysis stack, not a
complete solution.

## 8. Deployment Considerations

A few practical notes for using this pattern in production:
- **Verbatim passthrough.** Because the numbers are authoritative, the safest
  presentation renders the computed table directly and uses the model only for the
  surrounding prose, rather than letting the model reformat numbers.
- **Fail-safe on violation.** When validation flags an ungrounded number, redacting
  or refusing is preferable to shipping; a visible "[unverified]" is better than a
  confident wrong figure.
- **Auditability.** Keeping `metric_id`s and the computed table alongside the
  narrative lets a reviewer trace any statement back to its source.
- **Model-agnosticism.** Since the model only interprets, the approach is largely
  insensitive to model choice, which aids portability and cost control.

## 9. Limitations

Beyond the threat-model boundaries in §7: synthetic data may not capture the
messiness of production pipelines (missing data, skew, masking); the current
implementation covers two-arm and multi-arm conversion metrics rather than the
full range of experiment designs; and this report does not include an original
large-scale empirical study across models — that evaluation is left to
reproduction via the provided harness. These are deliberate scope choices, stated
plainly so the contribution is not overclaimed.

## 10. Conclusion and Future Work

For high-stakes, LLM-assisted experiment analysis, separating deterministic
computation from model explanation — and *verifying* that separation — is a simple,
measurable way to make output trustworthy. The reliability becomes a property of
the architecture rather than of the model. Future work includes extending the
compute layer to richer designs (continuous metrics, variance reduction,
sequential testing), studying grounding for qualitative claims (not only numbers),
and running the open benchmark across a range of models to characterize baseline
hallucination and error rates. We release the framework and benchmark to support
that study and adoption.

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
