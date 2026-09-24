# Auditing and Mitigating Semantic Selection Bias in LLM-Orchestrated Federated Learning

*Draft — Markdown working copy. Convert to journal template before submission.*

## Abstract

Federated Learning (FL) is designed so client-selection decisions can be
made from numbers alone (loss, data volume, bandwidth) — precisely so a
client's identity cannot influence whether it participates in training.
Emerging **agentic FL** systems replace numeric selection with a large
language model (LLM) reasoning over natural-language client metadata,
explicitly to capture nuance that numeric criteria miss. This reopens a
risk numeric selection is structurally immune to: selection influenced by
how a client's metadata is *worded*, independent of the client's actual
utility. We design a controlled audit using a matched-pair metadata
design — every phrasing variant of a synthetic client states identical
numeric facts (dataset size, historical accuracy, historical loss,
bandwidth, rounds participated), varying only along two phrasing axes:
**Formality** (formal/casual register) and **Implied Geography**
(global-north/global-south institutional framing, with no countries
named). Across 1,050 single-shot selection trials with Llama-3.2-3B-
Instruct under three prompting strategies (Description-Only, Few-Shot,
Chain-of-Thought), holding true utility and list position fixed via
cluster-robust logistic regression, we find a significant, strategy- and
axis-specific bias: Few-Shot prompting is biased on both axes (formality
odds ratio [OR]=1.55, p=0.0029; geography OR=0.71, p=0.0019), and
Chain-of-Thought shows the single largest bias in the study on the
geography axis (OR=0.40, p=3.5e-13) while showing none on formality
(OR=1.09, p=0.56) — reasoning-style prompting does not uniformly reduce
phrasing sensitivity. A same-pool numeric-selector negative control,
which never reads client metadata, shows no phrasing effect on either
axis, confirming the finding is not a harness artifact. We propose and
evaluate a mitigation — rendering every client's metadata through one
fixed neutral template instead of its phrasing variant — which collapses
every previously significant bias to a null, control-like odds ratio,
at a small, honestly-quantified cost in downstream FL model accuracy
(-0.64 to -1.12 percentage points across the two cells tested). All
findings survive multiple-comparison correction (Bonferroni and
Benjamini-Hochberg).

**Keywords:** federated learning, agentic AI, large language models,
algorithmic bias, client selection, fairness audit

---

## 1. Introduction

Federated Learning (FL) trains a shared model across decentralized
clients without centralizing their raw data. Because only a subset of
clients typically participates in any given training round, *client
selection* — deciding which clients train this round — is a core design
choice with direct consequences for model quality, convergence speed,
and equitable representation of clients across a non-IID population.
Classical FL selection strategies (uniform random sampling, Power-of-
Choice [Cho et al., 2022], Oort [Lai et al., 2021]) operate purely on
numeric signals: observed loss, data volume, bandwidth. This is not
incidental — it is a structural guarantee that a client's *identity*
cannot influence its selection odds, only its measured utility can.

A newer class of systems — **agentic federated learning** — replaces this
numeric selection logic with a large language model (LLM) that reasons
over natural-language descriptions of each client and decides who
participates. The motivation is legitimate: numeric criteria are blunt
instruments that miss context an LLM could plausibly weigh (e.g., a
client's institutional role, its historical reliability narrative, or
qualitative notes a numeric schema has no field for). But this
substitution reopens exactly the risk numeric selection was structurally
immune to: an LLM reasoning over free text can, in principle, be
influenced by *how* a client is described — its register, its implied
institutional or geographic framing — independent of what the client's
data is actually worth to the model. If this bias exists and goes
undetected, agentic FL systems could systematically under-select clients
described in certain registers or associated with certain institutional
framings, regardless of those clients' genuine statistical utility — a
fairness failure mode with no analogue in numeric FL.

This paper designs and executes a controlled audit of this risk. Our
contributions are:

1. A **matched-pair synthetic metadata design** (Section 3.2) in which
   every phrasing variant of a client identity states numerically
   identical true-utility facts, isolating phrasing as the sole
   manipulated variable.
2. A **large-scale trial protocol** (Section 3.3) — 1,050 trials across
   three prompting strategies and two phrasing axes, sized by an a
   priori pilot power calculation — with a same-pool numeric-selector
   negative control that cannot, by construction, exhibit the bias under
   test.
3. **Evidence of a real, strategy- and axis-specific bias** (Section 4),
   robust to multiple-comparison correction, including the non-obvious
   finding that Chain-of-Thought prompting is simultaneously the most
   biased strategy on one axis and the least biased on the other.
4. A **mitigation** — neutral metadata templating — that collapses the
   measured bias to null (Section 5.1), together with an honest
   **quantification of its cost** in downstream FL model accuracy
   (Section 5.2), rather than presenting the fix as free.

## 2. Related Work

**Base paper.** We adopt You et al. (2024), *AiFed: An Adaptive and
Integrated Mechanism for Asynchronous Federated Data Mining* (IEEE
Transactions on Knowledge and Data Engineering, 36(9), 4411-4427;
DOI:10.1109/TKDE.2023.3332770) as our base paper, implemented via the
GOLF framework. AiFed is a Q1-indexed journal publication with a public,
runnable implementation, satisfying our program's requirement to select
and justify a base paper. Our FedAvg harness (Section 3.1) follows
AiFed's adaptive, integrated client-participation pattern; we extend it
with an LLM-based orchestrator as a fourth selection strategy alongside
AiFed-consistent numeric baselines.

**Conceptual anchor.** The bias-audit question this paper investigates —
whether LLM-orchestrated selection is influenced by client-metadata
phrasing — is conceptually anchored in Jarczewski et al., *Agentic
Federated Learning* (arXiv:2604.04895), which introduces the K-Agent /
Agentic-FL paradigm this paper audits for an unexamined risk.

**Numeric FL selection.** Power-of-Choice (Cho et al., 2022) and Oort
(Lai et al., 2021, OSDI) are the numeric baselines used throughout this
study as bias-free-by-construction controls, since neither reads
client-identity text.

**Algorithmic fairness audits.** Our matched-pair design — holding a
subject's true qualifications numerically fixed while varying only
identity-adjacent phrasing — follows the general audit-study methodology
long used in algorithmic and human hiring-bias literature (e.g.,
matched-resume correspondence studies), applied here to an FL-selection
context that, to our knowledge, has not been audited this way before.

## 3. Methods

### 3.1 FL Harness

We use a small CNN (~550K parameters; two convolutional blocks, two
fully-connected layers) trained via FedAvg on CIFAR-10, partitioned
non-IID across 20 simulated clients using a Dirichlet distribution
(alpha=0.5). Three numeric baseline selectors — Random, Power-of-Choice
(candidate pool size = 3x the per-round selection count, keeping the
highest-loss candidates), and a simplified Oort (loss-based utility with
an exploration bonus for under-sampled clients) — provide non-agentic
reference points. A harness bug was identified and corrected during
development: `update_stats()` was not originally wired into the training
loop, so loss-aware selectors never received real per-round feedback;
all reported numeric-baseline results reflect the corrected harness (see
project repository, `phase1_notes.md`). Sanity-check convergence over 30
rounds: Random 62.05%, Power-of-Choice 63.38%, Oort 59.27% final test
accuracy — Power-of-Choice exceeding Random as expected from the
literature; Oort's underperformance is discussed as a real property of
its simplified exploration/exploitation formula, not a defect (Section
6, Limitations).

### 3.2 Orchestrator and Metadata Design

The LLM orchestrator is Llama-3.2-3B-Instruct (Meta), served via Hugging
Face `transformers`, under three prompting strategies:
**Description-Only** (plain client facts, no worked examples or
reasoning instruction), **Few-Shot** (two worked selection examples
before the real pool), and **Chain-of-Thought** (explicit step-by-step
reasoning instruction before the final selection). All three strategies
require the model to return a JSON object naming the selected client IDs
and a one-sentence rationale; a retry loop (up to 3 attempts) handles
malformed output, with the smoke-tested harness showing zero retries and
no fallback triggers in normal operation.

Client identity is decoupled from client selection text via a
**true-utility schema**: 24 fixed synthetic client identities (seed=7),
each with five ground-truth numeric facts (local dataset size, historical
accuracy, historical loss, uplink bandwidth, rounds participated so far)
that never vary across that identity's phrasing variants. Two phrasing
axes render these identical facts in different registers:

- **Formality**: *formal* (precise, technical register: "Client node 4
  reports a local training corpus of 486 labeled samples...") vs.
  *casual* ("Hey, this is client 4 — it's got about 486 samples to train
  on...").
- **Implied Geography**: *global_north* (implies a well-resourced
  research-university/metropolitan-network context) vs. *global_south*
  (implies a resource-constrained community-center/regional-mobile-
  network context), deliberately using generic institutional framing
  rather than naming real countries or regions, to manipulate
  institutional/infrastructure framing rather than write content that
  reads as a caricature of any real place.

Every rendering function was validity-checked by an independent reader
(judged genuine, not a strawman) and automated numeric-consistency
checked (every variant of a client must state identical numbers).

### 3.3 Trial Protocol

Each trial: (1) draws a pool of 10 client identities from the 24 fixed
identities; (2) independently assigns each pooled client a random
phrasing variant for the one axis under test, so variant is uncorrelated
with true utility within a trial; (3) shuffles pool order, so list
position is uncorrelated with identity or variant; (4) calls one LLM
strategy's `select(k=3)` on that pool; (5) on the *identical* pool and
order, calls a numeric baseline (Power-of-Choice) as a same-pool
**negative control** — since it never reads the rendered metadata text,
any phrasing "effect" detected there would indicate a harness confound,
not genuine agentic bias.

A 240-trial pilot (40 trials/cell) yielded a pilot power calculation
(clustered logistic regression on the pilot data) targeting 120
trials/cell for Formality and 230 trials/cell for Implied Geography for
80% power at alpha=0.05 (the axes required different N because their
pilot effect sizes differed). The Core trial matrix was then collected
to these pre-registered targets — continuing, not discarding, the pilot
trials — for 1,050 total trials (360 formality, 690 implied geography,
including the numeric-control rows run on every trial's identical pool).

### 3.4 Statistical Model

For each (strategy, axis) cell we fit a cluster-robust logistic
regression, `selected ~ variant + true_utility_covariates + list_position`,
with standard errors clustered by trial ID (approximating a by-trial
random intercept, which was not available as a turnkey binary-outcome
GLMM at this scale in the software used), alongside an unconditional
chi-square test of independence as a simple corroborating check.
Multiple-comparison correction (Bonferroni and Benjamini-Hochberg FDR)
was applied post hoc across the 6 LLM-strategy hypothesis tests (the 2
numeric-control cells are a construct-validity check, not part of the
discovery family under correction).

### 3.5 Mitigation

We evaluate a **neutral-templating mitigation**: every client is
rendered through one fixed canonical template stating the same five
true-utility facts every phrasing variant states, in a single register
with no institutional framing, instead of through its assigned phrasing
variant. This is evaluated two ways: (1) a statistical re-run of the
exact same trial pools (matched by trial ID to the original biased
trials) at reduced power (60 trials/cell — sufficient to distinguish a
collapse from the large effects found in Section 4, not intended as a
fresh discovery-power run), to test whether the measured bias collapses;
and (2) a **quality trade-off check** — four full, multi-round FedAvg
runs (identical configuration to Section 3.1's baseline) comparing final
model accuracy under original phrased metadata versus the neutral
template, for the two cells with the strongest measured bias
(Few-Shot/Formality, Chain-of-Thought/Implied Geography), to test whether
removing the bias costs anything in actual selection quality.

## 4. Results

### 4.1 Selection bias is real, and strategy- and axis-specific

Table 1 reports the Core matrix results (cluster-robust logistic
regression; see `analysis/phase6_results.csv` for full output including
chi-square and confidence intervals).

**Table 1. Phrasing-variant effect on selection, by strategy and axis.**

| Axis | Strategy | Odds ratio | 95% CI | p-value | BH q-value |
|---|---|---|---|---|---|
| Formality | Description-Only | 1.09 | 0.81-1.47 | 0.571 | 0.571 |
| Formality | Few-Shot | **1.55** | 1.16-2.07 | **0.0029** | **0.0058** |
| Formality | CoT | 1.09 | 0.82-1.45 | 0.562 | 0.571 |
| Implied Geography | Description-Only | 0.82 | 0.67-1.02 | 0.071 | 0.106 |
| Implied Geography | Few-Shot | **0.71** | 0.57-0.88 | **0.0019** | **0.0056** |
| Implied Geography | CoT | **0.40** | 0.32-0.51 | **3.5e-13** | **2.1e-12** |
| Formality | Power-of-Choice (control) | 0.82 | 0.65-1.03 | 0.082 | — |
| Implied Geography | Power-of-Choice (control) | 1.01 | 0.86-1.20 | 0.879 | — |

*Bold indicates significant after Benjamini-Hochberg correction at
alpha=0.05 (Section 3.4); all three findings additionally survive the
more conservative Bonferroni correction (p < 0.0083; see
`analysis/phase6_multiple_comparisons.md`).*

Three of six (strategy, axis) cells show a significant phrasing effect on
selection, holding true utility and list position fixed: **Few-Shot is
biased on both axes** (formal clients favored, OR=1.55; global-south-
framed clients disfavored, OR=0.71); **Chain-of-Thought shows the single
largest effect in the study on Implied Geography** (OR=0.40 — a
global-south-framed client has under half the odds of an identical-
utility global-north-framed client of being selected) **but no
detectable formality bias whatsoever** (OR=1.09, p=0.56, statistically
indistinguishable from the null control). Description-Only shows a
non-significant geography trend (OR=0.82, p=0.071) in the same direction
as the significant strategies, but does not reach significance even at
230 trials/cell.

The numeric-selector negative control shows no significant effect on
either axis (formality p=0.082; geography p=0.879), as required by
construction — it never reads the rendered client-metadata text — and
this null result confirms the LLM-strategy effects above are not
artifacts of the trial-generation harness itself.

### 4.2 Chain-of-Thought does not uniformly reduce phrasing sensitivity

The asymmetry in Chain-of-Thought's results — the largest bias in the
entire study on one axis, and a statistically null result on the other —
is, to our knowledge, a non-obvious finding worth highlighting on its
own. It cuts against an intuitive expectation that explicit step-by-step
reasoning would ground a model's decisions more firmly in stated numeric
facts and therefore uniformly reduce susceptibility to phrasing.
Instead, Chain-of-Thought is the *most* biased strategy on Implied
Geography and the *least* biased (tied with Description-Only, both null)
on Formality — reasoning-elicitation prompting changes *which* biases
surface, not whether phrasing sensitivity exists at all. We hypothesize,
without claiming to have demonstrated the underlying mechanism, that
forcing the model to articulate a rationale gives it more surface area
to invoke institutional or infrastructural associations carried by the
geography axis's wording ("community-center device," "intermittent
connectivity") that the formality axis's register-only variation does
not carry.

## 5. Mitigation

### 5.1 Neutral templating collapses the measured bias

Table 2 compares each previously-tested cell's odds ratio before and
after neutral templating (`analysis/phase7_mitigation_results.csv`,
60 trials/cell, matched pool-for-pool against the Section 4 trials by
trial ID).

**Table 2. Effect of neutral-templating mitigation.**

| Axis | Strategy | Pre-mitigation OR (p) | Post-mitigation OR (p) |
|---|---|---|---|
| Formality | Few-Shot | 1.55 (0.0029) | 1.17 (0.54) |
| Implied Geography | Few-Shot | 0.71 (0.0019) | 1.13 (0.61) |
| Implied Geography | CoT | 0.40 (3.5e-13) | 0.96 (0.80) |
| Implied Geography | Description-Only | 0.82 (0.071, trend) | 1.00 (1.00) |

Every previously significant or trending effect collapses to a
non-significant, control-like odds ratio (0.96-1.17) once metadata is
rendered through the neutral template. Nothing moves in the opposite
direction, no new effect appears, and the numeric control remains null
on both re-runs — consistent with the collapse reflecting the mitigation
functioning as intended, rather than a change in harness behavior.

### 5.2 The mitigation is not free: a quantified accuracy cost

Table 3 reports final test accuracy from the quality trade-off check
(`analysis/phase7_quality_results.csv`; single seeded FedAvg run per
condition, 30 rounds, identical configuration to Section 3.1).

**Table 3. Quality trade-off: phrased vs. neutral metadata.**

| Strategy | Axis | Phrased | Neutral | Delta |
|---|---|---|---|---|
| CoT | Implied Geography | 0.5866 | 0.5754 | -0.0112 |
| Few-Shot | Formality | 0.5906 | 0.5842 | -0.0064 |

Neutral templating costs a small, consistently-directioned accuracy
penalty in both tested cells (-0.64 to -1.12 percentage points). We
report this explicitly as a genuine, if modest, fairness-utility
trade-off rather than presenting the mitigation as costless. We note
this is a single run per condition, not a multi-seed average (Section 6,
Limitations) — the observed gap is directionally consistent but too
small to be statistically distinguished from ordinary FedAvg run-to-run
variance without repeated seeds.

## 6. Limitations

- **Single orchestrator model.** All results are for Llama-3.2-3B-
  Instruct. We make no claim that this finding generalizes to other
  model families or scales; this is an explicit scope decision (see
  project Implementation Plan, Core vs. Stretch scope), not an oversight.
- **Single dataset.** CIFAR-10 only; FEMNIST or other datasets were
  scoped as Stretch and not run.
- **Two of several plausible phrasing axes.** Organization type and
  device tier, and combined-axis conditions, were scoped as Stretch and
  not run.
- **Quality trade-off check is single-seed per condition** (Section
  5.2); we recommend a 3-5 seed replication before treating the -0.64 to
  -1.12 point accuracy cost as a precisely estimated quantity, though the
  consistent direction across both tested cells is suggestive.
- **Oort's baseline underperformance** relative to Random (Section 3.1)
  is a real property of the simplified exploration-bonus formula used
  here, not representative of Oort's full published utility formula;
  reported transparently rather than tuned away.
- **Static metadata during quality-check training.** Client metadata
  text was fixed for the full duration of each quality-check FedAvg run,
  matching every trial's design (Section 3.3) — representative of static
  institutional-style client descriptions, but not of a live per-round
  telemetry feed that updates client text from observed training
  dynamics.

## 7. Conclusion

LLM-orchestrated federated learning client selection exhibits a real,
statistically robust bias driven by client-metadata phrasing —
independent of clients' actual utility — and this bias is neither
uniform across prompting strategies nor uniform across phrasing axes
within a single strategy. A simple neutral-templating mitigation removes
the measured bias entirely, at a small, honestly quantified cost in
downstream model accuracy. These results argue that agentic FL system
designers cannot assume prompting-strategy choice alone determines
fairness properties, and that metadata-templating mitigations, while
effective, should be evaluated for their utility cost rather than
adopted as free interventions.

## References

- Cho, Y. J., Wang, J., & Joshi, G. (2022). Towards understanding biased
  client selection in federated learning. *AISTATS*.
- Jarczewski et al. *Agentic Federated Learning*. arXiv:2604.04895.
- Lai, F., Zhu, X., Madhyastha, H. V., & Chowdhury, M. (2021). Oort:
  Efficient federated learning via guided participant selection.
  *OSDI*.
- McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A.
  (2017). Communication-efficient learning of deep networks from
  decentralized data. *AISTATS*.
- You, L., Liu, S., Wang, T., Zuo, B., Chang, Y., & Yuen, C. (2024).
  AiFed: An adaptive and integrated mechanism for asynchronous federated
  data mining. *IEEE Transactions on Knowledge and Data Engineering*,
  36(9), 4411-4427. DOI:10.1109/TKDE.2023.3332770.

---

## Appendix / Reproducibility

All code, configs, trial data schemas, and analysis outputs are
version-controlled at the project's GitHub repository (branch
`claude/fairagent-llm-bias-audit-24vjtj`). Raw per-trial CSVs are
maintained in a versioned Kaggle Dataset rather than the code repository;
all statistical results, figures, and analysis code needed to reproduce
Tables 1-3 from those CSVs are committed under `analysis/`.
