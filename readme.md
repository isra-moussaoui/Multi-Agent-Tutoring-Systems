# Failure Modes & Recovery in Multi-Agent LLM Tutoring Pipelines

A research project by Moussaoui Isra & Chakroun Oussama


## What This Project Is About

Single-agent LLM tutors have a well-documented weakness: they tend to
validate wrong student answers rather than correct them, especially when the
student sounds confident. The natural fix is to add a second agent — a
Verifier — to independently check the Tutor's judgment before it reaches the
student.

**We find that it does work, with a measurable cost.** An independent
Verifier plus a tie-breaking Recovery role reduces over-validation by
7–11 percentage points across two independently implemented pipeline
orchestrations (McNemar's exact test, p<0.0001 / p=0.0001), at a 2.09×
latency cost per graded step relative to a single-agent Tutor call.

This project builds a multi-agent tutoring pipeline (Tutor → Verifier →
Recovery) and measures whether the Verifier genuinely catches the Tutor's
errors, or just agrees with it — a phenomenon known as conformity bias. We
benchmark both a single-agent baseline and our multi-agent pipeline on a
real, publicly available propositional-logic tutoring dataset and compare
failure rates.

**A note on scope:** both pipeline implementations are evaluated on
overlapping data drawn from a shared pool of ~516 proof states, so the two
experiments should be read as a robustness check across orchestration
frameworks (a linear script vs. a LangGraph implementation) rather than as
replication on independent samples. See [Limitations](#limitations) below
and the paper for the full discussion.

**Implementation note:** The tutoring core (Tutor → Verifier → Compare →
Recovery → Finalize) is a compiled LangGraph in `dt_code/graph/`. Student
simulation and KG-grounded labels run outside the graph as the evaluation
harness (`run_graph_pipeline.py`). The Verifier node is hard-constrained
never to read the Tutor's verdict when building its prompt (anti-conformity
invariant). All roles (Student, Tutor, Verifier, Recovery) are served by the
identical model, **Qwen2.5-7B-Instruct, run locally with no external API
dependency** — this is a deliberate methodological control to eliminate
model-capability confounds, so any measured improvement is attributable to
the architecture itself rather than a stronger model doing the judging.

## Results

Evaluated on matched proof states from the BEA 2026 propositional-logic
tutoring benchmark, comparing a single-agent Tutor baseline against the full
Tutor→Verifier→Recovery pipeline on identical paired states:

| Metric | Exp. 1 (script pipeline, n=503) | Exp. 2 (LangGraph, n=513) |
|---|---|---|
| Over-validation — baseline | 35.79% | 33.92% |
| Over-validation — pipeline | 24.65% | 26.90% |
| Over-validation — reduction | −11.14 pp (p=0.0001) | −7.02 pp (p<0.0001) |
| Over-rejection — baseline | 1.99% | 2.53% |
| Over-rejection — pipeline | 1.59% (n.s.) | 2.53% (n.s.) |
| Conformity rate (Tutor vs. Verifier) | 90.26% | 90.06% |
| Recovery invocation rate | 9.6% | 9.94% |
| Recovery precision | 85.7% | 86.27% |
| Mean latency, Tutor-only | — | 19.13s |
| Mean latency, full pipeline | — | 39.91s (2.09×) |

**The headline mechanism:** Tutor and Verifier agree on ~90% of steps
(correlated blind spots — not independent errors that simple aggregation
would catch), but on the ~10% where they disagree, Recovery resolves the
disagreement correctly ~86–87% of the time, disproportionately catching
over-validation rather than introducing new errors (a ~10:1 corrected-to-
introduced ratio in Experiment 2).

Full methodology, significance testing details, and the recovery-vs-cheap-
heuristic ablation are not available yet.

## The Research Gap We Are Filling

Three bodies of work are directly relevant to what we're doing — and none of
them connects to the others yet:

| Area | What exists | What's missing |
|---|---|---|
| LLM tutoring benchmarks | Documented over-validation / over-rejection errors in single-agent tutors (BEA 2026) | Nobody had tested whether a multi-agent architecture fixes this |
| Pedagogical sycophancy | Taxonomy of how and why LLM tutors become too agreeable (CS-SYC and others) | No architecture that actively detects and recovers from these failures |
| Multi-agent failure taxonomy | MAST documents 14 failure modes in agent systems (coding, business tasks) | Never applied to tutoring; pedagogical failures have a different cost structure |

**Our contribution:** a multi-agent verification architecture applied to
tutoring, evaluated on a real benchmark with a matched-sample, paired
significance test, and replicated across two independently implemented
orchestrations sharing an evaluation pool. We also report, for the first
time in this setting to our knowledge, the latency cost of the added
verification stages.

## Papers We Read & Key Takeaways

### 1. Confirming Correct, Missing the Rest (Yasir et al., BEA 2026 Workshop)

Benchmarks LLM tutoring agents on propositional logic exercises. Each
student solution step is labeled `optimal`, `valid_alternative`, or
`incorrect`, grounded in a knowledge graph of valid inference paths.

**Key finding:** LLM tutors score near-ceiling on confirming correct steps,
but systematically fail at over-validation (marking an incorrect step as
correct — the most dangerous failure in education) and over-rejection
(rejecting a valid-but-suboptimal step as wrong). These failures persist
across multiple models, suggesting an architectural rather than a
prompting or model-size problem.

**Why this matters for us:** this is our evaluation dataset (10,836
solution-feedback pairs with KG-grounded ground-truth labels). We run our
pipeline on the same benchmark and compare our over-validation and
over-rejection rates against their single-agent baselines — and confirm
their finding persists in our Tutor-alone baseline before showing the
multi-agent pipeline reduces it.

### 2. Sycophancy is an Educational Safety Risk (2026)

Defines and taxonomizes pedagogical sycophancy — when an LLM tutor becomes
too agreeable in ways that harm learning, including Context-Switch
Sycophancy (backing down when a student challenges a correction), authority
pressure, and face-saving.

**Why this matters for us:** these are the specific failure types our
Tutor agent is prone to. Our Verifier is designed to catch these patterns
independently rather than anchoring on the Tutor's judgment.

### 3. Why Do Multi-Agent LLM Systems Fail? — MAST Taxonomy (Cemri et al., arXiv:2503.13657)

Systematically analyzes failure modes across multi-agent LLM systems: 14
failure types across specification/design issues (~41.8%), inter-agent
misalignment (~36.9%), and verification failures (~21.3%). Critically,
accumulating good agents does not guarantee a good system — when agents
agree, they may be converging on a shared error rather than independently
confirming a correct answer (conformity bias).

**Why this matters for us:** this is our theoretical framework. Our
conformity-rate metric directly operationalizes this failure mode in the
tutoring domain, and Recovery is a targeted architectural response to the
verification-failure category.

### 4. Conformity Bias in Multi-Agent Systems

In a multi-agent pipeline, agents can stop reasoning independently and
converge on whatever the first agent said, especially under high confidence.
If the Verifier sees the Tutor's output before forming its own judgment, it
anchors on it — producing a pipeline that *looks* like double-checking but
is actually just repeating the Tutor's judgment with extra steps.

**Our architectural response:** the Verifier receives only the original
problem and the student's answer — never the Tutor's verdict. The Compare
node checks for agreement only after both agents have independently
committed to a verdict. Empirically, we still observe ~90% conformity
between Tutor and Verifier — consistent with correlated blind spots from
similar prompting/training rather than fully independent error patterns —
which is precisely why the tie-breaking Recovery role matters: the gain is
concentrated in the ~10% disagreement cases.

### 5. Hybrid Architecture Note (Internal)

The BEA 2026 paper uses a Knowledge Graph-grounded model for step-by-step
verification, which is precise but domain-specific to propositional logic.
Our pipeline uses KG-grounded ground truth **only for evaluation** (scoring
our pipeline's output against the correct label), never as an input to the
agents. The agents themselves see only the problem and the student's
answer, the way a real tutor would — keeping the setup realistic and
generalizable beyond propositional logic.

## Our Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                         INPUT                            │
│  problem_statement + student_solution_step               │
│  (ground truth label kept separate for eval only)         │
└───────────────────────┬───────────────────────────────────┘
                         │
             ┌───────────▼───────────┐
             │      TUTOR AGENT       │
             │  sees: problem + step  │
             │  out:  verdict +       │
             │        feedback_text + │
             │        confidence      │
             └───────────┬───────────┘
                         │
             ┌───────────▼───────────┐
             │    VERIFIER AGENT      │
             │  sees: problem + step  │
             │  (NOT Tutor's verdict) │
             │  out:  verdict +       │
             │        feedback_text + │
             │        confidence      │
             └───────────┬───────────┘
                         │
             ┌───────────▼───────────┐
             │    COMPARE NODE        │
             │  (pure code, no LLM)   │
             │  AGREE → pass through  │
             │  DISAGREE → Recovery   │
             └──────┬─────────┬───────┘
                    │         │
              AGREE │         │ DISAGREE
                    │         │
         ┌──────────▼┐    ┌───▼─────────────────┐
         │ FINAL      │    │   RECOVERY NODE      │
         │ OUTPUT     │    │  sees: both          │
         │ (~90% of   │    │  verdicts + input    │
         │  states)   │    │  tie-break LLM call  │
         └────────────┘    └────────┬─────────────┘
                                    │
                           ┌────────▼──────────┐
                           │    FINAL OUTPUT     │
                           │  verdict +           │
                           │  feedback_text +     │
                           │  recovery_flag=True  │
                           │  (~10% of states)    │
                           └───────────────────────┘
```

Two independent orchestrations of this architecture were implemented and
evaluated:

- **Experiment 1 — `run_pipeline.py`**: a linear, script-based orchestration.
- **Experiment 2 — this LangGraph implementation** (`dt_code/graph/`,
  `run_graph_pipeline.py`): Tutor → Verifier → Compare → Recovery (if
  needed) → Finalize.

Both share identical grading prompts and differ only in orchestration
framework. They are reported side-by-side in the paper as a check that the
effect is architectural rather than an artifact of one implementation's
control flow — not as independent-sample replication (see
[Limitations](#limitations)).

## Metrics We Measure

| Metric | Definition | Observed |
|---|---|---|
| Over-validation rate | % of incorrect student steps marked correct | 33.9–35.8% → 24.7–26.9% |
| Over-rejection rate | % of valid-but-suboptimal steps marked incorrect | 2.0–2.5%, no significant change |
| Conformity rate | % of cases where Verifier's verdict matches Tutor's, regardless of correctness | ~90% |
| Recovery rate | % of states where Tutor and Verifier disagreed (= 1 − conformity rate) | ~9.6–9.9% |
| Recovery precision | Among Recovery-invoked cases, % where the final verdict matched ground truth | 85.7–86.3% |
| Latency / cost overhead | Extra LLM calls vs. single-agent baseline | 2.09× mean per-state latency (19.1s → 39.9s) |

## Limitations

- **Shared evaluation pool.** Both experiments draw from the same
  underlying pool of ~516 proof states, so cross-experiment agreement
  should be read as an implementation-robustness check, not confirmation on
  independent samples.
- **Single model, all roles.** Using Qwen2.5-7B-Instruct for every role
  eliminates model-capability confounds but also means "independence"
  between Tutor and Verifier is prompt-level independence within one
  model's distribution, not architectural diversity — consistent with the
  ~90% conformity rate we observe.
- **Underpowered over-rejection comparison.** The ground-truth label
  distribution is heavily skewed toward `incorrect` steps (~95% of matched
  states), so the over-rejection test has few discordant pairs (16 and 2 in
  the two experiments) and little statistical power. We do not interpret
  the null result there as evidence of no effect.
- **No self-consistency baseline yet.** We haven't compared against a
  cheaper alternative such as majority-voting over repeated Tutor samples
  at the same model, which would isolate architectural gain from added
  inference compute. Planned as follow-up work.

## Target Output

- Runnable LangGraph pipeline with Tutor + Verifier + Recovery — see
  `dt_code/graph/` and `python run_graph_pipeline.py`
- Evaluation results table comparing baseline vs. multi-agent
  (`run_baseline.py` vs. `run_graph_pipeline.py`)
- Short research paper (4–6 pages, workshop format)
- arXiv preprint *(link once available)*

## Running the LangGraph Pipeline

From `Multi-Agent-Tutoring-Systems/dt_code/` (with a local Qwen2.5-7B-Instruct
endpoint configured, or an API key in `../.env` if running in dev/hosted
mode):

```bash
pip install -r ../../requirements.txt
python run_graph_pipeline.py --n 10
python run_baseline.py --n 10
```

Optional: `--checkpoint` enables an in-memory LangGraph checkpointer keyed
by case id. The older sequential script `run_pipeline.py` is Experiment 1's
implementation, kept as a reference of the same control flow under a
different orchestration framework — not a deprecated version of this one.

## References

1. T. Yasir, W. Li, S. Gilson, S. D. Tithi, X. Tian, and T. Barnes,
   "Confirming Correct, Missing the Rest: LLM Tutoring Agents Struggle
   Where Feedback Matters Most," *Proceedings of the 21st Workshop on
   Innovative Use of NLP for Building Educational Applications (BEA)*, 2026.
2. X. Theimer-Lienhard et al., "Sycophancy is an Educational Safety Risk:
   Why LLM Tutors Need Sycophancy Benchmarks," 2026.
3. M. Cemri et al., "Why Do Multi-Agent LLM Systems Fail?," arXiv:2503.13657, 2025.
4. Reid et al., "Risk Analysis Techniques for Governed LLM-based
   Multi-Agent Systems," arXiv:2508.05687.
5. LangGraph Documentation — Multi-Agent Supervisor Pattern.
6. Our paper: *Independent Verification Mitigates Over-Validation in
   Multi-Agent LLM Tutoring Systems* — arXiv preprint *(link once available)*.

---
ISIMM — Institut Supérieur d'Informatique et de Mathématiques de Monastir,
University of Monastir, Tunisia