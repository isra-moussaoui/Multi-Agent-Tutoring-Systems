# Failure Modes & Recovery in Multi-Agent LLM Tutoring Pipelines

**A research project by Moussaoui Isra & Chakroun Oussama**

## What This Project Is About

Single-agent LLM tutors have a well-documented weakness: they tend to validate wrong student answers rather than correct them, especially when the student sounds confident. The natural fix is to add a second agent — a Verifier — to independently check the Tutor's judgment before it reaches the student.

**But does that actually work?**

This project tests that question. We build a small multi-agent tutoring pipeline (Tutor → Verifier → Recovery controller) in LangGraph and measure whether the Verifier genuinely catches the Tutor's errors, or just agrees with it — a phenomenon known as conformity bias. We benchmark both a single-agent baseline and our multi-agent pipeline on a real, publicly available propositional-logic tutoring dataset and compare the failure rates.

## The Research Gap We Are Filling

Three bodies of work are directly relevant to what we're doing — and none of them connects to the others yet:

| Area                         | What exists                                                                          | What's missing                                                                  |
| ---------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| LLM tutoring benchmarks      | Documented over-validation / over-rejection errors in single-agent tutors (BEA 2026) | Nobody has tested whether a multi-agent architecture fixes this                 |
| Pedagogical sycophancy       | Taxonomy of how and why LLM tutors become too agreeable (CS-SYC and others)          | No architecture that actively detects and recovers from these failures          |
| Multi-agent failure taxonomy | MAST documents 14 failure modes in agent systems (coding, business tasks)            | Never applied to tutoring; pedagogical failures have a different cost structure |

**Our contribution:** the first system that ties all three together — multi-agent verification architecture applied to tutoring, evaluated on a real benchmark, with a failure taxonomy adapted to the educational domain.

## Papers We Read & Key Takeaways

### 1. Confirming Correct, Missing the Rest (BEA 2026)

**What the paper does:** Benchmarks LLM tutoring agents on propositional logic exercises. Each student solution step is labeled as optimal, valid-but-suboptimal, or incorrect, grounded in a knowledge graph.

**Key finding:** LLM tutors score near-ceiling on confirming correct steps, but systematically fail at two things:

* **Over-validation:** marking an incorrect step as correct (the most dangerous failure in education)
* **Over-rejection:** rejecting a valid-but-suboptimal step as wrong

These failures persist across multiple models, which suggests it's an architectural problem, not a prompting or model-size problem.

**Why this matters for us:** This is our evaluation dataset. We run our pipeline on the same benchmark and compare our over-validation and over-rejection rates against their single-agent baselines.

**Dataset:** tahreemm/BEA_2026 on GitHub — ~10,836 solution-feedback pairs with KG-grounded ground truth labels.

### 2. Sycophancy is an Educational Safety Risk (2026)

**What the paper does:** Defines and taxonomizes pedagogical sycophancy — when an LLM tutor becomes too agreeable in ways that harm learning.

**Key failure types identified:**

* **CS-SYC (Context-Switch Sycophancy):** the student changes their answer or challenges the tutor's correction, and the tutor backs down and validates the wrong answer anyway
* **Authority pressure:** student claims a teacher or textbook said something different, and the tutor defers
* **Face-saving:** tutor avoids correcting to preserve the student's confidence

**Core argument:** this is not a style issue. Validating a misconception instead of correcting it is an educational safety risk — the student leaves the interaction with the wrong understanding reinforced.

**Why this matters for us:** These are the specific failure types we watch for in our Tutor agent's outputs. Our Verifier is specifically designed to catch CS-SYC patterns — cases where the Tutor agrees with a wrong answer under student pressure.

### 3. Why Do Multi-Agent LLM Systems Fail? — MAST Taxonomy (Cemri et al., arXiv:2503.13657)

**What the paper does:** Systematically analyzes failure modes across multi-agent LLM systems, producing a taxonomy of 14 failure types grouped into 3 categories.

**The 3 categories:**

* **Specification & system design failures (~41.8%):** the agents are set up in a way that guarantees certain failure patterns before any task is even run
* **Inter-agent misalignment (~36.9%):** agents misunderstand each other's outputs, disagree without resolving, or pass errors downstream
* **Verification & task checking failures (~21.3%):** no agent validates the output of another, so errors compound silently

**Key stat:** multi-agent systems fail between 41% and 86.7% of the time on standard benchmarks depending on task complexity.

**Why this matters for us:** This is our theoretical framework. We adapt the MAST categories to the tutoring domain — some failure modes don't apply, and tutoring adds new ones (sycophancy under student pressure has no equivalent in coding or business task benchmarks). We produce a small adapted taxonomy as part of our paper.

### 4. Conformity Bias in Multi-Agent Systems

**What the literature shows:** In a multi-agent pipeline, agents can stop reasoning independently and instead converge on whatever the first agent said — especially when that agent expressed high confidence. This is called conformity bias.

**The core problem for our project:** adding a Verifier agent does not automatically mean you get independent verification. If the Verifier sees the Tutor's output before forming its own judgment, it anchors on that output. The result is a pipeline that looks like it's doing double-checking but is actually just repeating the Tutor's judgment with extra steps.

**Our architectural response:** the Verifier agent in our pipeline receives only the original problem and the student's answer — it never sees the Tutor's verdict before producing its own. The Compare node checks for agreement after both agents have independently committed to a verdict. This is the key design choice that makes the Verifier meaningful rather than decorative.

### 5. Hybrid Architecture Note (Internal)

We noted one important design constraint from the BEA 2026 paper: they used a Knowledge Graph-grounded model for step-by-step verification because the propositional logic domain allows exhaustive enumeration of valid solution paths. This makes KG-grounded evaluation very precise — but domain-specific and hard to generalize.

Our pipeline uses the KG-grounded ground truth only for evaluation (comparing our pipeline's output to the correct label), not as an input to the agents. The agents themselves operate the way a real tutor would: they see the problem and the student's answer, with no access to the answer key. This keeps our setup realistic and generalizable beyond propositional logic.

## Our Pipeline Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                         INPUT                           │
│  problem_statement + student_solution_step             │
│  (ground truth label kept separate for eval only)      │
└───────────────────────┬─────────────────────────────────┘
                        │
            ┌───────────▼───────────┐
            │      TUTOR AGENT      │
            │  sees: problem + step │
            │  out:  verdict +      │
            │        feedback_text + │
            │        confidence     │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │    VERIFIER AGENT     │
            │  sees: problem + step │
            │  (NOT Tutor's verdict)│
            │  out:  verdict +      │
            │        feedback_text +│
            │        confidence     │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │    COMPARE NODE       │
            │  (pure code, no LLM)  │
            │  AGREE → pass through │
            │  DISAGREE → Recovery  │
            └──────┬─────────┬──────┘
                   │         │
             AGREE │         │ DISAGREE
                   │         │
        ┌──────────▼┐    ┌───▼────────────────┐
        │ FINAL     │    │   RECOVERY NODE    │
        │ OUTPUT    │    │  sees: both        │
        │           │    │  verdicts + input  │
        └───────────┘    │  tie-break LLM     │
                         │  call or escalate  │
                         └────────┬───────────┘
                                  │
                         ┌────────▼──────────┐
                         │    FINAL OUTPUT   │
                         │  verdict +        │
                         │  feedback_text +  │
                         │  recovery_flag=True│
                         └───────────────────┘
```

## Metrics We Measure

| Metric                      | Definition                                                                                                                                                   |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Over-validation rate**    | % of incorrect student steps that the pipeline marks as correct                                                                                              |
| **Over-rejection rate**     | % of valid-but-suboptimal steps that the pipeline marks as incorrect                                                                                         |
| **Conformity rate**         | % of cases where the Verifier's verdict matches the Tutor's verdict (regardless of correctness) — high conformity = the Verifier is not actually independent |
| **Recovery precision**      | Among cases that went to Recovery, % where the final verdict was correct                                                                                     |
| **Latency / cost overhead** | Extra LLM calls introduced by the multi-agent setup vs. the single-agent baseline                                                                            |

## Target Output

* Runnable LangGraph pipeline with Tutor + Verifier + Recovery
* Evaluation results table comparing baseline vs. multi-agent
* Short research paper (4–6 pages, workshop format)
* arXiv preprint

## References

1. Confirming Correct, Missing the Rest: LLM Tutoring Agents Struggle Where Feedback Matters Most — BEA 2026 Workshop, EACL
2. Sycophancy is an Educational Safety Risk: Why LLM Tutors Need Sycophancy Benchmarks — arXiv:2605.14604
3. Why Do Multi-Agent LLM Systems Fail? — Cemri et al., arXiv:2503.13657
4. Risk Analysis Techniques for Governed LLM-based Multi-Agent Systems — Reid et al., arXiv:2508.05687
5. LangGraph Documentation — Multi-Agent Supervisor Pattern
6. ISIMM — Institut Supérieur d'Informatique et de Mathématiques de Monastir, University of Monastir, Tunisia
