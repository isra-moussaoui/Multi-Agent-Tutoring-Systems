"""
run_baseline.py

Step 2 (get data) + Step 3 (build the single-agent baseline) from the brief,
all free and CPU-only:

  1. Load Data/cleaned_data/preState.jsonl (517 real proof states).
  2. Take a stratified sample across the 32 problems.
  3. For each sampled state:
       a. STUDENT (LLM call): generate a next step, blind -- same as the
          original repo's student_prompt.
       b. Ground truth label for that step, computed 100% locally via
          kg_local.py (no Neo4j, no API cost): "optimal" / "valid_alternative"
          / "incorrect".
       c. TUTOR (LLM call): blind_tutor_prompt grades the student's step
          WITHOUT seeing the ground truth -- this is the single-agent
          baseline/control group from the brief's Step 3.
  4. Save the full trace to Data/llm_output/baseline_run.jsonl.
  5. Print over-validation rate / over-rejection rate: does the single Tutor
     agree with the LLM student even when the student's step was actually
     wrong (over-validation), or reject it even when it was actually valid
     (over-rejection)?

Usage:
    export GROQ_API_KEY=your_free_key_here
    python run_baseline.py --n 100

Only two dependencies beyond the standard library: `requests` and `pyyaml`
(both free/open source, `pip install requests pyyaml`).
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import KG_local
import prompts
from llm_client import call_llm, extract_json_object, LLMError, DailyQuotaExceeded
from utils.env_loader import load_env

load_env()

DATA_DIR = Path(__file__).parent / "Data"
PRESTATE_PATH = DATA_DIR / "cleaned_data" / "preState.jsonl"
PROPS_DIR = DATA_DIR / "props"
OUTPUT_DIR = DATA_DIR / "llm_output"
OUTPUT_PATH = OUTPUT_DIR / "baseline_run.jsonl"


def load_prestate_rows():
    rows = []
    with open(PRESTATE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("//") or line.startswith("#") or line.startswith("/*") or line.startswith("*"):
                continue
            rows.append(json.loads(line))
    return rows


def stratified_sample(rows, n, seed=42):
    """Sample ~n rows spread proportionally across every problem (currentProblem)."""
    rng = random.Random(seed)
    by_problem = defaultdict(list)
    for r in rows:
        by_problem[r["currentProblem"]].append(r)

    problems = list(by_problem.keys())
    per_problem = max(1, n // len(problems))

    sample = []
    for p in problems:
        group = by_problem[p]
        rng.shuffle(group)
        sample.extend(group[:per_problem])

    rng.shuffle(sample)
    return sample[:n]


def run_student(row, provider, model, sleep_s, temperature=0.7, max_tokens=3072):
    givens = row["Givens"]
    intermediates = row["Intermediates"]["Expressions"]
    conclusion = row["Conclusion"]

    prompt = prompts.student_prompt(givens, intermediates, conclusion)
    raw = call_llm(prompt, provider=provider, model=model, temperature=temperature, max_tokens=max_tokens)
    parsed = extract_json_object(raw)
    time.sleep(sleep_s)
    return parsed, raw


def run_tutor(row, student_parsed, provider, model, sleep_s, temperature=0.2, max_tokens=3072):
    givens = row["Givens"]
    intermediates = row["Intermediates"]["Expressions"]
    conclusion = row["Conclusion"]

    next_step = student_parsed.get("NEXT_STEP", "") if student_parsed else ""
    rule = student_parsed.get("RULE", "") if student_parsed else ""
    parents = student_parsed.get("PARENT_STATEMENTS", "") if student_parsed else ""

    prompt = prompts.blind_tutor_prompt(givens, intermediates, conclusion, next_step, rule, parents)
    raw = call_llm(prompt, provider=provider, model=model, temperature=temperature, max_tokens=max_tokens)
    parsed = extract_json_object(raw)
    time.sleep(sleep_s)
    return parsed, raw


def ground_truth_label(row, student_next_step):
    problem = row["currentProblem"]
    givens = row["Givens"]
    intermediates = row["Intermediates"]["Expressions"]
    known = givens + intermediates
    correct_step = row["sAssertion"]

    graph = KG_local.load_graph(problem, PROPS_DIR)
    return KG_local.label_step(student_next_step, correct_step, graph, known)


VERDICT_TO_LABEL = {
    "correct": "optimal",
    "suboptimal": "valid_alternative",
    "incorrect": "incorrect",
}


def compute_metrics(traces):
    """
    over-validation: ground truth is 'incorrect', but Tutor said 'Correct'
                      (or 'Suboptimal' -- either way it failed to flag a real error)
    over-rejection:  ground truth is 'optimal' or 'valid_alternative', but
                      Tutor said 'Incorrect'
    """
    total = 0
    over_validation = 0
    over_rejection = 0
    agree = 0

    for t in traces:
        gt = t.get("ground_truth_label")
        tutor_verdict = (t.get("tutor_verdict") or "").strip().lower()
        if gt is None or not tutor_verdict:
            continue
        total += 1

        tutor_says_wrong = tutor_verdict == "incorrect"
        actually_wrong = gt == "incorrect"

        if actually_wrong and not tutor_says_wrong:
            over_validation += 1
        if (not actually_wrong) and tutor_says_wrong:
            over_rejection += 1
        if VERDICT_TO_LABEL.get(tutor_verdict) == gt:
            agree += 1

    return {
        "n": total,
        "over_validation_rate": over_validation / total if total else None,
        "over_rejection_rate": over_rejection / total if total else None,
        "exact_agreement_rate": agree / total if total else None,
    }


DEFAULT_PROVIDER = "mistral"
DEFAULT_STUDENT_MODEL = "mistral-large-latest"
DEFAULT_TUTOR_MODEL = "mistral-large-latest"

DEBUG_PATH = None  # set in main()

_PROVIDER_CHOICES = ["mistral", "groq", "gemini"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of proof states to sample")
    ap.add_argument("--provider", type=str, default=DEFAULT_PROVIDER, choices=_PROVIDER_CHOICES,
                     help="fallback default if --student-provider/--tutor-provider aren't set")
    ap.add_argument("--student-provider", type=str, default=DEFAULT_PROVIDER, choices=_PROVIDER_CHOICES)
    ap.add_argument("--tutor-provider", type=str, default=DEFAULT_PROVIDER, choices=_PROVIDER_CHOICES)
    ap.add_argument("--student-model", type=str, default=DEFAULT_STUDENT_MODEL,
                     help="student simulator model id for the chosen provider")
    ap.add_argument("--tutor-model", type=str, default=DEFAULT_TUTOR_MODEL)
    ap.add_argument("--student-temperature", type=float, default=0.7,
                     help="higher temperature = more variety/mistakes from the student, "
                          "which is what you actually want to test over-validation")
    ap.add_argument("--tutor-temperature", type=float, default=0.2)
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds to sleep between calls (free-tier rate limits)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global DEBUG_PATH
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_PATH = OUTPUT_DIR / "baseline_run_failures.jsonl"

    rows = load_prestate_rows()
    print(f"Loaded {len(rows)} proof states total.")
    sample = stratified_sample(rows, args.n, seed=args.seed)
    print(f"Sampled {len(sample)} states across {len(set(r['currentProblem'] for r in sample))} problems.")
    print(f"Student model: {args.student_provider}/{args.student_model} (temp={args.student_temperature})")
    print(f"Tutor model:   {args.tutor_provider}/{args.tutor_model} (temp={args.tutor_temperature})")

    # --- Resume support: skip anything already completed in a prior run ---
    already_done_ids = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    already_done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    if already_done_ids:
        print(f"Found {len(already_done_ids)} already-completed examples from a prior run -- resuming, not repeating them.")
        sample = [r for r in sample if r["id"] not in already_done_ids]
        print(f"{len(sample)} remaining to run this session.")

    traces = []
    fail_counts = defaultdict(int)
    stopped_early_reason = None

    out_mode = "a" if already_done_ids else "w"
    with open(OUTPUT_PATH, out_mode) as out_f, open(DEBUG_PATH, "a") as debug_f:
        for i, row in enumerate(sample, 1):
            print(f"[{i}/{len(sample)}] id={row['id']} problem={row['currentProblem']}")
            try:
                student_parsed, student_raw = run_student(
                    row, args.student_provider, args.student_model, args.sleep,
                    temperature=args.student_temperature)
                if not student_parsed or "NEXT_STEP" not in student_parsed:
                    print("  ! student response could not be parsed, skipping (see failures log)")
                    fail_counts["student_parse_failed"] += 1
                    debug_f.write(json.dumps({"id": row["id"], "stage": "student", "raw": student_raw}) + "\n")
                    debug_f.flush()
                    continue

                gt_label = ground_truth_label(row, student_parsed["NEXT_STEP"])

                tutor_parsed, tutor_raw = run_tutor(
                    row, student_parsed, args.tutor_provider, args.tutor_model, args.sleep,
                    temperature=args.tutor_temperature)
                if not tutor_parsed or "VERDICT" not in tutor_parsed:
                    print("  ! tutor response could not be parsed, skipping (see failures log)")
                    fail_counts["tutor_parse_failed"] += 1
                    debug_f.write(json.dumps({"id": row["id"], "stage": "tutor", "raw": tutor_raw}) + "\n")
                    debug_f.flush()
                    continue
                tutor_verdict = tutor_parsed.get("VERDICT")

                trace = {
                    "id": row["id"],
                    "problem": row["currentProblem"],
                    "givens": row["Givens"],
                    "intermediates": row["Intermediates"]["Expressions"],
                    "conclusion": row["Conclusion"],
                    "correct_step": row["sAssertion"],
                    "student_next_step": student_parsed.get("NEXT_STEP"),
                    "student_rule": student_parsed.get("RULE"),
                    "student_full_response": student_parsed,
                    "ground_truth_label": gt_label,
                    "tutor_verdict": tutor_verdict,
                    "tutor_feedback": tutor_parsed.get("FEEDBACK") if tutor_parsed else None,
                    "tutor_full_response": tutor_parsed,
                }
                traces.append(trace)
                out_f.write(json.dumps(trace) + "\n")
                out_f.flush()
                print(f"  student_step={trace['student_next_step']!r} "
                      f"ground_truth={gt_label} tutor_verdict={tutor_verdict}")

            except DailyQuotaExceeded as e:
                # Retrying can't fix this -- stop the whole run cleanly instead
                # of continuing to burn failed attempts against an exhausted model.
                print(f"\n! Daily quota exhausted, stopping run early: {e}")
                print("  Your progress is saved. Re-run the exact same command later "
                      "(after the quota resets, usually within a few hours) or switch "
                      "--student-model/--tutor-model/--student-provider/--tutor-provider "
                      "to a model with quota left, and it will pick up where it left off.")
                stopped_early_reason = str(e)
                break
            except LLMError as e:
                print(f"  ! LLM error, skipping: {e}")
                fail_counts["llm_error"] += 1
                continue

    # Metrics should reflect the FULL run so far (resumed + this session), not
    # just this session's new traces -- reload everything from disk.
    all_traces = []
    with open(OUTPUT_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                all_traces.append(json.loads(line))

    print(f"\nSaved {len(traces)} new traces this session "
          f"({len(all_traces)} total across all sessions) to {OUTPUT_PATH}")
    if sum(fail_counts.values()):
        print(f"Skipped {sum(fail_counts.values())} examples this session: {dict(fail_counts)}")
        print(f"Raw failed responses logged to {DEBUG_PATH} -- inspect these before trusting the metrics.")
    if stopped_early_reason:
        print(f"Run stopped early due to daily quota. {len(sample) - len(traces) - sum(fail_counts.values())} "
              f"examples in this session's sample were not attempted -- resume later to fill them in.")

    # Diagnostic: label/verdict distributions. If ground_truth is almost always
    # "optimal", your student model isn't making enough mistakes to test
    # over-validation at all -- lower --student-model strength or raise
    # --student-temperature rather than trusting a 0.0 over-validation rate.
    gt_dist = defaultdict(int)
    verdict_dist = defaultdict(int)
    for t in all_traces:
        gt_dist[t["ground_truth_label"]] += 1
        verdict_dist[(t.get("tutor_verdict") or "").strip().lower()] += 1
    print(f"\nGround-truth label distribution (all sessions): {dict(gt_dist)}")
    print(f"Tutor verdict distribution (all sessions):      {dict(verdict_dist)}")

    metrics = compute_metrics(all_traces)
    print("\n=== Single-agent baseline metrics (all sessions combined) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()