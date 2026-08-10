"""
run_pipeline.py

REFERENCE / sequential implementation of the multi-agent pipeline.
Prefer `run_graph_pipeline.py` for the LangGraph tutoring core
(Tutor -> Verifier -> Compare -> Recovery -> Finalize in dt_code/graph/).

The full multi-agent pipeline described in the README (Tutor -> Verifier ->
Compare -> Recovery), built on top of the same free-API approach as
run_baseline.py. No local model inference anywhere -- Student, Tutor,
Verifier and Recovery are all REST calls to free-tier hosted APIs; the only
local computation is KG_local's in-memory BFS (instant).

Pipeline per sampled proof state:
  1. STUDENT   (LLM call): generate a next step (same as run_baseline.py).
  2. Ground truth label, computed 100% locally via KG_local.py.
  3. TUTOR     (LLM call): blind_tutor_prompt grades the step, sees NOTHING
                but the problem + student's step.
  4. VERIFIER  (LLM call): verifier_prompt independently grades the SAME
                step, sees the problem + student's step -- NEVER the Tutor's
                verdict. This is the design choice from the README that
                prevents conformity bias: the Verifier can't anchor on the
                Tutor's output because it never sees it.
  5. COMPARE   (pure code, no LLM call): do Tutor and Verifier agree?
       - AGREE    -> that verdict is final, recovery_flag=False.
       - DISAGREE -> go to Recovery.
  6. RECOVERY  (LLM call, only on disagreement): recovery_prompt sees BOTH
                verdicts + the original problem and produces a tie-broken
                FINAL_VERDICT, recovery_flag=True.

Recommended default split: everything runs on Groq (Gemini kept as an
optional --*-provider gemini override, but not the default -- Gemini's free
tier has been unstable this year, see llm_client.py). Each role still uses a
DIFFERENT MODEL so no single per-model daily cap on Groq's free tier gets
hit by all four roles at once, and so the Verifier is a genuinely different
model family from the Tutor:
  - Student:  Groq / openai/gpt-oss-20b   (small/fast -> makes realistic mistakes)
  - Tutor:    Groq / qwen/qwen3.6-27b      (different family from Student)
  - Verifier: Groq / openai/gpt-oss-120b   (different size from both
              Student and Tutor -- genuine independence, own quota bucket)
  - Recovery: Groq / openai/gpt-oss-120b   (only called on disagreements, so
              it's a small fraction of total calls -- fine to share
              Verifier's model)

NOTE (Aug 2026): llama-3.1-8b-instant and llama-3.3-70b-versatile are being
shut down by Groq on 08/16/26 -- don't reintroduce them as defaults even
though you'll still see them in older examples/blog posts. Check
https://console.groq.com/docs/deprecations if any model here errors out.

Usage:
    python run_pipeline.py --n 516 --seed 42

Multiple keys in ../.env (recommended for the full run):
    GROQ_API_KEY=key1
    GROQ_API_KEYS=key2,key3

The client rotates keys on rate/daily limits so you do not need long sleeps.
Same seed + same default models as run_baseline.py for a fair comparison.
Resume support mirrors run_baseline.py.

(GOOGLE_API_KEY / GOOGLE_API_KEYS only needed if you pass --*-provider gemini.)
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import KG_local
import prompts
from llm_client import call_llm, extract_json_object, LLMError, DailyQuotaExceeded, get_key_pool
from run_baseline import (
    load_prestate_rows,
    stratified_sample,
    run_student,
    ground_truth_label,
)
from utils.env_loader import load_env

load_env()

DATA_DIR = Path(__file__).parent / "Data"
PROPS_DIR = DATA_DIR / "props"
OUTPUT_DIR = DATA_DIR / "llm_output"
OUTPUT_PATH = OUTPUT_DIR / "pipeline_run.jsonl"
DEFAULT_N = 516

VERDICT_TO_LABEL = {
    "correct": "optimal",
    "suboptimal": "valid_alternative",
    "incorrect": "incorrect",
}


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


def run_verifier(row, student_parsed, provider, model, sleep_s, temperature=0.2, max_tokens=3072):
    """Independent audit -- deliberately takes no tutor_verdict argument.
    There is nothing to pass: the Verifier must never see the Tutor's output."""
    givens = row["Givens"]
    intermediates = row["Intermediates"]["Expressions"]
    conclusion = row["Conclusion"]
    next_step = student_parsed.get("NEXT_STEP", "") if student_parsed else ""
    rule = student_parsed.get("RULE", "") if student_parsed else ""
    parents = student_parsed.get("PARENT_STATEMENTS", "") if student_parsed else ""

    prompt = prompts.verifier_prompt(givens, intermediates, conclusion, next_step, rule, parents)
    raw = call_llm(prompt, provider=provider, model=model, temperature=temperature, max_tokens=max_tokens)
    parsed = extract_json_object(raw)
    time.sleep(sleep_s)
    return parsed, raw


def run_recovery(row, student_parsed, tutor_verdict, tutor_feedback,
                  verifier_verdict, verifier_feedback,
                  provider, model, sleep_s, temperature=0.2, max_tokens=3072):
    givens = row["Givens"]
    intermediates = row["Intermediates"]["Expressions"]
    conclusion = row["Conclusion"]
    next_step = student_parsed.get("NEXT_STEP", "") if student_parsed else ""
    rule = student_parsed.get("RULE", "") if student_parsed else ""
    parents = student_parsed.get("PARENT_STATEMENTS", "") if student_parsed else ""

    prompt = prompts.recovery_prompt(
        givens, intermediates, conclusion, next_step, rule, parents,
        tutor_verdict, tutor_feedback, verifier_verdict, verifier_feedback,
    )
    raw = call_llm(prompt, provider=provider, model=model, temperature=temperature, max_tokens=max_tokens)
    parsed = extract_json_object(raw)
    time.sleep(sleep_s)
    return parsed, raw


def compute_metrics(traces):
    """
    Same over-validation / over-rejection definitions as run_baseline.py, but
    computed on the PIPELINE's final_verdict (post Compare/Recovery), plus
    two multi-agent-specific metrics:

      conformity_rate:    % of cases where Verifier's verdict matches Tutor's
                           verdict. High conformity is the failure mode the
                           README warns about -- it would suggest the
                           Verifier isn't actually independent in practice,
                           even though it's architecturally blind to the
                           Tutor's output (models can still converge on the
                           "obvious" answer for a given step).
      recovery_rate:       % of cases that needed Recovery (Tutor and
                           Verifier disagreed).
      recovery_precision:  among Recovery cases, % where the FINAL verdict
                           matched ground truth. Low precision here means
                           the tie-breaker isn't earning its cost.
    """
    total = 0
    over_validation = 0
    over_rejection = 0
    agree_gt = 0
    conformity = 0
    recovery_cases = 0
    recovery_correct = 0

    for t in traces:
        gt = t.get("ground_truth_label")
        final_verdict = (t.get("final_verdict") or "").strip().lower()
        tutor_verdict = (t.get("tutor_verdict") or "").strip().lower()
        verifier_verdict = (t.get("verifier_verdict") or "").strip().lower()
        if gt is None or not final_verdict or not tutor_verdict or not verifier_verdict:
            continue
        total += 1

        final_says_wrong = final_verdict == "incorrect"
        actually_wrong = gt == "incorrect"

        if actually_wrong and not final_says_wrong:
            over_validation += 1
        if (not actually_wrong) and final_says_wrong:
            over_rejection += 1
        if VERDICT_TO_LABEL.get(final_verdict) == gt:
            agree_gt += 1
        if tutor_verdict == verifier_verdict:
            conformity += 1

        if t.get("recovery_flag"):
            recovery_cases += 1
            if VERDICT_TO_LABEL.get(final_verdict) == gt:
                recovery_correct += 1

    return {
        "n": total,
        "over_validation_rate": over_validation / total if total else None,
        "over_rejection_rate": over_rejection / total if total else None,
        "exact_agreement_rate": agree_gt / total if total else None,
        "conformity_rate": conformity / total if total else None,
        "recovery_rate": recovery_cases / total if total else None,
        "recovery_precision": (recovery_correct / recovery_cases) if recovery_cases else None,
    }


DEFAULT_PROVIDER = "mistral"
DEFAULT_STUDENT_MODEL = "mistral-large-latest"
DEFAULT_TUTOR_MODEL = "mistral-large-latest"
DEFAULT_VERIFIER_MODEL = "mistral-large-latest"
DEFAULT_RECOVERY_MODEL = "mistral-large-latest"

DEBUG_PATH = None

_PROVIDER_CHOICES = ["mistral", "groq", "gemini"]


def main():
    load_env()

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                     help=f"number of proof states to sample (default {DEFAULT_N} = full dataset). "
                          f"Use 0 to mean 'all rows'.")

    ap.add_argument("--student-provider", type=str, default=DEFAULT_PROVIDER, choices=_PROVIDER_CHOICES)
    ap.add_argument("--student-model", type=str, default=DEFAULT_STUDENT_MODEL)
    ap.add_argument("--student-temperature", type=float, default=0.7)

    ap.add_argument("--tutor-provider", type=str, default=DEFAULT_PROVIDER, choices=_PROVIDER_CHOICES)
    ap.add_argument("--tutor-model", type=str, default=DEFAULT_TUTOR_MODEL)
    ap.add_argument("--tutor-temperature", type=float, default=0.2)

    ap.add_argument("--verifier-provider", type=str, default=DEFAULT_PROVIDER, choices=_PROVIDER_CHOICES)
    ap.add_argument("--verifier-model", type=str, default=DEFAULT_VERIFIER_MODEL)
    ap.add_argument("--verifier-temperature", type=float, default=0.2)

    ap.add_argument("--recovery-provider", type=str, default=DEFAULT_PROVIDER, choices=_PROVIDER_CHOICES)
    ap.add_argument("--recovery-model", type=str, default=DEFAULT_RECOVERY_MODEL)
    ap.add_argument("--recovery-temperature", type=float, default=0.2)

    ap.add_argument("--sleep", type=float, default=0.0,
                     help="seconds to sleep between calls. Default 0 -- with multiple "
                          "GROQ_API_KEYS the client rotates on 429 instead of waiting.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global DEBUG_PATH
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_PATH = OUTPUT_DIR / "pipeline_run_failures.jsonl"

    rows = load_prestate_rows()
    print(f"Loaded {len(rows)} proof states total.")
    n = len(rows) if args.n <= 0 else min(args.n, len(rows))
    sample = stratified_sample(rows, n, seed=args.seed)
    print(f"Sampled {len(sample)} states across {len(set(r['currentProblem'] for r in sample))} problems "
          f"(seed={args.seed}).")
    print(f"Student model:   {args.student_provider}/{args.student_model} (temp={args.student_temperature})")
    print(f"Tutor model:     {args.tutor_provider}/{args.tutor_model} (temp={args.tutor_temperature})")
    print(f"Verifier model:  {args.verifier_provider}/{args.verifier_model} (temp={args.verifier_temperature})")
    print(f"Recovery model:  {args.recovery_provider}/{args.recovery_model} (temp={args.recovery_temperature})")
    providers = {args.student_provider, args.tutor_provider, args.verifier_provider, args.recovery_provider}
    print("API key pools: " + "; ".join(get_key_pool(p).summary() for p in sorted(providers)))
    print(f"Inter-call sleep: {args.sleep}s")

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
                # --- Student ---
                student_parsed, student_raw = run_student(
                    row, args.student_provider, args.student_model, args.sleep,
                    temperature=args.student_temperature)
                if not student_parsed or "NEXT_STEP" not in student_parsed:
                    print("  ! student response could not be parsed, skipping")
                    fail_counts["student_parse_failed"] += 1
                    debug_f.write(json.dumps({"id": row["id"], "stage": "student", "raw": student_raw}) + "\n")
                    debug_f.flush()
                    continue

                gt_label = ground_truth_label(row, student_parsed["NEXT_STEP"])

                # --- Tutor (blind) ---
                tutor_parsed, tutor_raw = run_tutor(
                    row, student_parsed, args.tutor_provider, args.tutor_model, args.sleep,
                    temperature=args.tutor_temperature)
                if not tutor_parsed or "VERDICT" not in tutor_parsed:
                    print("  ! tutor response could not be parsed, skipping")
                    fail_counts["tutor_parse_failed"] += 1
                    debug_f.write(json.dumps({"id": row["id"], "stage": "tutor", "raw": tutor_raw}) + "\n")
                    debug_f.flush()
                    continue
                tutor_verdict = tutor_parsed.get("VERDICT")
                tutor_feedback = tutor_parsed.get("FEEDBACK")

                # --- Verifier (blind, independent -- never sees tutor_verdict) ---
                verifier_parsed, verifier_raw = run_verifier(
                    row, student_parsed, args.verifier_provider, args.verifier_model, args.sleep,
                    temperature=args.verifier_temperature)
                if not verifier_parsed or "VERDICT" not in verifier_parsed:
                    print("  ! verifier response could not be parsed, skipping")
                    fail_counts["verifier_parse_failed"] += 1
                    debug_f.write(json.dumps({"id": row["id"], "stage": "verifier", "raw": verifier_raw}) + "\n")
                    debug_f.flush()
                    continue
                verifier_verdict = verifier_parsed.get("VERDICT")
                verifier_feedback = verifier_parsed.get("FEEDBACK")

                # --- Compare (pure code, no LLM call) ---
                agree = (str(tutor_verdict).strip().lower() == str(verifier_verdict).strip().lower())

                recovery_flag = False
                recovery_parsed = None
                if agree:
                    final_verdict = tutor_verdict
                    final_feedback = tutor_feedback
                else:
                    # --- Recovery (only on disagreement) ---
                    recovery_flag = True
                    recovery_parsed, recovery_raw = run_recovery(
                        row, student_parsed, tutor_verdict, tutor_feedback,
                        verifier_verdict, verifier_feedback,
                        args.recovery_provider, args.recovery_model, args.sleep,
                        temperature=args.recovery_temperature)
                    if not recovery_parsed or "FINAL_VERDICT" not in recovery_parsed:
                        print("  ! recovery response could not be parsed, skipping")
                        fail_counts["recovery_parse_failed"] += 1
                        debug_f.write(json.dumps({"id": row["id"], "stage": "recovery", "raw": recovery_raw}) + "\n")
                        debug_f.flush()
                        continue
                    final_verdict = recovery_parsed.get("FINAL_VERDICT")
                    final_feedback = recovery_parsed.get("REASONING")

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
                    "tutor_feedback": tutor_feedback,
                    "verifier_verdict": verifier_verdict,
                    "verifier_feedback": verifier_feedback,
                    "agree": agree,
                    "recovery_flag": recovery_flag,
                    "recovery_full_response": recovery_parsed,
                    "final_verdict": final_verdict,
                    "final_feedback": final_feedback,
                }
                traces.append(trace)
                out_f.write(json.dumps(trace) + "\n")
                out_f.flush()
                print(f"  gt={gt_label} tutor={tutor_verdict} verifier={verifier_verdict} "
                      f"{'AGREE' if agree else 'DISAGREE->recovery'} final={final_verdict}")

            except DailyQuotaExceeded as e:
                print(f"\n! Daily quota exhausted on every configured key: {e}")
                print("  Progress is saved. Add more keys to GROQ_API_KEYS (or GOOGLE_API_KEYS), "
                      "then re-run the exact same command -- finished ids are skipped. "
                      "Or wait for quota reset.")
                stopped_early_reason = str(e)
                break
            except LLMError as e:
                print(f"  ! LLM error, skipping: {e}")
                fail_counts["llm_error"] += 1
                continue

    # Reload everything from disk so metrics reflect the full run so far.
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
        print(f"Run stopped early due to daily quota. "
              f"{len(sample) - len(traces) - sum(fail_counts.values())} examples in this "
              f"session's sample were not attempted -- resume later to fill them in.")

    metrics = compute_metrics(all_traces)
    print("\n=== Multi-agent pipeline metrics (all sessions combined) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nTo compare against the single-agent baseline, also run:")
    print("    python run_baseline.py --n", n, "--seed", args.seed)
    print("and compare its over_validation_rate / over_rejection_rate against this "
          "pipeline's -- that comparison is the core result for the paper.")


if __name__ == "__main__":
    main()
