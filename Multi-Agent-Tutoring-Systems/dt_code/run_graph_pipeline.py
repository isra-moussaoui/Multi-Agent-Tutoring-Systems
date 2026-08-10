"""
run_graph_pipeline.py

Batch evaluation harness for the LangGraph tutoring core.

Student simulation + KG_local ground-truth labeling stay OUTSIDE the graph
(evaluation harness). The compiled graph owns only:

    Tutor → Verifier → Compare → (Recovery?) → Finalize

Usage (from dt_code/):
    python run_graph_pipeline.py --n 100

Output mirrors run_pipeline.py (pipeline_run.jsonl) plus per-node latencies
so latency/cost overhead can be computed against the baseline.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from graph import GraphRoleConfigs, RoleConfig, build_tutoring_graph
from llm_client import DailyQuotaExceeded, LLMError
from run_baseline import (
    ground_truth_label,
    load_prestate_rows,
    run_student,
    stratified_sample,
)
from run_pipeline import compute_metrics
from utils.env_loader import load_env

load_env()

DATA_DIR = Path(__file__).parent / "Data"
OUTPUT_DIR = DATA_DIR / "llm_output"
OUTPUT_PATH = OUTPUT_DIR / "pipeline_run.jsonl"
DEBUG_PATH = OUTPUT_DIR / "pipeline_run_failures.jsonl"

DEFAULT_PROVIDER = "mistral"
DEFAULT_STUDENT_MODEL = "mistral-large-latest"
DEFAULT_TUTOR_MODEL = "mistral-large-latest"
DEFAULT_VERIFIER_MODEL = "mistral-large-latest"
DEFAULT_RECOVERY_MODEL = "mistral-large-latest"
_PROVIDER_CHOICES = ["mistral", "groq", "gemini"]


def _trace_from_graph_result(row, student_parsed, gt_label, result: dict) -> dict:
    tutor = result.get("tutor") or {}
    verifier = result.get("verifier") or {}
    recovery = result.get("recovery") or {}
    compare = result.get("compare") or {}
    return {
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
        "tutor_verdict": tutor.get("verdict"),
        "tutor_feedback": tutor.get("feedback"),
        "tutor_latency_ms": tutor.get("latency_ms"),
        "tutor_model": tutor.get("model"),
        "verifier_verdict": verifier.get("verdict"),
        "verifier_feedback": verifier.get("feedback"),
        "verifier_latency_ms": verifier.get("latency_ms"),
        "verifier_model": verifier.get("model"),
        "agree": compare.get("agreed"),
        "recovery_flag": result.get("recovery_flag"),
        "recovery_full_response": {
            "FINAL_VERDICT": recovery.get("final_verdict"),
            "REASONING": recovery.get("reasoning"),
        } if recovery else None,
        "recovery_latency_ms": recovery.get("latency_ms"),
        "recovery_model": recovery.get("model"),
        "final_verdict": result.get("final_verdict"),
        "final_feedback": result.get("final_feedback"),
        "graph_status": result.get("status"),
        "graph_errors": result.get("errors") or [],
    }


def _latency_overhead(traces: list[dict]) -> dict:
    """Extra LLM-node latency from Verifier (+ Recovery when used) vs Tutor-only."""
    n = 0
    tutor_ms = 0.0
    pipeline_ms = 0.0
    for t in traces:
        tl = t.get("tutor_latency_ms")
        vl = t.get("verifier_latency_ms")
        if tl is None or vl is None:
            continue
        n += 1
        tutor_ms += float(tl)
        pipeline_ms += float(tl) + float(vl)
        if t.get("recovery_flag") and t.get("recovery_latency_ms") is not None:
            pipeline_ms += float(t["recovery_latency_ms"])
    if not n:
        return {"n_with_latency": 0}
    return {
        "n_with_latency": n,
        "mean_tutor_only_ms": tutor_ms / n,
        "mean_pipeline_ms": pipeline_ms / n,
        "mean_overhead_ms": (pipeline_ms - tutor_ms) / n,
        "mean_overhead_ratio": (pipeline_ms / tutor_ms) if tutor_ms else None,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Run multi-agent tutoring pipeline via LangGraph core",
    )
    ap.add_argument("--n", type=int, default=100, help="number of proof states to sample")
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
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between LLM calls")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--checkpoint",
        action="store_true",
        help="use in-memory LangGraph checkpointer (thread_id=case id)",
    )
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        from graph import build_tutoring_graph_with_memory
        graph = build_tutoring_graph_with_memory()
    else:
        graph = build_tutoring_graph()

    role_configs = GraphRoleConfigs(
        tutor=RoleConfig(
            provider=args.tutor_provider, model=args.tutor_model,
            temperature=args.tutor_temperature, sleep_s=args.sleep,
        ),
        verifier=RoleConfig(
            provider=args.verifier_provider, model=args.verifier_model,
            temperature=args.verifier_temperature, sleep_s=args.sleep,
        ),
        recovery=RoleConfig(
            provider=args.recovery_provider, model=args.recovery_model,
            temperature=args.recovery_temperature, sleep_s=args.sleep,
        ),
    )

    rows = load_prestate_rows()
    print(f"Loaded {len(rows)} proof states total.")
    sample = stratified_sample(rows, args.n, seed=args.seed)
    print(f"Sampled {len(sample)} states across {len(set(r['currentProblem'] for r in sample))} problems.")
    print(f"Orchestration: LangGraph tutoring core (Student + KG outside)")
    print(f"Student model:   {args.student_provider}/{args.student_model} (temp={args.student_temperature})")
    print(f"Tutor model:     {args.tutor_provider}/{args.tutor_model} (temp={args.tutor_temperature})")
    print(f"Verifier model:  {args.verifier_provider}/{args.verifier_model} (temp={args.verifier_temperature})")
    print(f"Recovery model:  {args.recovery_provider}/{args.recovery_model} (temp={args.recovery_temperature})")

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
        print(f"Found {len(already_done_ids)} already-completed examples -- resuming.")
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
                    temperature=args.student_temperature,
                )
                if not student_parsed or "NEXT_STEP" not in student_parsed:
                    print("  ! student response could not be parsed, skipping")
                    fail_counts["student_parse_failed"] += 1
                    debug_f.write(json.dumps({"id": row["id"], "stage": "student", "raw": student_raw}) + "\n")
                    debug_f.flush()
                    continue

                gt_label = ground_truth_label(row, student_parsed["NEXT_STEP"])

                initial_state = {
                    "case_id": str(row["id"]),
                    "givens": row["Givens"],
                    "intermediates": row["Intermediates"]["Expressions"],
                    "conclusion": row["Conclusion"],
                    "student_step": {
                        "next_step": student_parsed.get("NEXT_STEP", ""),
                        "rule": student_parsed.get("RULE", ""),
                        "parents": student_parsed.get("PARENT_STATEMENTS", ""),
                    },
                    "errors": [],
                    "status": "ok",
                    "eval_only": {"ground_truth_label": gt_label},
                }
                invoke_config = {
                    "configurable": {
                        **role_configs.to_configurable(),
                        "thread_id": str(row["id"]),
                    }
                }
                result = graph.invoke(initial_state, config=invoke_config)

                if result.get("status") == "parse_error" or not result.get("final_verdict"):
                    stage = "graph"
                    errs = result.get("errors") or []
                    if "tutor_parse_failed" in errs:
                        stage = "tutor"
                        fail_counts["tutor_parse_failed"] += 1
                        raw = (result.get("tutor") or {}).get("raw")
                    elif "verifier_parse_failed" in errs:
                        stage = "verifier"
                        fail_counts["verifier_parse_failed"] += 1
                        raw = (result.get("verifier") or {}).get("raw")
                    elif "recovery_parse_failed" in errs or "finalize_missing_recovery_verdict" in errs:
                        stage = "recovery"
                        fail_counts["recovery_parse_failed"] += 1
                        raw = (result.get("recovery") or {}).get("raw")
                    else:
                        fail_counts["graph_parse_failed"] += 1
                        raw = None
                    print(f"  ! graph parse/status error ({stage}), skipping")
                    debug_f.write(json.dumps({
                        "id": row["id"], "stage": stage,
                        "errors": errs, "raw": raw,
                        "status": result.get("status"),
                    }) + "\n")
                    debug_f.flush()
                    continue

                trace = _trace_from_graph_result(row, student_parsed, gt_label, result)
                traces.append(trace)
                out_f.write(json.dumps(trace) + "\n")
                out_f.flush()
                print(
                    f"  gt={gt_label} tutor={trace['tutor_verdict']} "
                    f"verifier={trace['verifier_verdict']} "
                    f"{'AGREE' if trace['agree'] else 'DISAGREE->recovery'} "
                    f"final={trace['final_verdict']}"
                )

            except DailyQuotaExceeded as e:
                print(f"\n! Daily quota exhausted, stopping run early: {e}")
                print("  Progress is saved. Re-run the same command later to resume.")
                stopped_early_reason = str(e)
                break
            except LLMError as e:
                print(f"  ! LLM error, skipping: {e}")
                fail_counts["llm_error"] += 1
                continue

    all_traces = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_traces.append(json.loads(line))

    print(f"\nSaved {len(traces)} new traces this session "
          f"({len(all_traces)} total across all sessions) to {OUTPUT_PATH}")
    if sum(fail_counts.values()):
        print(f"Skipped {sum(fail_counts.values())} examples this session: {dict(fail_counts)}")
        print(f"Raw failed responses logged to {DEBUG_PATH}")
    if stopped_early_reason:
        print("Run stopped early due to daily quota.")

    metrics = compute_metrics(all_traces)
    print("\n=== Multi-agent pipeline metrics (LangGraph core, all sessions) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    overhead = _latency_overhead(all_traces)
    print("\n=== Latency overhead (Tutor-only vs full pipeline nodes) ===")
    for k, v in overhead.items():
        print(f"  {k}: {v}")

    print("\nCompare against baseline:")
    print("    python run_baseline.py --n", args.n)


if __name__ == "__main__":
    main()
