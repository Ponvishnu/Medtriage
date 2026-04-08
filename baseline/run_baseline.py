#!/usr/bin/env python3
"""
MedTriageEnv — Baseline Inference Script

Reproduces reference scores for all three tasks using:
  • RuleBasedAgent  (default, no API key required, fully deterministic)
  • LLMAgent        (optional, requires GEMINI_API_KEY)

Usage:
  # Rule-based only (reproducible scores, no API key needed)
  python baseline/run_baseline.py

  # LLM-powered agent (requires GEMINI_API_KEY)
  GEMINI_API_KEY=AIza... python baseline/run_baseline.py --llm

  # Specify Gemini model
  GEMINI_API_KEY=AIza... python baseline/run_baseline.py --llm --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.baseline_agent import RuleBasedAgent, LLMAgent


def print_separator(char: str = "─", width: int = 60) -> None:
    print(char * width)


def print_results(results: dict) -> None:
    print_separator("═")
    print(f"  MedTriageEnv Baseline Results")
    print(f"  Agent: {results['agent']}")
    print_separator()

    for task_id, r in results["task_results"].items():
        print(f"\n  {task_id.upper().replace('_', ' ')}")
        print(f"    Episode score : {r['episode_score']:.4f}")
        print(f"    Num steps     : {r['num_steps']}")
        scores_str = ", ".join(f"{s:.3f}" for s in r["step_scores"])
        print(f"    Step scores   : [{scores_str}]")

    print_separator()
    print(f"  Overall score: {results['overall_score']:.4f}")
    print_separator("═")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline agent(s) against MedTriageEnv"
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Also run the LLM-powered agent (requires GEMINI_API_KEY)"
    )
    parser.add_argument(
        "--model", default="gemini-2.5-flash",
        help="Gemini model to use for LLM agent (default: gemini-2.5-flash)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to write JSON results (e.g. baseline_scores.json)"
    )
    args = parser.parse_args()

    all_results = {}

    # ── Rule-based baseline ───────────────────────────────────────────────────
    print("\n[1/2] Running RuleBasedAgent...")
    t0 = time.time()
    agent = RuleBasedAgent()
    rb_results = agent.run_all_tasks()
    elapsed = time.time() - t0
    rb_results["elapsed_seconds"] = round(elapsed, 2)
    all_results["rule_based"] = rb_results
    print_results(rb_results)

    # ── LLM baseline (optional) ───────────────────────────────────────────────
    if args.llm:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("\n[INFO] GEMINI_API_KEY not set — using built-in key.")
        print(f"\n[2/2] Running LLMAgent({args.model})...")
        t0 = time.time()
        try:
            llm_agent   = LLMAgent(model=args.model)
            llm_results = llm_agent.run_all_tasks()
            elapsed = time.time() - t0
            llm_results["elapsed_seconds"] = round(elapsed, 2)
            all_results["llm"] = llm_results
            print_results(llm_results)
        except Exception as e:
            print(f"\n[ERROR] LLM agent failed: {e}")
    else:
        print("\n[2/2] Skipping LLMAgent (pass --llm to enable).")

    # ── Write output ──────────────────────────────────────────────────────────
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "baseline_scores.json"
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to: {output_path}\n")


if __name__ == "__main__":
    main()
