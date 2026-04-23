#!/usr/bin/env python3
"""Batch runner for the COHERENT 40-task benchmark (paper subset).

Orchestrates `exp_sys.runner` subprocess invocations across (method, env) pairs,
optionally in parallel, and writes a CSV summary by scanning the UnifiedLogger
JSON outputs.

Defaults reproduce the paper's task subset (8 tasks × 5 envs = 40 tasks).

Usage:
    # Dry run: print what would be executed
    python -m exp_sys.run_suite --dry-run

    # Single method, sequential (default)
    python -m exp_sys.run_suite --methods pefa

    # All five methods, 2 subprocesses in parallel
    python -m exp_sys.run_suite --methods react pefa pefa_wo_history crms drms --parallel 2

    # Custom task subset for one env (overrides paper default for that env)
    python -m exp_sys.run_suite --methods react --envs env0 --tasks-env0 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXP_SYS = Path(__file__).resolve().parent
_SRC     = _EXP_SYS.parent
_REPO    = _SRC.parent

# Paper's 8 tasks per env (from experiment/PEFA/main.py docstring).
PAPER_TASKS: Dict[str, List[int]] = {
    "env0": [2, 4, 9, 10, 11, 15, 16, 20],
    "env1": [1, 3, 7, 8, 10, 11, 16, 20],
    "env2": [3, 5, 6, 7, 10, 11, 16, 17],
    "env3": [2, 4, 6, 7, 10, 16, 17, 19],
    "env4": [0, 1, 7, 10, 12, 17, 18, 19],
}

ALL_METHODS = ["react", "pefa", "pefa_wo_history", "crms", "drms"]
ALL_ENVS    = ["env0", "env1", "env2", "env3", "env4"]


# ---------------------------------------------------------------------------
# Job dataclass (plain dict for stdlib-only code)
# ---------------------------------------------------------------------------

def _make_job(method: str, env: str, tasks: List[int], lm_id: str,
              rounds: int, max_steps: Optional[int],
              logs_root: Path, reports_root: Path,
              max_tokens: Optional[int] = None,
              reasoning_effort: Optional[str] = None) -> Dict:
    return {
        "method":           method,
        "env":              env,
        "tasks":            tasks,
        "lm_id":            lm_id,
        "rounds":           rounds,
        "max_steps":        max_steps,
        "max_tokens":       max_tokens,
        "reasoning_effort": reasoning_effort,
        "logs_root":        str(logs_root),
        "reports_root":     str(reports_root),
    }


def _build_cmd(job: Dict) -> List[str]:
    # -u forces child Python to use unbuffered stdout/stderr, so parent sees
    # lines in real time instead of waiting for an 8KB pipe-buffer flush.
    cmd = [
        sys.executable, "-u", "-m", "exp_sys.runner",
        "--method",      job["method"],
        "--env",         job["env"],
        "--task",        *[str(t) for t in job["tasks"]],
        "--lm_id",       job["lm_id"],
        "--logs_root",   job["logs_root"],
        "--reports_root", job["reports_root"],
    ]
    if job["rounds"] > 1:
        cmd += ["--rounds", str(job["rounds"])]
    if job["max_steps"] is not None:
        cmd += ["--max_steps", str(job["max_steps"])]
    if job.get("max_tokens") is not None:
        cmd += ["--max_tokens", str(job["max_tokens"])]
    if job.get("reasoning_effort"):
        cmd += ["--reasoning_effort", job["reasoning_effort"]]
    return cmd


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------

def _run_job(job: Dict, stream_output: bool = True) -> Tuple[Dict, int, float]:
    """Run one (method, env) invocation as a subprocess. Returns (job, rc, elapsed_s).

    On any non-zero exit code the failure is logged but the suite continues
    (per user preference).
    """
    cmd = _build_cmd(job)
    label = f"[{job['method']}/{job['env']}]"
    print(f"{label} START  tasks={job['tasks']}  cmd={' '.join(cmd)}",
          flush=True)

    # Belt-and-suspenders: PYTHONUNBUFFERED also disables buffering for any
    # transitively-spawned Python child (in case -u is stripped somewhere).
    child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    t0 = time.time()
    try:
        if stream_output:
            # Prefix every child line so parallel outputs stay readable.
            proc = subprocess.Popen(
                cmd,
                cwd=str(_SRC),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                env=child_env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(f"{label} {line.rstrip()}", flush=True)
            rc = proc.wait()
        else:
            rc = subprocess.call(cmd, cwd=str(_SRC), env=child_env)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"{label} CRASHED: {exc}", flush=True)
        rc = -1

    elapsed = time.time() - t0
    status = "OK " if rc == 0 else "FAIL"
    print(f"{label} {status}  rc={rc}  elapsed={elapsed:.1f}s", flush=True)
    return job, rc, elapsed


# ---------------------------------------------------------------------------
# Result scanning
# ---------------------------------------------------------------------------

def _scan_results(logs_root: Path, suite_start: datetime,
                  jobs: List[Dict]) -> List[Dict]:
    """Walk logs_root/json/<method>/<env>/task_<id>/ and collect per-task results
    written after suite_start. Returns rows suitable for CSV."""
    rows: List[Dict] = []
    json_root = logs_root / "json"
    if not json_root.exists():
        return rows

    # Build set of (method, env, task) we expect to see, so we can mark missing.
    expected: Dict[Tuple[str, str, int], Dict] = {}
    for job in jobs:
        for task in job["tasks"]:
            expected[(job["method"], job["env"], task)] = job

    # Walk JSON logs and keep the newest per (method, env, task) written after suite_start.
    newest: Dict[Tuple[str, str, int], Tuple[float, Dict]] = {}
    for method_dir in json_root.iterdir():
        if not method_dir.is_dir():
            continue
        for env_dir in method_dir.iterdir():
            if not env_dir.is_dir():
                continue
            for task_dir in env_dir.iterdir():
                if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
                    continue
                for json_path in task_dir.glob("*.json"):
                    mtime = json_path.stat().st_mtime
                    if datetime.fromtimestamp(mtime) < suite_start:
                        continue
                    try:
                        with json_path.open() as f:
                            data = json.load(f)
                    except Exception:
                        continue
                    md = data.get("metadata", {})
                    summary = data.get("summary", {})
                    method = md.get("method", method_dir.name)
                    env = str(md.get("env_id", env_dir.name))
                    task_id = int(md.get("task_id", task_dir.name.removeprefix("task_")))
                    key = (method, env, task_id)
                    if key not in newest or mtime > newest[key][0]:
                        newest[key] = (mtime, {
                            "method":          method,
                            "env":             env,
                            "task_id":         task_id,
                            "success":         bool(summary.get("success", False)),
                            "steps":           int(summary.get("total_steps", 0)),
                            "gt_steps":        md.get("ground_truth_steps"),
                            "total_tokens_in": int(summary.get("total_tokens_in", 0)),
                            "total_tokens_out": int(summary.get("total_tokens_out", 0)),
                            "total_cost_usd":  float(summary.get("total_cost", 0.0)),
                            "log_path":        str(json_path),
                        })

    for key, (_, row) in newest.items():
        rows.append(row)

    # Add placeholder rows for expected (method, env, task) with no log file.
    seen_keys = set(newest.keys())
    for key in expected.keys() - seen_keys:
        method, env, task_id = key
        rows.append({
            "method":          method,
            "env":             env,
            "task_id":         task_id,
            "success":         False,
            "steps":           0,
            "gt_steps":        None,
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_cost_usd":  0.0,
            "log_path":        "(MISSING)",
        })

    rows.sort(key=lambda r: (r["method"], r["env"], r["task_id"]))
    return rows


def _write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "env", "task_id", "success", "steps", "gt_steps",
              "total_tokens_in", "total_tokens_out", "total_cost_usd", "log_path"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: List[Dict]) -> None:
    # Aggregate by (method, env).
    by_me: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        by_me.setdefault((r["method"], r["env"]), []).append(r)

    print("\n" + "=" * 78)
    print("SUITE SUMMARY  (success / total,  avg_steps,  total_cost_usd)")
    print("=" * 78)
    print(f"{'method':<18}{'env':<8}{'success':>10}{'avg_steps':>12}{'cost_usd':>14}")
    print("-" * 78)

    total_success = 0
    total_tasks   = 0
    total_cost    = 0.0
    for (method, env), group in sorted(by_me.items()):
        n = len(group)
        s = sum(1 for r in group if r["success"])
        avg_steps = sum(r["steps"] for r in group) / n if n else 0.0
        cost = sum(r["total_cost_usd"] for r in group)
        total_success += s
        total_tasks   += n
        total_cost    += cost
        print(f"{method:<18}{env:<8}{f'{s}/{n}':>10}{avg_steps:>12.1f}{cost:>14.4f}")

    print("-" * 78)
    overall = (total_success / total_tasks * 100) if total_tasks else 0.0
    print(f"{'TOTAL':<26}{f'{total_success}/{total_tasks}':>10}"
          f"{'':>12}{total_cost:>14.4f}  ({overall:.1f}%)")
    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the COHERENT 40-task benchmark across selected methods.",
    )
    p.add_argument("--methods", nargs="+", default=ALL_METHODS,
                   choices=ALL_METHODS,
                   help=f"Methods to run (default: all 5 = {ALL_METHODS})")
    p.add_argument("--envs", nargs="+", default=ALL_ENVS,
                   choices=ALL_ENVS,
                   help=f"Envs to run (default: all 5 = {ALL_ENVS})")
    for env in ALL_ENVS:
        p.add_argument(f"--tasks-{env}", nargs="+", type=int, default=None,
                       metavar="ID",
                       help=f"Override task IDs for {env} (default: paper subset)")
    p.add_argument("--lm_id", default="gpt-5-mini",
                   help="OpenAI model id (default: gpt-5-mini)")
    p.add_argument("--rounds", type=int, default=2,
                   help="DRMS dialogue rounds (default: 2; ignored for other methods)")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Optional hard step cap per task")
    p.add_argument("--max_tokens", type=int, default=None,
                   help="Per-LLM-call output token cap (runner default 512; "
                        "reasoning models auto-raise to 4096 if lower).")
    p.add_argument("--reasoning_effort", default=None,
                   choices=["minimal", "low", "medium", "high"],
                   help="Reasoning-model budget ('minimal' = closest to off). "
                        "Only applied when lm_id is gpt-5*/o1*/o3*.")
    p.add_argument("--parallel", type=int, default=1,
                   help="Max concurrent (method, env) subprocesses (default: 1)")
    p.add_argument("--logs_root", default=str(_EXP_SYS / "logs"),
                   help="Logs root dir (default: exp_sys/logs)")
    p.add_argument("--reports_root", default=str(_EXP_SYS / "reports"),
                   help="Reports root dir (default: exp_sys/reports)")
    p.add_argument("--summary_out", default=None,
                   help="Path to CSV summary (default: logs_root/suite_<ts>.csv)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned job list and exit without executing")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    logs_root    = Path(args.logs_root)
    reports_root = Path(args.reports_root)

    tasks_override = {env: getattr(args, f"tasks_{env}") for env in ALL_ENVS}
    jobs: List[Dict] = []
    for method in args.methods:
        for env in args.envs:
            tasks = tasks_override[env] if tasks_override[env] is not None else PAPER_TASKS[env]
            if not tasks:
                continue
            jobs.append(_make_job(method, env, tasks, args.lm_id,
                                  args.rounds, args.max_steps,
                                  logs_root, reports_root,
                                  max_tokens=args.max_tokens,
                                  reasoning_effort=args.reasoning_effort))

    suite_start = datetime.now()
    total_units = sum(len(j["tasks"]) for j in jobs)
    per_method  = total_units // len(args.methods) if args.methods else 0
    print(f"Suite started {suite_start.isoformat(timespec='seconds')}")
    print(f"  methods     : {args.methods}")
    print(f"  envs        : {args.envs}")
    print(f"  jobs        : {len(jobs)} runner invocations (method × env)")
    print(f"  work units  : {total_units} (= {per_method}/method × {len(args.methods)} methods)")
    print(f"  parallel    : {args.parallel}")
    print(f"  lm_id       : {args.lm_id}")
    print(f"  logs_root   : {logs_root}")
    print()

    if args.dry_run:
        print("Planned jobs (dry run):")
        for j in jobs:
            print(f"  {j['method']:<18} {j['env']}  tasks={j['tasks']}")
        return 0

    # --------------------------------------------------------------
    # Execute
    # --------------------------------------------------------------
    results: List[Tuple[Dict, int, float]] = []
    stream = args.parallel <= 1  # serial: show output live; parallel: also stream (prefixed)

    try:
        if args.parallel <= 1:
            for job in jobs:
                results.append(_run_job(job, stream_output=True))
        else:
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                futures = [pool.submit(_run_job, job, True) for job in jobs]
                for fut in as_completed(futures):
                    results.append(fut.result())
    except KeyboardInterrupt:
        print("\nInterrupted by user — scanning partial results before exit.")

    suite_end = datetime.now()
    print(f"\nSuite finished {suite_end.isoformat(timespec='seconds')}  "
          f"(wall {str(suite_end - suite_start).split('.')[0]})")

    # --------------------------------------------------------------
    # Scan results + write CSV summary
    # --------------------------------------------------------------
    rows = _scan_results(logs_root, suite_start, jobs)

    summary_out = Path(args.summary_out) if args.summary_out else \
        logs_root / f"suite_{suite_start.strftime('%Y%m%d_%H%M%S')}.csv"
    _write_csv(rows, summary_out)
    print(f"CSV summary → {summary_out}")

    _print_summary(rows)

    # Report any non-zero runner exits.
    failed_jobs = [(j, rc) for j, rc, _ in results if rc != 0]
    if failed_jobs:
        print(f"\n{len(failed_jobs)} runner invocation(s) exited non-zero "
              f"(individual tasks may still have logs):")
        for job, rc in failed_jobs:
            print(f"  - {job['method']}/{job['env']}  rc={rc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
