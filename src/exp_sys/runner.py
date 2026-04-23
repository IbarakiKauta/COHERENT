#!/usr/bin/env python3
"""
exp_sys runner — supports react / crms / pefa / pefa_wo_history / drms.

Usage:
    python src/exp_sys/runner.py --method react          --env env0 --task 2
    python src/exp_sys/runner.py --method crms           --env env0 --task 2
    python src/exp_sys/runner.py --method pefa           --env env0 --task 2
    python src/exp_sys/runner.py --method pefa_wo_history --env env0 --task 2
    python src/exp_sys/runner.py --method drms           --env env0 --task 2 --rounds 2

Logs   → src/exp_sys/logs/json/<method>/<env_id>/task_<id>/
Reports → src/exp_sys/reports/<method>/env{env_id}_task{task_id}_{ts}/report.html
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent  # src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from exp_sys.env.coherent_env import CoherentEnv
from exp_sys.agent.react_agent import ReactAgent
from exp_sys.agent.crms_agent import CRMSAgent
from exp_sys.agent.pefa_agent import PEFAAgent
from exp_sys.agent.pefa_wo_history_agent import PEFAWoHistoryAgent
from exp_sys.agent.drms_agent import DRMSAgent
from exp_sys.utils.reporting import generate_html_report, generate_multi_task_report
from exp_sys.utils.llm_log import generate_llm_log
from exp_sys.utils.unified_logger import UnifiedLogger

_EXP_SYS = Path(__file__).resolve().parent
_ENV_DIR = _EXP_SYS / "env" / "data"

_AGENT_CLASSES = {
    "react":           ReactAgent,
    "crms":            CRMSAgent,
    "pefa":            PEFAAgent,
    "pefa_wo_history": PEFAWoHistoryAgent,
    "drms":            DRMSAgent,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="exp_sys runner")
    p.add_argument("--method", default="react",
                   choices=list(_AGENT_CLASSES),
                   help="Agent method to run")
    p.add_argument("--env", required=True,
                   choices=["env0", "env1", "env2", "env3", "env4"])
    p.add_argument("--task", type=int, nargs="+", required=True,
                   help="Task indices, e.g. --task 2 or --task 2 4 9")
    p.add_argument("--lm_id",        default="gpt-4o-2024-11-20")
    p.add_argument("--source",       default="openai")
    p.add_argument("--api_key",      default="")
    p.add_argument("--organization", default="")
    p.add_argument("--base_url",     default="")
    p.add_argument("--max_tokens",   type=int,   default=512)
    p.add_argument("--t",            type=float, default=0.0)
    p.add_argument("--n",            type=int,   default=1)
    p.add_argument("--reasoning_effort", default="low",
                   choices=["minimal", "low", "medium", "high"],
                   help="Only used by reasoning models (gpt-5*/o1*/o3*). "
                        "'minimal' disables reasoning (and yields empty summaries); "
                        "'low' (default) is cheapest tier that still populates "
                        "reasoning.summary in the Responses API.")
    p.add_argument("--rounds",       type=int,   default=1,
                   help="Number of dialogue rounds per step (DRMS only)")
    p.add_argument("--max_steps",    type=int,   default=None,
                   help="Hard step limit (default: 2 × ground-truth steps)")
    p.add_argument("--debug",        action="store_true")
    p.add_argument("--logs_root",    default=str(_EXP_SYS / "logs"))
    p.add_argument("--reports_root", default=str(_EXP_SYS / "reports"))
    p.add_argument("--no_report",    action="store_true",
                   help="Skip HTML report generation")
    return p


def _build_task_data(raw: dict, task_id: int) -> dict:
    graph = raw["init_graph"]
    agent_list = [
        [n["class_name"], n["id"]]
        for n in graph["nodes"]
        if n["category"] == "Agents"
    ]
    return {
        "task_id":               task_id,
        "env_id":                raw["env_id"],
        "task_name":             raw["task_name"],
        "graph":                 graph,
        "task_goal":             raw["task_goal"],
        "goal_instruction":      raw["goal_instruction"],
        "ground_truth_step_num": raw["ground_truth_step_num"],
        "agent":                 agent_list,
        "num_agent":             len(agent_list),
    }


def _run_task(task_id: int, data: list, args: argparse.Namespace) -> tuple[bool, int, str]:
    raw      = data[task_id]
    env_id   = raw["env_id"]
    gt_steps = raw["ground_truth_step_num"]
    if isinstance(gt_steps, list):
        gt_steps = gt_steps[0]

    task_data = _build_task_data(raw, task_id)
    logs_root = Path(args.logs_root)

    logger = UnifiedLogger(
        task_id=task_id,
        env_id=env_id,
        task_name=raw["task_name"],
        method_name=args.method,
        log_dir=str(logs_root / "json"),
        ground_truth_steps=gt_steps,
    )
    logger.log_data["metadata"]["lm_id"] = args.lm_id

    env         = CoherentEnv(task_data)
    agent_cls   = _AGENT_CLASSES[args.method]
    agent       = agent_cls(env=env, args=args, logger=logger)

    steps     = 0
    success   = False
    json_path = ""
    try:
        success, steps = agent.run()
        json_path = logger.finalize(success, steps)
        print(f"  JSON log → {json_path}")

        # LLM-readable log (same dir, llm_json tree)
        stem      = Path(json_path).stem
        llm_dir   = logs_root / "llm_json" / args.method / str(env_id) / f"task_{task_id}"
        llm_dir.mkdir(parents=True, exist_ok=True)
        llm_path  = llm_dir / f"{stem}.json"
        generate_llm_log(json_path, str(llm_path))
        print(f"  LLM log → {llm_path}")

        if not args.no_report:
            report_name = f"{args.method}_env{env_id}_task{task_id}_{logger.timestamp}.html"
            report_dir  = Path(args.reports_root) / args.method
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / report_name
            generate_html_report(json_path, str(report_path))
            print(f"  Report  → {report_path}")

    except Exception as exc:
        print(f"  ERROR: {exc}")
        traceback.print_exc()
        json_path = logger.finalize(False, steps)

    return success, steps, json_path


def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    env_json = _ENV_DIR / f"{args.env}.json"
    if not env_json.exists():
        print(f"ERROR: env file not found: {env_json}")
        return 1

    with env_json.open() as f:
        data = json.load(f)

    text_dir = Path(args.logs_root) / "text" / args.method
    text_dir.mkdir(parents=True, exist_ok=True)
    text_log = text_dir / f"{args.env}.txt"

    success_tasks: list[int] = []
    failed_tasks:  list[int] = []
    steps_list:    list[int] = []
    json_paths:    list[str] = []

    for task_id in args.task:
        print(f"\n{'='*60}")
        print(f"  Task {task_id}  |  env {args.env}  |  method {args.method}")
        print(f"{'='*60}")
        success, steps, json_path = _run_task(task_id, data, args)
        steps_list.append(steps)
        (success_tasks if success else failed_tasks).append(task_id)
        if json_path:
            json_paths.append(json_path)
        print(f"  → {'SUCCESS' if success else 'FAILURE'} in {steps} steps")

    # Combined multi-task report
    if not args.no_report and len(json_paths) > 1:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_path = Path(args.reports_root) / args.method / f"{args.env}_combined_{ts}.html"
        try:
            generate_multi_task_report(json_paths, str(combined_path))
            print(f"\n  Combined report → {combined_path}")
        except Exception as exc:
            print(f"  WARNING: combined report failed: {exc}")

    avg = sum(steps_list) / len(steps_list) if steps_list else 0
    summary_lines = [
        f"\nSummary ({args.method}, {args.env})",
        f"  Tasks run : {args.task}",
        f"  Succeeded : {success_tasks}",
        f"  Failed    : {failed_tasks}",
        f"  Avg steps : {avg:.1f}",
    ]
    print("\n".join(summary_lines))
    with text_log.open("a") as f:
        f.write("\n".join(summary_lines) + "\n\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
