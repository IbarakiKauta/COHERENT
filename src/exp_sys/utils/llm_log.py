"""
LLM-readable JSON log generator.

Converts the UnifiedLogger visualization format into a structured,
self-explanatory JSON designed for LLM consumption (analysis, debugging,
prompt-tuning). Every field is named for readability without documentation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


_SYSTEM_MESSAGE = "You are a helpful assistant."

_METHOD_DESCRIPTIONS: Dict[str, str] = {
    "react": (
        "ReAct (Reasoning + Acting): Two LLM calls per step. "
        "Call 1 — Oracle sees a lightweight global state summary and dialogue history, "
        "then selects which agent to act and gives a brief instruction. "
        "Call 2 — The selected agent sees its full local observation and action list, "
        "then picks one concrete action. Oracle context is passed as conversation history."
    ),
    "crms": (
        "CRMS (Centralized Reasoning with Multi-agent System): Two LLM calls per step. "
        "Call 1 — Oracle sees all agents' full observations and a rolling action history, "
        "then reasons about the best next action including the target agent. "
        "Call 2 — Extracts the specific action from the oracle's reasoning."
    ),
    "pefa": (
        "PEFA (Planning with Explicit Feedback and Action): Up to five LLM calls per step. "
        "Call 1 — Oracle sees all agents' observations + dialogue history, produces reasoning. "
        "Call 2 — Extracts the target agent and sub-goal instruction. "
        "Call 3 — Target agent performs a feasibility check (YES I CAN / SORRY I CANNOT). "
        "Call 4 — If feasible, agent selects a concrete action. "
        "Call 5 — Judge verifies the action aligns with the instruction."
    ),
    "pefa_wo_history": (
        "PEFA w/o History: Ablation of PEFA. Identical flow but the oracle prompt "
        "does NOT receive dialogue history across steps, isolating the effect of memory."
    ),
    "drms": (
        "DRMS (Decentralized Reasoning with Multi-agent System): Multiple dialogue rounds per step. "
        "Agents are visited in random order for each round; each agent reasons with the "
        "accumulated dialogue record. Two LLM calls per agent per round "
        "(reasoning + action extraction). The last agent's final action is executed."
    ),
}

_CALL_PURPOSES: Dict[str, str] = {
    "agent_selection":    "Oracle observes the global state and selects which agent to act, issuing a brief instruction",
    "action_selection":   "Selected agent observes its local environment and picks one concrete action from available options",
    "oracle_reasoning":   "Oracle observes all agents' states and reasons about what instruction to issue next",
    "subgoal_extraction": "Extract the specific target agent and sub-goal instruction from the oracle's reasoning text",
    "action_extraction":  "Extract the specific action string from the agent's reasoning response",
    "judge_verify":       "Judge evaluates whether the selected action correctly fulfills the given instruction",
    "agent_reasoning":    "Agent reasons about the current situation, dialogue context, and available actions",
}


def _parse_prompt(raw: Any) -> List[Dict[str, str]]:
    """
    Parse the logged prompt into a structured messages list.
    Always prepends the system message if absent.
    """
    msgs: Optional[List[Dict[str, str]]] = None

    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "role" in parsed[0]:
                    msgs = parsed
            except (json.JSONDecodeError, Exception):
                pass
    elif isinstance(raw, list) and raw and isinstance(raw[0], dict) and "role" in raw[0]:
        msgs = raw

    if msgs is None:
        msgs = [{"role": "user", "content": str(raw) if raw is not None else ""}]

    if not msgs or msgs[0].get("role") != "system":
        msgs = [{"role": "system", "content": _SYSTEM_MESSAGE}] + msgs

    return msgs


def _convert_llm_call(call: Dict[str, Any], call_index: int) -> Dict[str, Any]:
    tokens = call.get("tokens") or {}
    prompt_toks = tokens.get("prompt") or 0
    completion_toks = tokens.get("completion") or 0
    latency_ms = call.get("latency_ms")

    return {
        "call_index":          call_index,
        "agent":               call.get("agent") or "unknown",
        "call_type":           call.get("call_type") or "unknown",
        "purpose":             _CALL_PURPOSES.get(call.get("call_type", ""), ""),
        "full_conversation":   _parse_prompt(call.get("prompt")),
        "model_response":      call.get("response") or "",
        "parsed_action":       call.get("parsed_action"),
        "execution_result":    call.get("execution_result") or "unknown",
        "tokens": {
            "input":    prompt_toks,
            "output":   completion_toks,
            "total":    prompt_toks + completion_toks,
        },
        "latency_seconds":     round(latency_ms / 1000, 3) if latency_ms is not None else None,
        "cost_usd":            call.get("cost") or 0.0,
    }


def _convert_step(step: Dict[str, Any]) -> Dict[str, Any]:
    obs = step.get("observation") or {}
    ec  = step.get("environment_change") or {}
    raw_calls = step.get("llm_calls") or []

    llm_calls = [_convert_llm_call(c, i + 1) for i, c in enumerate(raw_calls)]

    # Derive executed action and which agent ran it
    executed_action = ec.get("action") or None
    executing_agent: Optional[str] = None
    if executed_action:
        # Find the last call whose parsed_action matches (best-effort)
        for c in reversed(raw_calls):
            if c.get("parsed_action") == executed_action:
                executing_agent = c.get("agent")
                break

    env_step: Dict[str, Any] = {
        "action_executed":  executed_action,
        "executing_agent":  executing_agent,
        "action_succeeded": ec.get("success") if ec else None,
        "task_completed":   ec.get("success") if ec else None,
        "state_before":     (ec.get("before") or {}).get("obs_text"),
        "state_after":      (ec.get("after")  or {}).get("obs_text"),
        "changed_relations": ec.get("changed_relations") or [],
    }

    return {
        "step_number":         step.get("step_id") if step.get("step_id") is not None else 0,
        "context_at_step_start": obs,
        "llm_calls":           llm_calls,
        "total_calls_this_step": len(llm_calls),
        "environment_step":    env_step,
    }


def _build_llm_log(vis_log: Dict[str, Any]) -> Dict[str, Any]:
    meta = vis_log.get("metadata") or {}
    sm   = vis_log.get("summary")  or {}
    steps = vis_log.get("steps")   or []

    method_key  = (meta.get("method") or "").lower()
    method_desc = _METHOD_DESCRIPTIONS.get(method_key, "")

    total_calls = sum(len(s.get("llm_calls") or []) for s in steps)
    gt_steps    = meta.get("ground_truth_steps")
    taken_steps = sm.get("total_steps") or 0

    if gt_steps and taken_steps:
        steps_ratio = f"{taken_steps} steps taken, {gt_steps} optimal ({taken_steps / gt_steps:.1f}x overhead)"
    else:
        steps_ratio = f"{taken_steps} steps taken"

    lat_ms  = sm.get("total_latency_ms")
    success = sm.get("success") or False

    failure_reason: Optional[str] = None
    if not success:
        failure_reason = "exceeded_step_limit" if (gt_steps and taken_steps >= 2 * gt_steps) else "unknown"

    # Step-level outcome summary for quick scanning
    step_outcomes = []
    for s in steps:
        ec = s.get("environment_change") or {}
        step_outcomes.append({
            "step":    s.get("step_id") if s.get("step_id") is not None else 0,
            "action":  ec.get("action"),
            "success": ec.get("success") or False,
        })

    experiment: Dict[str, Any] = {
        "method":                    meta.get("method"),
        "method_description":        method_desc,
        "language_model":            meta.get("lm_id"),
        "environment":               "env" + str(meta.get("env_id") or "?"),
        "task_id":                   meta.get("task_id"),
        "task_name":                 meta.get("task_name"),
        "timestamp":                 meta.get("timestamp"),
        "ground_truth_optimal_steps": gt_steps,
    }

    outcome: Dict[str, Any] = {
        "success":              success,
        "failure_reason":       failure_reason,
        "steps_taken":          taken_steps,
        "steps_vs_optimal":     steps_ratio,
        "total_llm_calls":      total_calls,
        "total_input_tokens":   sm.get("total_tokens_in")  or 0,
        "total_output_tokens":  sm.get("total_tokens_out") or 0,
        "total_cost_usd":       round(sm.get("total_cost") or 0.0, 6),
        "total_latency_seconds": round(lat_ms / 1000, 2) if lat_ms is not None else None,
        "step_outcomes":        step_outcomes,
    }

    return {
        "experiment":  experiment,
        "outcome":     outcome,
        "step_trace":  [_convert_step(s) for s in steps],
    }


def generate_llm_log(json_file: str, output_json: str) -> None:
    """Convert a UnifiedLogger visualization JSON to LLM-readable format."""
    in_path  = Path(json_file)
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vis_log  = json.loads(in_path.read_text(encoding="utf-8"))
    llm_log  = _build_llm_log(vis_log)

    out_path.write_text(
        json.dumps(llm_log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
