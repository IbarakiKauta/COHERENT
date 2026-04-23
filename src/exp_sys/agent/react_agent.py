"""
ReactAgent: two-call-per-step coordinator (hierarchical subgoal variant).

Per-step flow
─────────────
Turn 1  Oracle (system = role/capabilities, user = task + state + env_structure + history)
        → picks ONE agent and gives a COARSE-GRAINED subgoal (not an atomic step).
        Oracle sees object locations/surfaces/heights so it can reason about reachability.

Turn 2  Selected agent (system = agent role, user = observation + actions + subgoal)
        → emits `Thought: ...` then EITHER `Action: <from list>` OR `CANNOT: <reason>`.
        A `CANNOT` is a feedback channel back to the oracle — no env.step is executed,
        and the refusal + reason is written to dialogue history for the next turn.

Env execution outcome (succeeded / failed) is also written back to dialogue history
so the oracle can react to failures instead of looping on the same subgoal.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import time

import backoff
from openai import OpenAI, OpenAIError

_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt"

# React-specific agent prompts (no YES I CAN / feasibility logic).
# The "reasoning" variant drops the forced Thought: block — reasoning models
# use native hidden reasoning (surfaced via Responses API summary) instead.
_AGENT_PROMPT_FILES = {
    "quadrotor":  "react_quadrotor_prompt.txt",
    "robot_dog":  "react_robot_dog_prompt.txt",
    "robot arm":  "react_robot_dog_prompt.txt",   # alias
    "robot_arm":  "react_robot_arm_prompt.txt",
}
_AGENT_PROMPT_FILES_REASONING = {
    "quadrotor":  "react_agent_prompt_reasoning.txt",
    "robot_dog":  "react_agent_prompt_reasoning.txt",
    "robot arm":  "react_agent_prompt_reasoning.txt",
    "robot_arm":  "react_agent_prompt_reasoning.txt",
}

# Static role/capability descriptions go into the system message for each agent type.
_AGENT_SYSTEM = {
    "quadrotor": (
        "You are quadrotor, an aerial robot with a transport basket.\n"
        "Capabilities: fly between rooms (door must be open), takeoff, movetowards, land on LANDABLE surfaces. "
        "You do NOT grab or place objects — you only transport them. "
        "Always follow the sequence: takeoff → movetowards → land (never skip a step)."
    ),
    "robot_dog": (
        "You are robot_dog, a wheeled robot with a robotic arm.\n"
        "Capabilities: move between rooms (door must be open), grab/place objects on LOW_HEIGHT surfaces "
        "(hand must be empty), open/close doors and containers (hand must be empty). "
        "Cannot reach HIGH_HEIGHT surfaces or ON_HIGH_SURFACE objects. "
        "Must movetowards an object before interacting with it."
    ),
    "robot arm": (
        "You are robot_arm, a fixed manipulator mounted on a table.\n"
        "Capabilities: grab/place objects on your own table, open/close containers on your table, "
        "interact with the quadrotor basket when it lands on your table. "
        "Cannot reach objects on other tables."
    ),
    "robot_arm": (
        "You are robot_arm, a fixed manipulator mounted on a table.\n"
        "Capabilities: grab/place objects on your own table, open/close containers on your table, "
        "interact with the quadrotor basket when it lands on your table. "
        "Cannot reach objects on other tables."
    ),
}

from exp_sys.utils.pricing import cost_for

# Models that burn completion-token budget on hidden reasoning — a too-small
# max_tokens will return empty `content`. We warn when this pattern is detected.
_REASONING_MODELS = {"gpt-5", "gpt-5-mini", "gpt-5-nano", "o1", "o1-mini", "o3", "o3-mini"}


class ReactAgent:
    """
    Centralized oracle + per-agent action selection.

    Usage:
        env   = CoherentEnv(task_data)
        agent = ReactAgent(env=env, args=args, logger=logger)
        success, steps = agent.run()
    """

    def __init__(self, env: Any, args: Any, logger: Any = None) -> None:
        self.env    = env
        self.args   = args
        self.logger = logger
        self._history_turns: List[str] = []
        self.dialogue_history: str = ""

        api_key      = getattr(args, "api_key",      "") or os.getenv("OPENAI_API_KEY")      or ""
        organization = getattr(args, "organization", "") or os.getenv("OPENAI_ORGANIZATION") or ""
        base_url     = (getattr(args, "base_url", "") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")

        client_kwargs: Dict[str, str] = {"api_key": api_key}
        if organization:
            client_kwargs["organization"] = organization
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

        # Reasoning models (gpt-5*, o1*, o3*) need a different sampling shape:
        #   - use `max_completion_tokens`, not `max_tokens`
        #   - no `temperature` override (only default 1.0 is accepted)
        #   - `reasoning_effort` controls hidden-thinking budget ("minimal"
        #     is the lowest; there is no hard "off")
        lm_id = str(getattr(args, "lm_id", ""))
        self._is_reasoning = lm_id in _REASONING_MODELS or any(
            lm_id.startswith(p) for p in ("gpt-5", "o1", "o3")
        )
        requested_tokens = getattr(args, "max_tokens", 512)
        if self._is_reasoning:
            # Reasoning models need headroom even at "minimal" — raise the
            # floor so `content` isn't emptied by the hidden reasoning budget.
            effective_tokens = max(requested_tokens, 4096)
            self._sampling = {
                "max_completion_tokens": effective_tokens,
                # Default to "low": minimal produces empty summaries, so we'd
                # have no reasoning trace in the log; "low" is the cheapest
                # tier that still populates reasoning.summary in the Responses API.
                "reasoning_effort":      getattr(args, "reasoning_effort", "low") or "low",
                "n":                     getattr(args, "n", 1),
            }
            if effective_tokens != requested_tokens:
                print(
                    f"[ReactAgent] reasoning model '{lm_id}': raised "
                    f"max_completion_tokens {requested_tokens} → {effective_tokens}; "
                    f"reasoning_effort={self._sampling['reasoning_effort']}",
                    flush=True,
                )
        else:
            self._sampling = {
                "max_tokens":  requested_tokens,
                "temperature": getattr(args, "t", 0.0),
                "n":           getattr(args, "n", 1),
            }

    # ──────────────────────────────────────────────────────────────────
    # LLM call
    # ──────────────────────────────────────────────────────────────────

    @backoff.on_exception(backoff.expo, OpenAIError, max_tries=3)
    def _llm(
        self,
        messages: List[Dict],
        system: str = "You are a helpful assistant.",
    ) -> Tuple[str, Optional[str], int, int, int, float, float]:
        """Returns (text, reasoning_summary, prompt_tok, completion_tok, reasoning_tok, cost, latency_ms).

        For reasoning models we use the Responses API, which exposes a
        (potentially multi-part) reasoning.summary alongside message content.
        For non-reasoning models we use Chat Completions; summary is None.
        """
        full_msgs = [{"role": "system", "content": system}] + messages
        t0 = time.perf_counter()

        summary: Optional[str] = None
        reasoning_tok = 0

        if self._is_reasoning:
            # Responses API: input replaces messages; reasoning + max_output_tokens are top-level.
            resp = self._client.responses.create(
                model=self.args.lm_id,
                input=full_msgs,
                reasoning={
                    "effort":  self._sampling["reasoning_effort"],
                    "summary": "auto",
                },
                max_output_tokens=self._sampling["max_completion_tokens"],
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            text = getattr(resp, "output_text", "") or ""

            # Collect reasoning summary parts across all reasoning items.
            parts: List[str] = []
            for item in (getattr(resp, "output", None) or []):
                if getattr(item, "type", None) != "reasoning":
                    continue
                for p in (getattr(item, "summary", None) or []):
                    ptxt = getattr(p, "text", None)
                    if ptxt:
                        parts.append(ptxt)
            summary = "\n\n".join(parts) if parts else None

            u = resp.usage
            p_tok = u.input_tokens
            c_tok = u.output_tokens
            details = getattr(u, "output_tokens_details", None)
            reasoning_tok = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
        else:
            resp = self._client.chat.completions.create(
                model=self.args.lm_id,
                messages=full_msgs,
                **self._sampling,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            text  = resp.choices[0].message.content or ""
            u     = resp.usage
            p_tok = u.prompt_tokens
            c_tok = u.completion_tokens
            details = getattr(u, "completion_tokens_details", None)
            reasoning_tok = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0

        cost  = cost_for(self.args.lm_id, p_tok, c_tok)

        if not text.strip():
            budget = self._sampling.get("max_completion_tokens") or self._sampling.get("max_tokens", 0)
            hint = ""
            if self._is_reasoning and budget and c_tok >= budget - 8:
                hint = (
                    f"  (reasoning model '{self.args.lm_id}' exhausted its "
                    f"{budget} output-token budget on hidden thinking at "
                    f"reasoning_effort={self._sampling.get('reasoning_effort','?')}; "
                    "bump --max_tokens further or drop effort)"
                )
            print(
                f"[ReactAgent] WARN: empty LLM response "
                f"(model={self.args.lm_id}, completion_tokens={c_tok}/{budget}, "
                f"reasoning_tokens={reasoning_tok}, latency={latency_ms:.0f}ms){hint}",
                flush=True,
            )
        return text, summary, p_tok, c_tok, reasoning_tok, cost, latency_ms

    # ──────────────────────────────────────────────────────────────────
    # Prompt helpers
    # ──────────────────────────────────────────────────────────────────

    def _read(self, filename: str) -> str:
        return (_PROMPT_DIR / filename).read_text(encoding="utf-8")

    def _oracle_system(self) -> str:
        return self._read("react_oracle_system.txt")

    def _oracle_user(self, obs: Dict) -> str:
        return (
            self._read("react_oracle_prompt.txt")
            .replace("#TASK_GOAL#",        self.env.goal_instruction)
            .replace("#ROBOT_STATES#",     self.env.global_summary(obs))
            .replace("#ENV_STRUCTURE#",    self.env.env_structure_summary(obs))
            .replace("#DIALOGUE_HISTORY#", self.dialogue_history or "(none)")
        )

    def _agent_user(self, class_name: str, obs_text: str, plans_str: str, instruction: str) -> str:
        files = _AGENT_PROMPT_FILES_REASONING if self._is_reasoning else _AGENT_PROMPT_FILES
        fname = files.get(class_name, files["robot_dog"])
        return (
            self._read(fname)
            .replace("#OBSERVATION#", obs_text)
            .replace("#ACTIONLIST#",  plans_str)
            .replace("#INSTRUCTION#", instruction)
        )

    def _agent_system(self, class_name: str) -> str:
        return _AGENT_SYSTEM.get(class_name, _AGENT_SYSTEM["robot_dog"])

    # ──────────────────────────────────────────────────────────────────
    # Parsing helpers
    # ──────────────────────────────────────────────────────────────────

    def _parse_oracle(self, text: str) -> Tuple[Optional[str], Optional[int], str]:
        """
        Extract (class_name, node_id, instruction) from one of:
          'Hello <class_name>(id): instruction.'   ← preferred
          'Hello class_name(id): instruction.'     ← common gpt-5-mini variant
        """
        # Strict form first (keeps back-compat and avoids matching object
        # references like '<milkbox>(30)' that may appear later in the line).
        m = re.search(r"<([\w\s]+)>\s*\((\d+)\)\s*:\s*(.*)", text, re.DOTALL)
        if not m:
            # Bracketless fallback, anchored on 'Hello' to avoid grabbing the
            # first random `word(id)` that appears in the body.
            m = re.search(
                r"Hello\s+([a-zA-Z][a-zA-Z_\s]*?)\s*\((\d+)\)\s*:\s*(.*)",
                text,
                re.DOTALL | re.IGNORECASE,
            )
        if not m:
            return None, None, text
        raw = m.group(1).strip().lower().replace(" ", "_")
        if raw == "robotdog":
            raw = "robot_dog"
        elif raw == "robotarm":
            raw = "robot_arm"
        return raw, int(m.group(2)), m.group(3).strip().rstrip(".")

    def _parse_agent_response(
        self, available: List[str], text: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse a T2 response into (action, thought, cannot_reason).

        Expected format (one of):
          Thought: ...
          Action: <one line from available list>
        OR
          Thought: ...
          CANNOT: <one sentence>

        Fallbacks:
          - If neither explicit marker is present, try to substring-match any
            action in `available` and return it (with thought=None).
          - If still nothing, return (None, thought, None) — main loop treats as skip.
        """
        thought: Optional[str] = None
        action: Optional[str] = None
        cannot: Optional[str] = None

        thought_m = re.search(r"(?is)Thought\s*:\s*(.+?)(?=\n\s*(?:Action|CANNOT)\s*:|\Z)", text)
        if thought_m:
            thought = thought_m.group(1).strip()

        cannot_m = re.search(r"(?is)CANNOT\s*:\s*(.+?)(?:\n{2,}|\Z)", text)
        if cannot_m:
            cannot = cannot_m.group(1).strip().splitlines()[0].strip()
            return None, thought, cannot

        action_m = re.search(r"(?is)Action\s*:\s*(.+?)(?:\n{2,}|\Z)", text)
        action_text = action_m.group(1).strip() if action_m else text

        action_up = action_text.upper().replace("TAKEOFF FROM", "TAKEOFF_FROM")
        for cand in available:
            if cand.upper() in action_up:
                action = cand
                break
        if action is None:
            full_up = text.upper().replace("TAKEOFF FROM", "TAKEOFF_FROM")
            for cand in available:
                if cand.upper() in full_up:
                    action = cand
                    break
        return action, thought, cannot

    def _agent_idx(self, class_name: str, node_id: int) -> Optional[int]:
        for idx, (cname, nid) in self.env.id_name_dict.items():
            if cname.replace(" ", "_") == class_name and nid == node_id:
                return idx
        return None

    # ──────────────────────────────────────────────────────────────────
    # Dialogue history
    # ──────────────────────────────────────────────────────────────────

    def _log_overview(
        self,
        subgoal: str,
        thought: Optional[str],
        action: Optional[str],
        cannot_reason: Optional[str],
        env_outcome: Optional[str],
    ) -> None:
        """Write a compact per-step overview for the HTML report."""
        if self.logger and hasattr(self.logger, "log_step_overview"):
            self.logger.log_step_overview(
                subgoal=subgoal,
                thought=thought,
                action=action,
                cannot_reason=cannot_reason,
                env_outcome=env_outcome,
            )

    def _push_history(self, turn: str) -> None:
        self._history_turns.append(turn)
        numbered = [f"[{i + 1}] {t}" for i, t in enumerate(self._history_turns)]
        self.dialogue_history = "\n".join(numbered[-10:])

    # ──────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────

    def run(self) -> Tuple[bool, int]:
        task_goal  = self.env.task_goal
        max_steps  = getattr(self.args, "max_steps", None)
        hard_limit = max_steps if max_steps else 2 * self.env.ground_truth_step_num + 1

        success = False
        obs     = self.env.get_observations()

        while True:
            step = self.env.steps

            if self.logger:
                self.logger.start_step(step, {
                    "global_summary":   self.env.global_summary(obs),
                    "dialogue_history": self.dialogue_history,
                })

            # ── Turn 1: Oracle selects agent ──────────────────────────
            t1_system = self._oracle_system()
            t1_user   = self._oracle_user(obs)
            t1_msgs   = [{"role": "user", "content": t1_user}]

            try:
                t1_resp, t1_summary, t1_in, t1_out, t1_rtok, t1_cost, t1_lat = \
                    self._llm(t1_msgs, system=t1_system)
            except Exception as exc:
                print(f"[ReactAgent] Step {step} Turn-1 LLM error: {exc}")
                traceback.print_exc()
                break

            if self.logger:
                logged_prompt = json.dumps(
                    [{"role": "system", "content": t1_system}] + t1_msgs,
                    ensure_ascii=False,
                )
                self.logger.log_llm_call(
                    agent="oracle",
                    call_type="agent_selection",
                    prompt=logged_prompt,
                    response=t1_resp,
                    prompt_tokens=t1_in,
                    completion_tokens=t1_out,
                    reasoning_tokens=t1_rtok,
                    cost=t1_cost,
                    latency_ms=t1_lat,
                    raw_reasoning=t1_summary,
                )

            class_name, node_id, instruction = self._parse_oracle(t1_resp)
            subgoal_line = (
                f"<{class_name}>({node_id}): {instruction}"
                if class_name and node_id is not None else t1_resp.strip()
            )
            print(f"[Step {step}] Oracle → {subgoal_line[:120]}")

            if class_name is None or node_id is None:
                reason = "oracle output did not match 'Hello <class>(id): ...'"
                print(f"[ReactAgent] Step {step}: {reason} — skipping.")
                self._push_history(f"Oracle: {t1_resp}")
                self._push_history(f"[system] Skipped step: {reason}")
                if self.logger:
                    self._log_overview(subgoal_line, None, None, None, f"skip: {reason}")
                self.env.skip_step()
                if self.env.steps > hard_limit:
                    break
                obs = self.env.get_observations()
                continue

            agent_idx = self._agent_idx(class_name, node_id)
            if agent_idx is None:
                reason = f"<{class_name}>({node_id}) is not a registered agent"
                print(f"[ReactAgent] Step {step}: {reason} — skipping.")
                self._push_history(f"Oracle: {t1_resp}")
                self._push_history(f"[system] Skipped step: {reason}")
                if self.logger:
                    self._log_overview(subgoal_line, None, None, None, f"skip: {reason}")
                self.env.skip_step()
                if self.env.steps > hard_limit:
                    break
                obs = self.env.get_observations()
                continue

            # ── Turn 2: Selected agent picks action ───────────────────
            # Completely independent call — no oracle context in conversation history.
            # The oracle's instruction is already embedded in the user prompt via #INSTRUCTION#.
            agent_obs_text        = self.env.obs2text(obs, agent_idx)
            plans_str, _, plans_list = self.env.get_available_plans(agent_idx, obs)

            t2_system = self._agent_system(class_name)
            t2_user   = self._agent_user(class_name, agent_obs_text, plans_str, instruction)
            t2_msgs   = [{"role": "user", "content": t2_user}]

            try:
                t2_resp, t2_summary, t2_in, t2_out, t2_rtok, t2_cost, t2_lat = \
                    self._llm(t2_msgs, system=t2_system)
            except Exception as exc:
                print(f"[ReactAgent] Step {step} Turn-2 LLM error: {exc}")
                traceback.print_exc()
                break

            action, parsed_thought, cannot_reason = self._parse_agent_response(plans_list, t2_resp)
            # Prefer the Responses-API reasoning summary when available
            # (reasoning models); fall back to Thought: parsed from content
            # (non-reasoning models).
            thought = t2_summary if t2_summary else parsed_thought
            print(
                f"[Step {step}] <{class_name}>({node_id}) → "
                f"{'CANNOT: ' + cannot_reason if cannot_reason else action}"
            )

            if self.logger:
                logged_t2 = json.dumps(
                    [{"role": "system", "content": t2_system}] + t2_msgs,
                    ensure_ascii=False,
                )
                self.logger.log_llm_call(
                    agent=f"{class_name}({node_id})",
                    call_type="action_selection",
                    prompt=logged_t2,
                    response=t2_resp,
                    prompt_tokens=t2_in,
                    completion_tokens=t2_out,
                    reasoning_tokens=t2_rtok,
                    cost=t2_cost,
                    latency_ms=t2_lat,
                    parsed_action=action,
                    raw_reasoning=thought,
                    execution_result=("cannot" if cannot_reason else ("success" if action else "unexpected_format")),
                )

            # ── Execute / handle CANNOT / handle parse failure ─────────
            env_outcome: Optional[str] = None  # human-readable line for history + report
            if cannot_reason:
                # Agent explicitly refused — surface back to oracle, skip env step.
                env_outcome = f"CANNOT (refused by <{class_name}>({node_id})): {cannot_reason}"
                self.env.skip_step()
            elif action is None:
                env_outcome = (
                    f"no valid action parsed from agent response "
                    f"(neither 'Action:' matching the list nor 'CANNOT:' found)"
                )
                self.env.skip_step()
            else:
                before_obs_text = self.env.obs2text(obs, agent_idx)
                try:
                    done, _results, _sat, _unsat, _steps = self.env.step(
                        class_name, node_id, action, task_goal
                    )
                    obs = self.env.get_observations()
                    if self.logger:
                        self.logger.log_environment_change(
                            before_state={"obs_text": before_obs_text},
                            after_state={"obs_text": self.env.obs2text(obs, agent_idx)},
                            action_executed=action,
                            action_success=done,
                        )
                    success = done
                    env_outcome = (
                        f"action {action} executed; task_done={done}"
                    )
                except Exception as exc:
                    env_outcome = f"env.step raised {type(exc).__name__}: {exc}"
                    print(f"[ReactAgent] Step {step}: env.step error: {exc}")
                    traceback.print_exc()
                    self._push_history(f"Oracle: {t1_resp}")
                    self._push_history(f"<{class_name}>({node_id}): {t2_resp}")
                    self._push_history(f"[env] {env_outcome}")
                    break

            # Update dialogue history: oracle subgoal, agent response, env feedback.
            self._push_history(f"Oracle: {subgoal_line}")
            agent_line = (
                f"<{class_name}>({node_id}) CANNOT: {cannot_reason}"
                if cannot_reason
                else f"<{class_name}>({node_id}) Thought: {thought or '(none)'} | Action: {action or '(unparsed)'}"
            )
            self._push_history(agent_line)
            if env_outcome:
                self._push_history(f"[env] {env_outcome}")

            if self.logger:
                self._log_overview(
                    subgoal_line, thought, action, cannot_reason, env_outcome
                )

            if success:
                print(f"[ReactAgent] Task succeeded in {self.env.steps} steps.")
                break
            if self.env.steps > hard_limit:
                print(f"[ReactAgent] Exceeded step limit ({hard_limit}). Stopping.")
                break

        return success, self.env.steps
