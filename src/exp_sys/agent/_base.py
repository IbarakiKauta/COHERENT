"""Shared LLM plumbing for all exp_sys agents."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import backoff
from openai import OpenAI, OpenAIError

from exp_sys.utils.pricing import cost_for

_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt"

# Extraction suffix shared by CRMS / DRMS
_EXTRACT_SUFFIX_CRMS = (
    "\nExtract from the above paragraph the content of the format "
    "'<agent>(id): [action] <class_name>(id)' such as "
    "'<robot dog>(23): [movetowards] <door>(9)'. "
    "Then output the contents of this section. "
    "Be careful not to output any superfluous content, exactly in the format given. "
    "If there are more than one action in this format, you only extract the best action "
    "to be done first in the next step."
)

_EXTRACT_SUFFIX_DRMS = (
    "\n\n\t\tExtract from the above paragraph the content of the format "
    "'<agent>(id): [action] <class name>(id)' such as "
    "'<robot dog>(23): [movetowards] <door>(9)' which means that the suggested action is to "
    "have <robot dog>(23) perform the action of [movetowards] <door>(9). "
    "Then only output the contents of this section in this format. "
    "Be careful not to output any superfluous content, exactly in the format given. "
    "If there are more than one action in this format, you only extract the best action "
    "to be done first in the next step.\n\t\t"
)

# Subgoal extraction suffix for PEFA oracle
_EXTRACT_SUFFIX_PEFA = (
    "\nExtract from the above paragraph the content of the format "
    '"Hello <class name>(id): message.". '
    "Then output the contents of this section. "
    "Be careful not to output any superfluous content, exactly in the format given. "
    "If the above paragraph is not exactly formatted as "
    '"Hello <class name>(id): #message#.", '
    "output similar content in this format. "
    "As an example, the output might read: "
    '"Hello <robot dog>(0): please movetowards the <door>(1), and then open the <door>(1)". '
    "If this format does not appear in the preceding text, please summarize the above content "
    "into this format for output. "
    "To emphasize once again, the names of all objects and agent robots must be enclosed in <>, "
    "and the (id) must not be omitted. "
    "Class name missing <> and (id) should be completed with these elements. "
    "Please strictly follow this format in the output content."
)


class AgentBase:
    """Base class providing OpenAI client, _llm(), and shared utilities."""

    def __init__(self, env: Any, args: Any, logger: Any = None) -> None:
        self.env = env
        self.args = args
        self.logger = logger

        api_key = getattr(args, "api_key", "") or os.getenv("OPENAI_API_KEY") or ""
        organization = getattr(args, "organization", "") or os.getenv("OPENAI_ORGANIZATION") or ""
        base_url = (getattr(args, "base_url", "") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if organization:
            client_kwargs["organization"] = organization
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

        self._sampling = {
            "max_tokens": getattr(args, "max_tokens", 2048),
            "temperature": getattr(args, "t", 0.0),
            "n": getattr(args, "n", 1),
        }

    @backoff.on_exception(backoff.expo, OpenAIError, max_tries=3)
    def _llm(self, messages: List[Dict]) -> Tuple[str, int, int, float, float]:
        """Returns (text, prompt_tokens, completion_tokens, cost, latency_ms)."""
        with_sys = [{"role": "system", "content": "You are a helper assistant."}] + messages
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.args.lm_id, messages=with_sys, **self._sampling
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = resp.choices[0].message.content or ""
        p_tok = resp.usage.prompt_tokens
        c_tok = resp.usage.completion_tokens
        cost  = cost_for(self.args.lm_id, p_tok, c_tok)
        return text, p_tok, c_tok, cost, latency_ms

    def _read(self, filename: str) -> str:
        return (_PROMPT_DIR / filename).read_text(encoding="utf-8")

    def _log(
        self,
        agent: str,
        call_type: str,
        prompt: Any,
        response: str,
        p_tok: int,
        c_tok: int,
        cost: float,
        lat: float,
        parsed_action: Optional[str] = None,
    ) -> None:
        if self.logger:
            self.logger.log_llm_call(
                agent=agent,
                call_type=call_type,
                prompt=prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False),
                response=response,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                cost=cost,
                latency_ms=lat,
                parsed_action=parsed_action,
            )

    # ── History helpers ──────────────────────────────────────────────

    @staticmethod
    def _push_history(history_list: List[str], entry: str, max_entries: int = 10) -> str:
        history_list.append(entry)
        numbered = [f"[{i + 1}]、{item}" for i, item in enumerate(history_list)]
        return "\n".join(numbered[-max_entries:])

    # ── Observation / action helpers ─────────────────────────────────

    def _all_obs_text(self, obs: Dict) -> str:
        parts = [self.env.obs2text(obs, idx) for idx in self.env.id_name_dict]
        return "\n".join(parts)

    def _combined_action_list(self, obs: Dict) -> Tuple[str, List[str]]:
        """Build '<class>(id): [action] …' prefixed list across all agents."""
        all_actions: List[str] = []
        for agent_idx, (class_name, node_id) in self.env.id_name_dict.items():
            _, _, local = self.env.get_available_plans(agent_idx, obs)
            for action in local:
                all_actions.append(f"<{class_name}>({node_id}): {action}")
        plans_str = "".join(f"{chr(ord('A') + i)}. {a}\n" for i, a in enumerate(all_actions))
        return plans_str, all_actions

    # ── Parsing helpers ──────────────────────────────────────────────

    @staticmethod
    def _parse_action_from_list(available: List[str], text: str) -> Optional[str]:
        text = text.replace("_", " ").replace("takeoff from", "takeoff_from").replace("land on", "land_on")
        for action in available:
            if action in text:
                return action
        for i, action in enumerate(available):
            letter = chr(ord("A") + i)
            words = text.split()
            if (
                f"option {letter}" in text.lower()
                or f"Option {letter}" in text
                or f"({letter})" in text
                or f"{letter}." in words
                or f"{letter}," in words
            ):
                return action
        return None

    @staticmethod
    def _parse_full_action(text: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """Parse '<class_name>(id): [action] …' → (class_name, node_id, action_str)."""
        m = re.search(r"<([\w\s]+)>\((\d+)\)\s*:\s*(\[.*)", text)
        if not m:
            return None, None, None
        return m.group(1).strip(), int(m.group(2)), m.group(3).strip()

    @staticmethod
    def _parse_oracle_target(text: str) -> Tuple[Optional[str], Optional[int], str]:
        """Parse 'Hello <class_name>(id): instruction' → (class_name, node_id, instruction)."""
        start_class_name = text.find("<") + 1
        end_class_name = text.find(">")
        start_id = text.find("(") + 1
        end_id = text.find(")")
        start_msg = text.find(":") + 1

        if (
            start_class_name <= 0
            or end_class_name <= start_class_name
            or start_id <= 0
            or end_id <= start_id
            or start_msg <= 0
        ):
            return None, None, text

        class_name = text[start_class_name:end_class_name].strip()
        node_id_str = text[start_id:end_id].strip()
        try:
            node_id = int(node_id_str)
        except ValueError:
            return None, None, text

        instruction = text[start_msg:].strip()
        return class_name, node_id, instruction

    def _agent_idx_by_id(self, node_id: int) -> Optional[int]:
        for idx, (_, nid) in self.env.id_name_dict.items():
            if nid == node_id:
                return idx
        return None
