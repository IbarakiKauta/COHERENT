"""
CRMSAgent: Centralized oracle with all-agent observations + combined action list.

Per-step flow
─────────────
1. Concat all agents' full observations
2. Build combined prefixed action list: <class>(id): [action]
3. LLM Call 1 — oracle_reasoning  (all-obs + action list → reasoning text)
4. LLM Call 2 — action_extraction (reasoning + suffix → single action)
5. Parse <class>(id): [action] → execute via env.step
6. Update action_history (rolling last-5)
"""

from __future__ import annotations

import traceback
from typing import Any, List, Tuple

from ._base import AgentBase, _EXTRACT_SUFFIX_CRMS


class CRMSAgent(AgentBase):

    def __init__(self, env: Any, args: Any, logger: Any = None) -> None:
        super().__init__(env, args, logger)
        self._history_list: List[str] = []
        self.action_history: str = ""

    def run(self) -> Tuple[bool, int]:
        import random  # local import — matches original CRMS arena_mp2.py usage

        task_goal = self.env.task_goal
        max_steps = getattr(self.args, "max_steps", None)
        gt_limit = 2 * self.env.ground_truth_step_num
        success = False

        while True:
            step = self.env.steps
            obs = self.env.get_observations()

            if self.logger:
                self.logger.start_step(step, {"action_history": self.action_history})

            obs_text = self._all_obs_text(obs)

            # Original arena_mp2.py:210-242 — shuffle agent order, accumulate
            # prefixed action list, then join with plain '\n' (NO letter prefixes).
            ids = list(self.env.id_name_dict.keys())
            random.shuffle(ids)
            all_actions: List[str] = []
            for agent_idx in ids:
                class_name, node_id = self.env.id_name_dict[agent_idx]
                _, _, local = self.env.get_available_plans(agent_idx, obs)
                all_actions += [f"<{class_name}>({node_id}): {a}" for a in local]
            plans_str = "\n".join(all_actions)

            # ── Call 1: Oracle reasoning ─────────────────────────────
            oracle_prompt = (
                self._read("crms_oracle_prompt.txt")
                .replace("#AGENT_OBSERVATIONS#", obs_text)
                .replace("#TASK_GOAL#", self.env.goal_instruction)
                .replace("#NUMBER_AGENTS#", str(len(self.env.id_name_dict)))
                .replace("#ACTION_LIST#", plans_str)
                .replace("#ACTION_HISTORY#", self.action_history)
            )
            try:
                reasoning, p_in, p_out, cost, lat = self._llm([{"role": "user", "content": oracle_prompt}])
            except Exception as exc:
                print(f"[CRMSAgent] Step {step} oracle LLM error: {exc}")
                traceback.print_exc()
                break
            self._log("oracle", "oracle_reasoning", oracle_prompt, reasoning, p_in, p_out, cost, lat)
            print(f"[Step {step}] Oracle: {reasoning[:100]}")

            # ── Call 2: Action extraction ────────────────────────────
            extract_prompt = reasoning + _EXTRACT_SUFFIX_CRMS
            try:
                extracted, p_in, p_out, cost, lat = self._llm([{"role": "user", "content": extract_prompt}])
            except Exception as exc:
                print(f"[CRMSAgent] Step {step} extraction LLM error: {exc}")
                traceback.print_exc()
                break
            self._log("oracle", "action_extraction", extract_prompt, extracted, p_in, p_out, cost, lat)
            print(f"[Step {step}] Extracted: {extracted}")

            # ── Parse & execute ──────────────────────────────────────
            plan = self._parse_action_from_list(all_actions, extracted)

            if plan is None:
                print(f"[CRMSAgent] Step {step}: no valid action — skipping.")
                wrong = (
                    f"Since the action ${extracted}$ you give is not in the list of available "
                    f"actions for all agents currently, no action is executed this round. "
                    f"This means that in the current state, the steps you gave to perform the "
                    f"task are problematic. Please think step by step."
                )
                self.action_history = self._push_history(self._history_list, wrong, max_entries=5)
                self.env.skip_step()
            else:
                self.action_history = self._push_history(self._history_list, reasoning + ".", max_entries=5)
                class_name, node_id, action_str = self._parse_full_action(plan)
                if class_name is None:
                    print(f"[CRMSAgent] Step {step}: cannot parse plan '{plan}' — skipping.")
                    self.env.skip_step()
                else:
                    agent_idx = self._agent_idx_by_id(node_id)
                    before_text = self.env.obs2text(obs, agent_idx) if agent_idx is not None else ""
                    try:
                        done, _, _, _, _ = self.env.step(class_name, node_id, action_str, task_goal)
                        after_obs = self.env.get_observations()
                        after_text = self.env.obs2text(after_obs, agent_idx) if agent_idx is not None else ""
                        if self.logger:
                            self.logger.log_environment_change(
                                before_state={"obs_text": before_text},
                                after_state={"obs_text": after_text},
                                action_executed=plan,
                                action_success=done,
                            )
                        success = done
                        obs = after_obs
                    except Exception as exc:
                        print(f"[CRMSAgent] Step {step}: env.step error: {exc}")
                        traceback.print_exc()
                        break
                    print(f"[Step {step}] → {plan}")

            if success:
                print(f"[CRMSAgent] Task succeeded in {self.env.steps} steps.")
                break

            if max_steps is not None and self.env.steps >= max_steps:
                print(f"[CRMSAgent] Early stop triggered by --max_steps={max_steps}.")
                break
            if self.env.steps > gt_limit:
                print(f"[CRMSAgent] Exceeded step limit ({gt_limit}). Stopping.")
                break

        return success, self.env.steps
