"""
DRMSAgent: Decentralized multi-round dialogue per step.

Per-step flow
─────────────
For round in range(args.rounds):
  Agents are visited in random order; each agent:
    1. Build combined action list (own + previous agents' actions)
    2. LLM Call A — agent reasoning (obs + task + dialogue_record → response)
    3. LLM Call B — action extraction (response + suffix → <class>(id): [action])
    4. Append own message to dialogue_record for next agent/round

After all rounds, execute the last agent's proposed action.
Update action_history (rolling last-5).
"""

from __future__ import annotations

import random
import traceback
from typing import Any, List, Optional, Tuple

from ._base import AgentBase, _EXTRACT_SUFFIX_DRMS

_AGENT_PROMPT_FILES = {
    "quadrotor":  "drms_quadrotor_prompt.txt",
    "robot_dog":  "drms_robot_dog_prompt.txt",
    "robot dog":  "drms_robot_dog_prompt.txt",
    "robot arm":  "drms_robot_arm_prompt.txt",
    "robot_arm":  "drms_robot_arm_prompt.txt",
}


class DRMSAgent(AgentBase):

    def __init__(self, env: Any, args: Any, logger: Any = None) -> None:
        super().__init__(env, args, logger)
        self._action_history_list: List[str] = []
        self.action_history: str = ""
        self.rounds: int = getattr(args, "rounds", 1)

    def run(self) -> Tuple[bool, int]:
        task_goal = self.env.task_goal
        max_steps = getattr(self.args, "max_steps", None)
        gt_limit = 2 * self.env.ground_truth_step_num
        success = False

        while True:
            step = self.env.steps
            obs = self.env.get_observations()

            if self.logger:
                self.logger.start_step(step, {
                    "action_history": self.action_history,
                    "rounds": self.rounds,
                })

            ids = list(self.env.id_name_dict.keys())
            random.shuffle(ids)

            dialogue_record: List[str] = []
            dialogue_record_str: str = ""
            total_actions: List[str] = []
            last_action: Optional[str] = None
            last_class_name: Optional[str] = None

            for round_i in range(self.rounds):
                for agent_idx in ids:
                    class_name, node_id = self.env.id_name_dict[agent_idx]
                    agent_obs_text = self.env.obs2text(obs, agent_idx)

                    # Combined action list: this agent's actions + previously collected
                    _, _, local_plans = self.env.get_available_plans(agent_idx, obs)
                    own_prefixed = [f"<{class_name}>({node_id}): {a}" for a in local_plans]
                    combined = list(set(own_prefixed + total_actions))
                    combined_str = "".join(f"{chr(ord('A') + i)}. {a}\n" for i, a in enumerate(combined))

                    prompt_file = _AGENT_PROMPT_FILES.get(class_name, "drms_robot_dog_prompt.txt")
                    agent_prompt = (
                        self._read(prompt_file)
                        .replace("#TASK_GOAL#", self.env.goal_instruction)
                        .replace("#ACTION_HISTORY#", self.action_history)
                        .replace("#OBSERVATION#", agent_obs_text)
                        .replace("#ACTION_LIST#", combined_str)
                        .replace("#DIALOGUE_RECORD#", dialogue_record_str)
                    )

                    # ── Call A: Agent reasoning ──────────────────────
                    try:
                        response, p_in, p_out, cost, lat = self._llm([{"role": "user", "content": agent_prompt}])
                    except Exception as exc:
                        print(f"[DRMSAgent] Step {step} round {round_i} agent LLM error: {exc}")
                        traceback.print_exc()
                        response = ""
                    self._log(
                        f"{class_name}({node_id})", "agent_reasoning",
                        agent_prompt, response, p_in, p_out, cost, lat,
                    )

                    # ── Call B: Action extraction ────────────────────
                    extract_prompt = response + _EXTRACT_SUFFIX_DRMS
                    try:
                        extracted, p_in, p_out, cost, lat = self._llm([{"role": "user", "content": extract_prompt}])
                    except Exception as exc:
                        print(f"[DRMSAgent] Step {step} round {round_i} extraction LLM error: {exc}")
                        traceback.print_exc()
                        extracted = ""
                    self._log(
                        f"{class_name}({node_id})", "action_extraction",
                        extract_prompt, extracted, p_in, p_out, cost, lat,
                    )

                    parsed = self._parse_action_from_list(combined, extracted)
                    last_action = parsed
                    last_class_name = class_name

                    dialogue_record.append(f"<{class_name}>({node_id}): {response}")
                    numbered = [f"[{i + 1}]、{item}" for i, item in enumerate(dialogue_record)]
                    dialogue_record_str = "\n".join(numbered)

                    total_actions = list(set(total_actions + own_prefixed))

                    print(f"[Step {step}] round {round_i} <{class_name}>({node_id}) → {parsed}")

            # ── Execute last round's action ──────────────────────────
            if last_action is None:
                print(f"[DRMSAgent] Step {step}: no valid action in final round — skipping.")
                # Match original arena_mp2.py:283-284 exactly. `wrong_action` is
                # the last-looped agent's dialogue entry.
                wrong_action = dialogue_record[-1] if dialogue_record else ""
                wrong = (
                    f"The final action to be performed in the last round of discussion is "
                    f"incorrect, and the result of the last round of discussion is: "
                    f"&&{wrong_action}&&. Because the proposed plan is not in the action list. "
                    f"So a rethink to discuss a new action is needed."
                )
                self.action_history = self._push_history(self._action_history_list, wrong, max_entries=5)
                self.env.skip_step()
            else:
                agent_msg = dialogue_record[-1] if dialogue_record else ""
                self.action_history = self._push_history(
                    self._action_history_list, agent_msg, max_entries=5
                )
                class_name_exec, node_id_exec, action_str = self._parse_full_action(last_action)
                if class_name_exec is None:
                    print(f"[DRMSAgent] Step {step}: cannot parse final action '{last_action}' — skipping.")
                    self.env.skip_step()
                else:
                    agent_idx_exec = self._agent_idx_by_id(node_id_exec)
                    before_text = self.env.obs2text(obs, agent_idx_exec) if agent_idx_exec is not None else ""
                    try:
                        done, _, _, _, _ = self.env.step(class_name_exec, node_id_exec, action_str, task_goal)
                        after_obs = self.env.get_observations()
                        after_text = self.env.obs2text(after_obs, agent_idx_exec) if agent_idx_exec is not None else ""
                        if self.logger:
                            self.logger.log_environment_change(
                                before_state={"obs_text": before_text},
                                after_state={"obs_text": after_text},
                                action_executed=last_action,
                                action_success=done,
                            )
                        success = done
                        obs = after_obs
                    except Exception as exc:
                        print(f"[DRMSAgent] Step {step}: env.step error: {exc}")
                        traceback.print_exc()
                        break
                    print(f"[Step {step}] → {last_action}")

            if success:
                print(f"[DRMSAgent] Task succeeded in {self.env.steps} steps.")
                break

            if max_steps is not None and self.env.steps >= max_steps:
                print(f"[DRMSAgent] Early stop triggered by --max_steps={max_steps}.")
                break
            if self.env.steps > gt_limit:
                print(f"[DRMSAgent] Exceeded step limit ({gt_limit}). Stopping.")
                break

        return success, self.env.steps
