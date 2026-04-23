"""
PEFAAgent: Centralized oracle → subgoal → agent → judge.

Per-step flow
─────────────
1. Oracle sees ALL agents' full observations + dialogue history
   → reasoning about which agent to instruct
2. Subgoal extraction  → "Hello <class>(id): instruction"
3. Target agent sees own observation + action list + instruction
   → "YES I CAN" / "SORRY I CANNOT" feasibility check
4. If YES I CAN: action selection call → pick one action
5. Judge call → verify instruction/action alignment
6. Execute action
7. Update dialogue history (rolling last-10)
"""

from __future__ import annotations

import traceback
from typing import Any, List, Optional, Tuple

from ._base import AgentBase, _EXTRACT_SUFFIX_PEFA

_AGENT_PROMPT_FILES = {
    "quadrotor":  "quadrotor_prompt.txt",
    "robot_dog":  "robot_dog_prompt.txt",
    "robot dog":  "robot_dog_prompt.txt",
    "robot arm":  "robot_arm_prompt.txt",
    "robot_arm":  "robot_arm_prompt.txt",
}


class PEFAAgent(AgentBase):

    _ORACLE_PROMPT = "pefa_oracle_prompt.txt"
    _JUDGE_PROMPT  = "pefa_judge_prompt.txt"
    _USE_HISTORY   = True

    def __init__(self, env: Any, args: Any, logger: Any = None) -> None:
        super().__init__(env, args, logger)
        self._history_list: List[str] = []
        self.dialogue_history: str = ""

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
                    "global_summary": self.env.global_summary(obs),
                    "dialogue_history": self.dialogue_history,
                })

            obs_text = self._all_obs_text(obs)

            # ── Call 1: Oracle reasoning ─────────────────────────────
            oracle_prompt = (
                self._read(self._ORACLE_PROMPT)
                .replace("#AGENT_OBSERVATIONS#", obs_text)
                .replace("#TASK_GOAL#", self.env.goal_instruction)
                .replace("#NUMBER_AGENTS#", str(len(self.env.id_name_dict)))
                .replace("#DIALOGUE_HISTORY#", self.dialogue_history if self._USE_HISTORY else "(none)")
            )
            try:
                oracle_resp, p_in, p_out, cost, lat = self._llm([{"role": "user", "content": oracle_prompt}])
            except Exception as exc:
                print(f"[PEFAAgent] Step {step} oracle LLM error: {exc}")
                traceback.print_exc()
                break
            self._log("oracle", "oracle_reasoning", oracle_prompt, oracle_resp, p_in, p_out, cost, lat)
            self._history_list.append("Oracle: " + oracle_resp)

            # ── Call 2: Subgoal extraction ───────────────────────────
            extract_prompt = oracle_resp + _EXTRACT_SUFFIX_PEFA
            try:
                subgoal, p_in, p_out, cost, lat = self._llm([{"role": "user", "content": extract_prompt}])
            except Exception as exc:
                print(f"[PEFAAgent] Step {step} extraction LLM error: {exc}")
                traceback.print_exc()
                break
            self._log("oracle", "subgoal_extraction", extract_prompt, subgoal, p_in, p_out, cost, lat)
            print(f"[Step {step}] Subgoal: {subgoal}")

            # Parse target agent
            class_name, node_id, instruction = self._parse_oracle_target(subgoal)
            if class_name is None or node_id is None:
                print(f"[PEFAAgent] Step {step}: cannot parse subgoal — skipping.")
                error_msg = (
                    "all robot agents: In the last step, the oracle's reasoning was incorrect, "
                    "and no instructions were given to any of the robot agents, therefore none of "
                    "the robot agents performed any actions. Please reassess the information in "
                    "the environment and give a correct instruction strictly following the template "
                    "'Hello <class name>(id): #message#.'"
                )
                self._history_list.append(error_msg)
                self.dialogue_history = self._format_history()
                self.env.skip_step()
                if (max_steps is not None and self.env.steps >= max_steps) or self.env.steps > gt_limit:
                    break
                continue

            agent_idx = self._agent_idx_by_name_id(class_name, node_id)
            if agent_idx is None:
                print(f"[PEFAAgent] Step {step}: agent <{class_name}>({node_id}) not found — skipping.")
                self.env.skip_step()
                if (max_steps is not None and self.env.steps >= max_steps) or self.env.steps > gt_limit:
                    break
                continue

            # ── Call 3: Agent feasibility check ─────────────────────
            agent_obs_text = self.env.obs2text(obs, agent_idx)
            plans_str, _, plans_list = self.env.get_available_plans(agent_idx, obs)
            agent_prompt_file = _AGENT_PROMPT_FILES.get(class_name, "robot_dog_prompt.txt")
            agent_prompt = (
                self._read(agent_prompt_file)
                .replace("#OBSERVATION#", agent_obs_text)
                .replace("#ACTIONLIST#", plans_str)
                .replace("#INSTRUCTION#", subgoal)
            )
            try:
                agent_resp, p_in, p_out, cost, lat = self._llm([{"role": "user", "content": agent_prompt}])
            except Exception as exc:
                print(f"[PEFAAgent] Step {step} agent LLM error: {exc}")
                traceback.print_exc()
                break

            first_sentence = agent_resp.split(".")[0].upper()
            exec_result = (
                "success" if "YES I CAN" in first_sentence
                else "failure" if "SORRY I CANNOT" in first_sentence
                else "unexpected_format"
            )
            self._log(
                f"{class_name}({node_id})", "oracle_reasoning",
                agent_prompt, agent_resp, p_in, p_out, cost, lat,
            )

            action: Optional[str] = None
            # Track `output` and `first_sentence` across calls — these are
            # reassigned on every LLM call so the post-processing block sees
            # the MOST RECENT response's first sentence (action or judge),
            # matching the original LLM.py variable reuse.
            output = agent_resp
            agent_message = agent_resp

            if "YES I CAN" in first_sentence:
                # ── Call 4: Action selection ─────────────────────────
                action_messages = [
                    {"role": "user", "content": agent_prompt},
                    {"role": "assistant", "content": agent_resp},
                    {"role": "user", "content": "Answer with only one best next action in the list of available actions. So the answer is"},
                ]
                try:
                    action_resp, p_in, p_out, cost, lat = self._llm(action_messages)
                except Exception as exc:
                    print(f"[PEFAAgent] Step {step} action LLM error: {exc}")
                    traceback.print_exc()
                    break

                output = action_resp
                first_sentence = action_resp.split(".")[0].upper()
                self._log(
                    f"{class_name}({node_id})", "action_selection",
                    action_messages, action_resp, p_in, p_out, cost, lat,
                )

                if "SORRY I CANNOT" not in first_sentence:
                    action = self._parse_action_from_list(plans_list, action_resp)
                    plan_str = action if action is not None else "no plan"
                    agent_message = f" The action I finally decided to perform is {plan_str}. "

                    # ── Call 5: Judge verification ───────────────────
                    judge_prompt = (
                        self._read(self._JUDGE_PROMPT)
                        .replace("#INSTRUCTION#", subgoal)
                        .replace("#PLAN#", plan_str)
                        .replace("#AGENT#", f"<{class_name}>")
                    )
                    try:
                        judge_resp, p_in, p_out, cost, lat = self._llm([{"role": "user", "content": judge_prompt}])
                    except Exception as exc:
                        judge_resp = ""
                        print(f"[PEFAAgent] Step {step} judge LLM error: {exc}")
                    output = judge_resp
                    agent_message += judge_resp
                    self._log(
                        f"{class_name}({node_id})", "judge_verify",
                        judge_prompt, judge_resp, p_in, p_out, cost, lat,
                        parsed_action=plan_str,
                    )

            # Post-processing: matches original LLM.py:348-355 exactly.
            # `first_sentence` is whichever response was LAST re-split:
            #   - feasibility's, if we never entered YES I CAN branch
            #   - action-selection's, if we entered YES I CAN branch
            # `output` is whichever response was LAST assigned:
            #   - feasibility's, if we never entered YES I CAN branch
            #   - action-selection's, if action had SORRY I CANNOT
            #   - judge's, otherwise
            # On the happy path (YES I CAN → valid action → judge),
            # action's first_sentence typically contains neither "YES I CAN"
            # nor "SORRY I CANNOT", so the `elif` branch OVERWRITES
            # agent_message with the "unexpected format" template whose
            # `{output}` is the judge response. This is a quirk of the
            # original we reproduce faithfully.
            if "SORRY I CANNOT" in first_sentence:
                reason_text = output[output.find("SORRY I CANNOT") + len("SORRY I CANNOT"):].strip()
                if reason_text:
                    reason_text = reason_text[0].lower() + reason_text[1:]
                agent_message = (
                    f"Sorry, the current actions I can perform cannot complete this instrcution. "
                    f"Possible reasons would be {reason_text} My current actionlist is: {plans_str}"
                )
            elif "YES I CAN" not in first_sentence:
                agent_message = (
                    f"The response format from the model was unexpected: {output}. "
                    f"My current actionlist is: {plans_str}"
                )

            print(f"[Step {step}] <{class_name}>({node_id}) → {action}")

            # ── Execute ──────────────────────────────────────────────
            before_text = self.env.obs2text(obs, agent_idx)
            if action is None:
                self.env.skip_step()
            else:
                try:
                    done, _, _, _, _ = self.env.step(class_name, node_id, action, task_goal)
                    after_obs = self.env.get_observations()
                    after_text = self.env.obs2text(after_obs, agent_idx)
                    if self.logger:
                        self.logger.log_environment_change(
                            before_state={"obs_text": before_text},
                            after_state={"obs_text": after_text},
                            action_executed=action,
                            action_success=done,
                        )
                    success = done
                    obs = after_obs
                except Exception as exc:
                    print(f"[PEFAAgent] Step {step}: env.step error: {exc}")
                    traceback.print_exc()
                    break

            # ── Update dialogue history ──────────────────────────────
            self._history_list.append(f"<{class_name}>({node_id}): {agent_message}")
            self.dialogue_history = self._format_history()

            if success:
                print(f"[PEFAAgent] Task succeeded in {self.env.steps} steps.")
                break

            if max_steps is not None and self.env.steps >= max_steps:
                print(f"[PEFAAgent] Early stop triggered by --max_steps={max_steps}.")
                break
            if self.env.steps > gt_limit:
                print(f"[PEFAAgent] Exceeded step limit ({gt_limit}). Stopping.")
                break

        return success, self.env.steps

    def _format_history(self) -> str:
        numbered = [f"[{i + 1}]、{item}" for i, item in enumerate(self._history_list)]
        return "\n".join(numbered[-10:])

    def _agent_idx_by_name_id(self, class_name: str, node_id: int) -> Optional[int]:
        cn_norm = class_name.replace(" ", "_").lower()
        for idx, (cname, nid) in self.env.id_name_dict.items():
            if cname.replace(" ", "_").lower() == cn_norm and nid == node_id:
                return idx
        return None
