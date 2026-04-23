"""
Unified JSON logging system for all experiment methods.
Supports multi-method comparison and web UI visualization.
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional


class UnifiedLogger:
    """
    Unified logger for all experiment methods (PEFA, CRMS, DRMS, MCTS, etc).
    Generates structured JSON logs with token counting per LLM call.
    """

    def __init__(
        self,
        task_id: int,
        env_id: str,
        task_name: str,
        method_name: str,
        log_dir: str = "./log",
        ground_truth_steps: Optional[int] = None,
        timestamp: Optional[str] = None,
    ):
        self.task_id = task_id
        self.env_id = env_id
        self.task_name = task_name
        self.method_name = method_name  # e.g., "PEFA", "CRMS", "DRMS"
        self.log_dir = log_dir

        # Normalize path components to avoid os.path.join type errors
        self._env_id_str = str(env_id)
        self._task_id_str = str(task_id)
        self._method_name_str = str(method_name)
        self._log_dir_str = str(log_dir)
        self.ground_truth_steps = ground_truth_steps
        # Use provided timestamp or generate one (YYYYMMDD_HHMM format)
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M")
        
        # Create hierarchical directory structure: log/<method>/<env>/<task>/
        self.log_subdir = os.path.join(
            self._log_dir_str,
            self._method_name_str.lower(),
            self._env_id_str,
            f"task_{self._task_id_str}",
        )
        os.makedirs(self.log_subdir, exist_ok=True)
        
        # Main structure
        self.log_data = {
            "metadata": {
                "task_id": task_id,
                "env_id": env_id,
                "task_name": task_name,
                "method": method_name,
                "ground_truth_steps": ground_truth_steps,
                "timestamp": datetime.now().isoformat(),
                "timestamp_compact": self.timestamp,
            },
            "steps": [],
            "summary": {
                "total_steps": 0,
                "success": False,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_cost": 0.0,
            }
        }
        
        self.current_step_idx = 0

    def start_step(self, step_id: int, observation: Dict[str, Any]):
        """Initialize a new step."""
        self.current_step_idx = step_id
        step_data = {
            "step_id": step_id,
            "observation": observation,
            "llm_calls": [],
            "environment_change": None,
        }
        self.log_data["steps"].append(step_data)

    def log_llm_call(
        self,
        agent: str,
        call_type: str,  # "oracle", "agent_plan", "judge", etc.
        prompt: str,
        response: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0.0,
        execution_result: str = "success",  # success, failure, unexpected_format
        parsed_action: Optional[str] = None,
        raw_reasoning: Optional[str] = None,
        latency_ms: Optional[float] = None,
        reasoning_tokens: int = 0,
    ):
        """
        Log a single LLM call with full details.
        
        Args:
            agent: e.g., "oracle", "robot_dog", "robot_arm", "quadrotor"
            call_type: type of call (oracle_reasoning, agent_plan, judge_verify, etc)
            prompt: the prompt sent to LLM
            response: the full response from LLM
            prompt_tokens: number of tokens in prompt
            completion_tokens: number of tokens in response
            cost: estimated cost of this call
            execution_result: whether this call led to valid action
            parsed_action: the parsed action from response (if any)
            raw_reasoning: intermediate reasoning (optional)
        """
        if self.current_step_idx < len(self.log_data["steps"]):
            call_record = {
                "agent": agent,
                "call_type": call_type,
                "prompt": prompt,
                "response": response,
                "raw_reasoning": raw_reasoning,
                "parsed_action": parsed_action,
                "execution_result": execution_result,
                "tokens": {
                    "prompt":     prompt_tokens,
                    "completion": completion_tokens,
                    "reasoning":  reasoning_tokens,
                    "total":      prompt_tokens + completion_tokens,
                },
                "cost": cost,
                "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
            }
            self.log_data["steps"][self.current_step_idx]["llm_calls"].append(call_record)

            self.log_data["summary"]["total_tokens_in"] += prompt_tokens
            self.log_data["summary"]["total_tokens_out"] += completion_tokens
            self.log_data["summary"]["total_cost"] += cost
            if latency_ms is not None:
                self.log_data["summary"]["total_latency_ms"] = (
                    self.log_data["summary"].get("total_latency_ms", 0.0) + latency_ms
                )

    def log_step_overview(
        self,
        subgoal: Optional[str] = None,
        thought: Optional[str] = None,
        action: Optional[str] = None,
        cannot_reason: Optional[str] = None,
        env_outcome: Optional[str] = None,
    ):
        """Attach a compact per-step overview (subgoal + thought + action/CANNOT + env feedback)
        to the current step, for display in the HTML report's Overview tab."""
        if self.current_step_idx < len(self.log_data["steps"]):
            self.log_data["steps"][self.current_step_idx]["overview"] = {
                "subgoal":       subgoal,
                "thought":       thought,
                "action":        action,
                "cannot_reason": cannot_reason,
                "env_outcome":   env_outcome,
            }

    def log_environment_change(
        self,
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
        action_executed: str,
        action_success: bool,
        changed_relations: Optional[List[Dict[str, Any]]] = None,
    ):
        """Log the environment state change after action execution."""
        if self.current_step_idx < len(self.log_data["steps"]):
            self.log_data["steps"][self.current_step_idx]["environment_change"] = {
                "action": action_executed,
                "success": action_success,
                "before": before_state,
                "after": after_state,
                "changed_relations": changed_relations or [],
            }

    def finalize(self, success: bool, final_steps: int):
        """Finalize the log and write to JSON file."""
        self.log_data["summary"]["success"] = success
        self.log_data["summary"]["total_steps"] = final_steps
        
        # Generate filename with timestamp: {env_id}_{method}_{timestamp}.json
        filename = f"{self._env_id_str}_{self._method_name_str.lower()}_{self.timestamp}.json"
        filepath = os.path.join(self.log_subdir, filename)
        
        # Write JSON
        with open(filepath, "w") as f:
            json.dump(self.log_data, f, indent=2)
        
        return filepath

    def get_current_log(self) -> Dict[str, Any]:
        """Return current log data (useful for debugging)."""
        return self.log_data


class MultiRunComparator:
    """
    Utility for loading and comparing multiple JSON logs.
    Supports cross-method analysis, cross-task aggregation, and unified log parsing.
    """

    def __init__(self, log_dir: str = "./log"):
        self.log_dir = log_dir

    def load_all_logs(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all JSON logs grouped by (task, method).
        Returns: {
            "task_2_PEFA": [...],
            "task_2_CRMS": [...],
            ...
        }
        """
        results = {}
        for root, dirs, files in os.walk(self.log_dir):
            for filename in files:
                if filename.endswith(".json"):
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, "r") as f:
                            data = json.load(f)
                            task_id = data["metadata"]["task_id"]
                            method = data["metadata"]["method"]
                            key = f"task_{task_id}_{method}"
                            if key not in results:
                                results[key] = []
                            results[key].append(data)
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
        return results

    def load_logs_for_method(self, method: str) -> Dict[int, List[Dict[str, Any]]]:
        """
        Load all logs for a specific method, grouped by task_id.
        Returns: {
            2: [log1, log2, ...],  # Multiple runs for task 2
            3: [log1, log2, ...],  # Multiple runs for task 3
            ...
        }
        """
        results = {}
        for root, dirs, files in os.walk(self.log_dir):
            for filename in files:
                if filename.endswith(".json"):
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, "r") as f:
                            data = json.load(f)
                            if data["metadata"]["method"] == method:
                                task_id = data["metadata"]["task_id"]
                                if task_id not in results:
                                    results[task_id] = []
                                results[task_id].append(data)
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
        return results

    def load_logs_for_task(self, task_id: int) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all logs for a specific task, grouped by method.
        Returns: {
            "PEFA": [log1, log2, ...],
            "CRMS": [log1, log2, ...],
            ...
        }
        """
        results = {}
        for root, dirs, files in os.walk(self.log_dir):
            for filename in files:
                if filename.endswith(".json"):
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, "r") as f:
                            data = json.load(f)
                            if data["metadata"]["task_id"] == task_id:
                                method = data["metadata"]["method"]
                                if method not in results:
                                    results[method] = []
                                results[method].append(data)
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
        return results

    def merge_task_logs(self, method: str, task_id: int) -> Dict[str, Any]:
        """
        Merge multiple runs of the same task/method into a unified log.
        Useful when the same task is run multiple times with the same method.
        
        Returns a merged structure with aggregated statistics.
        """
        logs = self.load_logs_for_task(task_id).get(method, [])
        if not logs:
            return {}
        
        merged = {
            "metadata": {
                "task_id": task_id,
                "method": method,
                "num_runs": len(logs),
                "merged_timestamp": datetime.now().isoformat(),
            },
            "runs": logs,
            "aggregated_summary": {
                "avg_steps": sum(log["summary"]["total_steps"] for log in logs) / len(logs),
                "avg_tokens_in": sum(log["summary"]["total_tokens_in"] for log in logs) / len(logs),
                "avg_tokens_out": sum(log["summary"]["total_tokens_out"] for log in logs) / len(logs),
                "avg_cost": sum(log["summary"]["total_cost"] for log in logs) / len(logs),
                "success_rate": sum(1 for log in logs if log["summary"]["success"]) / len(logs),
            }
        }
        return merged

    def compare_methods_on_task(self, task_id: int, methods: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple methods on a single task.
        If methods is None, compares all available methods for this task.
        """
        task_logs = self.load_logs_for_task(task_id)
        
        if methods is None:
            methods = list(task_logs.keys())
        
        comparison = {}
        for method in methods:
            logs = task_logs.get(method, [])
            if logs:
                # Use the first run, or average if multiple
                log = logs[0]  # Could also average across multiple runs
                comparison[method] = {
                    "success": log["summary"]["success"],
                    "steps": log["summary"]["total_steps"],
                    "tokens_in": log["summary"]["total_tokens_in"],
                    "tokens_out": log["summary"]["total_tokens_out"],
                    "cost": log["summary"]["total_cost"],
                    "num_llm_calls": sum(len(step.get("llm_calls", [])) for step in log["steps"]),
                    "num_runs": len(logs),
                }
        
        return comparison

    def get_method_summary(self, method: str) -> Dict[int, Dict[str, Any]]:
        """
        Get summary statistics for all tasks run with a specific method.
        Returns aggregated metrics per task.
        """
        method_logs = self.load_logs_for_method(method)
        summary = {}
        
        for task_id, logs in method_logs.items():
            summary[task_id] = {
                "num_runs": len(logs),
                "avg_success": sum(1 for log in logs if log["summary"]["success"]) / len(logs),
                "avg_steps": sum(log["summary"]["total_steps"] for log in logs) / len(logs),
                "avg_tokens_in": sum(log["summary"]["total_tokens_in"] for log in logs) / len(logs),
                "avg_tokens_out": sum(log["summary"]["total_tokens_out"] for log in logs) / len(logs),
                "avg_cost": sum(log["summary"]["total_cost"] for log in logs) / len(logs),
            }
        
        return summary
