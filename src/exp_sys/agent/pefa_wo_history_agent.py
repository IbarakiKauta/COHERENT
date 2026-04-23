"""
PEFAWoHistoryAgent: PEFA without dialogue history (ablation).

Identical to PEFAAgent except:
- Oracle prompt does not receive #DIALOGUE_HISTORY# context
- No dialogue history is accumulated across steps
"""

from __future__ import annotations

from typing import Any

from .pefa_agent import PEFAAgent


class PEFAWoHistoryAgent(PEFAAgent):

    _ORACLE_PROMPT = "pefa_wo_history_oracle_prompt.txt"
    _JUDGE_PROMPT  = "pefa_wo_history_judge_prompt.txt"
    _USE_HISTORY   = False

    def __init__(self, env: Any, args: Any, logger: Any = None) -> None:
        super().__init__(env, args, logger)
