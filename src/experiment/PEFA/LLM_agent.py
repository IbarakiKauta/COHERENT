import os
import sys

from LLM import LLM

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(curr_dir, ".."))

from agent.base_llm_agent import BaseLLMAgent


class LLM_agent(BaseLLMAgent):
	"""PEFA agent wrapper based on shared BaseLLMAgent."""

	def __init__(self, agent_id, args, agent_node, init_graph, logger=None):
		super().__init__(
			agent_id=agent_id,
			args=args,
			agent_node=agent_node,
			init_graph=init_graph,
			llm_cls=LLM,
			logger=logger,
		)

