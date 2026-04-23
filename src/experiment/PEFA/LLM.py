
import copy
import openai
import os

import json
from openai import OpenAIError, OpenAI
import backoff


class LLM:
	def __init__(self, source, lm_id, args, logger=None):

		self.args = args
		self.debug = args.debug
		self.source = args.source
		self.lm_id = args.lm_id
		self.chat = True
		self.total_cost = 0
		self.device = None
		self.record_dir = f'./log/{args.env}.txt'
		self.logger = logger

		if self.source == 'openai':

			api_key = args.api_key or os.getenv("OPENAI_API_KEY")
			organization = args.organization or os.getenv("OPENAI_ORGANIZATION")
			base_url = getattr(args, 'base_url', '') or os.getenv("OPENAI_BASE_URL")

			if not api_key:
				raise ValueError("Missing OpenAI API key. Set --api_key or OPENAI_API_KEY.")

			client_kwargs = {"api_key": api_key}
			if organization:
				client_kwargs["organization"] = organization
			if base_url:
				client_kwargs["base_url"] = base_url.rstrip('/')

			client = OpenAI(**client_kwargs)
			if self.chat:
				self.sampling_params = {
					"max_tokens": args.max_tokens,
                    "temperature": args.t,
                    # "top_p": 1.0,
                    "n": args.n
				}

		def lm_engine(source, lm_id, device):
			
			@backoff.on_exception(backoff.expo, OpenAIError)
			def _generate(prompt, sampling_params):
				usage = 0
				if source == 'openai':
					try:
						if self.chat:
							prompt.insert(0,{"role":"system", "content":"You are a helper assistant."})
							response = client.chat.completions.create(
                                model=lm_id, messages=prompt, **sampling_params
                            )
							if self.debug:
								with open(f"./chat_raw.json", 'a') as f:
									f.write(json.dumps(response, indent=4))
									f.write('\n')
							generated_samples = [response.choices[i].message.content for i in
                                                    range(sampling_params['n'])]
							if 'gpt-4-0125-preview' in self.lm_id:
								usage = response.usage.prompt_tokens * 0.01 / 1000 + response.usage.completion_tokens * 0.03 / 1000
							elif 'gpt-3.5-turbo-1106' in self.lm_id:
								usage = response.usage.prompt_tokens * 0.0015 / 1000 + response.usage.completion_tokens * 0.002 / 1000
							elif 'gpt-4o-2024-11-20' in self.lm_id:
								usage = response.usage.prompt_tokens * 0.005 / 1000 + response.usage.completion_tokens * 0.015 / 1000
							else:
								usage = 0
						# mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
						# 				  range(sampling_params['n'])]
						else:
							raise ValueError(f"{lm_id} not available!")
					except OpenAIError as e:
						print(e)
						raise e
				
				else:
					raise ValueError("invalid source")
				return generated_samples, usage, response.usage.prompt_tokens, response.usage.completion_tokens

			return _generate

		self.generator = lm_engine(self.source, self.lm_id, self.device)

	def parse_answer(self, available_actions, text):
		
		text = text.replace("_", " ")
		text = text.replace("takeoff from", "takeoff_from")
		text = text.replace("land on", "land_on")

		for i in range(len(available_actions)):
			action = available_actions[i]
			if action in text:
				return action
		self.write_log_to_file('\nThe first action parsing failed!!!')

		for i in range(len(available_actions)):
			action = available_actions[i]
			option = chr(ord('A') + i)
			if f"option {option}" in text or f"{option}." in text.split(' ') or f"{option}," in text.split(' ') or f"Option {option}" in text or f"({option})" in text:
				return action
		self.write_log_to_file('\nThe second action parsing failed!!!')

		print("WARNING! No available action parsed!!! Output plan NONE!\n")
		return None

	def get_available_plans(self, agent_node, next_rooms, all_landable_surfaces, landable_surfaces, on_surfaces, 
						 grabbed_objects, reached_objects, unreached_objecs, on_same_surface_objects
						 ):
		"""
		'quadrotor':
		[land_on] <surface>
		[movetowards] <surface>/<next_room>
		[takeoff_from] <surface>

		'robot dog':
		[open] <container>/<door>
		[close] <container>/<door>
		[grab] <object>
		[putinto] <object> into <container>
		[puton] <object> on <surface>
		[movetowards] <object>

		'robot arm':
		[open] <container>
		[close] <container>
		[grab] <object>
		[putinto] <object> into <container>
		[puton] <object> on <surface>

		"""
		available_plans = []
		if agent_node["class_name"] == "quadrotor":
			other_landable_surfaces = []
			if "FLYING" in agent_node["states"]:
				if landable_surfaces is not None:
					available_plans.append(f"[land_on] <{landable_surfaces['class_name']}>({landable_surfaces['id']})")
					all_landable_surfaces.remove(landable_surfaces)
					other_landable_surfaces = copy.deepcopy(all_landable_surfaces)
				if len(other_landable_surfaces) != 0:
					for surface in other_landable_surfaces :
						available_plans.append(f"[movetowards] <{surface['class_name']}>({surface['id']})")
				for next_room in next_rooms:
					if 'OPEN' in next_room[1]['states'] or "OPEN_FOREVER" in next_room[1]['states']:
						available_plans.append(f"[movetowards] <{next_room[0]['class_name']}>({next_room[0]['id']})")

			if "LAND" in agent_node["states"]:
				if on_surfaces is not None:
					available_plans.append(f"[takeoff_from] <{on_surfaces['class_name']}>({on_surfaces['id']})")

		if agent_node["class_name"] == "robot dog" or agent_node["class_name"] == "robot_dog":
			# if grabbed_objects is not None:
			# 	available_plans.append(f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) on <{on_surfaces['class_name']}>({on_surfaces['id']})")
			# The robotic dog is not allowed to put things on the floor. If it needs to open the door and has something in its hand, it needs to find a low surface to put things first
			if len(reached_objects) != 0:
				for reached_object in reached_objects:
					if grabbed_objects is None:
						if 'CONTAINERS' in reached_object['properties'] and 'CLOSED' in reached_object['states'] or \
							reached_object['class_name'] == 'door' and 'CLOSED' in reached_object['states']:
							available_plans.append(f"[open] <{reached_object['class_name']}>({reached_object['id']})")
						if 'CONTAINERS' in reached_object['properties'] and 'OPEN' in reached_object['states'] or \
							reached_object['class_name'] == 'door' and 'OPEN' in reached_object['states']:
							available_plans.append(f"[close] <{reached_object['class_name']}>({reached_object['id']})")
						if 'GRABABLE' in reached_object['properties']:
							available_plans.append(f"[grab] <{reached_object['class_name']}>({reached_object['id']})")
					if grabbed_objects is not None:
						if 'CONTAINERS' in reached_object['properties'] and ('OPEN' in reached_object['states'] or "OPEN_FOREVER" in reached_object['states']):
							available_plans.append(f"[putinto] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) into <{reached_object['class_name']}>({reached_object['id']})")
						if 'SURFACES' in reached_object['properties']:
							available_plans.append(f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) on <{reached_object['class_name']}>({reached_object['id']})")
			
			if len(unreached_objecs) != 0:
				for unreached_object in unreached_objecs:
					available_plans.append(f"[movetowards] <{unreached_object['class_name']}>({unreached_object['id']})")
			for next_room in next_rooms:
					if 'OPEN' in next_room[1]['states'] or "OPEN_FOREVER" in next_room[1]['states']:
						available_plans.append(f"[movetowards] <{next_room[0]['class_name']}>({next_room[0]['id']})")


		if agent_node['class_name'] == 'robot arm' or agent_node['class_name'] == 'robot_arm':
			if grabbed_objects is not None:
				available_plans.append(f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) on <{on_surfaces['class_name']}>({on_surfaces['id']})")
			for on_same_surface_object in on_same_surface_objects:
				if grabbed_objects is None:
					if 'CONTAINERS' in on_same_surface_object['properties'] and 'OPEN' in on_same_surface_object['states']:
						available_plans.append(f"[close] <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")
					if 'CONTAINERS' in on_same_surface_object['properties'] and 'CLOSED' in on_same_surface_object['states']:
						available_plans.append(f"[open] <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")
					if 'GRABABLE' in on_same_surface_object['properties']:
						available_plans.append(f"[grab] <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")

				if grabbed_objects is not None:
					
					if 'CONTAINERS' in on_same_surface_object['properties'] and ('OPEN' in on_same_surface_object['states'] or "OPEN_FOREVER" in on_same_surface_object['states']):
						available_plans.append(f"[putinto] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) into <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")
					if 'SURFACES' in on_same_surface_object['properties']:
						available_plans.append(f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) on <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")

		plans = ""
		for i, plan in enumerate(available_plans):
			plans += f"{chr(ord('A') + i)}. {plan}\n"
		print(agent_node["class_name"],agent_node['id'])
		print(available_plans)
		return plans, len(available_plans), available_plans

		
	def run(self, agent_node, chat_agent_info,current_room, next_rooms, all_landable_surfaces,landable_surfaces, on_surfaces, grabbed_objects, reached_objects,unreached_objecs, on_same_surface_objects):
		info = {"num_available_actions": None,
			"prompts": None,
			"outputs": None,
			"plan": None,
			"action_list": None,
			"cost":self.total_cost, 
			f"<{agent_node['class_name']}>({agent_node['id']}) total_cost": self.total_cost}

		prompt_path = chat_agent_info['prompt_path']
		with open(prompt_path, 'r') as f:
			agent_prompt = f.read()

		available_plans, num, available_plans_list = self.get_available_plans(agent_node, next_rooms, all_landable_surfaces,landable_surfaces, on_surfaces, grabbed_objects, reached_objects,unreached_objecs, on_same_surface_objects,
																		 )
		
		agent_prompt = agent_prompt.replace('#OBSERVATION#', chat_agent_info['observation'])
		agent_prompt = agent_prompt.replace('#ACTIONLIST#', available_plans)
		agent_prompt = agent_prompt.replace('#INSTRUCTION#', chat_agent_info['instruction'])
		
		if self.debug:
			print(f"cot_prompt:\n{agent_prompt}")
		chat_prompt = [{"role": "user", "content": agent_prompt}]
		result = self.generator(chat_prompt, self.sampling_params)
		if len(result) == 4:
			outputs, usage, prompt_tokens, completion_tokens = result
		else:
			outputs, usage = result[:2]
			prompt_tokens, completion_tokens = 0, 0
		output = outputs[0]
		message = output

		self.write_log_to_file(output+'\n111111111')
		self.total_cost += usage
		info['cot_outputs'] = outputs

		if self.debug:
			print(f"cot_output:\n{output}")
			print(f"total cost: {self.total_cost}")
		sentences = output.split(".")
		first_sentence = sentences[0].upper()
		if self.logger:
			execution_result = "success" if "YES I CAN" in first_sentence else "failure" if "SORRY I CANNOT" in first_sentence else "unexpected_format"
			self.logger.log_llm_call(
				agent=f"{agent_node['class_name']}({agent_node['id']})",
				call_type="oracle_reasoning",
				prompt=agent_prompt,
				response=output,
				prompt_tokens=prompt_tokens,
				completion_tokens=completion_tokens,
				cost=usage,
				execution_result=execution_result,
			)

		if "YES I CAN" in first_sentence:
			chat_prompt = [{"role": "user", "content": agent_prompt},
							{"role": "assistant", "content": output},
							{"role": "user", "content": "Answer with only one best next action in the list of available actions. So the answer is"}]

			result = self.generator(chat_prompt, self.sampling_params)
			if len(result) == 4:
				outputs, usage, prompt_tokens, completion_tokens = result
			else:
				outputs, usage = result[:2]
				prompt_tokens, completion_tokens = 0, 0
			output = outputs[0]
			self.total_cost += usage
			self.write_log_to_file(output+'\n2222222222222')
			sentences = output.split(".")
			first_sentence = sentences[0].upper()
			if self.logger:
				execution_result = "failure" if "SORRY I CANNOT" in first_sentence else "success"
				self.logger.log_llm_call(
					agent=f"{agent_node['class_name']}({agent_node['id']})",
					call_type="action_selection",
					prompt=json.dumps(chat_prompt, ensure_ascii=False),
					response=output,
					prompt_tokens=prompt_tokens,
					completion_tokens=completion_tokens,
					cost=usage,
					execution_result=execution_result,
				)
			if "SORRY I CANNOT" not in first_sentence: 

				if self.debug:
					print(f"cot_output:\n{output}")
					print(f"total cost: {self.total_cost}")

				plan = self.parse_answer(available_plans_list, output)
				if plan is None:
					plan_str = 'no plan'
				else:
					plan_str = plan
				print(plan)
				if self.debug:
					print(f"plan: {plan}\n")
				info.update({"num_available_actions": num,
						"prompts": chat_prompt,
						# "outputs": outputs,
						"plan": plan,
						"action_list": available_plans_list,
						f"<{agent_node['class_name']}>({agent_node['id']}) total_cost": self.total_cost})
				message = f" The action I finally decided to perform is {plan_str}. "

				prompt_path = self.args.judge_prompt_path
				with open(prompt_path, 'r') as f:
					prompt = f.read()
				prompt = prompt.replace('#INSTRUCTION#', chat_agent_info['instruction'])
				prompt = prompt.replace('#PLAN#', plan_str)
				prompt = prompt.replace('#AGENT#', f"<{agent_node['class_name']}>")
				prompt = [{"role": "user", "content": prompt}]
				result = self.generator(prompt, self.sampling_params)
				if len(result) == 4:
					outputs, usage, prompt_tokens, completion_tokens = result
				else:
					outputs, usage = result[:2]
					prompt_tokens, completion_tokens = 0, 0
				output = outputs[0]
				self.total_cost += usage
				message += output
				self.write_log_to_file(output+'\n333333333333333333')
				if self.logger:
					self.logger.log_llm_call(
						agent=f"{agent_node['class_name']}({agent_node['id']})",
						call_type="judge_verify",
						prompt=json.dumps(prompt, ensure_ascii=False),
						response=output,
						prompt_tokens=prompt_tokens,
						completion_tokens=completion_tokens,
						cost=usage,
						execution_result="success",
						parsed_action=plan_str,
					)
				info.update({"outputs": message})


		if "SORRY I CANNOT" in first_sentence:
			reason_text = output[output.find("SORRY I CANNOT") + len("SORRY I CANNOT"):].strip()
			if reason_text:
				reason_text = reason_text[0].lower() + reason_text[1:] if reason_text else ""
			message = f"Sorry, the current actions I can perform cannot complete this instrcution. Possible reasons would be {reason_text} My current actionlist is: {available_plans}"
			self.write_log_to_file(message+'\n4444444444444')
		elif "YES I CAN" not in first_sentence:
			message = f"The response format from the model was unexpected: {output}. My current actionlist is: {available_plans}"
		info['cost'] = self.total_cost	
		self.write_log_to_file(f"total cost: {self.total_cost}")
		info.update({"outputs": message})
		return message, info

	def write_log_to_file(self,log_message, file_name=None):
		file_name = self.record_dir
		with open(file_name, 'a') as file:  
			file.write(log_message + '\n')  