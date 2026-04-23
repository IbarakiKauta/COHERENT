"""
Thin wrapper around the COHERENT graph-based simulation environment.

Provides:
  - obs2text(obs, agent_idx)      — full observation text for one agent
  - global_summary(obs)           — lightweight per-agent state summary for the oracle
  - get_available_plans(agent_idx, obs) — enumerate valid actions for one agent
  - step / get_observations / task properties forwarded from Get_env_info
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from exp_sys.env.get_env_info import Get_env_info


class CoherentEnv:
    """Wraps Get_env_info and exposes text-observation utilities for exp_sys agents."""

    def __init__(self, task_data: Dict[str, Any]) -> None:
        self._env = Get_env_info(
            task_id=task_data["task_id"],
            env_id=task_data["env_id"],
            task_name=task_data["task_name"],
            graph=task_data["graph"],
            task_goal=task_data["task_goal"],
            goal_instruction=task_data["goal_instruction"],
            ground_truth_step_num=task_data["ground_truth_step_num"],
            agent=task_data["agent"],
            num_agent=task_data["num_agent"],
        )

    # ------------------------------------------------------------------
    # Forwarded properties
    # ------------------------------------------------------------------

    @property
    def steps(self) -> int:
        return self._env.steps

    @property
    def task_goal(self):
        return self._env.task_goal

    @property
    def goal_instruction(self) -> str:
        return self._env.goal_instruction

    @property
    def ground_truth_step_num(self) -> int:
        return self._env.ground_truth_step_num

    @property
    def id_name_dict(self) -> Dict[int, List]:
        """Maps agent_idx → [class_name, node_id]."""
        return self._env.id_name_dict

    @property
    def graph(self):
        return self._env.graph

    @property
    def task_id(self):
        return self._env.task_id

    @property
    def env_id(self):
        return self._env.env_id

    @property
    def task_name(self):
        return self._env.task_name

    def get_observations(self) -> Dict[int, Dict]:
        return self._env.get_observations()

    def step(self, class_name: str, agent_id: int, action: str, task_goal):
        return self._env.step(class_name, agent_id, action, task_goal)

    def skip_step(self) -> int:
        """Increment step counter without executing an action."""
        self._env.steps += 1
        return self._env.steps

    # ------------------------------------------------------------------
    # Full text observation for a single agent
    # ------------------------------------------------------------------

    def obs2text(self, obs: Dict[int, Dict], agent_idx: int) -> str:
        """Convert raw observation dict to natural language for agent at agent_idx."""
        observation = obs[agent_idx]
        id2node = {node["id"]: node for node in observation["nodes"]}
        agent_node_id = self._env.id_name_dict[agent_idx][1]
        agent_class = id2node.get(agent_node_id, {}).get("class_name", "")
        with_basket_id: Optional[int] = None
        text = ""

        for node in observation["nodes"]:
            if node["category"] == "Agents" and node["id"] == agent_node_id:
                text += f"I am <{node['class_name']}>({node['id']}). "
                if node["states"]:
                    text += "Now my state is: " + ", ".join(node["states"]) + ". "
                for edge in observation["edges"]:
                    if edge["from_id"] == node["id"]:
                        text += (
                            f"I am {edge['relation_type']} the "
                            f"<{id2node[edge['to_id']]['class_name']}>({edge['to_id']}). "
                        )
                    if edge["relation_type"] == "WITH":
                        with_basket_id = edge["to_id"]
                text += "\n"

        for node in observation["nodes"]:
            if node["category"] == "Rooms" and node["id"] == observation["agent_in_room_id"]:
                text += f"Now I am in the <{node['class_name']}>({node['id']}). In this room, I can see:\n"

        for node in observation["nodes"]:
            if node["id"] == agent_node_id or node["category"] == "Rooms":
                continue
            text += f"<{node['class_name']}>({node['id']}). "
            if node["properties"]:
                text += "Its properties are: " + ", ".join(node["properties"]) + ". "
            if node["states"]:
                text += "Now its state is: " + ", ".join(node["states"]) + ".\n"
            else:
                text += "\n"

        text += "These objects have a certain position relationship with each other:\n"
        for node in observation["nodes"]:
            if node["id"] == agent_node_id or node["category"] == "Rooms":
                continue
            for edge in observation["edges"]:
                if edge["from_id"] != node["id"]:
                    continue
                to_node = id2node.get(edge["to_id"])
                if to_node is None:
                    continue
                if edge["from_id"] == with_basket_id and agent_class == "quadrotor":
                    text += (
                        f"The <{node['class_name']}>({node['id']}) is with me LAND "
                        f"{edge['relation_type']} the <{to_node['class_name']}>({to_node['id']}).\n"
                    )
                else:
                    text += (
                        f"The <{node['class_name']}>({node['id']}) is "
                        f"{edge['relation_type']} the <{to_node['class_name']}>({to_node['id']}).\n"
                    )

        for edge in observation["edges"]:
            if edge["relation_type"] == "WITH" and agent_class == "quadrotor":
                basket = id2node.get(edge["to_id"])
                if basket is None:
                    continue
                in_basket = False
                text += f"I have a <{basket['class_name']}>({basket['id']}) with me. "
                for ee in observation["edges"]:
                    if ee["to_id"] == edge["to_id"] and ee["relation_type"] == "INSIDE":
                        inner = id2node.get(ee["from_id"])
                        if inner:
                            text += (
                                f"<{inner['class_name']}>({inner['id']}) is in my "
                                f"<{basket['class_name']}>({basket['id']}).\n"
                            )
                            in_basket = True
                if not in_basket:
                    text += f"But nothing is in my <{basket['class_name']}>({basket['id']}).\n"
            if edge["relation_type"] == "HOLD" and agent_class != "quadrotor":
                obj = id2node.get(edge["to_id"])
                if obj:
                    text += f"I am holding a <{obj['class_name']}>({obj['id']}) in my hand.\n"

        return text

    # ------------------------------------------------------------------
    # Lightweight global summary for oracle (Turn 1)
    # ------------------------------------------------------------------

    def global_summary(self, obs: Dict[int, Dict]) -> str:
        """
        Produce a concise per-agent status line:
          <class_name>(id): In <room>(room_id), state: X[, holding/basket info]
        """
        lines = []
        for agent_idx, (class_name, node_id) in self._env.id_name_dict.items():
            agent_obs = obs[agent_idx]
            id2node = {n["id"]: n for n in agent_obs["nodes"]}

            room_id = agent_obs.get("agent_in_room_id")
            room_name = id2node[room_id]["class_name"] if room_id is not None and room_id in id2node else "unknown"

            agent_node = id2node.get(node_id, {})
            states = agent_node.get("states", [])
            state_str = ", ".join(states) if states else "idle"

            held: Optional[str] = None
            basket_contents: List[str] = []
            for edge in agent_obs["edges"]:
                if edge["from_id"] != node_id:
                    continue
                if edge["relation_type"] == "HOLD":
                    obj = id2node.get(edge["to_id"])
                    if obj:
                        held = f"<{obj['class_name']}>({obj['id']})"
                if edge["relation_type"] == "WITH":
                    basket_id = edge["to_id"]
                    for ee in agent_obs["edges"]:
                        if ee["to_id"] == basket_id and ee["relation_type"] == "INSIDE":
                            inner = id2node.get(ee["from_id"])
                            if inner:
                                basket_contents.append(f"<{inner['class_name']}>({inner['id']})")

            extra = ""
            if held:
                extra = f", holding {held}"
            elif basket_contents:
                extra = f", basket: {', '.join(basket_contents)}"
            elif class_name == "quadrotor" and "LAND" in states:
                extra = ", basket: empty"

            lines.append(
                f"<{class_name}>({node_id}): In <{room_name}>({room_id}), state: {state_str}{extra}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Environment-structure summary for the oracle (coarse subgoal planning)
    # ------------------------------------------------------------------

    def env_structure_summary(self, obs: Dict[int, Dict]) -> str:
        """
        Compact per-object listing so the oracle can pick the right agent.
        Lists non-agent, non-floor objects with: room, surface it rests on,
        height class, container/landable/grab flags, and current state.

        Uses the first agent's observation as the source of truth for the
        shared global graph; all agents see the full node/edge set in this env.
        """
        if not obs:
            return "(no observation)"
        # Each agent's obs is limited to its own room; union all agents' views
        # so the oracle sees the full cross-room picture.
        id2node: Dict[int, Dict] = {}
        all_edges: List[Dict] = []
        seen_edge_keys: set = set()
        for agent_obs in obs.values():
            for n in agent_obs.get("nodes", []):
                id2node.setdefault(n["id"], n)
            for e in agent_obs.get("edges", []):
                key = (e["from_id"], e["relation_type"], e["to_id"])
                if key not in seen_edge_keys:
                    seen_edge_keys.add(key)
                    all_edges.append(e)

        room_of: Dict[int, int] = {}
        surface_of: Dict[int, int] = {}
        inside_of: Dict[int, int] = {}
        for e in all_edges:
            fid, rt, tid = e["from_id"], e["relation_type"], e["to_id"]
            if rt == "INSIDE" and id2node.get(tid, {}).get("category") == "Rooms":
                room_of[fid] = tid
            elif rt == "ON":
                surface_of[fid] = tid
            elif rt == "INSIDE" and id2node.get(tid, {}).get("category") != "Rooms":
                inside_of[fid] = tid

        def _room_of(obj_id: int) -> Optional[int]:
            seen = set()
            cur = obj_id
            while cur is not None and cur not in seen:
                seen.add(cur)
                if cur in room_of:
                    return room_of[cur]
                nxt = surface_of.get(cur, inside_of.get(cur))
                if nxt is None:
                    return None
                cur = nxt
            return None

        lines: List[str] = []
        for node in id2node.values():
            cat = node.get("category")
            if cat in ("Rooms", "Agents", "Floor"):
                continue
            props = node.get("properties", []) or []
            states = node.get("states", []) or []
            room_id = _room_of(node["id"])
            room_name = id2node[room_id]["class_name"] if room_id in id2node else "?"

            surf_id = surface_of.get(node["id"]) or inside_of.get(node["id"])
            if surf_id is not None and surf_id in id2node and id2node[surf_id].get("category") != "Rooms":
                surf = id2node[surf_id]
                rel = "on" if node["id"] in surface_of else "inside"
                surf_desc = f"{rel} <{surf['class_name']}>({surf['id']})"
            else:
                surf_desc = "in-room"

            flags: List[str] = []
            if "HIGH_HEIGHT" in props:
                flags.append("HIGH")
            elif "LOW_HEIGHT" in props:
                flags.append("LOW")
            if "ON_HIGH_SURFACE" in props:
                flags.append("ON_HIGH")
            if "CONTAINERS" in props:
                flags.append("container")
            if "SURFACES" in props:
                flags.append("surface")
            if "LANDABLE" in props:
                flags.append("landable")
            if "GRABABLE" in props:
                flags.append("grabable")
            flag_str = f" [{', '.join(flags)}]" if flags else ""

            state_str = f" state={{{', '.join(states)}}}" if states else ""
            lines.append(
                f"<{node['class_name']}>({node['id']}): room=<{room_name}>({room_id}), {surf_desc}{flag_str}{state_str}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Available action enumeration for one agent
    # ------------------------------------------------------------------

    def get_available_plans(
        self, agent_idx: int, obs: Dict[int, Dict]
    ) -> Tuple[str, int, List[str]]:
        """
        Returns (plans_str, count, plans_list) for agent at agent_idx.
        Derives agent state from the observation graph, then enumerates valid actions.
        """
        agent_obs = obs[agent_idx]
        node_id = self._env.id_name_dict[agent_idx][1]
        id2node = {n["id"]: n for n in agent_obs["nodes"]}
        agent_node = id2node.get(node_id, {})
        init_id2node = {x["id"]: x for x in self._env.graph["nodes"]}

        grabbed_objects: Optional[Dict] = None
        reachable_objects: List[Dict] = []
        on_surfaces: Optional[Dict] = None
        landable_surfaces: Optional[Dict] = None
        all_landable_surfaces = [n for n in agent_obs["nodes"] if "LANDABLE" in n.get("properties", [])]
        on_same_surfaces: List[Dict] = []
        current_room: Optional[Dict] = None

        on_same_surfaces_ids: set = set()

        for e in agent_obs["edges"]:
            x, r, y = e["from_id"], e["relation_type"], e["to_id"]
            if x != node_id:
                continue
            if r == "INSIDE":
                current_room = id2node.get(y)
            elif r == "ON":
                on_surfaces = id2node.get(y)
                if agent_node.get("class_name") in ("robot_arm", "robot arm"):
                    self._collect_on_same_surfaces(
                        agent_obs, id2node, node_id, y, on_same_surfaces_ids
                    )
            elif r == "HOLD":
                grabbed_objects = id2node.get(y)
            elif r == "CLOSE":
                if y in id2node:
                    reachable_objects.append(id2node[y])
            elif r == "ABOVE" and "LANDABLE" in id2node.get(y, {}).get("properties", []):
                landable_surfaces = id2node.get(y)

        on_same_surfaces = [id2node[nid] for nid in on_same_surfaces_ids if nid in id2node]

        grabbed_id = grabbed_objects["id"] if grabbed_objects else None
        reachable_ids = {o["id"] for o in reachable_objects}
        unreached_objects = [
            n for n in agent_obs["nodes"]
            if n["id"] != grabbed_id
            and n["id"] not in reachable_ids
            and n["category"] not in ("Rooms", "Agents", "Floor")
            and "HIGH_HEIGHT" not in n.get("properties", [])
            and "ON_HIGH_SURFACE" not in n.get("properties", [])
        ]

        doors = [n for n in agent_obs["nodes"] if n["class_name"] == "door"]
        current_room_id = current_room["id"] if current_room else None
        next_rooms: List[List] = []
        for door in doors:
            for edge in self._env.graph["edges"]:
                if (
                    edge["relation_type"] == "LEADING TO"
                    and edge["from_id"] == door["id"]
                    and edge["to_id"] != current_room_id
                    and edge["to_id"] in init_id2node
                ):
                    next_rooms.append([init_id2node[edge["to_id"]], door])

        available = self._enumerate_plans(
            agent_node=agent_node,
            grabbed_objects=grabbed_objects,
            reachable_objects=reachable_objects,
            on_surfaces=on_surfaces,
            landable_surfaces=landable_surfaces,
            all_landable_surfaces=all_landable_surfaces,
            on_same_surfaces=on_same_surfaces,
            unreached_objects=unreached_objects,
            next_rooms=next_rooms,
        )

        plans_str = "".join(f"{chr(ord('A') + i)}. {p}\n" for i, p in enumerate(available))
        return plans_str, len(available), available

    def _collect_on_same_surfaces(
        self,
        agent_obs: Dict,
        id2node: Dict,
        agent_node_id: int,
        surface_id: int,
        result: set,
    ) -> None:
        """Populate result with node ids that are on the same surface as robot_arm."""
        for _ in range(3):
            for edge in agent_obs["edges"]:
                fid, rt, tid = edge["from_id"], edge["relation_type"], edge["to_id"]
                if fid == agent_node_id:
                    continue
                if (tid == surface_id and rt == "ON") or (tid in result and rt in ("ON", "INSIDE")):
                    result.add(fid)
                    node = id2node.get(fid, {})
                    if "SURFACES" in node.get("properties", []) or "CONTAINERS" in node.get("properties", []):
                        for ee in agent_obs["edges"]:
                            if ee["to_id"] == fid and ee["relation_type"] in ("INSIDE", "ON"):
                                result.add(ee["from_id"])

    def _enumerate_plans(
        self,
        agent_node: Dict,
        grabbed_objects: Optional[Dict],
        reachable_objects: List[Dict],
        on_surfaces: Optional[Dict],
        landable_surfaces: Optional[Dict],
        all_landable_surfaces: List[Dict],
        on_same_surfaces: List[Dict],
        unreached_objects: List[Dict],
        next_rooms: List[List],
    ) -> List[str]:
        available: List[str] = []
        class_name = agent_node.get("class_name", "")
        states = agent_node.get("states", [])

        if class_name == "quadrotor":
            other_landable = [s for s in all_landable_surfaces if s != landable_surfaces]
            if "FLYING" in states:
                if landable_surfaces:
                    available.append(
                        f"[land_on] <{landable_surfaces['class_name']}>({landable_surfaces['id']})"
                    )
                for surface in other_landable:
                    available.append(f"[movetowards] <{surface['class_name']}>({surface['id']})")
                for room, door in next_rooms:
                    if "OPEN" in door.get("states", []) or "OPEN_FOREVER" in door.get("states", []):
                        available.append(f"[movetowards] <{room['class_name']}>({room['id']})")
            if "LAND" in states and on_surfaces:
                available.append(
                    f"[takeoff_from] <{on_surfaces['class_name']}>({on_surfaces['id']})"
                )

        elif class_name in ("robot_dog", "robot dog"):
            for obj in reachable_objects:
                props = obj.get("properties", [])
                obj_states = obj.get("states", [])
                if grabbed_objects is None:
                    if "CONTAINERS" in props and "CLOSED" in obj_states:
                        available.append(f"[open] <{obj['class_name']}>({obj['id']})")
                    if "CONTAINERS" in props and "OPEN" in obj_states:
                        available.append(f"[close] <{obj['class_name']}>({obj['id']})")
                    if obj["class_name"] == "door" and "CLOSED" in obj_states:
                        available.append(f"[open] <{obj['class_name']}>({obj['id']})")
                    if obj["class_name"] == "door" and "OPEN" in obj_states:
                        available.append(f"[close] <{obj['class_name']}>({obj['id']})")
                    if "GRABABLE" in props:
                        available.append(f"[grab] <{obj['class_name']}>({obj['id']})")
                else:
                    if "CONTAINERS" in props and ("OPEN" in obj_states or "OPEN_FOREVER" in obj_states):
                        available.append(
                            f"[putinto] <{grabbed_objects['class_name']}>({grabbed_objects['id']})"
                            f" into <{obj['class_name']}>({obj['id']})"
                        )
                    if "SURFACES" in props:
                        available.append(
                            f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']})"
                            f" on <{obj['class_name']}>({obj['id']})"
                        )
            for obj in unreached_objects:
                available.append(f"[movetowards] <{obj['class_name']}>({obj['id']})")
            for room, door in next_rooms:
                if "OPEN" in door.get("states", []) or "OPEN_FOREVER" in door.get("states", []):
                    available.append(f"[movetowards] <{room['class_name']}>({room['id']})")

        elif class_name in ("robot_arm", "robot arm"):
            if grabbed_objects and on_surfaces:
                available.append(
                    f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']})"
                    f" on <{on_surfaces['class_name']}>({on_surfaces['id']})"
                )
            for obj in on_same_surfaces:
                props = obj.get("properties", [])
                obj_states = obj.get("states", [])
                if grabbed_objects is None:
                    if "CONTAINERS" in props and "OPEN" in obj_states:
                        available.append(f"[close] <{obj['class_name']}>({obj['id']})")
                    if "CONTAINERS" in props and "CLOSED" in obj_states:
                        available.append(f"[open] <{obj['class_name']}>({obj['id']})")
                    if "GRABABLE" in props:
                        available.append(f"[grab] <{obj['class_name']}>({obj['id']})")
                else:
                    if "CONTAINERS" in props and ("OPEN" in obj_states or "OPEN_FOREVER" in obj_states):
                        available.append(
                            f"[putinto] <{grabbed_objects['class_name']}>({grabbed_objects['id']})"
                            f" into <{obj['class_name']}>({obj['id']})"
                        )
                    if "SURFACES" in props:
                        available.append(
                            f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']})"
                            f" on <{obj['class_name']}>({obj['id']})"
                        )

        return available
