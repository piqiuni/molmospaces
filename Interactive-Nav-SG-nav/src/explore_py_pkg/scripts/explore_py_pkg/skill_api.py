from __future__ import annotations


class ExplorationSkillApi:
    """Small facade intended for future agent/LLM control surfaces."""

    def __init__(self, explorer_node):
        self._node = explorer_node

    def get_explorer_summary(self) -> dict:
        return self._node.build_status_payload()

    def set_exploration_mode(self, mode: str) -> None:
        self._node.value_fusion.strategy_bias["mode"] = str(mode)

    def set_interest_bias(self, target_object=None, room_type=None, container_type=None) -> None:
        if target_object is not None:
            self._node.value_fusion.strategy_bias["target_object"] = str(target_object)
        if room_type is not None:
            self._node.value_fusion.strategy_bias["room_type"] = str(room_type)
        if container_type is not None:
            self._node.value_fusion.strategy_bias["container_type"] = str(container_type)

    def set_llm_value_grid(self, grid) -> None:
        self._node.value_fusion.set_llm_value_grid(grid)

    def request_next_subgoal(self):
        return self._node.compute_next_subgoal(force=True)
