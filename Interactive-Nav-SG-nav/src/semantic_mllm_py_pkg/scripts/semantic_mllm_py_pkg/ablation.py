from __future__ import annotations

from dataclasses import dataclass


MODULE1_MODES = {"static_semantic", "dynamic_rule", "dynamic_mllm"}
MODULE2_MODES = {"rule_cost", "mllm_score"}
MODULE3_MODES = {"direct_atomic", "rule_verified", "mllm_skill_verified"}


@dataclass(frozen=True)
class AblationConfig:
    """Independent experimental switches for the three navigation modules."""

    module1: str = "dynamic_rule"
    module2: str = "rule_cost"
    module3: str = "rule_verified"

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("module1", self.module1, MODULE1_MODES),
            ("module2", self.module2, MODULE2_MODES),
            ("module3", self.module3, MODULE3_MODES),
        ):
            normalized = str(value).casefold()
            if normalized not in allowed:
                raise ValueError(f"unsupported {name} ablation mode: {value}")

    @property
    def uses_mllm(self) -> bool:
        return any(
            value.endswith("_mllm") or value.startswith("mllm_")
            for value in (self.module1, self.module2, self.module3)
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "module1": self.module1,
            "module2": self.module2,
            "module3": self.module3,
        }
