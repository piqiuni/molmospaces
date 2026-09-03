from dataclasses import dataclass

@dataclass(frozen=True)
class AblationConfig:
    module1: str = "dynamic_rule"
    module2: str = "rule_cost"
    module3: str = "rule_verified"

    def __post_init__(self) -> None:
        allowed = {"module1": {"static_semantic", "dynamic_rule", "dynamic_mllm"}, "module2": {"rule_cost", "mllm_score"}, "module3": {"direct_atomic", "rule_verified", "mllm_skill_verified", "external_mllm_verified"}}
        for name, value in (("module1", self.module1), ("module2", self.module2), ("module3", self.module3)):
            if str(value).casefold() not in allowed[name]: raise ValueError(f"unsupported {name} ablation mode: {value}")

    @property
    def uses_mllm(self) -> bool:
        return any(str(v).endswith("_mllm") or str(v).startswith("mllm_") or str(v) == "external_mllm_verified" for v in (self.module1, self.module2, self.module3))
