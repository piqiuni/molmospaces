"""Configuration module for MolmoSpaces experiments.

This module provides configuration classes organized by category:
- abstract_config: Base Config class
- abstract_exp_config: Base experiment configuration
- camera_configs: Camera-related configurations
- robot_configs: Robot-related configurations
- task_configs: Task-related configurations
- task_sampler_configs: Task sampler-related configurations
- policy_configs: Policy-related configurations
"""

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "Config": "molmo_spaces.configs.abstract_config",
    "MlSpacesExpConfig": "molmo_spaces.configs.abstract_exp_config",
    "CameraSystemConfig": "molmo_spaces.configs.camera_configs",
    "CameraConfig": "molmo_spaces.configs.camera_configs",
    "MjcfCameraConfig": "molmo_spaces.configs.camera_configs",
    "RobotMountedCameraConfig": "molmo_spaces.configs.camera_configs",
    "FixedExocentricCameraConfig": "molmo_spaces.configs.camera_configs",
    "RandomizedExocentricCameraConfig": "molmo_spaces.configs.camera_configs",
    "RBY1MjcfCameraSystem": "molmo_spaces.configs.camera_configs",
    "RBY1GoProD455CameraSystem": "molmo_spaces.configs.camera_configs",
    "FrankaRandomizedD405D455CameraSystem": "molmo_spaces.configs.camera_configs",
    "FrankaDroidCameraSystem": "molmo_spaces.configs.camera_configs",
    "BaseRobotConfig": "molmo_spaces.configs.robot_configs",
    "FrankaRobotConfig": "molmo_spaces.configs.robot_configs",
    "BaseMujocoTaskConfig": "molmo_spaces.configs.task_configs",
    "PickTaskConfig": "molmo_spaces.configs.task_configs",
    "BaseMujocoTaskSamplerConfig": "molmo_spaces.configs.task_sampler_configs",
    "PickTaskSamplerConfig": "molmo_spaces.configs.task_sampler_configs",
    "BasePolicyConfig": "molmo_spaces.configs.policy_configs",
    "ObjectManipulationPlannerPolicyConfig": "molmo_spaces.configs.policy_configs",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "Config",
    "MlSpacesExpConfig",
    # Camera configs - new unified system
    "CameraSystemConfig",
    "CameraConfig",
    "MjcfCameraConfig",
    "RobotMountedCameraConfig",
    "FixedExocentricCameraConfig",
    "RandomizedExocentricCameraConfig",
    "RBY1MjcfCameraSystem",
    "RBY1GoProD455CameraSystem",
    "FrankaRandomizedD405D455CameraSystem",
    "FrankaDroidCameraSystem",
    # Robot configs
    "BaseRobotConfig",
    "FrankaRobotConfig",
    # Task configs
    "BaseMujocoTaskConfig",
    "PickTaskConfig",
    # Task sampler configs
    "BaseMujocoTaskSamplerConfig",
    "PickTaskSamplerConfig",
    # Policy configs
    "BasePolicyConfig",
    "ObjectManipulationPlannerPolicyConfig",
]
