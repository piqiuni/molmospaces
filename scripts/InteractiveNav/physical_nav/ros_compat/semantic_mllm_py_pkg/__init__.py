"""Minimal ROS-only compatibility surface for semantic_mapping_py_pkg.

The physical mapping node only needs AblationConfig. Keeping this shim avoids
loading the optional HTTP MLLM client into the system ROS Python process; all
Qwen calls still run through the dedicated web gateway.
"""
from .ablation import AblationConfig
__all__ = ["AblationConfig"]
