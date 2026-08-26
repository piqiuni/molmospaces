"""Read-only Go2 physical interactive-navigation platform.

This package is deliberately separate from the simulator launch tree.  The
Go2 process only publishes sensor data over WebSocket; the policy machine owns
ROS, inference, mapping, diagnostics and the web UI.
"""

__all__ = ["physical_protocol", "safety_gate"]
