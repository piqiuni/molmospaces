import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from explore_py_pkg.initial_scan import initial_scan_rgb_step_gate


def test_initial_scan_gate_requires_fresh_rgb_and_prior_step_sync() -> None:
    # The first RGB observation can start the scan before a prior action
    # exists.  Every later command must wait for the prior action's sync.
    assert initial_scan_rgb_step_gate(
        last_sent_rgb_step_seq=None,
        current_rgb_step_seq=0,
        latest_step_sync_index=None,
        nonzero_commands_sent=0,
        max_control_steps=28,
    ) == "send"
    assert initial_scan_rgb_step_gate(
        last_sent_rgb_step_seq=0,
        current_rgb_step_seq=1,
        latest_step_sync_index=None,
        nonzero_commands_sent=1,
        max_control_steps=28,
    ) == "wait"
    assert initial_scan_rgb_step_gate(
        last_sent_rgb_step_seq=0,
        current_rgb_step_seq=1,
        latest_step_sync_index=0,
        nonzero_commands_sent=1,
        max_control_steps=28,
    ) == "send"


def test_initial_scan_gate_rejects_repeated_or_reset_rgb_and_enforces_cap() -> None:
    assert initial_scan_rgb_step_gate(
        last_sent_rgb_step_seq=4,
        current_rgb_step_seq=4,
        latest_step_sync_index=4,
        nonzero_commands_sent=5,
        max_control_steps=28,
    ) == "wait"
    assert initial_scan_rgb_step_gate(
        last_sent_rgb_step_seq=4,
        current_rgb_step_seq=3,
        latest_step_sync_index=4,
        nonzero_commands_sent=5,
        max_control_steps=28,
    ) == "stop"
    assert initial_scan_rgb_step_gate(
        last_sent_rgb_step_seq=27,
        current_rgb_step_seq=28,
        latest_step_sync_index=27,
        nonzero_commands_sent=28,
        max_control_steps=28,
    ) == "stop"


def test_initial_scan_cap_waits_for_last_command_acknowledgement() -> None:
    # Sending action 27 must not be followed immediately by a stop command:
    # its pose update arrives with the next observation/step-sync pair.
    assert initial_scan_rgb_step_gate(
        last_sent_rgb_step_seq=27,
        current_rgb_step_seq=27,
        latest_step_sync_index=26,
        nonzero_commands_sent=28,
        max_control_steps=28,
    ) == "wait"
