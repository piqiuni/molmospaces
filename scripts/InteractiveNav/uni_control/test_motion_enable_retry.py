import pathlib
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:
    import websocket  # noqa: F401
except ModuleNotFoundError:
    sys.modules["websocket"] = types.SimpleNamespace()

from go2_control_bridge import Go2Driver


class FakeObstaclesClient:
    def __init__(self, switch_results):
        self.switch_results = list(switch_results)
        self.remote_enable_calls = 0

    def SwitchGet(self):
        return self.switch_results.pop(0)

    def SwitchSet(self, _enabled):
        return 0

    def UseRemoteCommandFromApi(self, enabled):
        if enabled:
            self.remote_enable_calls += 1
        return 0


def driver_with_client(client):
    driver = object.__new__(Go2Driver)
    driver._client = client
    driver._velocity_control_enabled = False
    driver._last_enable_attempt = 0.0
    return driver


def test_motion_enable_retries_transient_switch_failure():
    client = FakeObstaclesClient([(3104, False), (0, True)])
    driver = driver_with_client(client)

    driver.enable_velocity_control(attempts=3, retry_delay_s=0.0)

    assert driver._velocity_control_enabled is True
    assert client.remote_enable_calls == 1


def test_motion_enable_reports_exhausted_retries():
    client = FakeObstaclesClient([(3104, False), (3104, False)])
    driver = driver_with_client(client)

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        driver.enable_velocity_control(attempts=2, retry_delay_s=0.0)

    assert driver._velocity_control_enabled is False
