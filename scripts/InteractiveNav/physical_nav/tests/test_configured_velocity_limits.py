from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from velocity_safety import VelocitySafetyConfig, VelocitySafetyLimiter


def limiter():
    config = yaml.safe_load((Path(__file__).parents[1] / "config/physical_nav.yaml").read_text())["velocity_safety"]
    names = {field.name for field in fields(VelocitySafetyConfig)}
    return VelocitySafetyLimiter(VelocitySafetyConfig(**{key: value for key, value in config.items() if key in names}))


@pytest.mark.parametrize("sign", [-1, 1])
def test_configured_pure_turn_floor_and_linear_cap(sign):
    assert limiter().limit(0, 0, sign * .1, now=1) == pytest.approx((0, 0, sign * .8))
    assert limiter().limit(sign * 1., 0, 0, now=1) == pytest.approx((sign * .5, 0, 0))


def test_turn_floor_does_not_force_moving_arc_and_stop_stays_immediate():
    safety = limiter()
    assert safety.limit(.4, 0, .2, now=1) == pytest.approx((.4, 0, .2))
    assert safety.limit(0, 0, 0, now=1.01) == (0, 0, 0)
