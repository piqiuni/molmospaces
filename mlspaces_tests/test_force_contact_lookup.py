import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from scripts.InteractiveNav import force_interaction_runtime as runtime


class Contacts(list):
    @property
    def geom(self):
        return np.asarray([(item.geom1, item.geom2) for item in self], dtype=np.int32).reshape(-1, 2)


def model():
    names = ["world", "target", "robot_hand", "", "other", "robot_root"]
    return SimpleNamespace(nbody=len(names), geom_bodyid=np.arange(len(names)),
        body_rootid=np.array([0, 1, 5, 1, 4, 5]),
        body=lambda index: SimpleNamespace(name=names[index]))


@pytest.mark.parametrize("root", [0, 1, 4, 5, 100])
@pytest.mark.parametrize("distances", [[], [.2, -.1], [float("nan"), -.1], [-.1, float("nan")]])
def test_cached_contact_filter_preserves_membership_min_order_and_names(root, distances):
    source = model()
    contacts = Contacts([SimpleNamespace(geom1=1, geom2=2, dist=d) for d in distances])
    contacts.extend([SimpleNamespace(geom1=a, geom2=b, dist=-.02) for a, b in
                     [(1, 4), (3, 2), (2, 3), (0, 4), (-1, 3), (5, 2)]])
    data = SimpleNamespace(contact=contacts, ncon=len(contacts))
    expected = runtime._robot_articulation_contact_stats(source, data, root)
    actual = runtime._robot_articulation_contact_stats(source, data, root,
                lookup=runtime._build_robot_contact_lookup(source))
    expected_distance = expected.pop("minimum_distance")
    actual_distance = actual.pop("minimum_distance")
    assert actual == expected
    if expected_distance is not None and math.isnan(expected_distance):
        assert math.isnan(actual_distance)
    else:
        assert actual_distance == expected_distance


def test_lookup_owns_read_only_topology_but_never_caches_contacts():
    source = model()
    lookup = runtime._build_robot_contact_lookup(source)
    assert not lookup.root_ids.flags.writeable and not lookup.robot_geoms.flags.writeable
    contacts = Contacts([SimpleNamespace(geom1=1, geom2=2, dist=-.1)])
    data = SimpleNamespace(contact=contacts, ncon=1)
    assert runtime._robot_articulation_contact_stats(source, data, 1, lookup=lookup)["count"] == 1
    contacts[0].geom2 = 4
    assert runtime._robot_articulation_contact_stats(source, data, 1, lookup=lookup)["count"] == 0
    data.ncon = 0
    assert runtime._robot_articulation_contact_stats(source, data, 1, lookup=lookup) == {
        "count": 0, "minimum_distance": None, "body_pairs": []}
    source.body_rootid[:] = 100
    assert lookup.root_ids[1] == 1
    assert runtime._build_robot_contact_lookup(source).root_ids[1] == 100


def test_drive_trajectory_and_callbacks_match_uncached_contact_checks(monkeypatch):
    xml = """<mujoco><option gravity="0 0 0" timestep=".002"/><worldbody>
      <body name="target"><joint name="hinge" type="hinge" axis="0 0 1" range="0 90" damping="1"/>
        <geom type="box" size=".3 .03 .15" pos=".3 0 0" mass="1"/></body>
      <body name="robot_probe" pos=".3 .06 0"><geom type="sphere" size=".06"/></body>
      </worldbody></mujoco>"""
    optimized = runtime._robot_articulation_contact_stats
    results, trajectories = [], []
    for cached in (False, True):
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert data.ncon > 0
        if cached:
            monkeypatch.setattr(runtime, "_robot_articulation_contact_stats", optimized)
        else:
            monkeypatch.setattr(runtime, "_robot_articulation_contact_stats",
                                lambda m, d, r, **_kwargs: optimized(m, d, r))
        trajectory = []
        def lock():
            trajectory.append(np.concatenate([data.qpos.copy(), data.qvel.copy()]))
            mujoco.mj_forward(model, data)
        result = runtime.drive_joint_group_to_targets(model, data, {"hinge": .4},
                    config=runtime.ForceDriveConfig(max_physics_substeps=100), robot_lock_callback=lock)
        assert len(trajectory) == 2 * result["physics_substeps"]
        results.append(result)
        trajectories.append(np.asarray(trajectory))
    assert results[0] == results[1]
    assert np.array_equal(trajectories[0], trajectories[1])
