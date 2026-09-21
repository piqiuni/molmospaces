"""Guards for simulator-only collection, without experiment-branch imports."""
import ast
import gzip
import hashlib
from pathlib import Path

import h5py
import numpy as np

from scripts.InteractiveNav import collect_full_gt
from scripts.InteractiveNav.collection import gt_map
from scripts.InteractiveNav.collection.full_rollout_recorder import H5StepRolloutRecorder
from scripts.InteractiveNav.evaluation.benchmark_io import load_benchmark_episodes
from scripts.InteractiveNav.navigation_posture import ROS_NAVIGATION_ARM_QPOS

ROOT = Path(__file__).resolve().parents[2]


def test_no_experiment_or_ros_imports_in_collection():
    paths = list((ROOT/'scripts/InteractiveNav/collection').glob('*.py'))
    paths += [ROOT/'scripts/InteractiveNav/collect_full_gt.py']
    forbidden = ('explore_molmo_interactions', 'benchmark_runner', 'rospy', 'semantic_decision',
                 'ros_navigation_factory', 'habitat_v2_adapter', 'collection.config')
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ''] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            assert not any(token in name for name in names for token in forbidden), (path, names)
    assert Path(collect_full_gt.__file__).resolve().is_relative_to(ROOT)


def test_native_release_selects_same_ten_cases():
    archive = ROOT/'scripts/InteractiveNav/benchmarks/interactive_nav_v3_procthor10k_val_release_v1_2/mixed.json.gz'
    _, episodes = load_benchmark_episodes(archive)
    assert hashlib.sha256(gzip.decompress(archive.read_bytes())).hexdigest() == '07b0c27c7664fd14c633ec042f8eda31ef1844fd4911fc04cce1c0729187a1a8'
    assert collect_full_gt.select_samples(episodes, 10, 20260921) == [765, 809, 279, 168, 696, 468, 981, 328, 718, 883]


def test_posture_matches_existing_sim_protocol_without_importing_runner():
    source = ROOT/'scripts/InteractiveNav/evaluation/benchmark_runner.py'
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name) and n.target.id == 'ROS_NAVIGATION_ARM_QPOS')
    assert ast.literal_eval(node.value) == ROS_NAVIGATION_ARM_QPOS


def test_extracted_geometry_preserves_wall_slice_and_rasterization():
    triangle = np.array([[0, 0, 0], [2, 0, 1], [0, 2, 1]])
    segment = gt_map._triangle_horizontal_slice_segment(triangle, .5)
    np.testing.assert_allclose(sorted(segment.tolist()), [[0, 1], [1, 0]])
    mask = gt_map.rasterize_world_xy_segments(np.array([[[1, 2], [3, 2]]]),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]]), (5, 5))
    assert mask[1, 2] and mask[2, 2] and mask[3, 2]


def test_standalone_recorder_alignment(tmp_path):
    path = tmp_path/'trajectory.h5'
    recorder = H5StepRolloutRecorder(path, episode_id='test', camera_names=['head', 'external'])
    for i in range(3):
        images = {k: np.full((4, 6, 3), i, np.uint8) for k in ['head', 'external']}
        recorder.record_step(images=images, action={'type':'navigate'}, state={'qpos':[i], 'qvel':[1]},
            segment='0', phase='navigate', timestamp_seconds=i*.2, dt_seconds=.2, terminal=i==2)
    recorder.finalize(success=True, terminal_reason='complete')
    with h5py.File(path) as h:
        assert h.attrs['step_count'] == 3
        np.testing.assert_allclose(h['steps/timestamp_seconds'][:], [0, .2, .4])
        assert h['steps/images/head'][2].mean() == 2
        assert h['steps/terminal'][:].tolist() == [False, False, True]
