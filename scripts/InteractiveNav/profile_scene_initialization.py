"""Profile benchmark scene initialization only; never construct or run a policy."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class Profiler:
    def __init__(self, output):
        self.output = output
        self.starts = {}
        self.totals = {}

    def start(self, name):
        self.starts[name] = time.perf_counter()
        print(f"START {name}", flush=True)

    def end(self, name):
        self.record(name, time.perf_counter() - self.starts.pop(name))

    def record(self, name, seconds):
        self.totals[name] = self.totals.get(name, 0) + seconds
        self.output.write_text(json.dumps(self.totals, indent=2))
        print(f"END {name} {seconds:.6f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settle-steps", type=int, default=None,
                        help="Diagnostic override only; omitted preserves the benchmark default")
    parser.add_argument("--isolate-robot-during-settle", action="store_true",
                        help="Diagnostic causal control; restore collision masks after settling")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["INTERACTIVE_NAV_SCENE_MIRROR"] = str(args.output / "mirror")
    faulthandler.dump_traceback_later(60, repeat=True)
    profiler = Profiler(args.output / "timings.json")
    profiler.start("imports")
    from scripts.InteractiveNav.evaluation import benchmark_runner as br
    profiler.end("imports")
    benchmark = Path("/home/ldl/molmospaces/scripts/InteractiveNav/output/interactive_nav_v3_procthor10k_val_release_v1_2/benchmark/benchmark.json")
    profiler.start("benchmark_read")
    episode = json.loads(benchmark.read_text())[args.episode]
    episode.setdefault("scene_modifications", {}).setdefault("articulation_states", [])
    profiler.end("benchmark_read")
    config = br.BenchmarkEvaluationConfig(benchmark=benchmark, output_dir=args.output,
                                          policy="ros_object_goal_rule", max_steps=2000)
    spec = br.EpisodeSpec.model_validate(episode)
    spec.cameras = [camera for camera in spec.cameras if camera.name == "head_camera"]
    spec.img_resolution = (640, 480)
    for camera in spec.cameras:
        camera.record_depth = True
    br._apply_ros_navigation_arm_posture(spec)
    horizon, _ = br.episode_step_budget(config, episode)
    replay = br._build_replay_config(config, args.output, task_horizon=horizon)
    if args.settle_steps is not None:
        replay.task_sampler_config.sim_settle_timesteps = args.settle_steps
    print(f"settle_steps={replay.task_sampler_config.sim_settle_timesteps}", flush=True)
    import mujoco
    real_step = mujoco.mj_step
    physics = []
    saved_masks = []
    def measured_step(model, data, *a, **kw):
        if not physics and args.isolate_robot_during_settle:
            for g in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
                if name.startswith("robot_"):
                    saved_masks.append((g, int(model.geom_contype[g]), int(model.geom_conaffinity[g])))
                    model.geom_contype[g] = 0
                    model.geom_conaffinity[g] = 0
        before = time.perf_counter()
        real_step(model, data, *a, **kw)
        item = {"step": len(physics), "seconds": time.perf_counter() - before,
                "ncon": int(data.ncon), "nefc": int(data.nefc),
                "solver_niter": data.solver_niter.tolist(), "sim_time": float(data.time)}
        physics.append(item)
        if len(physics) == 1:
            pairs = Counter()
            for contact in data.contact:
                bodies = [int(model.geom_bodyid[g]) for g in contact.geom]
                names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies]
                pairs[str(tuple(sorted(str(n) for n in names)))] += 1
            (args.output / "initial_contacts.json").write_text(json.dumps({
                "nv": int(model.nv), "ngeom": int(model.ngeom),
                "solver": int(model.opt.solver), "iterations": int(model.opt.iterations),
                "timestep": float(model.opt.timestep), "jacobian": int(model.opt.jacobian),
                "pairs": pairs.most_common(25)}, indent=2))
        with (args.output / "physics.jsonl").open("a") as stream:
            stream.write(json.dumps(item) + "\n")
        if len(physics) <= 10 or len(physics) % 50 == 0:
            print("PHYSICS " + json.dumps(item), flush=True)
        if len(physics) == replay.task_sampler_config.sim_settle_timesteps:
            for g, contype, affinity in saved_masks:
                model.geom_contype[g] = contype
                model.geom_conaffinity[g] = affinity
    mujoco.mj_step = measured_step
    profiler.start("sampler_construct")
    sampler = br.V3BenchmarkTaskSampler(replay, spec, episode["interactive_nav"])
    profiler.end("sampler_construct")
    sampler.set_datagen_profiler(profiler)
    import molmo_spaces.tasks.task_sampler as ts
    ts.install_scene_with_objects_and_grasps_from_path = lambda *a, **k: None
    variants = sampler._get_dataset_index_map()[spec.data_split][spec.house_index]
    source = variants["base"]
    profiler.start("prepare_mirror")
    variants["base"] = br.probe.prepare_writable_scene_path(Path(source))
    profiler.end("prepare_mirror")
    try:
        profiler.start("task_sample_total")
        task = sampler.sample_task(house_index=spec.house_index)
        profiler.end("task_sample_total")
        profiler.start("task_reset")
        task.reset()
        profiler.end("task_reset")
        print(f"POST_RESET ncon={task.env.current_data.ncon} nefc={task.env.current_data.nefc}", flush=True)
        import numpy as np
        from scipy.spatial.transform import Rotation
        from molmo_spaces.env.data_views import create_mlspaces_body
        from molmo_spaces.utils.pose import pos_quat_to_pose_mat
        view = task.env.current_robot.robot_view
        group_errors = {name: float(np.max(np.abs(view.get_move_group(name).joint_pos - np.asarray(qpos))))
                        for name, qpos in spec.robot.init_qpos.items() if name != "base"}
        base = spec.task["robot_base_pose"]
        base_error = float(np.max(np.abs(view.base.pose - pos_quat_to_pose_mat(base[:3], base[3:7]))))
        object_errors = {}
        for name, pose in spec.scene_modifications.object_poses.items():
            body = create_mlspaces_body(task.env.current_data, name)
            object_errors[name] = {
                "position_m": float(np.max(np.abs(body.position - np.asarray(pose[:3])))),
                "rotation_rad": float((Rotation.from_quat(body.quat).inv() * Rotation.from_quat(pose[3:7])).magnitude())}
        audit = {"robot_joint_errors": group_errors, "base_pose_matrix_error": base_error,
                 "objects": object_errors,
                 "collision_masks_isolated": args.isolate_robot_during_settle}
        audit["passed"] = (base_error < 1e-5 and max(group_errors.values(), default=0) < 1e-5
                           and all(v["position_m"] <= 1e-3 and v["rotation_rad"] <= 1e-2
                                   for v in object_errors.values()))
        (args.output / "state_audit.json").write_text(json.dumps(audit, indent=2))
        print(f"STATE_AUDIT passed={audit['passed']} objects={len(object_errors)}", flush=True)
        if not audit["passed"]:
            raise RuntimeError("Initial state differs from benchmark; inspect state_audit.json")
        print("INITIALIZATION COMPLETE: no policy, ROS or algorithm steps executed", flush=True)
    finally:
        variants["base"] = source
        if sampler._env is not None:
            sampler._env.close()
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
