from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.env.env import BaseMujocoEnv
from molmo_spaces.tasks.task import BaseMujocoTask

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig

log = logging.getLogger(__name__)


class NavToObjTask(BaseMujocoTask):
    """Navigation to object task implementation."""

    def __init__(self, env: BaseMujocoEnv, exp_config: MlSpacesExpConfig) -> None:
        super().__init__(env, exp_config)
        self.exp_config = exp_config

        # For eval mode: reconstruct candidate list from category if needed
        self._reconstruct_candidate_list_if_needed(env)

        self.nav_objs = self._get_nav_objects()

        # BaseMujocoTask queries reward/terminal/info/success together for one
        # MuJoCo state. Keep the expensive segmentation-based visibility result
        # only for that dynamic scope, so external calls always observe live state.
        self._visibility_reward_cache_active = False
        self._visibility_cache: dict[int, bool] = {}
        self._reward_cache_for_current_state: np.ndarray | None = None
        # A segmentation frame produced while evaluating this task state can be
        # consumed by the simulator-local realtime-GT publisher.  Keep it
        # private to the task/publisher boundary: it is never put in an
        # observation, ROS payload, or recorder manifest.  The state token is
        # checked before every reuse so an external qpos edit cannot make GT
        # consume an old image.
        self._realtime_gt_segmentation_snapshot: np.ndarray | None = None
        self._realtime_gt_segmentation_snapshot_token: dict[str, Any] | None = None
        # The bridge requests this only when its next policy frame will publish
        # realtime-GT.  Leaving it false preserves the normal visibility path
        # and avoids copying a segmentation frame on non-GT steps.
        self._realtime_gt_snapshot_requested = False

    def _invalidate_visibility_reward_cache(self) -> None:
        """Discard metrics tied to the current simulated state."""
        self._visibility_cache = {}
        self._reward_cache_for_current_state = None

    def invalidate_private_realtime_gt_segmentation_snapshot(self) -> None:
        """Invalidate the simulator-local GT segmentation snapshot.

        Interaction helpers that edit MuJoCo state outside ``task.step`` may
        call this hook explicitly.  Normal task stepping/resetting also calls
        it, while the state-token check below provides a defensive fallback.
        """
        self._realtime_gt_segmentation_snapshot = None
        self._realtime_gt_segmentation_snapshot_token = None

    def request_private_realtime_gt_segmentation_snapshot(self, requested: bool) -> None:
        """Enable same-state snapshot capture for the next metric transaction.

        This is a simulator-local scheduling hint from ``RosBridgePolicy``.
        It is deliberately opt-in: only steps that will immediately publish a
        realtime-GT frame pay the copy/state-token cost.
        """
        self._realtime_gt_snapshot_requested = bool(requested)
        if not requested:
            self.invalidate_private_realtime_gt_segmentation_snapshot()

    def _segmentation_snapshot_state_token(
        self, camera_name: str, batch_index: int
    ) -> dict[str, Any] | None:
        """Capture the minimal state needed to prove a segmentation is fresh."""
        env = self._env
        try:
            model = env.current_model
            current_batch_index = int(getattr(env, "current_batch_index", batch_index))
            if current_batch_index != int(batch_index):
                return None
            data = env.current_data
            camera = env.camera_manager.registry[camera_name]
            qpos = np.asarray(data.qpos, dtype=np.float64).copy()
            data_time = getattr(data, "time", None)
            data_time = None if data_time is None else float(data_time)
            return {
                "model_id": id(model),
                "data_id": id(data),
                "batch_index": int(batch_index),
                "camera_name": str(camera_name),
                "data_time": data_time,
                "qpos": qpos,
                "camera_pos": np.asarray(camera.pos, dtype=np.float64).copy(),
                "camera_forward": np.asarray(camera.forward, dtype=np.float64).copy(),
                "camera_up": np.asarray(camera.up, dtype=np.float64).copy(),
                "camera_fov": float(getattr(camera, "fov", 0.0)),
            }
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _segmentation_snapshot_tokens_equal(
        first: dict[str, Any] | None, second: dict[str, Any] | None
    ) -> bool:
        if first is None or second is None:
            return False
        if (
            first["model_id"] != second.get("model_id")
            or first["data_id"] != second.get("data_id")
            or first["batch_index"] != second.get("batch_index")
            or first["camera_name"] != second.get("camera_name")
            or first["data_time"] != second.get("data_time")
            or first["camera_fov"] != second.get("camera_fov")
        ):
            return False
        return all(
            np.array_equal(first[key], second[key])
            for key in ("qpos", "camera_pos", "camera_forward", "camera_up")
        )

    def _segmentation_snapshot_matches_current_state(
        self, token: dict[str, Any] | None
    ) -> bool:
        if token is None:
            return False
        current = self._segmentation_snapshot_state_token(
            str(token.get("camera_name", "")), int(token.get("batch_index", 0))
        )
        return self._segmentation_snapshot_tokens_equal(token, current)

    def _capture_private_realtime_gt_segmentation(
        self, camera_name: str = "head_camera", batch_index: int | None = None
    ) -> np.ndarray | None:
        """Render and retain one private segmentation frame for this state."""
        env = self._env
        if batch_index is None:
            batch_index = int(getattr(env, "current_batch_index", 0))
        token_before = self._segmentation_snapshot_state_token(camera_name, int(batch_index))
        if token_before is None or not hasattr(env, "render_segmentation_frame"):
            return None
        try:
            frame = np.asarray(env.render_segmentation_frame(camera_name))
            if frame.ndim < 3 or frame.shape[-1] < 2:
                return None
            # Renderer buffers are implementation-defined; own a read-only
            # copy so a later render cannot mutate the GT input asynchronously.
            snapshot = np.ascontiguousarray(frame[..., :2]).copy()
            snapshot.setflags(write=False)
        except Exception:
            return None
        token_after = self._segmentation_snapshot_state_token(camera_name, int(batch_index))
        if not self._segmentation_snapshot_tokens_equal(token_before, token_after):
            return None
        self._realtime_gt_segmentation_snapshot = snapshot
        self._realtime_gt_segmentation_snapshot_token = token_after
        return snapshot

    def get_private_realtime_gt_segmentation_snapshot(
        self, camera_name: str = "head_camera"
    ) -> np.ndarray | None:
        """Return a fresh same-state frame for the realtime-GT publisher only.

        This method intentionally exposes no snapshot metadata and the caller
        must not serialize the returned array.  A state mismatch invalidates
        the frame and returns ``None`` so the publisher falls back to a fresh
        render.
        """
        if camera_name != "head_camera":
            return None
        snapshot = getattr(self, "_realtime_gt_segmentation_snapshot", None)
        token = getattr(self, "_realtime_gt_segmentation_snapshot_token", None)
        if snapshot is None or not self._segmentation_snapshot_matches_current_state(token):
            self.invalidate_private_realtime_gt_segmentation_snapshot()
            return None
        return snapshot

    def get_and_cache_all_step_information(self):
        """Cache navigation metrics only while BaseMujocoTask reads one state."""
        # A new metric snapshot follows the current state.  Keep the new
        # segmentation frame alive after this method returns so policy GT can
        # consume it before the next physics step.
        self.invalidate_private_realtime_gt_segmentation_snapshot()
        self._invalidate_visibility_reward_cache()
        self._visibility_reward_cache_active = True
        try:
            return super().get_and_cache_all_step_information()
        finally:
            self._visibility_reward_cache_active = False
            self._invalidate_visibility_reward_cache()

    def reset(self):
        self._realtime_gt_snapshot_requested = False
        self.invalidate_private_realtime_gt_segmentation_snapshot()
        return super().reset()

    def step(self, action):
        # Invalidate before any first-step observation checks or physics work;
        # the post-physics metric pass will create the next valid snapshot.
        self.invalidate_private_realtime_gt_segmentation_snapshot()
        return super().step(action)

    def _selection_mode(self) -> str:
        return getattr(self.config.task_config, "selection_mode", "any_candidate")

    def _reconstruct_candidate_list_if_needed(self, env: BaseMujocoEnv) -> None:
        """Reconstruct candidate list from category for eval mode.

        This is needed because saved benchmarks may contain pickup_obj_candidates
        that include objects of multiple categories. We need to filter them to only
        include objects of the target category.
        """
        # Check if we have a pickup_obj_name to work with
        if (
            not hasattr(self.config.task_config, "pickup_obj_name")
            or self.config.task_config.pickup_obj_name is None
        ):
            return

        if self._selection_mode() == "specific_instance":
            self.config.task_config.pickup_obj_candidates = [
                self.config.task_config.pickup_obj_name
            ]
            return

        # Check if candidates list exists and needs filtering
        if (
            not hasattr(self.config.task_config, "pickup_obj_candidates")
            or not self.config.task_config.pickup_obj_candidates
        ):
            return

        om = env.object_managers[env.current_batch_index]

        # Try to get category from config, or infer from pickup_obj_name
        target_category = None
        target_synset = None

        if (
            hasattr(self.config.task_config, "pickup_obj_category")
            and self.config.task_config.pickup_obj_category is not None
        ):
            # Category already saved in config (new data)
            target_category = self.config.task_config.pickup_obj_category
            target_synset = getattr(self.config.task_config, "pickup_obj_synset", None)
            log.info(f"[NavTask] Using saved category: {target_category} (synset: {target_synset})")
        else:
            # Infer category from pickup_obj_name (old data)
            try:
                target_category = om.category_from_name(self.config.task_config.pickup_obj_name)
                target_synset = om.get_annotation_synset(self.config.task_config.pickup_obj_name)
                # Save for future reference
                self.config.task_config.pickup_obj_category = target_category
                self.config.task_config.pickup_obj_synset = target_synset
                log.info(
                    f"[NavTask] Inferred category '{target_category}' (synset: {target_synset}) from pickup_obj_name"
                )
            except Exception as e:
                log.warning(f"[NavTask] Could not infer category from pickup_obj_name: {e}")
                return

        # Filter saved candidates to only include same category
        saved_candidates = self.config.task_config.pickup_obj_candidates
        filtered_candidates = []

        for obj_name in saved_candidates:
            try:
                obj_category = om.category_from_name(obj_name)
                obj_synset = om.get_annotation_synset(obj_name)

                # Match by category or synset
                if obj_category == target_category or (
                    target_synset and obj_synset == target_synset
                ):
                    filtered_candidates.append(obj_name)
            except Exception:
                # Skip objects that can't be categorized
                continue

        # Update config with filtered candidates if we found any
        if len(filtered_candidates) > 0:
            log.info(
                f"[NavTask] ✅ Filtered candidates: {len(saved_candidates)} → {len(filtered_candidates)} "
                f"(category: '{target_category}')"
            )
            log.info(f"[NavTask]    First 5: {filtered_candidates[:5]}")
            self.config.task_config.pickup_obj_candidates = filtered_candidates

            # Use first candidate as default pickup_obj_name if original is not in filtered list
            if self.config.task_config.pickup_obj_name not in filtered_candidates:
                old_name = self.config.task_config.pickup_obj_name
                self.config.task_config.pickup_obj_name = filtered_candidates[0]
                log.warning(
                    f"[NavTask] Original object '{old_name}' not in filtered candidates. "
                    f"Using '{filtered_candidates[0]}' as default."
                )
        else:
            log.warning(
                f"[NavTask] ⚠️  No objects in saved candidates match category '{target_category}'. "
                f"Keeping all {len(saved_candidates)} saved candidates."
            )

    def _get_nav_objects(self) -> list[list[MlSpacesObject]]:
        """Get all navigation objects of the target type for each batch.

        Returns:
            List of lists, where nav_objs[batch_idx] contains all candidate objects for that batch.
        """
        nav_objs_per_batch = []

        for i in range(self._env.n_batch):
            data = self._env.mj_datas[i]

            if self._selection_mode() == "specific_instance":
                pickup_obj = MlSpacesObject(
                    data=data, object_name=self.config.task_config.pickup_obj_name
                )
                nav_objs_per_batch.append([pickup_obj])
                continue

            # If pickup_obj_candidates exists, create objects for all candidates
            if (
                hasattr(self.config.task_config, "pickup_obj_candidates")
                and self.config.task_config.pickup_obj_candidates is not None
                and len(self.config.task_config.pickup_obj_candidates) > 0
            ):
                objs = []
                for obj_name in self.config.task_config.pickup_obj_candidates:
                    try:
                        obj = MlSpacesObject(data=data, object_name=obj_name)
                        objs.append(obj)
                    except Exception as e:
                        print(f"Warning: Could not create MlSpacesObject for {obj_name}: {e}")

                nav_objs_per_batch.append(objs)
            else:
                # Backward compatibility: single object mode
                pickup_obj = MlSpacesObject(
                    data=data, object_name=self.config.task_config.pickup_obj_name
                )
                nav_objs_per_batch.append([pickup_obj])

        return nav_objs_per_batch

    def get_nav_object_priority(self, batch_index: int) -> list[MlSpacesObject]:
        """Get the nearest navigation object for the given batch.

        Args:
            batch_index: Index of the environment batch

        Returns:
            The MlSpacesObject instance that is nearest to the robot
        """
        robot_base_pos = self._env.robots[batch_index].robot_view.base.pose[:3, 3]

        if len(self.nav_objs[batch_index]) == 1:
            return self.nav_objs[batch_index][:]

        priority = [
            (np.linalg.norm(obj.position[:2] - robot_base_pos[:2]), obj)
            for obj in self.nav_objs[batch_index]
        ]

        return [dist_obj[1] for dist_obj in sorted(priority, key=lambda x: x[0])]

    def get_nearest_nav_object(self, batch_index: int) -> MlSpacesObject:
        """Get the nearest navigation object for the given batch.

        Args:
            batch_index: Index of the environment batch

        Returns:
            The MlSpacesObject instance that is nearest to the robot
        """
        priority = self.get_nav_object_priority(batch_index)
        return priority[0] if priority else None

    def get_task_description(self) -> str:
        """Get the task description for this navigation task."""
        pickup_obj_name = self.config.task_config.pickup_obj_name

        om = self.env.object_managers[self.env.current_batch_index]

        # Get natural name if available
        try:
            object_name = om.fallback_expression(pickup_obj_name)
        except Exception:
            # Fallback to raw name if natural name lookup fails
            object_name = pickup_obj_name.replace("_", " ").title()

        # Include candidate count if multiple objects
        num_candidates = (
            len(self.config.task_config.pickup_obj_candidates)
            if hasattr(self.config.task_config, "pickup_obj_candidates")
            and self.config.task_config.pickup_obj_candidates is not None
            else 1
        )

        if num_candidates > 1:
            return f"Navigate to any {object_name} ({num_candidates} available)"
        else:
            return f"Navigate to the {object_name}"

    def _create_sensor_suite_from_config(self, exp_config: MlSpacesExpConfig) -> SensorSuite:
        """Create a sensor suite from configuration using the centralized get_nav_task_sensors function."""
        from molmo_spaces.env.sensors import get_nav_task_sensors

        sensors = get_nav_task_sensors(exp_config)
        return SensorSuite(sensors)

    def calculate_distance(self, index: int) -> float:
        """Calculate the distance to the NEAREST navigation object of the target type.

        Args:
            index: Index of the environment batch

        Returns:
            Distance in meters to the nearest object of the target type
        """
        robot = self._env.robots[index]
        robot_base_pose = robot.robot_view.base.pose
        robot_base_pos = robot_base_pose[:3, 3]

        # Get the nearest object dynamically
        nearest_obj = self.get_nearest_nav_object(index)

        return np.linalg.norm(nearest_obj.position[:2] - robot_base_pos[:2])

    def check_object_visible(self, index: int) -> bool:
        """Check if the nearest navigation object is visible from head camera."""
        if self._visibility_reward_cache_active:
            cached_visibility = self._visibility_cache.get(index)
            if cached_visibility is not None:
                return cached_visibility

        nearest_obj = self.get_nearest_nav_object(index)

        # Use 'head_camera' (registry name), not 'robot_0/head_camera' (MJCF
        # name).  During the task metric transaction, retain the exact frame
        # used for this visibility query so realtime GT can reuse it.  For
        # unsupported/batched fake environments, preserve the generic env API.
        segmentation = None
        if (
            self._visibility_reward_cache_active
            and bool(getattr(self, "_realtime_gt_snapshot_requested", False))
            and int(
                getattr(self._env, "current_batch_index", index)
            ) == int(index)
        ):
            segmentation = self._capture_private_realtime_gt_segmentation(
                "head_camera", batch_index=index
            )
        if segmentation is not None and hasattr(self._env, "segmentation_fraction"):
            try:
                visibility = self._env.segmentation_fraction(segmentation, nearest_obj.name)
            except Exception:
                visibility = 0.0
        else:
            visibility = self._env.check_visibility("head_camera", nearest_obj.name)
        object_visible = visibility > 0.0  # Any non-zero visibility fraction
        if self._visibility_reward_cache_active:
            self._visibility_cache[index] = object_visible
        return object_visible

    def get_reward(self) -> np.ndarray:
        """Calculate reward based on distance to target object.

        Returns:
            Array of rewards for each environment in the batch
        """
        if self._visibility_reward_cache_active and self._reward_cache_for_current_state is not None:
            # Preserve the previous API behavior: callers receive a fresh array.
            return self._reward_cache_for_current_state.copy()

        rewards = []

        for i in range(self._env.n_batch):
            # Calculate distance-based reward (negative distance)
            distance = self.calculate_distance(i)

            # Success: robot is close enough AND object is visible (visual navigation)
            if not self.check_object_visible(i):
                reward = 0.0
            else:
                # Linearly scale reward from 1 → 0 as distance goes from 0 → threshold
                reward = max(0.0, 1.0 - distance / self.config.task_config.succ_pos_threshold)
                # TODO this can be detrimental for RL training, as the reward goes up for locations
                #  with potentially vanishing visibility. We might want to make it maximum for
                #  distances under some smaller threshold than `succ_pos_threshold`

            rewards.append(reward)

        reward_array = np.array(rewards, dtype=np.float32)
        if self._visibility_reward_cache_active:
            self._reward_cache_for_current_state = reward_array
            return reward_array.copy()
        return reward_array

    def judge_success(self) -> bool:
        """Judge whether the task is successfully completed.

        Returns:
            Boolean indicating success for the first environment
        """
        success = self.get_reward()[0] > 0.0
        if not success:
            distance = self.calculate_distance(0)
            object_visible = self.check_object_visible(0)
            log.info(f"[Nav fail] Distance: {distance:.2f}m, Object visible: {object_visible}")

        return success

    def get_info(self) -> list[dict[str, Any]]:
        """Get additional metrics for each environment."""
        metrics = []

        # Calculate rewards once for all environments
        rewards = self.get_reward()

        for i in range(self._env.n_batch):
            distance = self.calculate_distance(i)
            # Use pre-calculated reward
            success = rewards[i] > 0.0

            metrics.append(
                {
                    "position_error": distance,
                    "success": success,
                    "episode_step": self.episode_step_count,
                }
            )

        return metrics

    def get_obs_scene(self):
        """
        This is for observations that are constant over all time steps of an env.
        """
        # Get base observation from parent class (includes frozen_config handling)
        obs_scene = super().get_obs_scene()

        text = self.config.task_type + " " + self.config.task_config.pickup_obj_name
        obs_extra = dict(text=text, object_name=self.config.task_config.pickup_obj_name)
        obs_scene.update(obs_extra)

        return obs_scene
