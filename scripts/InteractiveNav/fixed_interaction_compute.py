"""Opt-in fixed-test camera batching; scoped to one synchronous force drive."""


class BatchForceCameras:
    def __init__(self, runtime, env):
        self.runtime, self.env = runtime, env
        self.original_drive = runtime.drive_joint_group_to_targets
        self.registry = env.camera_manager.registry

    def __enter__(self):
        self.runtime.drive_joint_group_to_targets = self.drive
        return self

    def drive(self, *args, **kwargs):
        original_update = self.registry.update_all_cameras
        dirty = False

        def defer(env):
            nonlocal dirty
            if env is not self.env:
                return original_update(env)
            dirty = True
            return []

        self.registry.update_all_cameras = defer
        try:
            return self.original_drive(*args, **kwargs)
        finally:
            self.registry.update_all_cameras = original_update
            if dirty:
                original_update(self.env)

    def __exit__(self, *exc):
        self.runtime.drive_joint_group_to_targets = self.original_drive
