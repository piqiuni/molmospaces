# Go2 physical interactive-navigation platform

This is an additional, read-only platform for the D435i-equipped Go2. It does
not modify the simulator or the existing interactive-navigation launch files.
The Go2 runs only `go2_readonly_sensor_bridge.py`; ROS, YOLOE, 3-D lifting,
semantic mapping, graph construction, consistency checks and the web UI run on
the local policy machine.

## Data path

```text
D435i + Unitree state subscribers (Go2)
  -> JSON WebSocket (RGB JPEG, depth PNG16, intrinsics, timestamps, pose)
  -> physical_sensor_ros_bridge.py (direct latest-only ROS sensor publisher)
       -> physical_yoloe_bridge.py (algorithm Python, ROS RGB-D input)
       YOLOE-26l PF Seg -> RGB-D 3-D boxes/pointcloud -> ROS detections
  -> semantic_mapping_py_pkg/semantic_mapping_node.py
       object map/room segmentation/interaction graph
  -> physical_consistency_node.py -> /physical_nav/consistency

Optional observer path (does not carry algorithm input):
  physical_sensor_ros_bridge.py -> POST /api/raw-frame (latest-only mirror)
  physical_six_panel_server.py -> LAN dashboard on :8765
  physical_ros_gateway.py -> ROS state/debug mirror and overlays
```

The Go2 client has no motion publisher or control client. Keyboard/web commands
are handled by `ReadOnlySafetyGate`, recorded as `READ_ONLY_BLOCKED`, and never
sent to the robot.

For the physical M1 path, raw YOLO observations first pass through the 3-D
`ObjectMapStore`. M1 consumes `/physical_nav/tracked_detections`, so repeated
views share the map-owned identity (`track_0042`) instead of a frame-local or
position-quantized detector ID. The dashboard stores the exact source RGB for
each completed call and redraws only that track's box as `door #42`. Physical
M1 admission requires at least 1024 segmented pixels, a 1600 px bounding box,
and the configured visible-fraction/distance checks.

The room-segmentation grid uses the complete occupancy-grid geometry. Colored
cells are discovered free space assigned to a stable room; occupied and
unknown/unexplored cells stay unassigned (`-1`). They are shown for spatial
context but are not fabricated as exterior rooms.

The physical local costmap retains obstacle observations (and therefore their
inflation) for 2 seconds. Its source keeps `marking` and `clearing` enabled.
The mapper's existing local-overwrite path is also enabled for physical runs:
observed hit-free/no-return rays clear OCC up to 7.9 m, while directions with
no camera evidence remain unchanged. Free/occupied overwrite evidence expires
after 2 seconds unless observed again.

## Physical interaction policy and armed velocity path

The physical branch contains a composable interaction policy in
`interaction_policy.py`. The default `physical_human` profile speaks a
target-specific request through `uni_control`, calls Qwen as M3, and publishes
success only after the same target is visually open for three seconds.
`simulation_force` and `physical_vla` reuse the same request/result contract
with different actuator and verifier adapters. Select the profile with the
physical launch parameter; simulator launch files are not changed.

The semantic executor uses `external_mllm_verified` for this profile. The
policy owns the postcondition VLM call and the executor consumes its terminal
`/physical_nav/interaction_result` without issuing a duplicate M3 request.
Intermediate M3 records are published on `/physical_nav/mllm_events` and shown
by the web dashboard.

The final physical velocity filter is `physical_velocity_safety.py`. Its
default profile uses scale `1.0`, limits commands to `0.50 m/s` and
`1.20 rad/s`, applies acceleration slew limits, and zeros stale input. The
read-only launch publishes the filtered result on
`/physical_nav/actuated_cmd_vel` but deliberately does not connect it to Go2.
For a separately armed deployment, run the policy-side control server with:

```bash
python3 scripts/InteractiveNav/uni_control/policy_control_server.py \
  --control-mode continuous --source ros \
  --cmd-vel-topic /physical_nav/actuated_cmd_vel \
  --speech-request-topic /physical_nav/speech_request
```

The Go2 launcher must separately be started with `--enable-motion`; the
read-only sensor bridge never gains motion capability.

Motion initialization retries the obstacle-avoidance switch/API lease up to
five times by default. The one-command wrapper waits for a bridge-owned ready
PID file and fails startup if the remote bridge exits or is not fully
initialized within 30 seconds; a launcher PID alone is not treated as ready.

## Running profiles

All commands below assume the policy host is `/home/user/ldl/molmospaces`.
Use the detached service controller for normal runs. It keeps the stack alive
after SSH/Codex terminals close and supervises the gateway, YOLOE, ROS launch
and the input-progress watchdog as one unit:

```bash
cd /home/user/ldl/molmospaces
PHYSICAL_NAV_START_QWEN_TUNNEL=1 \
bash scripts/InteractiveNav/physical_nav/physical_nav_service.sh start

bash scripts/InteractiveNav/physical_nav/physical_nav_service.sh status
bash scripts/InteractiveNav/physical_nav/physical_nav_service.sh logs
```

`stop` also performs a PID-file-independent residual cleanup for this physical
pipeline. It removes stale gateway, detector, watchdog, OCC/SLAM, navigation,
semantic-decision and interaction processes left behind by an abnormal launch
exit. The shared ROS master/rosout and an independently launched Foxglove
bridge are deliberately preserved.

### One-command startup (policy host)

To start the complete read-only chain from the policy machine, use
`physical_nav_all.sh`. It SSHes to the Go2, starts its reverse WebSocket tunnel
and D435i/state bridge, opens the Qwen tunnel, and then starts the local
gateway, YOLOE and ROS nodes. No motion/control API is started by default:

```bash
cd /home/user/ldl/molmospaces
bash scripts/InteractiveNav/physical_nav/physical_nav_all.sh start

# dashboard and component status
bash scripts/InteractiveNav/physical_nav/physical_nav_all.sh status

# stop only processes started by the wrapper
bash scripts/InteractiveNav/physical_nav/physical_nav_all.sh stop
```

To run the full ROS perception, mapping and navigation chain without starting
the dashboard, use:

```bash
PHYSICAL_NAV_START_WEB=0 bash scripts/InteractiveNav/physical_nav/physical_nav_all.sh start
```

In this mode Go2 still sends frames to the direct sensor socket on port
`12335`; only the optional raw-frame and ROS-state mirrors are disabled.

To start the ROS velocity control path from the same wrapper, pass the
explicit second argument `enable_motion`:

```bash
bash scripts/InteractiveNav/physical_nav/physical_nav_all.sh start enable_motion
```

This additionally starts `policy_control_server.py` on port `12333` and
launches `/home/unitree/uni_control/start_go2_control.py --enable-motion
--no-restart` on the Go2. The plain `start` command remains read-only. The
wrapper stops the policy server before the Go2 bridge, allowing disconnect
cleanup to send zero velocity and release the API control lease.

To inject an object goal into the semantic navigation algorithm at startup,
append the target object label after `enable_motion`:

```bash
bash scripts/InteractiveNav/physical_nav/physical_nav_all.sh restart enable_motion door
```

The wrapper publishes a JSON target context once to `/semantic_decision/target`.
The candidate/decision nodes then match the detected object, run M1/M2, and
generate the physical navigation subgoal. Without an object label, no target
is injected.

The default SSH aliases are `unitree` (policy host -> Go2) and `zgca_gpu`
(Go2 -> policy host). Override them with `PHYSICAL_NAV_GO2_SSH_TARGET` and
`PHYSICAL_NAV_GO2_TUNNEL_TARGET` when the local SSH config uses different
names. Logs are kept in `/tmp/molmospaces-physical-nav-all-$UID/logs` on the
policy host and `/home/unitree/physical_nav/go2_sensor_tunnel.log` plus
`go2_readonly_sensor_bridge.log` on the Go2.

`start_physical_nav.sh` remains the foreground/debug entry point. Do not use
that foreground form for a long-lived run from a disposable terminal.

Start the read-only sensor client on Go2 in another terminal:

```bash
python3 /home/unitree/uni_control/start_go2_readonly_sensor.py \
  --policy-host <policy-host-ip>
```

### Autonomous navigation with human-assisted interaction

On the policy host, run the control server (one process only on port 12333):

```bash
python3 scripts/InteractiveNav/uni_control/policy_control_server.py \
  --control-mode continuous --source ros \
  --cmd-vel-topic /physical_nav/actuated_cmd_vel \
  --speech-request-topic /physical_nav/speech_request
```

On Go2, first use the telemetry-only bridge for a dry check:

```bash
python3 /home/unitree/uni_control/start_go2_control.py --no-restart
```

After checking `hello`, telemetry, STOP and the zero-velocity path, arm motion
explicitly:

```bash
python3 /home/unitree/uni_control/start_go2_control.py \
  --enable-motion --no-restart
```

The default physical launch uses M1=`dynamic_mllm`, M2=`mllm_score`,
M3=`external_mllm_verified`, and policy profile `physical_human`.

### Pure navigation/exploration (skip all interaction)

Stop the current service and restart it with the no-interaction override:

```bash
bash scripts/InteractiveNav/physical_nav/physical_nav_service.sh stop
PHYSICAL_NAV_SEMANTIC_OVERRIDE_CONFIG=\
scripts/InteractiveNav/physical_nav/config/semantic_no_interaction_override.yaml \
PHYSICAL_NAV_ENABLE_M1=false \
PHYSICAL_NAV_ENABLE_INTERACTION_POLICY=false \
bash scripts/InteractiveNav/physical_nav/physical_nav_service.sh start
```

The YOLOE, OCC, costmaps, move_base and exploration nodes remain active, but
door/fridge/drawer interaction candidates and MLLM interaction calls are
disabled. Keep the continuous control-server and Go2 bridge commands above.

### Human takeover / keyboard teleoperation

Stop the continuous `policy_control_server` first, but leave the Go2 launcher
running. Start a discrete keyboard server on the policy host:

```bash
python3 scripts/InteractiveNav/uni_control/policy_control_server.py \
  --control-mode discrete --source keyboard \
  --wait-discrete-completion
```

Keys are `w/s` forward/backward, `a/d` lateral, `q/e` turn, and `x` immediate
STOP. The dashboard at `http://<policy-lan-ip>:8765/` also exposes the
physical-nav shell controls: `Start`, `Restart + Goal` (the adjacent input is
passed as `obj_goal`), and `Stop`. `Esc` or `S` triggers the same Stop action;
the control endpoint only accepts the fixed `start`, `restart`, and `stop`
actions. Web `stop` terminates only the local policy-control transport (the
Go2 bridge then zeros its command); it does not stop ROS, the navigation stack,
or the web gateway. The `开启运动控制` button runs
`physical_nav_all.sh start_control enable_motion` to restore only the control
transport. `start`/`restart` retain the original shell behavior.
When switching back to autonomy, stop this server and restart the continuous
server. The Go2 bridge reconnects to the new server and zeros speed during the
transition.

### Manual speech test without interaction execution

With the continuous control server running, publish a speech request from the
policy host:

```bash
rostopic pub -1 /physical_nav/speech_request std_msgs/String \
  "{data: '{\"text\":\"请帮我打开前面的门\",\"wait\":true,\"volume\":4}'}"
```

### Complete stop

Press `x` in the keyboard server or stop the ROS source server first, then stop
the policy stack and the Go2 launcher:

```bash
bash scripts/InteractiveNav/physical_nav/physical_nav_service.sh stop
```

The Go2 launcher sends zero velocity and releases the Go2 API control lease
before exiting.

The live six-panel image reuses the established offline interactive-navigation
renderer: `build_semantic_video_offline.py`'s canonical panel assembly and
`offline_semantic_renderer.py`'s `OfflineSixPanelRenderer`. The physical adapter
feeds it recorder-shaped `step` and `RawGrid` values. The panel order is
`camera / OCC / room` on the first row and `global+local costmaps / semantic XY
/ topology` on the second row. Real hardware has no simulator GT stream, so no
GT overlay is fabricated.

The current measured standing-pose camera extrinsic is used by default:
`base -> camera = (x=+0.03 m, y=0 m, z=+0.62 m)` with zero roll/pitch/yaw.
This interprets the 38 cm camera location as 3 cm forward of the 70 cm body
centre; with the Go2 standing-pose base height of about 0.43 m, this gives a
1.05 m camera-to-ground height.
approximately 0.43 m base-to-ground offset. Override the six
`PHYSICAL_NAV_CAMERA_*` variables if the mounting reference is different.

The extrinsic is only the rigid rod offset; live orientation is no longer
forced to zero. The Go2 read-only state subscriber forwards the Unitree IMU
quaternion, and the ROS bridge publishes that full quaternion (roll/pitch/yaw)
on `/physical_nav/odom` and `tf_frame_odom -> tf_frame_base_link`. YOLOE RGB-D
lifting and consistency projection use the same dynamic transform. The D435i
source also forwards gyro/accelerometer samples as `telemetry.camera_imu` when
the librealsense driver exposes them. D435i itself has no absolute pose/VO
stream, so an explicit `camera_pose`/`d435i_pose` quaternion, if supplied by a
tracking wrapper, takes precedence over the body IMU automatically.

Physical YOLOE detections are filtered before mask point-cloud lifting and 3D
box construction using `object_detection.detection_filter` in
`config/physical_nav.yaml`. Set `PHYSICAL_NAV_DETECTOR_CONFIG` to use another
full semantic-mapping YAML or a filter-only YAML without changing code.

## Low-level/manual start on the policy machine

Install the side-specific transport dependencies first. The Go2 image uses
Python 3.8, while the policy machine uses Python 3.10+:

```bash
python3 -m pip install -r /home/user/ldl/molmospaces/scripts/InteractiveNav/physical_nav/requirements_go2.txt  # on Go2
python3 -m pip install -r /home/user/ldl/molmospaces/scripts/InteractiveNav/physical_nav/requirements_policy.txt  # on policy machine
```

`unitree_sdk2py`, `librealsense/pyrealsense2`, ROS Noetic and Ultralytics are
not vendored. They must already be provided by the corresponding platform
environment; YOLOE is loaded only on the policy machine.

The following commands are for component-level debugging. For a complete run,
prefer `physical_nav_service.sh start` above. The dashboard is an optional
observer and binds to all interfaces so a LAN browser can connect:

```bash
cd /home/user/ldl/molmospaces
# In zsh use the zsh setup files; in bash use the corresponding setup.bash files.
source /opt/ros/noetic/setup.zsh
source /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/devel/setup.zsh
export PYTHONPATH="$PWD/scripts/InteractiveNav/physical_nav/ros_compat:$PWD/scripts/InteractiveNav/physical_nav:$PWD/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts:$PYTHONPATH"
python3 scripts/InteractiveNav/physical_nav/physical_six_panel_server.py \
  --ws-host 0.0.0.0 --ws-port 12334 --http-host 0.0.0.0 --http-port 8765 \
  --qwen-url http://127.0.0.1:18080/v1
```

`physical_sensor_ros_bridge.py` is started by
`physical_nav_readonly.launch`; it accepts Go2 frames directly on port `12335`
and publishes RGB, depth, camera info, odometry, TF and the point cloud to ROS.
YOLO, mapping and navigation therefore continue when the dashboard is stopped
or never started. `physical_ros_gateway.py` consumes detector output from
`/physical_nav/yolo_report`, publishes the mapped detections, and only mirrors
graph, consistency and occupancy snapshots to `/api/ros-state` when the
dashboard is enabled.
Before republishing detections, it resolves each RGB-D 3-D center from the
D435i frame into `tf_frame_map` through the live GMapping TF. If that TF is not
available during startup, the detector's Unitree-odometry estimate is retained
with `map_transform_status: telemetry_fallback` until TF becomes available.

`start_physical_nav.sh` also starts `physical_yoloe_bridge.py` in the local
algorithm Python environment. Set `PHYSICAL_NAV_ALGORITHM_PYTHON` to the
environment containing Ultralytics; set `PHYSICAL_NAV_START_YOLO_WORKER=0`
only when using a separately managed detector.

In another local terminal, after sourcing the catkin workspace, start the
physical mapping nodes:

```bash
source /opt/ros/noetic/setup.zsh
source /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/devel/setup.zsh
export PYTHONPATH="/home/user/ldl/molmospaces/scripts/InteractiveNav/physical_nav/ros_compat:/home/user/ldl/molmospaces/scripts/InteractiveNav/physical_nav:$PYTHONPATH"
roslaunch physical_nav physical_nav_readonly.launch \
  model_path:=/home/user/ldl/molmospaces/detection_models/yoloe/weights/yoloe-26l-seg-pf.pt
```

On the Go2:

```bash
python3 /home/user/ldl/molmospaces/scripts/InteractiveNav/physical_nav/start_go2_readonly_sensor.py --policy-host <policy-lan-ip>
```

For a development-only protocol test without a Go2 or camera:

```bash
python3 physical_six_panel_server.py --ws-host 127.0.0.1 --http-host 127.0.0.1 &
python3 go2_readonly_sensor_bridge.py --dry-run --url ws://127.0.0.1:12334
```

Open `http://<policy-lan-ip>:8765/`. `/stream.mjpg` is the six-panel live
image, `/api/state` includes raw detections, `mapped_detections` after TF
alignment, graph, telemetry, consistency and Qwen request/result history, and
`/api/health` is suitable for a smoke test.

The web gateway is persistent and is no longer a critical child of the
navigation supervisor. Therefore a watchdog-triggered navigation stop or a
`physical_nav_service.sh restart` keeps port 8765 and the control bar online;
the page reports the navigation stack as offline until it is started again.
Use `physical_nav_service.sh web-stop` only when the web gateway itself must
be shut down.

The service watchdog polls the constant-size `/api/health` response. After the
startup grace period, a disconnected Go2 link, a stale D435i frame or a stale
YOLOE receipt causes the supervisor to stop the entire local stack instead of
letting ROS and the dashboard consume the last frame indefinitely. A D435i
frame may now remain stale for 15 seconds, followed by five consecutive failed
checks (2 seconds apart), so a brief camera interruption recovers without
terminating the stack while a sustained interruption still stops it. Thresholds
can be changed with `PHYSICAL_NAV_WATCHDOG_*`; set
`PHYSICAL_NAV_WATCHDOG_ENABLED=0` only for component-level debugging. Runtime
state and component logs default to
`/tmp/molmospaces-physical-nav-$UID/`.

Before trusting map-frame 3-D geometry, verify the measured D435i-to-base
extrinsic through `PHYSICAL_NAV_CAMERA_X/Y/Z/ROLL/PITCH/YAW` (metres/radians).
The defaults are the current Go2 standing-pose measurement above; override them
when the camera mount or base-frame convention changes.

## RViz perception inspection

With the physical stack running, open the prepared map-frame view:

```bash
source /opt/ros/noetic/setup.zsh
export ROS_PACKAGE_PATH=/home/user/ldl/molmospaces/Interactive-Nav-SG-nav/src:$ROS_PACKAGE_PATH
rviz -d /home/user/ldl/molmospaces/scripts/InteractiveNav/physical_nav/config/physical_perception.rviz
```

The view overlays the raw D435i cloud (`/physical_nav/points`), category-coloured
YOLOE instance cloud (`/physical_nav/segmented_cloud_world`), stable labelled
3-D box markers (`/physical_nav/boxes_3d_world`), and the exact detector RGB
receipt with YOLO boxes (`/physical_nav/detections_overlay`). The same topics
are available in Foxglove. The world box display uses two-frame confirmation, EMA geometry
smoothing, stable marker IDs, and a three-second missed-detection hold; this
does not modify raw detections or semantic-map tracking. Camera-frame
equivalents are `/physical_nav/segmented_cloud` and
`/physical_nav/boxes_3d`. Point and marker payloads are live-only and are not
written to disk.

## Qwen over SSH

The Qwen service is not run on the Go2. Forward its remote HTTP port to the
policy machine (the remote SSH endpoint is the one supplied for this project):

```bash
python3 /home/user/ldl/molmospaces/scripts/InteractiveNav/physical_nav/qwen_ssh_tunnel.py --ssh-port 41051 --user root \
  --host 115.190.90.101 --local-port 18080 --remote-port 8000
```

The remote vLLM service currently listens on `127.0.0.1:8000`; the remote
service port remains configurable because the SSH port (`41051`) and the Qwen
HTTP port are independent. Its advertised model id is
`qwen3.6-35b-a3b-fp8`. The web page exposes `/api/qwen` and shows all
requests/results and latency in the status area.

`start_physical_nav.sh` can manage this tunnel too:

```bash
PHYSICAL_NAV_START_QWEN_TUNNEL=1 \
PHYSICAL_NAV_QWEN_REMOTE_PORT=8000 \
bash scripts/InteractiveNav/physical_nav/start_physical_nav.sh
```

## Safety and acceptance gates

1. No `SportClient`, `ObstaclesAvoidClient`, motion publisher or manipulation
   API is imported by the Go2 read-only bridge.
2. `actuation_enabled` is always false and keyboard intents receive a 202
   `read_only_blocked` response.
3. D435i RGB/Depth/CameraInfo and Unitree state share sequence/timestamp data.
4. YOLOE output is lifted to map-frame 3-D boxes and fed to the existing global
   semantic graph implementation through `/physical_nav/*` topics.
5. The consistency report projects each matched global 3-D graph box back into
   the current RGB frame using the live D435i `CameraInfo` and Go2 odometry,
   then checks image IoU/pixel offset, camera-Z depth, RGB-D lift, map
   association and temporal/spatial relations; it is displayed in panel 6.
6. `curl http://<policy-lan-ip>:8765/api/health` returns JSON and a browser can
   render `/stream.mjpg` while the dog remains stationary.
