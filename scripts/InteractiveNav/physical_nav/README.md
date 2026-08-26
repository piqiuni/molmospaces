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
  -> physical_six_panel_server.py (WebSocket gateway + LAN web page)
  -> physical_ros_gateway.py (system-Python ROS bridge via /api/raw-frame)
  -> physical_yoloe_bridge.py (algorithm Python, no ROS dependency)
       YOLOE-26l PF Seg -> RGB-D 3-D boxes/pointcloud -> ROS detections
  -> semantic_mapping_py_pkg/semantic_mapping_node.py
       object map/room segmentation/interaction graph
  -> physical_consistency_node.py -> /physical_nav/consistency
```

The Go2 client has no motion publisher or control client. Keyboard/web commands
are handled by `ReadOnlySafetyGate`, recorded as `READ_ONLY_BLOCKED`, and never
sent to the robot.

## Start on the policy machine

Install the side-specific transport dependencies first. The Go2 image uses
Python 3.8, while the policy machine uses Python 3.10+:

```bash
python3 -m pip install -r /home/user/ldl/molmospaces/scripts/InteractiveNav/physical_nav/requirements_go2.txt  # on Go2
python3 -m pip install -r /home/user/ldl/molmospaces/scripts/InteractiveNav/physical_nav/requirements_policy.txt  # on policy machine
```

`unitree_sdk2py`, `librealsense/pyrealsense2`, ROS Noetic and Ultralytics are
not vendored. They must already be provided by the corresponding platform
environment; YOLOE is loaded only on the policy machine.

Start the WebSocket/web gateway before the Go2 client. It binds to all interfaces so a LAN
browser can connect:

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

The ROS bridge (`physical_ros_gateway.py`) is started by
`physical_nav_readonly.launch`; it uses the system ROS Python and polls the
gateway's encoded frame endpoint. This process separation is required on ROS
Noetic hosts whose `rospy` is not compatible with the Python environment used
by YOLOE. It posts graph, consistency and occupancy snapshots back to
`/api/ros-state` for the detailed web area. The ROS bridge republishes
the detector JSON from the web gateway to `/physical_nav/detections`; graph,
consistency and occupancy messages flow in the opposite direction.
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
image, `/api/state` includes graph, detections, telemetry, consistency and
Qwen request/result history, and `/api/health` is suitable for a smoke test.

Before trusting map-frame 3-D geometry, set the measured D435i-to-base
extrinsic through `PHYSICAL_NAV_CAMERA_X/Y/Z/ROLL/PITCH/YAW` (metres/radians). The
defaults are identity solely for protocol smoke tests; they are not a camera
calibration.

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
