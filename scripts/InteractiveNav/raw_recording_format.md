# Image + JSON raw recording format

Runtime recording stores only source data. Six-panel images and MP4 files are derived offline.

```text
raw/
  step_boundaries.jsonl
  camera/manifest.jsonl
  maps/{raw_occ,planning_occ,global_costmap,local_costmap,room_segment}/
  navigation/{odom,subgoal,global_plan,local_global_plan,local_plan}.jsonl
  semantic/{gt_observations,unified_graph,candidates,selection,execution_state}.jsonl
```

Grid receipts use lossless PNG plus one JSONL metadata record. Values are encoded as `png_value = source_value + png_value_offset` (normally offset 1); ordinary occupancy maps use 8-bit PNG, while room labels and costmaps automatically use 16-bit PNG when their range requires it. Metadata retains the bit depth, header sequence/stamp, receipt order, frame, resolution, dimensions and origin.

Each step-boundary record contains the simulator step/stamp and the latest accepted receipt id for every source. Offline alignment is causal: a renderer may use only data received at or before that boundary and must never substitute a future receipt.

Raw map PNG encoding and JSONL manifest appends run in a dedicated local writer thread. The recorder's full-map ROS subscriptions use a bounded `--map-receive-queue-size` transport backlog. ROS callbacks only detach an occupancy raster and enqueue it; they never call PNG encoding or `flush()`. The default queue policy is `block`, so a full queue applies explicit backpressure instead of silently deleting a map. `summary.json.raw_recording_stats` records per-stage `source_received`, `accepted`, `persisted`, `queue_dropped`, and `write_failed` counts plus queue latency/peak depth. A valid complete recording has equal source/accepted/persisted counts and zero drops/failures for every required stage.
