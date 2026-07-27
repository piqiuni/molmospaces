# semantic_mapping_py_pkg

Python semantic mapping pipeline for MolmoSpaces navigation.

The package keeps the same downstream contract expected by `explore_pkg`:

- `/semantic_mapping/obj_map`
- `/semantic_mapping/scene_id_grid`
- `/semantic_mapping/scene_confidence_grid`

Internally the pipeline is split into replaceable nodes:

1. `object_detection_node.py`: object/target detection from RGB-D or an external model.
2. `room_attribute_node.py`: room attribute inference from the object layer.
3. `semantic_mapping_node.py`: accumulated object map and scene-id grid publication.

The default object detector backend is `mock_empty`, and `no_detection` is available when a test should explicitly publish no detections.

## Two-level object detection design

`object_detection_node.py` is now split into two internal stages:

1. A 2D provider that returns detections with `bbox`, and optionally `mask`.
2. A 3D lifting backend that converts those detections into camera-frame and world-frame geometry.

In short:

- `provider`: detection source, responsible for returning raw per-frame detections
- `backend`: geometry interpretation, responsible for turning raw detections into 3D object observations

Current backends:

- `no_detection`: explicit no-detection mode.
- `mock_empty`: no detections.
- `external_http`: direct passthrough from an external model service.
- `yolo_world_center_projection`: use 2D `bbox` center plus depth to estimate a 3D point.
- `yolo_world_sam_box3d`: use `mask` when available, otherwise `bbox` depth samples, and fit an axis-aligned 3D box.
- `sam3_box3d`: assume the provider returns SAM3-style masks, then fit an axis-aligned 3D box from the masked depth/point cloud.
- `yoloe_pf_box3d`: run local YOLOE prompt-free segmentation and fit an axis-aligned 3D box from masked depth/point cloud.

Current providers:

- `external_http`
- `mock_empty`
- `yoloe_local`

This keeps the node interface stable while letting us swap:

- 2D detector: YOLO-World, GT detector, GroundingDINO, Molmo VLM
- 3D lifter: center projection, mask point cloud box fitting, future oriented box fitting

## Detection message contract

`/semantic_mapping/object_detections` publishes a JSON object inside `std_msgs/String`:

```json
{
  "stamp_sec": 123,
  "stamp_nsec": 456,
  "detections": [
    {
      "semantic_class": "apple",
      "semantic_class_raw": "apple",
      "confidence": 0.92,
      "bbox": [120, 80, 180, 150],
      "position": {"x": 1.2, "y": 0.1, "z": 0.7},
      "world_position": {"x": 3.4, "y": -1.2, "z": 0.7},
      "size": {"x": 0.08, "y": 0.08, "z": 0.10},
      "box3d_center": {"x": 1.2, "y": 0.1, "z": 0.7},
      "box3d_size": {"x": 0.08, "y": 0.08, "z": 0.10},
      "instance_id": "apple_001",
      "projection_method": "mask_box3d_projection",
      "source_frame": "tf_frame_lidar",
      "mask_area": 164
    }
  ]
}
```

Field meaning:

- `bbox`: 2D detection box in `[x1, y1, x2, y2]`
- `position`: 3D position in the source camera frame
- `world_position`: 3D position transformed into `frames.world_frame` when TF is available
- `size`: estimated axis-aligned 3D size, present for box-fitting backends
- `projection_method`: documents whether the detection came from center projection or mask/box 3D fitting

For `yolo_world_sam_box3d`, the external provider is expected to return `mask` when possible. Right now we support either:

- a dense binary mask with image shape
- a sparse mask dictionary with `rows` and `cols`

If `mask` is missing, the backend falls back to sampling depth points inside `bbox`.

For `sam3_box3d`, we expect the provider to primarily return masks. It still accepts `bbox` for clipping and fallback, but the intended use is "SAM3 first" rather than "YOLO first".

For `yoloe_local`, the provider loads a local Ultralytics YOLOE prompt-free segmentation model and returns:

- `semantic_class_raw`: original YOLOE open-vocabulary class
- `semantic_class`: mapped navigation-friendly class
- `bbox`
- sparse `mask` using `rows` and `cols`
- `mask_area` (`provider` stage stores mask pixels, `box3d` stage overwrites it with valid projected point count)
- `source_model`

Unknown open-set classes are dropped by default. Set `keep_unknown_open_set: true` to keep them as
`semantic_class: unknown_open_set`.

The default raw-to-navigation mapping now lives in:

- [`config/object_class_mapping.json`](./config/object_class_mapping.json)

Downstream nodes already consume the normalized `semantic_class`:

- `semantic_mapping_node.py` uses it to build `/semantic_mapping/obj_map`
- `explore_pkg` consumes `/semantic_mapping/obj_map` and therefore sees the normalized navigation label

## Offline detector RViz test

`object_detection_visual_test.py` reads one RGB image and one depth image, calls the configured object detector backend, and republishes the result in a camera-centered RViz frame.

Example:

```bash
roslaunch semantic_mapping_py_pkg object_detection_visual_test.launch \
  rgb_path:=/path/to/rgb.png \
  depth_path:=/path/to/depth.png \
  backend:=external_http \
  provider:=external_http \
  external_url:=http://127.0.0.1:8000/detect
```

Supported `backend` values are the same as `object_detection_node.py`: `no_detection`,
`mock_empty`, `external_http`, `yolo_world_center_projection`, `yolo_world_sam_box3d`,
`sam3_box3d`, and `yoloe_pf_box3d`.
Use `include_depth:=true` when the external provider needs the depth array in the HTTP payload.

Example local YOLOE prompt-free test:

```bash
roslaunch semantic_mapping_py_pkg object_detection_visual_test.launch \
  rgb_path:=/home/user/ldl/molmospaces/detection_models/tum_rgbd_scribble_samples/bedroom_12_image.png \
  depth_path:=/home/user/ldl/molmospaces/detection_models/tum_rgbd_scribble_samples/bedroom_12_depth.png \
  backend:=yoloe_pf_box3d \
  provider:=yoloe_local \
  model_path:=/home/user/ldl/molmospaces/detection_models/yoloe/weights/yoloe-26x-seg-pf.pt
```

RViz fixed frame can be set to `semantic_test_camera`. Published topics:

- `/semantic_mapping/test/rgb_image`
- `/semantic_mapping/test/depth_viz`
- `/semantic_mapping/test/rgb_depth_cloud`
- `/semantic_mapping/test/segmented_object_cloud`
- `/semantic_mapping/test/boxes_3d`
- `/semantic_mapping/test/detections`

## Incremental interaction graph

`semantic_mapping_node.py` now maintains the legacy semantic mapping outputs and an incremental
interaction-aware scene graph in parallel.

Additional topics:

- `/semantic_mapping/unified_graph`
- `/semantic_mapping/navigation_hints`

Online data flow:

`detections or GT replay -> normalized observations -> InteractionGraphStore -> semantic_mapping outputs`

### Observation contract

Detector output and realtime GT observations are normalized before entering the graph store. The
realtime GT contract is intentionally limited to perfect instance identity, class naming, 2D
localization/segmentation, and a world-frame 3D box:

```json
{
  "id": "double_door_root",
  "name": "Door",
  "bbox_2d": [120, 80, 180, 220],
  "segmentation": {
    "rows": [80, 80, 81],
    "cols": [120, 121, 120]
  },
  "box_3d": {
    "center": [1.0, 2.0, 0.9],
    "size": [0.9, 0.1, 2.0],
    "frame_id": "world"
  }
}
```

GT observations do not publish simulator articulation metadata, room IDs, containment relations,
interaction flags, object state, joint names/ranges/values, approach axes, or parent/child links.
The mapping pipeline derives node type from the normalized name, computes visibility evidence from
the mask, associates objects with rooms geometrically, infers portal connectivity and containment
from the 3D boxes, and keeps interaction state `unknown` until semantic attribute inference or an
executor result explicitly supplies `state`/`post_state`.

For realtime GT records, normalization is allowlist-based: even if a legacy producer accidentally
includes flags, parent/child links, room IDs, poses, confidence, or joint metadata, those values are
discarded and cannot affect graph type, relations, or interaction state.

Joint readback may be used privately inside an oracle interaction executor. Downstream graph and
decision messages retain only semantic commands/results (`object_id`, action, optional visual open
regions, `state`/`post_state`, success, and cost), not raw joint metadata. The simulator executor
maps `object_id` to its articulation and, for drawer scans, maps visual regions to slide joints.

### Unified graph JSON

`/semantic_mapping/unified_graph` publishes:

```json
{
  "scene_id": "semantic_mapping_scene",
  "source_mode": "detector_online",
  "timestamp": 123.4,
  "nodes": [],
  "edges": [],
  "views": {
    "semantic_view": {"node_ids": [], "edge_ids": []},
    "interaction_view": {"node_ids": [], "edge_ids": []},
    "navigation_view": {"node_ids": [], "edge_ids": [], "hints": []}
  }
}
```

Node types are fixed to `room`, `portal`, `support`, `container`, and `object`.
Doors are always represented as `portal` nodes.

### Navigation hints

`/semantic_mapping/navigation_hints` is a lighter navigation-facing view:

```json
[
  {
    "hint_id": "hint_0001",
    "type": "interactive_portal",
    "node_id": "portal_door_12",
    "position": [1.0, 2.0, 0.9],
    "room_id": 2,
    "priority": 1.0,
    "confidence": 1.0,
    "requires_interaction": true,
    "interaction_node_id": "portal_door_12",
    "interaction_mode": "open_close",
    "state": "closed",
    "reason": "door_may_unlock_room"
  }
]
```

### Full scene JSON export

`read_scene_room_properties.py` should now be treated as a scene inspection and plotting tool.
When you want data for later graph or replay tasks, export the full scene JSON:

```bash
python scripts/InteractiveNav/read_scene_room_properties.py \
  --scene_dataset procthor-10k \
  --house_ind 0 \
  --export_scene_json /tmp/scene_full.json \
  --no_plot
```

### GT detector replay with graph visualization

Replay the full scene JSON as incrementally published detection messages, unified graph JSON,
navigation hints, and RViz graph markers:

```bash
python Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts/semantic_mapping_gt_replay.py \
  /tmp/scene_full.json \
  --batch-size 4 \
  --publish-rate 1.0
```

This now publishes to:

- `/semantic_mapping/object_detections`
- `/semantic_mapping/unified_graph`
- `/semantic_mapping/navigation_hints`
- `/semantic_mapping/unified_graph_markers`

So `semantic_mapping_node.py` and RViz can consume GT replay exactly like the online pipeline.

Current limitation:

- The legacy full-scene replay format may carry room and joint metadata, but the interaction graph
  does not use parent links, interaction flags, or joints for graph construction or state inference.
- Support/container assignment from detector-only observations is still heuristic.
