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

Current providers:

- `external_http`
- `mock_empty`

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
and `sam3_box3d`.
Use `include_depth:=true` when the external provider needs the depth array in the HTTP payload.

RViz fixed frame can be set to `semantic_test_camera`. Published topics:

- `/semantic_mapping/test/rgb_image`
- `/semantic_mapping/test/depth_viz`
- `/semantic_mapping/test/rgb_depth_cloud`
- `/semantic_mapping/test/boxes_3d`
- `/semantic_mapping/test/detections`
