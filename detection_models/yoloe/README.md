# YOLOE Prompt-Free Smoke Test

This directory contains a minimal local setup for Ultralytics YOLOE prompt-free models.

Models planned for local download:

- `yoloe-26m-seg-pf.pt`
- `yoloe-26l-seg-pf.pt`
- `yoloe-26x-seg-pf.pt`

Main script:

- `run_yoloe_prompt_free.py`
- `download_yoloe_weights.py`

Example:

```bash
conda run -n yolo_world python /home/user/ldl/molmospaces/detection_models/yoloe/run_yoloe_prompt_free.py \
  --model /home/user/ldl/molmospaces/detection_models/yoloe/weights/yoloe-26m-seg-pf.pt \
  --image /home/user/ldl/molmospaces/detection_models/tum_rgbd_scribble_samples/bedroom_12_image.png \
  --device cuda:0
```

Download the official 26-series prompt-free weights:

```bash
conda run -n yolo_world python /home/user/ldl/molmospaces/detection_models/yoloe/download_yoloe_weights.py
```
