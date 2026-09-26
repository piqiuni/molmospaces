# Physical Go2 recording and offline showcase

The physical gateway remains read-only.  Recording is a side channel owned by
the gateway and is independent of browser repainting or navigation restarts.

## Start/stop

One-command launch with recording (does not enable robot motion):

The port 8765 home page also has **开始录制 / 停止录制** buttons for the same
`raw_plus_panels` session, with duration, destination, queue depth and drop/error
counters. Status polling runs at 1 Hz. All recorder ingress is non-blocking,
including model/state events; overflow marks the session degraded. Explicit stop
may wait for queued disk writes to finish. Recording does not bypass the normal
six-panel rendering rate or unchanged-state cache.

```bash
PHYSICAL_NAV_YOLO_GPU=2 bash scripts/InteractiveNav/physical_nav/physical_nav_all.sh restart --record
```

`--record` can also be combined with the existing `enable_motion` argument.
Without this option recording remains off by default. Set `PHYSICAL_NAV_RECORD_DIR`
to choose the destination. The launcher checks that recording is active and prints
the session directory; `stop`/`restart` drain the old session even when the web
gateway is retained. A browser is not required, but the visualization service is.

New recordings also store the exact six-panel JPEG shown on port 8765 as panel 0,
all individual panels including panel 5, and `raw/camera/first_person.mjpeg`:
the original JPEG frames concatenated without live re-encoding. This native RGB
video has no detection overlays; its authoritative timing and depth/calibration
are in `raw/camera/manifest.jsonl`. MJPEG has no embedded timestamps; use the
offline exporter for correctly timed MP4 playback. Queue drops/write failures
remain visible in session statistics; recording is best-effort and must not block
navigation when storage is slow.

After stopping, export the original six-panel layout (not the dark showcase theme)
and first-person MP4 with a separate offline command:

```bash
conda run -n mlspaces python scripts/InteractiveNav/build_physical_six_panel_video.py \
  /home/user/ldl/recordings/go2_physical/SESSION_ID --fps 10
```

Outputs: `derived/six-panel/overview_6panel.mp4`, `first_person.mp4`, and a manifest
containing source quality statistics. Existing outputs are never overwritten;
use `--output /new/directory` for another export. Each output frame uses only
the latest preceding receipt, preserving real elapsed time and holding frames
during source gaps. The six-panel image content/layout matches the live JPEG;
browser controls/chrome are not recorded. Export is silent, uses MP4 encoding,
and does not invoke ROS, Qwen or a running web server.

For a review copy, add `--step-overlay --overview-only --output /new/preview-directory`.
The exporter adds a separate top strip containing the recorded step and source
time without covering the six panels. Steps are matched to panel receipts in
order, including multiple steps captured from the same camera frame.

The default root is:

```text
/home/user/ldl/recordings/go2_physical/<session-id>/
```

On `showcase-dark`, **后台录制到本机** starts/stops a session through
`POST /api/recording/start` and `POST /api/recording/stop`.  The phone page has
**同步录制到本机**; it joins the same session and sends its JPEG frames and
16-bit PCM audio with the session token.  Closing the showcase does not stop a
session.  The gateway may also start one at launch:

```bash
PHYSICAL_NAV_RECORD_ON_START=1 \
PHYSICAL_NAV_RECORD_DIR=/home/user/ldl/recordings/go2_physical \
bash scripts/InteractiveNav/physical_nav/start_physical_nav.sh
```

The HTTP status and session list are available at `/api/recording/status` and
`/api/recording/sessions`.  No Go2 control endpoint is involved.

## Session contents

`raw/` is immutable source data:

```text
camera/manifest.jsonl        # every accepted RGB-D receipt; JPEG + depth PNG
camera/rgb/, camera/depth/
map_manifest.jsonl           # lossless occupancy/cost/room grid receipts
maps/<stage>/
panels/manifest.jsonl         # source-resolution panel rasters (1–4 and 6)
panels/<panel-name>/
state/right_panel.jsonl       # Go2 state + Agent/MLLM right-rail snapshots
semantic/mllm_events.jsonl   # M1, M2 and M3 requests/results
semantic/qwen_events.jsonl   # explicit Qwen requests/results
navigation/*.jsonl            # plans and execution state receipts
phone/frames/, phone/manifest.jsonl
phone/audio.pcm, phone/audio.wav
step_boundaries.jsonl         # causal alignment receipts
```

The dark showcase sections are recorded explicitly in `session.json`:

* perception: panel 1 / RGB + box overlay;
* spatial understanding: panel 3;
* interaction graph: panel 6;
* right rail: `state/right_panel.jsonl` (Go2 status, MLLM calls and Agent
  timeline).

Panel 2 (OCC) and panel 4 (global/local costmap) are also retained as useful
chain evidence.  Masks/point clouds are not duplicated in right-rail JSON;
their authoritative RGB-D/map receipts remain available for replay.

Modes:

* `raw_plus_panels` (default): RGB-D, maps, events, panels and phone media;
* `raw_replay`: source RGB-D/maps/events without presentation JPEGs;
* `page_capture`: panels/events/phone media only, for a small presentation
  capture when raw RGB-D is not needed.

## Offline synthesis

The live service is not needed after a session is complete:

```bash
python scripts/InteractiveNav/build_physical_showcase_video.py <session-id> \
  --theme dark --phone-mode empty

# Produce all three themes and replace the D435i panel with an edited phone clip:
python scripts/InteractiveNav/build_physical_showcase_video.py <session-id> \
  --theme all --phone-video /path/to/phone.mp4 --phone-mode replace

# Use the synchronized JPEG receipts captured by the phone page directly
# (an editor-supplied MP4 is optional):
python scripts/InteractiveNav/build_physical_showcase_video.py <session-id> \
  --theme dark --phone-mode pip
```

Derived files are written under `derived/showcase-<theme>/` and include
`showcase.mp4`, `panel1.mp4` … `panel6.mp4`, `right_rail.jsonl` and a manifest.
The phone modes are `empty`, `replace`, `pip` and `side-by-side`.  With no
`--phone-video`, `pip`/`replace`/`side-by-side` use the synchronized JPEG
receipts under `raw/phone/`; an editor-supplied MP4 takes precedence when it is
provided.  Audio comes from the recorded phone WAV by default; `--audio`
selects another phone audio file.  The compositor uses causal timestamps and
automatically inserts the measured delay before the first phone PCM packet;
it never substitutes a future receipt.  If `ffmpeg` is installed it muxes
H.264/AAC; otherwise the OpenCV MP4 remains usable without audio.
