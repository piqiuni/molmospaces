# Physical Go2 recording and offline showcase

The physical gateway remains read-only.  Recording is a side channel owned by
the gateway and is independent of browser repainting or navigation restarts.

## Start/stop

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
