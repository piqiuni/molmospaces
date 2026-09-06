#!/usr/bin/env python3
"""Local/LAN six-panel viewer and read-only sensor WebSocket gateway."""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import io
import json
import math
import os
from pathlib import Path
import socket
import ssl
import subprocess
import shutil
import secrets
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from physical_protocol import decode_wire_packet, validate_packet
from runtime_state import RuntimeState
from safety_gate import ReadOnlySafetyGate
from qwen_client import QwenClient
from physical_raw_recorder import DEFAULT_RECORD_DIR, PhysicalRawRecorder
from showcase_pages import ACADEMIC_SHOWCASE_HTML, DARK_SHOWCASE_HTML, LIGHT_SHOWCASE_HTML
try:
    from offline_semantic_renderer import (
        OfflineSixPanelRenderer,
        RawGrid,
        TransformResolver,
        draw_camera_title,
        draw_task_subgoal_header,
        known_world_bounds,
        _node_label,
    )
except ModuleNotFoundError:  # imported from physical_nav/tests
    _interactive_nav_dir = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    if _interactive_nav_dir not in sys.path:
        sys.path.insert(0, _interactive_nav_dir)
    from offline_semantic_renderer import (
        OfflineSixPanelRenderer,
        RawGrid,
        TransformResolver,
        draw_camera_title,
        draw_task_subgoal_header,
        known_world_bounds,
        _node_label,
    )

try:
    import cv2
    import numpy as np
    # The viewer is a diagnostic stream; OpenCV's default worker pool can
    # create one thread per CPU and starve the HTTP/WebSocket event loops.
    cv2.setNumThreads(1)
except ImportError:  # pragma: no cover - useful for protocol-only testing
    cv2 = None
    np = None


def _phone_stream_page(stream_url: str) -> str:
    """Mobile camera publisher kept independent from the navigation process."""
    escaped_url = html_lib.escape(stream_url, quote=True)
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1,viewport-fit=cover'>
<title>手机实时视频推流</title><style>
*{{box-sizing:border-box}}html,body{{margin:0;min-height:100%;background:#06101d;color:#edf7ff;font-family:Inter,'Noto Sans SC',system-ui,sans-serif}}main{{min-height:100vh;padding:18px;display:flex;flex-direction:column;gap:14px;max-width:820px;margin:auto}}h1{{font-size:24px;margin:0;color:#79caff}}.sub{{color:#8da7c3;font-size:13px}}.preview{{position:relative;min-height:260px;flex:1;border:1px solid #315d88;border-radius:14px;background:#02060b;overflow:hidden;display:grid;place-items:center}}video{{width:100%;height:100%;object-fit:contain;background:#000}}.badge{{position:absolute;left:12px;top:12px;padding:6px 10px;border-radius:99px;background:#14283de8;color:#ffd46c;font-size:12px}}.badge.live{{color:#72efad}}.record-badge{{position:absolute;right:12px;top:12px;padding:6px 10px;border-radius:99px;background:#3a1118e8;color:#ff9ca8;font-size:12px}}.record-badge[hidden]{{display:none}}.controls{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}button,.rate-control{{min-height:48px;border:1px solid #3b78ad;border-radius:10px;background:#10263c;color:#e9f5ff;font-size:16px;font-weight:700}}button.primary{{background:#1474ad}}button.recording{{background:#8f2533;border-color:#ff6d7c}}button:disabled{{opacity:.45}}.rate-control{{display:flex;align-items:center;justify-content:center;gap:7px;padding:0 8px;font-size:13px}}.rate-control select{{border:0;border-radius:6px;background:#071625;color:#79caff;font:700 15px system-ui;padding:5px 7px}}.notice{{padding:12px;border:1px solid #315d88;border-radius:10px;background:#0b1a2a;color:#9fb4ca;font-size:12px;line-height:1.55}}code{{color:#72d7ff;word-break:break-all}}canvas{{display:none}}@media(max-width:620px){{.controls{{grid-template-columns:1fr 1fr}}}}
</style><main><h1>手机实时视频推流</h1><div class='sub'>连接具身智能交互导航展示平台 · 540p 视频 + 48 kHz 高质量音频</div><section class='preview'><video id='video' autoplay muted playsinline></video><span id='status' class='badge'>等待相机与麦克风授权</span><span id='record-status' class='record-badge' hidden>● REC 00:00</span></section><div class='controls'><button id='start' class='primary'>开启音视频推流</button><button id='flip' disabled>切换摄像头</button><label class='rate-control'>推流帧率<select id='fps'><option value='5'>5 FPS</option><option value='10' selected>10 FPS</option><option value='15'>15 FPS</option><option value='20'>20 FPS</option></select></label><button id='sync-record' disabled>同步录制到本机</button><button id='record' disabled>录制 MP4（备份）</button></div><div class='notice'>推流地址：<code>{escaped_url}</code><br>默认 960×540 / 10 FPS，可实时切换帧率。点击“同步录制到本机”后，网关后台保存原始视频、手机音频、四路展示面板和右侧状态；浏览器关闭也不会中断。下方 MP4 按钮仅作为手机端本地备份，转换文件不会留在服务器。</div><canvas id='capture'></canvas></main><script>
const video=document.getElementById('video'),canvas=document.getElementById('capture'),statusBox=document.getElementById('status'),startButton=document.getElementById('start'),flipButton=document.getElementById('flip'),recordButton=document.getElementById('record'),syncRecordButton=document.getElementById('sync-record'),recordStatus=document.getElementById('record-status'),fpsSelect=document.getElementById('fps');
let stream=null,facing='environment',targetFps=10,uploadTimer=0,uploadBusy=false,frameCount=0,recorder=null,recordChunks=[],recordStartedAt=0,recordTimer=0,audioContext=null,audioSource=null,audioProcessor=null,audioQueue=[],audioUploadBusy=false,audioSequence=0,recordingSession=null,syncRecordingBusy=false;
function status(text,live=false){{statusBox.textContent=text;statusBox.classList.toggle('live',live)}}
function recordingTypes(){{return ['video/mp4;codecs=avc1.42E01E,mp4a.40.2','video/mp4','video/webm;codecs=vp9,opus','video/webm;codecs=vp8,opus','video/webm']}}
function createRecorder(mediaStream,videoRate){{for(const mimeType of recordingTypes()){{if(!MediaRecorder.isTypeSupported(mimeType))continue;try{{return new MediaRecorder(mediaStream,{{mimeType,videoBitsPerSecond:videoRate,audioBitsPerSecond:192000}})}}catch(_){{}}}}return new MediaRecorder(mediaStream,{{videoBitsPerSecond:videoRate,audioBitsPerSecond:192000}})}}
function downloadBlob(blob,name){{const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download=name;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(link.href),30000)}}
async function savePhoneMp4(blob){{const stamp=new Date().toISOString().replace(/[:.]/g,'-'),name='go2-phone-'+stamp;if(blob.type.includes('mp4')){{downloadBlob(blob,name+'.mp4');return}}status('正在转换 MP4，请保持页面开启…');recordButton.textContent='转换 MP4…';recordButton.disabled=true;try{{const response=await fetch('/api/recording-to-mp4',{{method:'POST',headers:{{'Content-Type':blob.type||'video/webm'}},body:blob}});if(!response.ok)throw new Error((await response.json()).error||('HTTP '+response.status));downloadBlob(await response.blob(),name+'.mp4');status('MP4 已保存',true)}}catch(error){{downloadBlob(blob,name+'.webm');status('MP4 转换失败，已保存 WebM：'+error.message)}}}}
function updateRecordClock(){{const elapsed=Math.max(0,Math.floor((Date.now()-recordStartedAt)/1000)),minutes=String(Math.floor(elapsed/60)).padStart(2,'0'),seconds=String(elapsed%60).padStart(2,'0');recordStatus.textContent='● REC '+minutes+':'+seconds}}
function finishRecording(){{if(recordTimer)clearInterval(recordTimer);recordTimer=0;recordStatus.hidden=true;recordButton.classList.remove('recording');recordButton.textContent='录制 MP4';recordButton.disabled=!stream}}
function stopRecording(){{if(recorder&&recorder.state!=='inactive')recorder.stop();else finishRecording()}}
function startRecording(){{if(!stream||typeof MediaRecorder==='undefined'){{status('当前浏览器不支持视频录制');return}}try{{recordChunks=[];recorder=createRecorder(stream,3500000);recorder.ondataavailable=event=>{{if(event.data?.size)recordChunks.push(event.data)}};recorder.onerror=event=>{{status('录制失败：'+(event.error?.message||'未知错误'));finishRecording()}};recorder.onstop=async()=>{{const actualMime=recorder.mimeType||'video/webm',blob=new Blob(recordChunks,{{type:actualMime}});recordChunks=[];if(blob.size)await savePhoneMp4(blob);finishRecording()}};recorder.start(4000);recordStartedAt=Date.now();recordStatus.hidden=false;recordButton.classList.add('recording');recordButton.textContent='停止并保存 MP4';updateRecordClock();recordTimer=setInterval(updateRecordClock,1000)}}catch(error){{status('无法录制：'+error.message);finishRecording()}}}}
function encodePcm16(samples){{const output=new ArrayBuffer(samples.length*2),view=new DataView(output);for(let index=0;index<samples.length;index++){{const sample=Math.max(-1,Math.min(1,samples[index]));view.setInt16(index*2,sample<0?sample*32768:sample*32767,true)}}return output}}
function recordingQuery(prefix){{if(!recordingSession)return '';return '&recording_session='+encodeURIComponent(recordingSession.session_id||'')+'&recording_token='+encodeURIComponent(recordingSession.token||'')+'&ts='+(Date.now()/1000).toFixed(3)}}
async function pumpAudio(){{if(audioUploadBusy||!audioQueue.length)return;audioUploadBusy=true;const packet=audioQueue.shift();try{{const response=await fetch('/api/phone-audio?seq='+(++audioSequence)+'&rate='+packet.rate+recordingQuery(),{{method:'POST',headers:{{'Content-Type':'audio/pcm;format=s16le;channels=1'}},body:packet.data,cache:'no-store'}});if(response.status===403){{recordingSession=null;syncRecordButton.classList.remove('recording');syncRecordButton.textContent='同步录制到本机'}}}}catch(_){{}}finally{{audioUploadBusy=false;if(audioQueue.length)pumpAudio()}}}}
async function startAudioUpload(){{const AudioContextClass=window.AudioContext||window.webkitAudioContext;if(!AudioContextClass||!stream?.getAudioTracks().length)return;try{{audioContext=new AudioContextClass({{sampleRate:48000,latencyHint:'interactive'}})}}catch(_){{audioContext=new AudioContextClass()}}await audioContext.resume();audioSource=audioContext.createMediaStreamSource(stream);audioProcessor=audioContext.createScriptProcessor(8192,1,1);audioProcessor.onaudioprocess=event=>{{const samples=event.inputBuffer.getChannelData(0);audioQueue.push({{rate:event.inputBuffer.sampleRate,data:encodePcm16(samples)}});if(!recordingSession&&audioQueue.length>3)audioQueue.splice(0,audioQueue.length-3);else if(audioQueue.length>100)audioQueue.splice(0,audioQueue.length-100);pumpAudio()}};audioSource.connect(audioProcessor);const silentSink=audioContext.createGain();silentSink.gain.value=0;audioProcessor.connect(silentSink);silentSink.connect(audioContext.destination)}}
async function stopAudioUpload(){{audioQueue=[];if(audioProcessor){{audioProcessor.onaudioprocess=null;audioProcessor.disconnect()}}if(audioSource)audioSource.disconnect();audioProcessor=null;audioSource=null;if(audioContext){{try{{await audioContext.close()}}catch(_){{}}}}audioContext=null}}
async function stopCamera(){{if(uploadTimer)clearInterval(uploadTimer);uploadTimer=0;if(recorder&&recorder.state!=='inactive')stopRecording();await stopAudioUpload();if(stream)stream.getTracks().forEach(track=>track.stop());stream=null;video.srcObject=null;startButton.textContent='开启音视频推流';flipButton.disabled=true;recordButton.disabled=true;syncRecordButton.disabled=!recordingSession;status(recordingSession?'音视频已停止 · 后台同步录制仍在继续':'音视频推流已停止')}}
function scheduleFrameUploads(){{if(uploadTimer)clearInterval(uploadTimer);uploadTimer=0;if(!stream)return;uploadTimer=setInterval(uploadFrame,Math.round(1000/targetFps));uploadFrame()}}
async function startCamera(){{try{{await stopCamera();if(!navigator.mediaDevices?.getUserMedia)throw new Error('当前浏览器仅允许在安全页面使用相机和麦克风');stream=await navigator.mediaDevices.getUserMedia({{video:{{facingMode:{{ideal:facing}},width:{{ideal:960,max:960}},height:{{ideal:540,max:540}},frameRate:{{ideal:30,max:30}}}},audio:{{channelCount:{{ideal:1,max:1}},sampleRate:{{ideal:48000}},sampleSize:{{ideal:16}},echoCancellation:false,noiseSuppression:false,autoGainControl:false}}}});video.srcObject=stream;await video.play();await startAudioUpload();startButton.textContent='停止音视频推流';flipButton.disabled=false;recordButton.disabled=false;syncRecordButton.disabled=false;status('正在连接音视频…');scheduleFrameUploads()}}catch(error){{status('无法开启：'+error.message);startButton.textContent='重试开启'}}}}
async function uploadFrame(){{if(uploadBusy||!stream||video.readyState<2)return;uploadBusy=true;try{{const sourceWidth=video.videoWidth||640,sourceHeight=video.videoHeight||480,width=Math.min(960,sourceWidth),height=Math.max(1,Math.round(width*sourceHeight/sourceWidth)),quality=targetFps>=15?.58:.66;canvas.width=width;canvas.height=height;canvas.getContext('2d',{{alpha:false}}).drawImage(video,0,0,width,height);const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',quality));if(!blob)throw new Error('JPEG 编码失败');const response=await fetch('/api/phone-frame?seq='+(++frameCount)+'&fps='+targetFps+recordingQuery(),{{method:'POST',headers:{{'Content-Type':'image/jpeg'}},body:blob,cache:'no-store'}});if(response.status===403){{recordingSession=null;syncRecordButton.classList.remove('recording');syncRecordButton.textContent='同步录制到本机'}}else if(!response.ok)throw new Error('上传 '+response.status);status('实时推流中 · '+width+'×'+height+' · '+targetFps+' FPS',true)}}catch(error){{status('推流重试：'+error.message)}}finally{{uploadBusy=false}}}}
async function startSynchronizedRecording(){{if(syncRecordingBusy||!stream)return;syncRecordingBusy=true;try{{const response=await fetch('/api/recording/phone-join',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{mode:'raw_plus_panels',label:'phone-sync',metadata:{{source:'phone',width:video.videoWidth||960,height:video.videoHeight||540,fps:targetFps,audio:'phone'}}}})}});const value=await response.json();if(!response.ok)throw new Error(value.error||('HTTP '+response.status));recordingSession={{session_id:value.session_id,token:value.token||''}};syncRecordButton.classList.add('recording');syncRecordButton.textContent='停止同步录制';status('后台同步录制中 · 手机音频已保存',true)}}catch(error){{status('同步录制失败：'+error.message)}}finally{{syncRecordingBusy=false}}}}
async function stopSynchronizedRecording(){{if(syncRecordingBusy||!recordingSession)return;syncRecordingBusy=true;try{{const current=recordingSession;const response=await fetch('/api/recording/stop',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{session_id:current.session_id,token:current.token,reason:'phone_button'}})}});const value=await response.json();if(!response.ok)throw new Error(value.error||('HTTP '+response.status));recordingSession=null;syncRecordButton.classList.remove('recording');syncRecordButton.textContent='同步录制到本机';status('后台录制已保存 · '+(value.record_dir||'本机'),true)}}catch(error){{status('停止同步录制失败：'+error.message)}}finally{{syncRecordingBusy=false}}}}
startButton.onclick=()=>stream?stopCamera():startCamera();flipButton.onclick=async()=>{{facing=facing==='environment'?'user':'environment';await startCamera()}};fpsSelect.onchange=()=>{{targetFps=Number(fpsSelect.value)||10;scheduleFrameUploads()}};recordButton.onclick=()=>recorder&&recorder.state==='recording'?stopRecording():startRecording();syncRecordButton.onclick=()=>recordingSession?stopSynchronizedRecording():startSynchronizedRecording();window.addEventListener('pagehide',()=>{{if(recordingSession){{navigator.sendBeacon('/api/recording/stop',new Blob([JSON.stringify({{session_id:recordingSession.session_id,token:recordingSession.token,reason:'phone_pagehide'}})],{{type:'application/json'}}));recordingSession=null}}stopCamera()}});startCamera();
</script></html>"""


def _ensure_phone_tls_certificate(cert_path: str, key_path: str, host_ip: str = "10.100.5.3") -> None:
    """Create a short-lived LAN-only certificate outside the repository."""
    cert, key = Path(cert_path), Path(key_path)
    if cert.is_file() and key.is_file():
        return
    cert.parent.mkdir(parents=True, exist_ok=True)
    key.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-sha256",
            "-nodes", "-days", "30", "-subj",
            f"/CN={host_ip}/O=MolmoSpaces Local Demo",
            "-addext", f"subjectAltName=IP:{host_ip}",
            "-addext", "keyUsage=digitalSignature,keyEncipherment",
            "-addext", "extendedKeyUsage=serverAuth",
            "-keyout", str(key), "-out", str(cert),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    key.chmod(0o600)


class _ReusableHTTPServer(ThreadingHTTPServer):
    # Allow an immediate service restart after a browser/MJPEG disconnect.
    allow_reuse_address = True


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np is None or np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _point_xy(value: Any) -> tuple[float, float] | None:
    """Read either graph point dictionaries or the mapper's XYZ arrays."""
    if isinstance(value, dict):
        return _safe_float(value.get("x")), _safe_float(value.get("y"))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _safe_float(value[0]), _safe_float(value[1])
    return None


def _decode(value: str, encoding: str) -> Any:
    raw = base64.b64decode(value)
    if cv2 is None:
        return raw
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not decode {encoding}")
    return image


class SixPanelRenderer:
    def __init__(self, state: RuntimeState, width: int = 640, height: int = 360) -> None:
        self.state, self.width, self.height = state, width, height
        # Use the exact six-panel renderer shared by build_raw_overview.  The
        # live physical adapter supplies in-memory RawGrid/step receipts in
        # the same shape as record_explore_debug.py's offline artifacts.
        # Render the six-panel source at 640x360 per panel.  The browser may
        # scale it down, but preserving this raster resolution keeps box text
        # readable and makes the enlarged semantic panels substantially sharper.
        self.panel_size = (640, 360)
        self._canonical = OfflineSixPanelRenderer(
            # Physical telemetry is odometry-frame pose. Mapping is currently
            # odom-locked, so the empty resolver supplies the correct identity
            # map<->odom transform while retaining both frame names.
            transforms=TransformResolver([], map_frame="tf_frame_map", odom_frame="tf_frame_odom")
        )
        # Native outputs of the canonical renderer used by the presentation
        # pages.  Keeping these separate avoids browser-side redraws and avoids
        # lossy crop/resize cycles through the composite six-panel JPEG.
        self.latest_original_panels: dict[int, bytes] = {}
        # Full panel streams are encoded only while a recorder session is
        # active.  Keeping this off during ordinary viewing avoids six extra
        # JPEG encodes on every 5 Hz render tick.
        self.capture_panel_streams = False
        self.latest_panel_streams: dict[int, bytes] = {}
        # OCC viewport is locked once at startup.  Its size is copied from
        # the first room-panel world bounds, while its center is the initial
        # robot pose, so later frontier/trajectory points cannot zoom panel 2.
        self._occ_view_bounds: tuple[float, float, float, float] | None = None

    def set_capture_panel_streams(self, enabled: bool) -> None:
        self.capture_panel_streams = bool(enabled)

    @staticmethod
    def _raw_grid(payload: Any, default_frame: str = "tf_frame_map") -> RawGrid | None:
        if not isinstance(payload, dict) or not payload.get("data"):
            return None
        try:
            width, height = int(payload.get("width", 0)), int(payload.get("height", 0))
            values = np.asarray(payload.get("data", []), dtype=np.int32)
            if width <= 0 or height <= 0 or values.size < width * height:
                return None
            values = values[: width * height].reshape((height, width))
            origin = payload.get("origin") if isinstance(payload.get("origin"), dict) else {}
            qz, qw = float(origin.get("qz", 0.0) or 0.0), float(origin.get("qw", 1.0) or 1.0)
            origin_yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
            return RawGrid(
                values=values,
                width=width,
                height=height,
                resolution=float(payload.get("resolution", 0.0) or 0.0),
                frame_id=str(payload.get("frame_id") or default_frame).lstrip("/"),
                origin_x=float(origin.get("x", 0.0) or 0.0),
                origin_y=float(origin.get("y", 0.0) or 0.0),
                origin_yaw=origin_yaw,
            )
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _global_costmap_for_display(planning: RawGrid | None, costmap: RawGrid | None) -> RawGrid | None:
        """Use a causally aligned global costmap, synthesizing stale frames."""
        if planning is None:
            return costmap
        geometry_matches = bool(
            costmap is not None
            and costmap.width == planning.width
            and costmap.height == planning.height
            and abs(costmap.resolution - planning.resolution) <= 1e-6
            and abs(costmap.origin_x - planning.origin_x) <= 1e-4
            and abs(costmap.origin_y - planning.origin_y) <= 1e-4
        )
        occupied = planning.values >= 50
        if geometry_matches and costmap is not None:
            lethal = costmap.values >= 100
            occupied_count = int(np.count_nonzero(occupied))
            overlap = int(np.count_nonzero(occupied & lethal))
            if occupied_count == 0 or overlap / occupied_count >= 0.90:
                return costmap

        # The ROS maps are delivered on independent callbacks. When the
        # costmap receipt is older than the OCC receipt, derive only the
        # display frame from that exact OCC so panel 4 never compares two
        # different map epochs. Navigation continues using the ROS costmap.
        values = np.full(planning.values.shape, -1, dtype=np.int32)
        known = planning.values >= 0
        values[known] = 0
        values[occupied] = 100
        if np.any(occupied) and planning.resolution > 0.0:
            distances = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, 5)
            distances *= float(planning.resolution)
            inscribed = known & ~occupied & (distances <= 0.25)
            soft = known & ~occupied & ~inscribed & (distances <= 0.40)
            values[inscribed] = 99
            values[soft] = np.clip(
                np.rint(1.0 + 97.0 * (0.40 - distances[soft]) / 0.15),
                1,
                98,
            ).astype(np.int32)
        return RawGrid(
            values=values,
            width=planning.width,
            height=planning.height,
            resolution=planning.resolution,
            frame_id=planning.frame_id,
            origin_x=planning.origin_x,
            origin_y=planning.origin_y,
            origin_yaw=planning.origin_yaw,
        )

    @staticmethod
    def _draw_live_detections(panel: Any, detections: list[dict[str, Any]], source_shape: tuple[int, int] | None, *, include_masks: bool = True, include_labels: bool = True) -> None:
        """Overlay live YOLOE boxes and optionally segmentation masks.

        The shared offline renderer intentionally draws only recorder/GT
        overlays.  Physical detections arrive asynchronously, so they are
        added by this adapter after the canonical camera title is rendered.
        """
        if source_shape is None:
            return
        src_h, src_w = source_shape
        if src_w <= 0 or src_h <= 0:
            return
        sx, sy = panel.shape[1] / float(src_w), panel.shape[0] / float(src_h)
        palette = (
            (70, 210, 70), (235, 165, 45), (210, 80, 210), (60, 190, 235),
            (235, 95, 70), (185, 210, 55), (220, 125, 45), (100, 130, 245),
        )

        def detection_color(det: dict[str, Any]) -> tuple[int, int, int]:
            label = str(det.get("semantic_class", det.get("class", "?")))
            return palette[sum(ord(char) for char in label) % len(palette)]

        # Fill every retained boundary per instance. Sparse points are kept as
        # a compatibility fallback, but the plural polygon field mirrors the
        # unified mask used by 3-D segmentation and box estimation.
        for det in detections if include_masks else ():
            display_mask = np.zeros(panel.shape[:2], dtype=np.uint8)
            polygons = det.get("rgb_mask_polygons") or det.get("mask_polygons")
            if not isinstance(polygons, list) or not polygons:
                legacy_polygon = det.get("mask_polygon")
                polygons = [legacy_polygon] if isinstance(legacy_polygon, list) else []
            scaled_polygons: list[np.ndarray] = []
            for polygon in polygons:
                if not isinstance(polygon, list) or len(polygon) < 3:
                    continue
                points = np.asarray(polygon, dtype=np.float32).reshape((-1, 2))
                points[:, 0] = np.clip(points[:, 0] * sx, 0, panel.shape[1] - 1)
                points[:, 1] = np.clip(points[:, 1] * sy, 0, panel.shape[0] - 1)
                scaled_polygons.append(np.rint(points).astype(np.int32))
            if scaled_polygons:
                cv2.fillPoly(display_mask, scaled_polygons, 255)
            else:
                mask = det.get("mask")
                if not isinstance(mask, dict):
                    continue
                rows = np.asarray(mask.get("rows") or [], dtype=np.int32)
                cols = np.asarray(mask.get("cols") or [], dtype=np.int32)
                if rows.size == 0 or rows.size != cols.size:
                    continue
                valid = (rows >= 0) & (rows < src_h) & (cols >= 0) & (cols < src_w)
                rows, cols = rows[valid], cols[valid]
                if rows.size == 0:
                    continue
                ys = np.clip(np.rint(rows * sy).astype(np.int32), 0, panel.shape[0] - 1)
                xs = np.clip(np.rint(cols * sx).astype(np.int32), 0, panel.shape[1] - 1)
                display_mask[ys, xs] = 255
                sampling_ratio = max(1.0, _safe_float(det.get("mask_area"), rows.size) / rows.size)
                kernel_size = min(9, max(3, int(round(math.sqrt(sampling_ratio))) | 1))
                display_mask = cv2.morphologyEx(
                    display_mask,
                    cv2.MORPH_CLOSE,
                    np.ones((kernel_size, kernel_size), dtype=np.uint8),
                    iterations=2,
                )
                contours, _hierarchy = cv2.findContours(
                    display_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                display_mask.fill(0)
                if contours:
                    cv2.fillPoly(display_mask, contours, 255)
            active = display_mask > 0
            color = np.asarray(detection_color(det), dtype=np.float32)
            panel[active] = np.clip(
                panel[active].astype(np.float32) * 0.70 + color * 0.30,
                0,
                255,
            ).astype(np.uint8)

        for det in detections:
            box = det.get("bbox") or det.get("bbox_2d")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(value) for value in box]
            except (TypeError, ValueError):
                continue
            x1, x2 = sorted((max(0.0, min(src_w - 1.0, x1)), max(0.0, min(src_w - 1.0, x2))))
            y1, y2 = sorted((max(0.0, min(src_h - 1.0, y1)), max(0.0, min(src_h - 1.0, y2))))
            confidence = _safe_float(det.get("confidence"), 0.0)
            color = detection_color(det)
            p1, p2 = (int(round(x1 * sx)), int(round(y1 * sy))), (int(round(x2 * sx)), int(round(y2 * sy)))
            cv2.rectangle(panel, p1, p2, color, 2, cv2.LINE_AA)
            if include_labels:
                label = f"{det.get('semantic_class', det.get('class', '?'))} {confidence:.2f}"
                cv2.putText(panel, label[:34], (p1[0], max(48, p1[1] - 7)), cv2.FONT_HERSHEY_SIMPLEX, .58, color, 2, cv2.LINE_AA)

    def render_camera_overlay(self, *, include_masks: bool = True) -> bytes:
        """Render detections directly at the D435i source resolution."""
        if cv2 is None:
            raise RuntimeError("physical viewer requires opencv-python and numpy")
        with self.state._lock:
            rgb = None if self.state.rgb is None else self.state.rgb.copy()
            detections = [dict(item) for item in self.state.detections if isinstance(item, dict)]
        if rgb is None:
            rgb = np.full((480, 640, 3), 25, dtype=np.uint8)
            cv2.putText(rgb, "NO RGB YET", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)
        self._draw_live_detections(
            rgb,
            detections,
            (rgb.shape[0], rgb.shape[1]),
            include_masks=include_masks,
            include_labels=True,
        )
        # Quality 92 preserves small labels while avoiding the high CPU and
        # large payload spikes of JPEG quality 100 on every 5 Hz refresh.
        ok, encoded = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("camera overlay JPEG encoding failed")
        return bytes(encoded)

    @staticmethod
    def _display_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        """Give physical subgoals a readable label without changing identity."""
        display = dict(candidate)
        metadata = display.get("metadata") if isinstance(display.get("metadata"), dict) else {}
        family = str(metadata.get("semantic_name") or metadata.get("node_type") or "").casefold()
        if family in {"portal", "doorway", "entrance", "gate"}:
            family = "door"
        raw_name = str(display.get("target_name") or "")
        if family and (not raw_name or "track_" in raw_name.casefold()):
            stable_id = str(display.get("target_id") or raw_name)
            suffix = stable_id.rsplit("_", 1)[-1]
            display["target_name"] = f"{family} #{suffix}" if suffix.isdigit() else family
        return display

    @staticmethod
    def _physical_step(snapshot: dict[str, Any]) -> dict[str, Any]:
        telemetry = snapshot.get("telemetry") if isinstance(snapshot.get("telemetry"), dict) else {}
        position = telemetry.get("map_position") or telemetry.get("position") or [0.0, 0.0, 0.0]
        try:
            pose = [float(position[0]), float(position[1]), float(telemetry.get("map_yaw", telemetry.get("yaw", 0.0)) or 0.0)]
        except (IndexError, TypeError, ValueError):
            pose = [0.0, 0.0, 0.0]
        graph = snapshot.get("graph") if isinstance(snapshot.get("graph"), dict) else {}
        navigation = snapshot.get("navigation") if isinstance(snapshot.get("navigation"), dict) else {}
        current = navigation.get("current_subgoal") if isinstance(navigation.get("current_subgoal"), dict) else {}
        selected = navigation.get("selection") if isinstance(navigation.get("selection"), dict) else {}
        selected = SixPanelRenderer._display_candidate(selected)
        decision_trace = navigation.get("decision_trace") if isinstance(navigation.get("decision_trace"), dict) else {}
        pending = (
            decision_trace.get("pending_post_interaction_traversal")
            if isinstance(
                decision_trace.get("pending_post_interaction_traversal"), dict
            )
            else {}
        )
        terminal = selected.get("terminal") if isinstance(selected.get("terminal"), dict) else {}
        if (
            selected.get("active") is False
            and pending
            and str(terminal.get("candidate_id") or "")
            != str(pending.get("candidate_id") or "")
        ):
            selected = SixPanelRenderer._display_candidate(pending)
            selected["active"] = True
            selected["selection_state"] = str(
                decision_trace.get("phase") or "post_interaction_refresh"
            )
        candidates = dict(navigation.get("candidates")) if isinstance(navigation.get("candidates"), dict) else {}
        candidates["candidates"] = [
            SixPanelRenderer._display_candidate(item)
            for item in candidates.get("candidates", [])
            if isinstance(item, dict)
        ]
        goal = current.get("point") or current.get("position") or current.get("goal") or []
        if not isinstance(goal, (list, tuple)) or len(goal) < 2:
            goal = []
        if goal:
            selected = dict(selected)
            selected.setdefault(
                "goal_xyyaw",
                [
                    float(goal[0]),
                    float(goal[1]),
                    _safe_float(current.get("yaw", current.get("theta", 0.0))),
                ],
            )
            selected.setdefault("active", True)
        observed = []
        for node in graph.get("nodes", []):
            if not isinstance(node, dict):
                continue
            observed.extend(
                str(value)
                for value in (
                    node.get("id"), node.get("name"), node.get("object_id"),
                    (node.get("attributes") or {}).get("instance_id"),
                )
                if value not in {None, ""}
            )
        return {
            "step_index": int(snapshot.get("navigation_step", 0) or 0),
            "stamp_sec": float(snapshot.get("frame_stamp", 0.0) or 0.0),
            "pose": pose,
            "pose_frame_id": "tf_frame_map",
            "active_goal": list(goal[:2]) if goal else [],
            "active_goal_yaw": _safe_float(current.get("yaw", current.get("theta", 0.0))),
            "distance_m": 0.0,
            "trajectory": [pose + [float(snapshot.get("frame_stamp", 0.0) or 0.0)]],
            "global_plan": navigation.get("global_plan") or {},
            "local_global_plan": navigation.get("local_global_plan") or {},
            "local_plan": navigation.get("local_plan") or {},
            "unified_graph": graph,
            "physical_detections": list(snapshot.get("detections") or []),
            "observed_instance_ids": sorted(set(observed)),
            "semantic_candidates": candidates,
            "semantic_selection": selected or {"active": False},
            "semantic_execution_state": navigation.get("execution_state") or {},
            "semantic_behavior_feedback": navigation.get("behavior_feedback") or {},
            "semantic_decision_trace": decision_trace,
        }

    def _placeholder(self, title: str) -> Any:
        panel = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        panel[:, :, :] = (32, 32, 32)
        cv2.putText(panel, title, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
        return panel

    def _image_panel(self, image: Any, title: str) -> Any:
        if image is None:
            return self._placeholder(title)
        if image.ndim == 2:
            valid = image > 0
            if np.any(valid):
                norm = np.zeros_like(image, dtype=np.uint8)
                clipped = np.clip(image.astype(np.float32), 0, np.percentile(image[valid], 98))
                norm = (clipped / max(1.0, float(clipped.max())) * 255).astype(np.uint8)
                image = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
            else:
                image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        panel = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        cv2.putText(panel, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
        return panel

    def _detection_panel(self, rgb: Any, detections: list[dict[str, Any]]) -> Any:
        panel = self._image_panel(rgb, f"3 Original M1 / YOLOE detections | n={len(detections)}")
        if rgb is None:
            return panel
        sx, sy = self.width / rgb.shape[1], self.height / rgb.shape[0]
        for det in detections:
            box = det.get("bbox") or det.get("bbox_2d")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(box)]
            color = (0, 255, 0) if float(det.get("confidence", 0)) >= .5 else (0, 165, 255)
            cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
            text = f"{det.get('semantic_class', det.get('class', '?'))} {float(det.get('confidence', 0)):.2f}"
            cv2.putText(panel, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1)
        return panel

    def _depth_panel(self, depth: Any, depth_scale: float) -> Any:
        if depth is None:
            return self._placeholder("2 Public depth | clearance=N/A")
        valid = np.asarray(depth) > 0
        clearance = 0.0
        if np.any(valid):
            center = np.asarray(depth)[..., max(0, np.asarray(depth).shape[1] // 2 - 8): np.asarray(depth).shape[1] // 2 + 8]
            center_valid = center > 0
            if np.any(center_valid):
                clearance = float(np.median(center[center_valid])) * float(depth_scale)
        return self._image_panel(depth, f"2 Public depth | clearance={clearance:.2f}m")

    def _json_panel(self, title: str, value: Any) -> Any:
        panel = self._placeholder(title)
        lines = json.dumps(value, ensure_ascii=False, indent=2, default=str).splitlines()[:16]
        for i, line in enumerate(lines):
            cv2.putText(panel, line[:78], (12, 62 + i * 19), cv2.FONT_HERSHEY_PLAIN, 1.0, (220, 220, 220), 1)
        return panel

    def _map_panel(self, occupancy: Any, telemetry: dict[str, Any]) -> Any:
        if not isinstance(occupancy, dict) or not occupancy.get("data"):
            return self._json_panel("4 Public RGB-D occupancy / route", telemetry)
        panel = np.full((self.height, self.width, 3), 127, dtype=np.uint8)
        width, height = int(occupancy.get("width", 0)), int(occupancy.get("height", 0)); values = np.asarray(occupancy.get("data", []), dtype=np.int16)
        if width > 0 and height > 0 and values.size >= width * height:
            grid = values[:width * height].reshape(height, width); image = np.full((height, width, 3), 127, dtype=np.uint8)
            image[grid == 0] = (235, 235, 235); image[grid > 50] = (30, 30, 30)
            panel = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        cv2.putText(panel, "4 Public RGB-D occupancy / route | no-control", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
        pose = telemetry.get("position", [0, 0, 0]); cv2.putText(panel, f"pose={pose[:2]} yaw={telemetry.get('yaw', '?')}", (12, self.height - 15), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 100, 255), 1)
        return panel

    def _graph_panel(self, graph: dict[str, Any]) -> Any:
        panel = np.full((self.height, self.width, 3), 35, dtype=np.uint8); nodes = graph.get("nodes", []) if isinstance(graph, dict) else []; edges = graph.get("edges", []) if isinstance(graph, dict) else []
        positions = {}
        for index, node in enumerate(nodes):
            value = node.get("world_position") or node.get("position") or node.get("centroid") or node.get("aabb_center") or node.get("center") or {}
            point = _point_xy(value)
            x, y = point if point is not None else (float(index % 5), float(index // 5))
            positions[str(node.get("id", node.get("node_id", index)))] = (x, y)
        if positions:
            xs, ys = [p[0] for p in positions.values()], [p[1] for p in positions.values()]; minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
            def point(x: float, y: float) -> tuple[int, int]: return (int(40 + (x - minx) / max(1e-6, maxx - minx) * (self.width - 80)), int(self.height - 40 - (y - miny) / max(1e-6, maxy - miny) * (self.height - 80)))
            for edge in edges:
                a, b = positions.get(str(edge.get("source", edge.get("from", edge.get("src_id", ""))))), positions.get(str(edge.get("target", edge.get("to", edge.get("dst_id", "")))))
                if a and b: cv2.line(panel, point(*a), point(*b), (110, 110, 110), 1)
            for node in nodes:
                key = str(node.get("id", node.get("node_id", nodes.index(node)))); xy = positions.get(key); 
                if not xy: continue
                kind = str(node.get("type", node.get("node_type", "object"))); color = {"room": (255, 160, 30), "portal": (30, 220, 255), "container": (180, 80, 220), "support": (200, 200, 60)}.get(kind, (60, 220, 80)); px = point(*xy); cv2.circle(panel, px, 7, color, -1); cv2.putText(panel, _node_label(node)[:18], (px[0] + 8, px[1]), cv2.FONT_HERSHEY_PLAIN, .9, color, 1)
        cv2.putText(panel, f"5 Original semantic graph / room context | nodes={len(nodes)} edges={len(edges)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 255, 255), 2)
        return panel

    def _topdown_panel(self, occupancy: Any, graph: dict[str, Any], telemetry: dict[str, Any], consistency: dict[str, Any]) -> Any:
        """Physical replacement for the canonical six-panel topdown/GT view.

        The simulator's panel 6 is explicitly posthoc GT.  A real Go2 has no
        GT target stream, so this panel keeps the same slot and layout while
        showing the public map-frame audit: occupancy, robot pose and mapped
        semantic nodes. Detailed consistency metrics remain in the side panel.
        """
        if not isinstance(occupancy, dict) or not occupancy.get("data"):
            return self._json_panel("6 Physical topdown / map audit", consistency)
        width, height = int(occupancy.get("width", 0)), int(occupancy.get("height", 0))
        values = np.asarray(occupancy.get("data", []), dtype=np.int16)
        if width <= 0 or height <= 0 or values.size < width * height:
            return self._json_panel("6 Physical topdown / map audit", consistency)
        grid = values[: width * height].reshape(height, width)
        canvas = np.full((height, width, 3), 127, dtype=np.uint8)
        canvas[grid == 0] = (235, 235, 235)
        canvas[grid > 50] = (25, 25, 25)
        origin = occupancy.get("origin") if isinstance(occupancy.get("origin"), dict) else {}
        resolution = float(occupancy.get("resolution", 0.05) or 0.05)
        ox, oy = float(origin.get("x", 0.0) or 0.0), float(origin.get("y", 0.0) or 0.0)

        def world_pixel(value: Any) -> tuple[int, int] | None:
            point = _point_xy(value)
            if point is None or resolution <= 0:
                return None
            px = int(round((point[0] - ox) / resolution))
            py = int(round(height - 1 - (point[1] - oy) / resolution))
            return (px, py) if 0 <= px < width and 0 <= py < height else None

        robot = world_pixel(telemetry.get("position"))
        if robot is not None:
            cv2.circle(canvas, robot, max(3, min(width, height) // 90), (0, 0, 255), -1)
            yaw = _safe_float(telemetry.get("yaw"))
            tip = (int(robot[0] + 18 * math.cos(yaw)), int(robot[1] - 18 * math.sin(yaw)))
            cv2.arrowedLine(canvas, robot, tip, (255, 80, 0), 2, tipLength=0.3)
        for node in graph.get("nodes", []) if isinstance(graph, dict) else []:
            if not isinstance(node, dict):
                continue
            point = world_pixel(node.get("centroid") or node.get("aabb_center") or node.get("world_position") or node.get("position"))
            if point is None:
                continue
            kind = str(node.get("type", node.get("node_type", "object")))
            color = {"room": (255, 160, 30), "portal": (30, 220, 255), "container": (180, 80, 220)}.get(kind, (60, 190, 80))
            if kind == "portal":
                center = node.get("aabb_center") or node.get("centroid") or []
                size = node.get("aabb_size") or []
                attrs = node.get("attributes") if isinstance(node.get("attributes"), dict) else {}
                orientation = attrs.get("orientation") or node.get("orientation")
                yaw = attrs.get("yaw")
                if isinstance(orientation, (list, tuple)) and len(orientation) >= 4:
                    try:
                        yaw = math.atan2(
                            2.0 * float(orientation[3]) * float(orientation[2]),
                            1.0 - 2.0 * float(orientation[2]) * float(orientation[2]),
                        )
                    except (TypeError, ValueError):
                        yaw = None
                try:
                    if len(center) >= 2 and len(size) >= 2 and yaw is not None:
                        half_x, half_y = abs(float(size[0])) * 0.5, abs(float(size[1])) * 0.5
                        cos_yaw, sin_yaw = math.cos(float(yaw)), math.sin(float(yaw))
                        corners = []
                        for local_x, local_y in ((-half_x, -half_y), (-half_x, half_y), (half_x, half_y), (half_x, -half_y)):
                            world_x = float(center[0]) + cos_yaw * local_x - sin_yaw * local_y
                            world_y = float(center[1]) + sin_yaw * local_x + cos_yaw * local_y
                            corners.append(world_pixel([world_x, world_y, 0.0]))
                        if all(corner is not None for corner in corners):
                            cv2.polylines(canvas, [np.asarray(corners, dtype=np.int32)], True, color, max(2, min(width, height) // 180))
                            continue
                except (TypeError, ValueError, IndexError):
                    pass
            cv2.circle(canvas, point, max(2, min(width, height) // 140), color, -1)
        panel = cv2.resize(canvas, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        status = str(consistency.get("status", "waiting")) if isinstance(consistency, dict) else "waiting"
        counts = consistency.get("counts", {}) if isinstance(consistency, dict) else {}
        cv2.putText(panel, f"6 Physical topdown / map audit | consistency={status}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .58, (0, 255, 255), 2)
        cv2.putText(panel, f"pass={counts.get('pass', 0)} warn={counts.get('warn', 0)} fail={counts.get('fail', 0)}", (12, self.height - 15), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 180, 255), 1)
        return panel

    def render(self) -> bytes:
        if cv2 is None:
            raise RuntimeError("physical viewer requires opencv-python and numpy")
        navigation_step = self.state.advance_navigation_step()
        with self.state._lock:
            snapshot = {
                "frame_seq": self.state.frame_seq,
                "navigation_step": navigation_step,
                "frame_stamp": self.state.frame_stamp,
                "telemetry": dict(self.state.telemetry),
                "graph": dict(self.state.graph),
                "consistency": dict(self.state.consistency),
                "navigation": dict(self.state.navigation),
                "detections": [dict(item) for item in self.state.detections if isinstance(item, dict)],
            }
            rgb = None if self.state.rgb is None else self.state.rgb.copy()
            planning = self._raw_grid(self.state.occupancy)
            room = self._raw_grid(self.state.room_grid)
            global_grid = self._raw_grid(self.state.global_costmap)
            local_grid = self._raw_grid(self.state.local_costmap)
        step = self._physical_step(snapshot)
        if planning is None:
            global_grid = global_grid or planning
            local_grid = local_grid or planning
        else:
            global_grid = global_grid or planning
            local_grid = local_grid or planning
        # Crop the OCC view to the currently known cells.  The SLAM grid is
        # preallocated much larger than the explored area (often <2% known),
        # so using a multi-metre margin makes panel 2 mostly unknown space.
        world_bounds = known_world_bounds(planning, margin_m=2.5) if planning is not None else None
        if self._occ_view_bounds is None and world_bounds is not None:
            pose = step.get("pose") or []
            try:
                cx, cy = float(pose[0]), float(pose[1])
            except (TypeError, ValueError, IndexError):
                cx = (world_bounds[0] + world_bounds[2]) * 0.5
                cy = (world_bounds[1] + world_bounds[3]) * 0.5
            half_w = max(1.0, (world_bounds[2] - world_bounds[0]) * 0.5)
            half_h = max(1.0, (world_bounds[3] - world_bounds[1]) * 0.5)
            self._occ_view_bounds = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
        # Keep the OCC viewport aligned with the current explored extent. A
        # startup-locked viewport can leave the robot/path outside the panel
        # after the map grows or the origin shifts.
        occ_view_bounds = world_bounds
        width, height = self.panel_size
        if rgb is None:
            camera = np.full((height, width, 3), 235, dtype=np.uint8)
            cv2.putText(camera, "NO RGB YET", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .9, (80, 80, 80), 2, cv2.LINE_AA)
        else:
            camera = cv2.resize(rgb, self.panel_size, interpolation=cv2.INTER_AREA)
        draw_camera_title(camera, step, int(step["step_index"]))
        # The canonical six-panel camera stays box-only. The separate right
        # side enlargement uses the source-resolution box+seg endpoint.
        self._draw_live_detections(camera, snapshot["detections"], None if rgb is None else (rgb.shape[0], rgb.shape[1]), include_masks=False)
        occ = self._canonical.render_map_panel(
            planning, self.panel_size, step, int(step["step_index"]),
            title="OCC", kind="occupancy", world_bounds=occ_view_bounds,
            draw_global_plan=True, draw_local_plan=False, draw_frontiers=True,
            draw_semantic_candidates=True, draw_route_plan=False,
            draw_interaction_target_links=True,
            # Match panel 3 (room segments) so OCC and room views use the
            # same viewport framing and can be compared directly.
            # Panel 2 should show a slightly wider OCC context than the
            # room/costmap panels; lower scale means less zoom and more area.
            view_scale=1.45,
        )
        draw_task_subgoal_header(occ, step, box_width_px=width // 2 - 10, background_alpha=.55)
        # Showcase spatial-understanding panel should show room geometry and
        # the global route only; suppress the two exploratory subgoal markers
        # that belong to the debug/navigation view.
        room_step = dict(step)
        room_step["current_subgoal"] = None
        room_step["selected_subgoal"] = None
        room_step["semantic_candidates"] = {"candidates": []}
        # OfflineSixPanelRenderer reads the canonical semantic fields rather
        # than the convenience keys above. Mark the selection inactive and
        # clear executor geometry so panel 03 cannot draw either selected or
        # alternate subgoal markers.
        room_step["semantic_selection"] = {"active": False}
        room_step["semantic_execution_state"] = {}
        room_panel = self._canonical.render_room_panel(
            planning, room, self.panel_size, room_step, int(step["step_index"]), world_bounds,
            view_scale=1.75, draw_global_plan=True,
        )
        global_grid = self._global_costmap_for_display(planning, global_grid)
        global_width = width // 2
        global_panel = self._canonical.render_map_panel(
            global_grid, (global_width, height), step, int(step["step_index"]),
            title="GLOBAL COSTMAP", kind="costmap", world_bounds=None,
            draw_global_plan=True, draw_local_plan=False, draw_frontiers=True,
            draw_semantic_candidates=True,
            # Keep panel 4 framed identically to panel 3 (room segments) and
            # panel 2 (OCC); only the raster values differ.
            view_scale=1.75,
        )
        local_panel = self._canonical.render_map_panel(
            local_grid, (width - global_width, height), step, int(step["step_index"]),
            title="LOCAL COSTMAP", kind="costmap", draw_global_plan=False,
            draw_local_global_plan=True, draw_local_plan=True, draw_frontiers=False,
            draw_semantic_candidates=False,
        )
        if local_grid is not None:
            # Full local grid is square and letterboxed into this half-panel.
            # Keep the scale bar on the right, away from the cost legend.
            rendered_side = min(width - global_width, height)
            extent_m = max(local_grid.width * local_grid.resolution, 1e-6)
            metre_px = max(8, int(round(rendered_side / extent_m)))
            bar_right, bar_y = local_panel.shape[1] - 10, local_panel.shape[0] - 14
            bar_left = max(local_panel.shape[1] // 2, bar_right - metre_px)
            cv2.line(local_panel, (bar_left, bar_y), (bar_right, bar_y), (20, 20, 20), 2, cv2.LINE_AA)
            cv2.line(local_panel, (bar_left, bar_y - 4), (bar_left, bar_y + 4), (20, 20, 20), 2, cv2.LINE_AA)
            cv2.line(local_panel, (bar_right, bar_y - 4), (bar_right, bar_y + 4), (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(local_panel, "1 m", (bar_left, bar_y - 7), cv2.FONT_HERSHEY_PLAIN, .8, (20, 20, 20), 1, cv2.LINE_AA)
        costmaps = np.concatenate([global_panel, local_panel], axis=1)
        spatial = self._canonical.render_semantic_xy(
            planning, self.panel_size, step, int(step["step_index"]), world_bounds,
            view_scale=1.8, label_mode="all", draw_overview_inset=False,
        )
        topology_step = dict(step)
        topology_candidates = dict(step.get("semantic_candidates") or {})
        raw_topology_candidates = list(topology_candidates.get("candidates") or [])
        allowed_topology_families = ("door", "portal", "gate", "fridge", "refrigerator")
        topology_candidates["candidates"] = [
            candidate
            for candidate in raw_topology_candidates
            if str(candidate.get("behavior_type") or candidate.get("type") or "").upper() != "INTERACT"
            or any(
                marker in " ".join(
                    str(value or "").casefold()
                    for value in (
                        candidate.get("target_name"),
                        (candidate.get("metadata") or {}).get("semantic_name"),
                        (candidate.get("metadata") or {}).get("node_type"),
                    )
                )
                for marker in allowed_topology_families
            )
        ]
        topology_step["semantic_candidates"] = topology_candidates
        # Keep panel 6's interaction layer aligned with the physical policy:
        # generic detector boxes are useful evidence in panels 1/5, but they
        # are not interaction targets.  Preserve rooms/ordinary objects and
        # retain only door/portal and fridge containers in the topology's
        # interaction layer. Cabinets/drawers remain detector evidence only.
        # Panel 6 is a historical semantic interaction graph, not a current
        # camera-view graph.  Keep every accumulated node and edge (including
        # ordinary objects and previously observed containers); only the
        # interaction-candidate list above is policy-filtered.
        topology_graph = step.get("unified_graph")
        if isinstance(topology_graph, dict):
            topology_step["unified_graph"] = topology_graph
        topology = self._canonical.render_topology(self.panel_size, topology_step, int(step["step_index"]))
        original_panels: dict[int, bytes] = {}
        for panel_index, panel in ((3, room_panel), (6, topology)):
            panel_ok, panel_encoded = cv2.imencode(
                ".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 96]
            )
            if panel_ok:
                original_panels[panel_index] = bytes(panel_encoded)
        self.latest_original_panels = original_panels
        if self.capture_panel_streams:
            panel_streams: dict[int, bytes] = {}
            for panel_index, panel in (
                (1, camera),
                (2, occ),
                (3, room_panel),
                (4, costmaps),
            ):
                panel_ok, panel_encoded = cv2.imencode(
                    ".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                if panel_ok:
                    panel_streams[panel_index] = bytes(panel_encoded)
            self.latest_panel_streams = panel_streams
        else:
            self.latest_panel_streams = {}
        canvas = np.vstack([
            np.concatenate([camera, occ, room_panel], axis=1),
            np.concatenate([costmaps, spatial, topology], axis=1),
        ])
        ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError("six-panel JPEG encoding failed")
        return bytes(encoded)


class _WebHandler(BaseHTTPRequestHandler):
    # Browsers can reuse one TLS connection for the 5 Hz JPEG stream and the
    # audio packets.  HTTP/1.0 forced a new handshake/thread for every packet
    # and made the phone publisher stall after audio was enabled.
    protocol_version = "HTTP/1.1"
    state: RuntimeState
    renderer: SixPanelRenderer
    gate: ReadOnlySafetyGate
    recorder: PhysicalRawRecorder
    qwen_submit = None
    frame_lock = threading.Lock()
    latest_jpeg: bytes = b""
    latest_camera_jpeg: bytes = b""
    latest_camera_box_jpeg: bytes = b""
    latest_panel3_jpeg: bytes = b""
    latest_panel6_jpeg: bytes = b""
    phone_lock = threading.Lock()
    latest_phone_jpeg: bytes = b""
    latest_phone_at = 0.0
    latest_phone_seq = 0
    latest_phone_client = ""
    latest_phone_fps = 10
    phone_audio_packets: deque[tuple[int, int, bytes]] = deque(maxlen=128)
    latest_phone_audio_seq = 0
    latest_phone_audio_at = 0.0
    latest_phone_audio_rate = 48_000
    phone_stream_url = ""
    control_lock = threading.Lock()
    control_process: subprocess.Popen[str] | None = None
    control_action = ""
    control_goal = ""
    control_started_at = 0.0
    control_log = ""

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _json(self, value: Any, status: int = 200) -> None:
        data = json.dumps(value, ensure_ascii=False, default=str).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def _read_json_payload(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            length = 0
        if length < 0 or length > 16 * 1024 * 1024:
            raise ValueError("JSON payload too large")
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("JSON payload must be an object")
        return payload

    def _phone_stream_url(self) -> str:
        configured = str(self.phone_stream_url or "").strip()
        if configured:
            return configured
        host = str(self.headers.get("Host") or "10.100.5.3:8765").strip()
        if host.split(":", 1)[0].casefold() in {"127.0.0.1", "localhost", "0.0.0.0"}:
            host = "10.100.5.3:8765"
        if not host or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-:[]" for character in host):
            host = "10.100.5.3:8765"
        return f"http://{host}/phone-stream"

    @classmethod
    def _phone_snapshot(cls) -> dict[str, Any]:
        with cls.phone_lock:
            age = max(0.0, time.time() - cls.latest_phone_at) if cls.latest_phone_at else None
            return {
                "connected": bool(cls.latest_phone_jpeg and age is not None and age <= 3.0),
                "frame_seq": cls.latest_phone_seq,
                "last_frame_at": cls.latest_phone_at,
                "age_s": age,
                "client": cls.latest_phone_client,
                "target_fps": cls.latest_phone_fps,
                "audio_connected": bool(cls.latest_phone_audio_at and time.time() - cls.latest_phone_audio_at <= 3.0),
                "audio_seq": cls.latest_phone_audio_seq,
                "audio_rate": cls.latest_phone_audio_rate,
            }

    def _recording_credentials(self) -> tuple[str, str]:
        """Read optional session credentials used by the phone publisher."""
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        session_id = str(
            self.headers.get("X-Recording-Session")
            or (query.get("recording_session") or [""])[0]
            or ""
        )
        token = str(
            self.headers.get("X-Recording-Token")
            or (query.get("recording_token") or [""])[0]
            or ""
        )
        return session_id, token

    def _recording_allowed(self) -> bool:
        recorder = getattr(self, "recorder", None)
        if recorder is None:
            return False
        session_id, token = self._recording_credentials()
        # Existing phone clients remain compatible when no session is active;
        # credentials are checked whenever a caller supplies them.
        return not recorder.is_active() or recorder.authorize(session_id, token)

    @classmethod
    def _control_snapshot(cls) -> dict[str, Any]:
        with cls.control_lock:
            process = cls.control_process
            running = process is not None and process.poll() is None
            return {"running": running, "pid": process.pid if running else None,
                    "action": cls.control_action, "goal": cls.control_goal,
                    "started_at": cls.control_started_at, "log": cls.control_log}

    @classmethod
    def _run_control(cls, action: str, goal: str) -> dict[str, Any]:
        if action == "stop":
            # Stop only the policy/control transport.  The ROS navigation and
            # web gateway stay alive; the Go2 bridge zeros its command on
            # websocket disconnect/TTL expiry and Start/Restart can recreate
            # the policy server later.
            pid_file = Path(os.environ.get(
                "PHYSICAL_NAV_POLICY_PID_FILE",
                f"/tmp/molmospaces-physical-nav-all-{os.getuid()}/policy_control.pid",
            ))
            try:
                pid = int(pid_file.read_text().strip())
            except (OSError, ValueError):
                return {"accepted": True, "action": "stop", "control_only": True,
                        "message": "控制服务已停止或尚未运行"}
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            except OSError:
                cmdline = ""
            if "policy_control_server.py" not in cmdline:
                return {"accepted": False, "error": "控制服务 PID 无效，未执行停止", "control_only": True}
            try:
                os.kill(pid, 15)
            except OSError as exc:
                return {"accepted": False, "error": f"控制服务停止失败: {exc}", "control_only": True}
            return {"accepted": True, "action": "stop", "control_only": True,
                    "message": "已中断控制，网页和导航栈保持运行", "pid": pid}
        script = Path(__file__).resolve().with_name("physical_nav_all.sh")
        args = ["bash", str(script), "start_control", "enable_motion"] if action == "enable_motion" else ["bash", str(script), action]
        if action in {"start", "restart"}:
            args.append("enable_motion")
            if goal:
                args.append(goal)
        log_dir = Path(os.environ.get("PHYSICAL_NAV_ALL_LOG_DIR", f"/tmp/molmospaces-physical-nav-all-{os.getuid()}/logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "web_control.log"
        with cls.control_lock:
            if cls.control_process is not None and cls.control_process.poll() is None:
                process = cls.control_process
                return {"accepted": False, "error": "已有操控命令正在执行",
                        "running": True, "pid": process.pid, "action": cls.control_action,
                        "goal": cls.control_goal, "started_at": cls.control_started_at,
                        "log": cls.control_log}
            log_file = log_path.open("ab")
            process = subprocess.Popen(args, cwd=str(script.parent.parent.parent.parent),
                                       stdout=log_file, stderr=subprocess.STDOUT,
                                       start_new_session=True, text=False)
            cls.control_process, cls.control_action, cls.control_goal = process, action, goal
            cls.control_started_at, cls.control_log = time.time(), str(log_path)
            threading.Thread(target=lambda: (process.wait(), log_file.close()), daemon=True).start()
            return {"accepted": True, "pid": process.pid, "action": action, "goal": goal, "log": str(log_path)}

    @staticmethod
    def _state_summary(snapshot: dict[str, Any], *, include_ros_clouds: bool = False) -> dict[str, Any]:
        """Return a browser-sized view of state (full masks stay on /api/state)."""
        detection_meta = snapshot.get("detection_meta")
        compact_detection_meta = (
            {
                key: detection_meta.get(key)
                for key in (
                    "seq",
                    "stamp",
                    "camera_frame",
                    "width",
                    "height",
                    "count",
                    "inference_ms",
                    "cycle_ms",
                    "timings_ms",
                )
                if detection_meta.get(key) is not None
            }
            if isinstance(detection_meta, dict)
            else {}
        )
        def detection_view(item: Any) -> dict[str, Any]:
            if not isinstance(item, dict):
                return {"value": item}
            view = {
                "semantic_class": item.get("semantic_class", item.get("raw_class", "")),
                "semantic_class_raw": item.get("semantic_class_raw"),
                "raw_class": item.get("raw_class"),
                "confidence": item.get("confidence"),
                "bbox": item.get("bbox"),
                "depth_median_m": item.get("depth_median_m"),
                # The ROS gateway needs the sensor-frame geometry to apply
                # the same TF as segmented_cloud_world.  Omitting these
                # fields forces it to fall back to the worker's legacy,
                # telemetry-frame world coordinates.
                "camera_position": item.get("camera_position"),
                "camera_box3d_center": item.get("camera_box3d_center"),
                "camera_box3d_size": item.get("camera_box3d_size"),
                "position": item.get("world_position", item.get("position")),
                "world_position": item.get("world_position"),
                "world_box3d_center": item.get("world_box3d_center"),
                "world_box3d_size": item.get("world_box3d_size"),
                "world_box3d_orientation": item.get("world_box3d_orientation"),
                # Preserve the fitted OBB yaw through the compact ROS-state
                # endpoint.  The gateway must pass this to semantic mapping;
                # otherwise the portal path sees yaw=None and reconstructs a
                # default yaw=0, which can rotate the interaction normal by
                # 90 degrees for doors whose long edge is along Y.
                "yaw": item.get("yaw"),
                "world_box3d_yaw": item.get("world_box3d_yaw", item.get("yaw")),
                "world_box3d_marker_size": item.get("world_box3d_marker_size"),
                "aabb_center": item.get("aabb_center"),
                "aabb_size": item.get("aabb_size"),
                "source_frame": item.get("source_frame"),
                "capture_seq": item.get("capture_seq"),
            }
            if include_ros_clouds:
                # Already capped per instance and packed as float32 base64.
                # Keep these off the browser's 5 Hz summary endpoint.
                view.update({
                    "segment_point_count": item.get("segment_point_count"),
                    "camera_segment_points_f32": item.get("camera_segment_points_f32"),
                    "world_segment_points_f32": item.get("world_segment_points_f32"),
                })
            return view

        def candidate_view(item: Any) -> dict[str, Any]:
            if not isinstance(item, dict):
                return {"id": str(item)}
            return {
                key: item.get(key)
                for key in ("id", "candidate_id", "target_name", "action", "subject_type")
                if item.get(key) not in (None, "")
            }

        def event_view(item: dict[str, Any]) -> dict[str, Any]:
            # Instructions, image references and full candidate scoring traces
            # can be tens of kilobytes per call.  The dashboard only needs the
            # compact card fields below; /api/state retains the complete trace.
            compact = {
                key: item.get(key)
                for key in (
                    "event_type", "stage", "module", "timestamp", "role", "model",
                    "episode_id", "object_id", "target_id", "target_name", "target_kind",
                    "request_sequence", "latency_s", "error", "raw_text",
                    "sample_period_s", "next_call_at", "deadline_at", "phase", "result",
                    "m1_input_image_key", "m1_input_bbox", "m1_input_label",
                    "m1_input_track_id", "m1_input_object_id",
                )
                if item.get(key) not in (None, "")
            }
            context = item.get("context") if isinstance(item.get("context"), dict) else {}
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            candidates = context.get("candidates", payload.get("candidates", item.get("candidate_options", [])))
            candidates = candidates if isinstance(candidates, list) else []
            mission = context.get("mission", payload.get("mission", {}))
            mission = mission if isinstance(mission, dict) else {}
            recent = context.get("recent_decisions", payload.get("recent_decisions", []))
            compact["context"] = {
                "semantic_class": context.get("semantic_class"),
                "mission": {
                    key: mission.get(key)
                    for key in ("target_name", "target", "mode")
                    if mission.get(key) not in (None, "")
                },
                "candidates": [candidate_view(value) for value in candidates[:20]],
                "recent_decision_count": len(recent) if isinstance(recent, list) else 0,
            }
            if "raw_text" not in compact:
                response = item.get("response") if isinstance(item.get("response"), dict) else {}
                result = payload.get("result", response.get("raw_text"))
                if result not in (None, ""):
                    compact["raw_text"] = result
            return compact

        def navigation_view(value: Any) -> dict[str, Any]:
            nav = value if isinstance(value, dict) else {}
            candidates = nav.get("candidates") if isinstance(nav.get("candidates"), dict) else {}
            candidate_items = candidates.get("candidates", [])
            candidate_items = candidate_items if isinstance(candidate_items, list) else []
            trace = nav.get("decision_trace") if isinstance(nav.get("decision_trace"), dict) else {}
            return {
                "current_subgoal": nav.get("current_subgoal", {}),
                "candidates": {
                    "count": len(candidate_items),
                    "candidates": [candidate_view(item) for item in candidate_items[:20]],
                },
                "selection": nav.get("selection", {}),
                "execution_state": nav.get("execution_state", {}),
                "behavior_feedback": nav.get("behavior_feedback", {}),
                "interaction_result": nav.get("interaction_result", {}),
                "decision_trace": {
                    key: trace.get(key)
                    for key in (
                        "timestamp", "active_candidate_id", "model_selected_candidate_id",
                        "executed_candidate_id", "model_reason", "selection_override_reason",
                        "model_result_source", "model_error",
                    )
                    if trace.get(key) not in (None, "")
                },
            }

        graph = snapshot.get("graph")
        graph_summary = graph
        if isinstance(graph, dict):
            graph_summary = {
                "scene_id": graph.get("scene_id"),
                "graph_revision": graph.get("graph_revision"),
                "capture_step": graph.get("capture_step"),
                "node_count": len(graph.get("nodes") or []),
                "edge_count": len(graph.get("edges") or []),
            }
        mllm = [event_view(item) for item in (snapshot.get("mllm_events") or []) if isinstance(item, dict)]
        stage_limits = {"M1": 8, "M2": 8, "M3": 10}
        stages = {
            stage: [
                item for item in mllm
                if str(item.get("stage") or item.get("module") or "").upper() == stage
            ][-stage_limits[stage]:]
            for stage in ("M1", "M2", "M3")
        }
        latest_m3 = stages["M3"][-1] if stages["M3"] else {}
        navigation = navigation_view(snapshot.get("navigation"))
        qwen = snapshot.get("qwen") if isinstance(snapshot.get("qwen"), dict) else {}
        return {
            "frame_seq": snapshot.get("frame_seq"),
            "navigation_step": snapshot.get("navigation_step"),
            "frame_stamp": snapshot.get("frame_stamp"),
            # Never inline overlay_jpeg here: the ROS gateway polls this
            # endpoint under a short deadline to feed temporal tracking.
            "detection_meta": compact_detection_meta,
            "generated_at": snapshot.get("generated_at"),
            "read_only": snapshot.get("read_only", True),
            "status": snapshot.get("status"),
            "link": snapshot.get("link"),
            "counters": snapshot.get("counters"),
            "telemetry": snapshot.get("telemetry"),
            "detections": [detection_view(item) for item in (snapshot.get("detections") or [])],
            "mapped_detections": [detection_view(item) for item in (snapshot.get("mapped_detections") or [])],
            "graph": graph_summary,
            "consistency": snapshot.get("consistency"),
            "safety": snapshot.get("safety"),
            "qwen": {
                "request_count": len(qwen.get("requests") or []),
                "result_count": len(qwen.get("results") or []),
            },
            "mllm": stages,
            "m3": {
                "stage": "M3",
                "model_call": bool(stages.get("M3")),
                "events": stages.get("M3", []),
                "sample_period_s": latest_m3.get("sample_period_s", 1.0),
                "next_call_at": latest_m3.get("next_call_at"),
                "interaction_result": navigation.get("interaction_result", {}),
                "behavior_feedback": navigation.get("behavior_feedback", {}),
                "execution_state": navigation.get("execution_state", {}),
            },
            "navigation": navigation,
            "last_error": snapshot.get("last_error", ""),
        }

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path == "/phone-stream":
            if self.phone_stream_url and not isinstance(self.connection, ssl.SSLSocket):
                self.send_response(302); self.send_header("Location", self._phone_stream_url()); self.send_header("Content-Length", "0"); self.send_header("Cache-Control", "no-store"); self.end_headers(); return
            data = _phone_stream_page(self._phone_stream_url()).encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-cache, no-store, must-revalidate"); self.send_header("Permissions-Policy", "camera=(self), microphone=(self)"); self.end_headers(); self.wfile.write(data); return
        if path == "/phone-qr.png":
            try:
                import qrcode
                image = qrcode.make(self._phone_stream_url())
                output = io.BytesIO(); image.save(output, format="PNG"); data = output.getvalue()
            except Exception as exc:
                self._json({"ok": False, "error": f"QR generation failed: {exc}"}, 503); return
            self.send_response(200); self.send_header("Content-Type", "image/png"); self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "private, max-age=60"); self.end_headers(); self.wfile.write(data); return
        if path == "/api/phone-status":
            self._json(self._phone_snapshot()); return
        if path == "/api/recording/status":
            recorder = getattr(self, "recorder", None)
            self._json(recorder.status() if recorder is not None else {"active": False, "error": "recorder unavailable"})
            return
        if path == "/api/recording/sessions":
            recorder = getattr(self, "recorder", None)
            self._json(recorder.list_sessions() if recorder is not None else [])
            return
        if path == "/phone-frame.jpg":
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                after = int((query.get("after") or [0])[0])
            except (TypeError, ValueError):
                after = 0
            with self.phone_lock:
                frame, sequence = self.latest_phone_jpeg, self.latest_phone_seq
            if not frame:
                self._json({"ok": False, "error": "phone frame not ready"}, 503); return
            if after >= sequence:
                self.send_response(204); self.send_header("Content-Length", "0"); self.send_header("X-Frame-Seq", str(sequence)); self.send_header("Cache-Control", "no-store"); self.end_headers(); return
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(frame))); self.send_header("X-Frame-Seq", str(sequence)); self.send_header("Cache-Control", "no-cache, no-store, must-revalidate"); self.send_header("Pragma", "no-cache"); self.end_headers(); self.wfile.write(frame); return
        if path == "/phone-audio.pcm":
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                after = int((query.get("after") or [0])[0])
            except (TypeError, ValueError):
                after = 0
            cls = type(self)
            with cls.phone_lock:
                packets = [packet for packet in cls.phone_audio_packets if packet[0] > after]
                # A resumed display skips stale packets and rejoins the live
                # edge instead of replaying a long audio backlog.
                packets = packets[-2:]
            if not packets:
                self.send_response(204); self.send_header("Content-Length", "0"); self.send_header("Cache-Control", "no-store"); self.end_headers(); return
            data = b"".join(packet[2] for packet in packets)
            self.send_response(200); self.send_header("Content-Type", "audio/pcm; format=s16le; channels=1")
            self.send_header("Content-Length", str(len(data))); self.send_header("X-Audio-Seq", str(packets[-1][0])); self.send_header("X-Audio-Rate", str(packets[-1][1])); self.send_header("Cache-Control", "no-cache, no-store, must-revalidate"); self.end_headers(); self.wfile.write(data); return
        if path in {"/assets/go2-dark-reference.png", "/assets/go2-light-reference.png", "/assets/go2-user-reference.png"}:
            asset = Path(__file__).resolve().parent / "assets" / path.removeprefix("/assets/")
            # Only the two bundled, read-only reference assets are exposed.
            try:
                data = asset.read_bytes()
            except OSError:
                self.send_error(404); return
            self.send_response(200); self.send_header("Content-Type", "image/png"); self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "public, max-age=3600"); self.end_headers(); self.wfile.write(data); return
        if path in {"/showcase-dark", "/showcase-light", "/showcase-academic"}:
            if path != "/showcase-dark":
                self.send_error(404, "Only showcase-dark is enabled during physical testing")
                return
            if path.endswith("academic"):
                html = ACADEMIC_SHOWCASE_HTML.encode()
            else:
                html = (DARK_SHOWCASE_HTML if path.endswith("dark") else LIGHT_SHOWCASE_HTML).encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(html))); self.send_header("Cache-Control", "no-cache, no-store, must-revalidate"); self.send_header("Pragma", "no-cache"); self.end_headers(); self.wfile.write(html); return
        if path == "/showcase":
            html = _SHOWCASE_HTML.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(html))); self.send_header("Cache-Control", "no-cache, no-store, must-revalidate"); self.send_header("Pragma", "no-cache"); self.end_headers(); self.wfile.write(html); return
        if path == "/api/state":
            self._json({**self.state.snapshot(), "safety": self.gate.snapshot()}); return
        if path == "/api/state-summary":
            summary = self._state_summary({**self.state.snapshot(), "safety": self.gate.snapshot()})
            recorder = getattr(self, "recorder", None)
            if recorder is not None:
                status = recorder.status()
                summary["recording"] = {
                    key: status.get(key)
                    for key in ("active", "session_id", "mode", "duration_s", "queue_size", "degraded")
                }
            self._json(summary); return
        if path == "/api/perception-summary":
            # Keep high-rate browser polling separate from the full state/MLLM
            # summary, which is intentionally refreshed at a lower rate.
            snapshot = self.state.snapshot()
            meta = snapshot.get("detection_meta")
            meta = meta if isinstance(meta, dict) else {}
            self._json({
                "frame_seq": snapshot.get("frame_seq"),
                "frame_stamp": snapshot.get("frame_stamp"),
                "detection_meta": {
                    key: meta.get(key)
                    for key in ("seq", "stamp", "camera_frame", "width", "height", "count", "inference_ms", "cycle_ms", "timings_ms")
                    if meta.get(key) is not None
                },
                "detections": self._state_summary(snapshot).get("detections", []),
            }); return
        if path == "/api/ros-state":
            self._json(self._state_summary(
                {**self.state.snapshot(), "safety": self.gate.snapshot()},
                include_ros_clouds=True,
            )); return
        if path == "/api/visualization-data":
            # Raw map/graph receipts are polled separately by showcase pages;
            # this keeps the normal state endpoint lightweight while allowing
            # independent canvas redraws from semantic data.
            self._json(self.state.visualization_snapshot()); return
        if path == "/api/occupancy":
            # Dedicated latest-only map endpoint.  Do not make the browser
            # download RGB/depth/detections when it only needs OCC.
            self._json(self.state.occupancy_snapshot()); return
        if path == "/api/raw-frame":
            self._json(self.state.raw_frame()); return
        if path == "/api/health":
            self._json(self.state.health_snapshot()); return
        if path == "/api/control-status":
            self._json(self._control_snapshot()); return
        if path == "/api/m1-input-image":
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            key = str((query.get("key") or [""])[0])
            data = self.state.m1_input_image(key)
            if not data:
                self._json({"ok": False, "error": "M1 input image not found"}, 404); return
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "private, max-age=300"); self.end_headers(); self.wfile.write(data); return
        if path == "/snapshot.jpg":
            # Short-lived JPEG requests are more reliable than a long-lived
            # multipart stream through some LAN proxies/browser setups.
            with self.frame_lock:
                frame = self.latest_jpeg
            if not frame:
                self._json({"ok": False, "error": "frame not ready"}, 503); return
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(frame))); self.send_header("Cache-Control", "no-cache, no-store, must-revalidate"); self.send_header("Pragma", "no-cache"); self.end_headers(); self.wfile.write(frame); return
        if path == "/camera-overlay.jpg":
            with self.frame_lock:
                frame = self.latest_camera_jpeg
            if not frame:
                self._json({"ok": False, "error": "camera overlay not ready"}, 503); return
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(frame))); self.send_header("Cache-Control", "no-cache, no-store, must-revalidate"); self.send_header("Pragma", "no-cache"); self.end_headers(); self.wfile.write(frame); return
        if path == "/camera-box-overlay.jpg":
            with self.frame_lock:
                frame = self.latest_camera_box_jpeg
            if not frame:
                self._json({"ok": False, "error": "box-only camera overlay not ready"}, 503); return
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(frame))); self.send_header("Cache-Control", "no-cache, no-store, must-revalidate"); self.send_header("Pragma", "no-cache"); self.end_headers(); self.wfile.write(frame); return
        if path in {"/original-panel3.jpg", "/original-panel6.jpg"}:
            with self.frame_lock:
                frame = self.latest_panel3_jpeg if path.endswith("panel3.jpg") else self.latest_panel6_jpeg
            if not frame:
                self._json({"ok": False, "error": "original renderer panel not ready"}, 503); return
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(frame))); self.send_header("Cache-Control", "no-cache, no-store, must-revalidate"); self.send_header("Pragma", "no-cache"); self.end_headers(); self.wfile.write(frame); return
        if path in {"/panel1.jpg", "/panel5.jpg"}:
            panel_index = 0 if path.startswith("/panel1") else 4
            with self.frame_lock:
                frame = self.latest_jpeg
            if not frame or cv2 is None:
                self._json({"ok": False, "error": "frame not ready"}, 503); return
            image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                self._json({"ok": False, "error": "frame decode failed"}, 503); return
            panel_width, panel_height = image.shape[1] // 3, image.shape[0] // 2
            x = (panel_index % 3) * panel_width; y = (panel_index // 3) * panel_height
            ok, encoded = cv2.imencode(".jpg", image[y:y + panel_height, x:x + panel_width], [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                self._json({"ok": False, "error": "panel encode failed"}, 500); return
            data = bytes(encoded)
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-cache, no-store, must-revalidate"); self.send_header("Pragma", "no-cache"); self.end_headers(); self.wfile.write(data); return
        if path == "/stream.mjpg":
            # Browser refreshes can leave an old multipart request half-open.
            # A write timeout ensures those abandoned stream threads are
            # reclaimed instead of accumulating until the viewer stalls.
            self.connection.settimeout(2.0)
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.send_header("Cache-Control", "no-cache, no-store, must-revalidate"); self.send_header("Pragma", "no-cache"); self.send_header("Connection", "close"); self.end_headers()
            while True:
                with self.frame_lock: frame = self.latest_jpeg
                if frame:
                    try:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"); self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError): return
                # A physical audit view does not need camera-rate streaming;
                # limiting this to 5 Hz reduces network pressure for remote
                # browsers while still showing continuous state changes.
                time.sleep(0.1)
        elif path in {"/panel1.mjpg", "/panel5.mjpg"}:
            panel_index = 0 if path.startswith("/panel1") else 4
            self.connection.settimeout(2.0)
            self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.send_header("Cache-Control", "no-cache, no-store, must-revalidate"); self.send_header("Connection", "close"); self.end_headers()
            while True:
                with self.frame_lock: frame = self.latest_jpeg
                if frame and cv2 is not None:
                    try:
                        image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if image is not None:
                            panel_width, panel_height = image.shape[1] // 3, image.shape[0] // 2
                            x = (panel_index % 3) * panel_width; y = (panel_index // 3) * panel_height
                            crop = image[y:y + panel_height, x:x + panel_width]
                            ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                            if ok:
                                data = bytes(encoded)
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"); self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError): return
                time.sleep(0.1)
        else:
            html = _HTML.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(html))); self.send_header("Cache-Control", "no-cache, no-store, must-revalidate"); self.send_header("Pragma", "no-cache"); self.end_headers(); self.wfile.write(html)

    def _convert_recording_to_mp4(self) -> None:
        """Transcode a browser WebM fallback and return an ephemeral MP4."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 1_000_000_000:
            self._json({"accepted": False, "error": "recording size must be 1..1000000000 bytes"}, 413); return
        with tempfile.TemporaryDirectory(prefix="molmospaces-recording-") as directory:
            source = Path(directory) / "recording.webm"
            output = Path(directory) / "recording.mp4"
            remaining = length
            with source.open("wb") as destination:
                while remaining:
                    chunk = self.rfile.read(min(1_048_576, remaining))
                    if not chunk:
                        self._json({"accepted": False, "error": "incomplete recording upload"}, 400); return
                    destination.write(chunk); remaining -= len(chunk)
            try:
                result = subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-threads", "2",
                        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                        "-movflags", "+faststart", str(output),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=3600,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                self._json({"accepted": False, "error": f"MP4 conversion unavailable: {exc}"}, 503); return
            if result.returncode or not output.is_file():
                error = result.stderr.decode("utf-8", "replace")[-500:]
                self._json({"accepted": False, "error": f"MP4 conversion failed: {error}"}, 422); return
            size = output.stat().st_size
            self.send_response(200); self.send_header("Content-Type", "video/mp4"); self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", 'attachment; filename="recording.mp4"'); self.send_header("Cache-Control", "no-store"); self.end_headers()
            with output.open("rb") as source_file:
                shutil.copyfileobj(source_file, self.wfile, length=1_048_576)

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        recorder = getattr(self, "recorder", None)
        if path == "/api/recording/start":
            if recorder is None:
                self._json({"accepted": False, "error": "recorder unavailable"}, 503); return
            try:
                payload = self._read_json_payload()
                mode = str(payload.get("mode") or "raw_plus_panels")
                label = str(payload.get("label") or "web")
                metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                result = recorder.start(mode=mode, label=label, metadata=metadata)
                # The token is returned only at explicit start/join time; it is
                # intentionally omitted from the polling status endpoint.
                result = {**result, "token": recorder.token()}
                self._json(result, 201)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self._json({"accepted": False, "error": str(exc)}, 400)
            return
        if path == "/api/recording/phone-join":
            if recorder is None:
                self._json({"accepted": False, "error": "recorder unavailable"}, 503); return
            try:
                payload = self._read_json_payload()
                session_id = str(payload.get("session_id") or "")
                token = str(payload.get("token") or "")
                if recorder.is_active() and not recorder.authorize(session_id, token):
                    self._json({"accepted": False, "error": "recording session token rejected"}, 403); return
                if not recorder.is_active():
                    result = recorder.start(
                        mode=str(payload.get("mode") or "raw_plus_panels"),
                        label="phone_sync",
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                else:
                    result = recorder.status()
                self._json({**result, "token": recorder.token(), "joined": True}, 200)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self._json({"accepted": False, "error": str(exc)}, 400)
            return
        if path == "/api/recording/stop":
            if recorder is None:
                self._json({"accepted": False, "error": "recorder unavailable"}, 503); return
            try:
                payload = self._read_json_payload()
                session_id = str(payload.get("session_id") or "")
                token = str(payload.get("token") or "")
                if recorder.is_active() and not recorder.authorize(session_id, token):
                    self._json({"accepted": False, "error": "recording session token rejected"}, 403); return
                self._json(recorder.stop(reason=str(payload.get("reason") or "user")), 200)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self._json({"accepted": False, "error": str(exc)}, 400)
            return
        if path == "/api/recording/event":
            if recorder is None:
                self._json({"accepted": False, "error": "recorder unavailable"}, 503); return
            try:
                payload = self._read_json_payload()
                session_id = str(payload.pop("session_id", "") or "")
                token = str(payload.pop("token", "") or "")
                if recorder.is_active() and not recorder.authorize(session_id, token):
                    self._json({"accepted": False, "error": "recording session token rejected"}, 403); return
                accepted = recorder.record_event("ui_event", payload, critical=True)
                self._json({"accepted": accepted}, 202 if accepted else 409)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self._json({"accepted": False, "error": str(exc)}, 400)
            return
        if path == "/api/recording-to-mp4":
            self._convert_recording_to_mp4(); return
        if path == "/api/phone-audio":
            if not self._recording_allowed():
                self._json({"accepted": False, "error": "recording session token rejected"}, 403); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > 262_144:
                self._json({"accepted": False, "error": "PCM packet size must be 1..262144 bytes"}, 413); return
            if not str(self.headers.get("Content-Type") or "").casefold().startswith("audio/pcm"):
                self._json({"accepted": False, "error": "Content-Type must be audio/pcm"}, 415); return
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                client_seq = int((query.get("seq") or [0])[0]); sample_rate = int((query.get("rate") or [48_000])[0])
            except (TypeError, ValueError):
                self._json({"accepted": False, "error": "invalid audio sequence or sample rate"}, 400); return
            client_timestamp = (query.get("ts") or [""])[0]
            if not 8_000 <= sample_rate <= 96_000 or length % 2:
                self._json({"accepted": False, "error": "unsupported PCM format"}, 400); return
            packet = self.rfile.read(length)
            if len(packet) != length:
                self._json({"accepted": False, "error": "incomplete PCM packet"}, 400); return
            cls = type(self)
            with cls.phone_lock:
                sequence = max(cls.latest_phone_audio_seq + 1, client_seq)
                cls.latest_phone_audio_seq = sequence; cls.latest_phone_audio_at = time.time(); cls.latest_phone_audio_rate = sample_rate
                cls.phone_audio_packets.append((sequence, sample_rate, packet))
            recorder = getattr(self, "recorder", None)
            if recorder is not None:
                recorder.record_phone_audio(
                    packet,
                    sequence=sequence,
                    sample_rate=sample_rate,
                    client_timestamp=client_timestamp,
                )
            self._json({"accepted": True, "audio_seq": sequence}, 202); return
        if path == "/api/phone-frame":
            if not self._recording_allowed():
                self._json({"accepted": False, "error": "recording session token rejected"}, 403); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > 2_000_000:
                self._json({"accepted": False, "error": "JPEG frame size must be 1..2000000 bytes"}, 413); return
            if not str(self.headers.get("Content-Type") or "").casefold().startswith("image/jpeg"):
                self._json({"accepted": False, "error": "Content-Type must be image/jpeg"}, 415); return
            frame = self.rfile.read(length)
            if len(frame) != length or not frame.startswith(b"\xff\xd8") or not frame.endswith(b"\xff\xd9"):
                self._json({"accepted": False, "error": "invalid JPEG frame"}, 400); return
            if cv2 is not None:
                decoded = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is None or decoded.shape[0] > 2160 or decoded.shape[1] > 3840:
                    self._json({"accepted": False, "error": "unsupported JPEG dimensions"}, 400); return
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                client_seq = int((query.get("seq") or [0])[0])
            except (TypeError, ValueError):
                client_seq = 0
            client_timestamp = (query.get("ts") or [""])[0]
            try:
                target_fps = min(20, max(5, int((query.get("fps") or [10])[0])))
            except (TypeError, ValueError):
                target_fps = 10
            cls = type(self)
            with cls.phone_lock:
                cls.latest_phone_jpeg = frame
                cls.latest_phone_at = time.time()
                cls.latest_phone_seq = max(cls.latest_phone_seq + 1, client_seq)
                cls.latest_phone_client = str(self.client_address[0])
                cls.latest_phone_fps = target_fps
                sequence = cls.latest_phone_seq
            recorder = getattr(self, "recorder", None)
            if recorder is not None:
                recorder.record_phone_frame(
                    frame,
                    sequence=sequence,
                    client_timestamp=client_timestamp,
                    metadata={"target_fps": target_fps, "client_ip": str(self.client_address[0])},
                )
            self._json({"accepted": True, "frame_seq": sequence}, 202); return
        if self.path == "/api/control":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                action = str(payload.get("action", "")).strip().lower()
                goal = str(payload.get("goal", "")).strip()
                if action not in {"start", "stop", "restart", "enable_motion"}:
                    self._json({"accepted": False, "error": "action 必须是 start、stop、restart 或 enable_motion"}, 400); return
                if len(goal) > 80 or any(ch in goal for ch in "\r\n\x00"):
                    self._json({"accepted": False, "error": "goalname 无效"}, 400); return
                self._json(self._run_control(action, goal), 202)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json({"accepted": False, "error": f"请求格式错误: {exc}"}, 400)
            return
        if self.path == "/api/ros-state":
            length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
            name, value = str(payload.get("name", "")), payload.get("value")
            if name not in {"detections", "mapped_detections", "graph", "consistency", "occupancy", "room_grid", "global_costmap", "local_costmap", "global_plan", "local_global_plan", "local_plan", "telemetry", "mllm_events", "explore_status", "current_subgoal", "candidates", "selection", "execution_state", "behavior_feedback", "interaction_result", "decision_trace"}:
                self._json({"accepted": False, "error": "unsupported ROS state"}, 400); return
            if name in {"explore_status", "current_subgoal", "candidates", "selection", "execution_state", "behavior_feedback", "interaction_result", "decision_trace", "global_plan", "local_global_plan", "local_plan"}:
                self.state.navigation[name] = value
            else:
                self.state.update_topic(name, value)
            recorder = getattr(self, "recorder", None)
            if recorder is not None:
                if name == "mllm_events":
                    events = value if isinstance(value, list) else [value]
                    for event in events:
                        if isinstance(event, dict):
                            recorder.record_mllm_event(event, source="ros_state")
                else:
                    recorder.record_ros_state(name, value, payload=payload)
            self._json({"accepted": True}); return
        if self.path == "/api/mllm-event":
            length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
            recorder = getattr(self, "recorder", None)
            if recorder is not None:
                recorder.record_mllm_event(payload, source="trace_http")
            self.state.add_mllm_event(payload); self._json({"accepted": True}); return
        if self.path == "/api/qwen":
            length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
            if self.qwen_submit is None: self._json({"error": "Qwen client is disabled"}, 503); return
            self._json(self.qwen_submit(payload), 202); return
        if self.path != "/api/teleop-intent": self._json({"error": "not found"}, 404); return
        length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
        self._json(self.gate.handle_intent(payload), 202)


_HTML = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Go2 Physical Interactive Navigation</title>
<style>
:root{color-scheme:dark;--bg:#0b0d11;--card:#141821;--line:#2e3748;--blue:#67a7ff;--green:#54d68b;--amber:#ffca58;--red:#ff6c67;--muted:#9ba8ba}*{box-sizing:border-box}body{font-family:Inter,"Noto Sans SC",system-ui,sans-serif;background:var(--bg);color:#edf2fa;margin:0;padding:14px}.title{display:flex;align-items:center;gap:12px;margin:0 0 10px;font-size:23px}.readonly{font-size:13px;color:#101418;background:var(--amber);padding:4px 9px;border-radius:99px}.dashboard{display:grid;grid-template-columns:minmax(640px,1fr) 430px;gap:12px;align-items:start}.left{min-width:0}.overview-wrap{position:relative;width:100%;overflow:hidden;border:1px solid var(--line);border-radius:8px;background:#111}.overview{display:block;width:100%;border:0;background:#111}.overview-camera{position:absolute;z-index:2;left:0;top:0;width:33.333333%;height:50%;object-fit:fill;background:#111;border-right:1px solid #aaa;border-bottom:1px solid #aaa}.ratebar{display:flex;gap:14px;flex-wrap:wrap;color:var(--green);font-size:13px;padding:7px 2px}.mllm-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.right{display:grid;gap:10px}.card{border:1px solid var(--line);border-radius:10px;background:var(--card);padding:10px;min-width:0;box-shadow:0 4px 14px #0004}.card h3{display:flex;align-items:center;justify-content:space-between;margin:0 0 8px;color:var(--blue);font-size:15px}.hint{color:var(--muted);font-size:11px;font-weight:400}.visual{width:100%;display:block;border-radius:6px;border:1px solid #343c49;background:#0d0f13}.events{max-height:390px;overflow:auto;display:grid;gap:8px}.event{display:grid;grid-template-columns:minmax(76px,.8fr) minmax(105px,1.25fr) minmax(90px,1fr);gap:7px;padding:7px;border:1px solid #313947;border-radius:8px;background:#0e1218;font-size:11px}.event-col{min-width:0;overflow:hidden}.label{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}.thumb{width:100%;height:auto;aspect-ratio:16/9;object-fit:contain;border-radius:4px;border:1px solid #354053;background:#070d15}.chips{display:flex;gap:4px;flex-wrap:wrap}.chip{display:inline-block;max-width:100%;padding:2px 5px;border-radius:5px;background:#263248;color:#cfe1ff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.prompt,.result{line-height:1.35;word-break:break-word}.result{color:#d9f7e6}.meta{grid-column:1/-1;color:#7f8da1;font-size:10px}.empty{height:110px;display:grid;place-items:center;color:#768397;border:1px dashed #354052;border-radius:8px}.state-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.metric{padding:9px 7px;border:1px solid #303a4a;border-radius:8px;background:#0e1218;text-align:center}.metric .icon{font-size:19px}.metric .value{font-size:16px;font-weight:700;margin-top:2px}.metric .name{font-size:10px;color:var(--muted)}.statusline{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}.badge{padding:3px 7px;border-radius:99px;font-size:11px;background:#263248;color:#cfe1ff}.badge.good{background:#173b2b;color:#7ff0ae}.badge.warn{background:#493b14;color:#ffd46c}.badge.bad{background:#4b2021;color:#ff9692}.m3box{display:grid;gap:8px}.m3head{display:flex;align-items:center;gap:8px}.m3status{font-size:22px;font-weight:800}.m3details{display:grid;grid-template-columns:repeat(2,1fr);gap:6px}.mini{background:#0e1218;border:1px solid #303a4a;border-radius:7px;padding:7px}.mini b{display:block;color:#eaf1fc;font-size:12px}.mini span{font-size:10px;color:var(--muted)}@media(max-width:1150px){.dashboard{grid-template-columns:1fr}.right{grid-template-columns:repeat(3,minmax(0,1fr));grid-row:2}.mllm-grid{grid-row:3}}@media(max-width:800px){body{padding:8px}.dashboard{display:block}.right,.mllm-grid{grid-template-columns:1fr;margin-top:10px}.event{grid-template-columns:1fr 1fr}.meta{grid-column:1/-1}}
</style>
<style>
/* The dashboard always stays within the viewport. Resizing either zoom card
   changes the shared right rail, so cards 4/5/6 retain one aligned width. */
@media(min-width:1151px){
  .dashboard{grid-template-columns:minmax(640px,1fr) var(--right-width,430px);width:100%;max-width:100%}
  .right{position:relative;border-left:8px solid transparent;margin-left:-8px}
  .right:before{content:"";position:absolute;z-index:20;left:-8px;top:0;bottom:0;width:8px;cursor:col-resize;background:linear-gradient(90deg,transparent,#67a7ff66,transparent);opacity:.25}
  .right:hover:before,.right.is-resizing:before{opacity:1}
}
.right .card:nth-child(2),.right .card:nth-child(3){resize:both;overflow:hidden;min-width:360px;min-height:260px;width:100%;max-width:900px;height:360px;justify-self:end}
.right .card:nth-child(2) .visual,.right .card:nth-child(3) .visual{height:calc(100% - 30px);object-fit:contain}
.m3schedule{display:flex;gap:6px;flex-wrap:wrap}
.m3history-title{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-top:2px}
.m3history{display:grid;gap:5px;max-height:132px;overflow:auto}
.m3event{display:grid;grid-template-columns:74px 58px minmax(0,1fr) auto;gap:6px;align-items:center;background:#0e1218;border:1px solid #303a4a;border-radius:7px;padding:6px;font-size:10px}
.m3event .object{color:#a8cfff;font-weight:700}.m3event .state{color:#d9f7e6;font-weight:700}.m3event .reason{color:#aeb9c8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.m3event time{color:#77869b}
.mini.question{grid-column:1/-1}
.controlbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 12px;padding:9px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card)}
.controlbar button,.controlbar input{font:inherit;border:1px solid #43536c;border-radius:6px;padding:7px 10px;background:#1b2636;color:#edf2fa}.controlbar button{cursor:pointer}.controlbar button:hover{border-color:var(--blue);background:#263752}.controlbar .danger{border-color:#a94b4b;background:#482326;color:#ffd8d8}.controlbar input{width:180px}.control-status{color:var(--muted);font-size:11px;margin-left:auto}
.go2-layout{display:grid;grid-template-columns:minmax(0,1fr) 162px;gap:10px;align-items:start}.go2-controls{display:grid;gap:7px}.go2-controls button,.go2-controls input{font:inherit;font-size:12px;border:1px solid #43536c;border-radius:6px;padding:7px 8px;background:#1b2636;color:#edf2fa;min-height:34px}.go2-controls button{cursor:pointer}.go2-controls button:hover{border-color:var(--blue);background:#263752}.go2-controls .danger{border-color:#a94b4b;background:#482326;color:#ffd8d8}.go2-controls input{width:100%}.go2-controls .control-status{font-size:11px;line-height:1.25;overflow-wrap:anywhere;margin:0}
@media(max-width:1150px){.right .card:nth-child(2),.right .card:nth-child(3){width:100%!important;max-width:none}}
</style>
<h1 class='title'>Go2 Physical Interactive Navigation <span class='readonly'>动作控制需显式启动</span></h1>
<main class='dashboard'><div class='left'><div class='overview-wrap'><img id='overview' class='overview' src='/snapshot.jpg' alt='实时六面板'><img id='overview-camera' class='overview-camera' src='/camera-box-overlay.jpg' alt='10 Hz 实时感知'></div><div class='ratebar'><span>● 导航 5 Hz</span><span>● YOLOE / 建图输入 10 Hz</span><span>● 网页 10 Hz</span><span>● Go2 人工遥控</span></div><div class='mllm-grid'><section class='card'><h3>1 · M1 VLM 感知 <span class='hint'>图片 → 简化问题 → 结果</span></h3><div id='m1' class='events'></div></section><section class='card'><h3>2 · M2 LLM 子目标 <span class='hint'>历史 + 候选 + 目标</span></h3><div id='m2' class='events'></div></section><section class='card'><h3>3 · M3 交互评价 <span class='hint'>对象类别 · 周期 · 历史</span></h3><div id='m3' class='m3box'></div></section></div></div><aside class='right'><section class='card'><h3>4 · Go2 当前状态 <span id='stamp' class='hint'>连接中</span></h3><div class='go2-layout'><div id='go2'></div><div class='go2-controls' aria-label='运动控制'><button id='nav-start'>Start</button><button id='nav-restart'>重启导航栈 + Goal</button><input id='nav-goal' placeholder='goalname' maxlength='80' autocomplete='off'><button id='nav-stop' class='danger'>Stop</button><button id='nav-enable'>开启运动</button><span class='control-status' id='control-status'>Esc / S：Stop</span></div></div></section><section class='card panel-suppressed'><h3>5 · 图 1 放大 <span class='hint'>当前测试已关闭</span></h3><div class='empty'>暂时关闭以测试网页负载</div></section><section class='card panel-suppressed'><h3>6 · 图 5 放大 <span class='hint'>当前测试已关闭</span></h3><div class='empty'>暂时关闭以测试网页负载</div></section></aside></main>
<script>
const q=s=>document.querySelector(s), text=v=>String(v??'').replace(/\\s+/g,' ').trim(), clip=(v,n=150)=>{v=text(v);return v.length>n?v.slice(0,n)+'…':v};
async function navControl(action, goal=''){
  const status=q('#control-status'); status.textContent=action+' 提交中…';
  try{
    const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,goal})});
    const value=await r.json();
    status.textContent=value.accepted?(value.message||((action==='stop'?'已中断控制':'已启动')+(value.pid?' · pid '+value.pid:''))):(value.error||'命令未接受');
  }catch(error){status.textContent='控制接口不可用：'+error}
}
q('#nav-start')?.addEventListener('click',()=>navControl('start'));
q('#nav-restart')?.addEventListener('click',()=>navControl('restart',q('#nav-goal').value.trim()));
q('#nav-stop')?.addEventListener('click',()=>navControl('stop'));
q('#nav-enable')?.addEventListener('click',()=>navControl('enable_motion'));
document.addEventListener('keydown',event=>{
  if(event.target instanceof HTMLInputElement||event.target instanceof HTMLTextAreaElement)return;
  if(event.key==='Escape'||event.key.toLowerCase()==='s'){event.preventDefault();navControl('stop')}
});
async function refreshControlStatus(){try{const r=await fetch('/api/control-status?ts='+Date.now(),{cache:'no-store'});const s=await r.json();if(s.running)q('#control-status').textContent=s.action+' 执行中 · pid '+s.pid; }catch(_){} }
setInterval(refreshControlStatus,2000);
refreshControlStatus();
function setupResizableLayout(){
  const dashboard=q('.dashboard'),rail=q('.right'),cards=[...document.querySelectorAll('.right .card:nth-child(2),.right .card:nth-child(3)')];
  if(!dashboard||!rail)return;
  const clampWidth=value=>{const available=Math.max(360,dashboard.getBoundingClientRect().width-652);return Math.max(360,Math.min(900,available,Number(value)||430))};
  const setRailWidth=(value,persist=false)=>{const width=clampWidth(value);dashboard.style.setProperty('--right-width',width+'px');if(persist)try{localStorage.setItem('physicalRightWidth',Math.round(width))}catch(_){}return width};
  try{const width=Number(localStorage.getItem('physicalRightWidth'));setRailWidth(width>0?width:430);cards.forEach((card,index)=>{const height=Number(localStorage.getItem('physicalZoomHeight'+index));if(height>=260)card.style.height=height+'px';card.style.width='100%'})}catch(_){}
  let railDragging=false,cardDragging=null;
  rail.addEventListener('pointerdown',event=>{if(window.innerWidth<=1150||event.clientX-rail.getBoundingClientRect().left>12)return;railDragging=true;rail.classList.add('is-resizing');rail.setPointerCapture(event.pointerId);event.preventDefault()});
  rail.addEventListener('pointermove',event=>{if(railDragging)setRailWidth(dashboard.getBoundingClientRect().right-event.clientX)});
  const finishRail=event=>{if(!railDragging)return;railDragging=false;rail.classList.remove('is-resizing');setRailWidth(rail.getBoundingClientRect().width,true);if(event?.pointerId!=null&&rail.hasPointerCapture(event.pointerId))rail.releasePointerCapture(event.pointerId)};
  rail.addEventListener('pointerup',finishRail);rail.addEventListener('pointercancel',finishRail);
  cards.forEach(card=>card.addEventListener('pointerdown',event=>{const bounds=card.getBoundingClientRect();if(bounds.right-event.clientX<=24&&bounds.bottom-event.clientY<=24)cardDragging=card}));
  const finishCard=()=>{if(!cardDragging)return;setRailWidth(cardDragging.getBoundingClientRect().width,true);cardDragging=null;cards.forEach(card=>card.style.width='100%')};
  window.addEventListener('pointerup',finishCard);window.addEventListener('pointercancel',finishCard);
  if(window.ResizeObserver){const observer=new ResizeObserver(entries=>entries.forEach(entry=>{const index=cards.indexOf(entry.target);if(index<0)return;const bounds=entry.target.getBoundingClientRect();try{localStorage.setItem('physicalZoomHeight'+index,Math.round(bounds.height))}catch(_){}if(cardDragging===entry.target&&window.innerWidth>1150){setRailWidth(bounds.width);cards.forEach(card=>{if(card!==entry.target)card.style.width='100%'})}}));cards.forEach(card=>observer.observe(card))}
  window.addEventListener('resize',()=>{if(window.innerWidth>1150)setRailWidth(rail.getBoundingClientRect().width)})
}
function make(tag,cls,value){const e=document.createElement(tag);if(cls)e.className=cls;if(value!==undefined)e.textContent=value;return e}
function contextCandidates(e){const c=e?.context?.candidates||e?.payload?.candidates||[];return Array.isArray(c)?c:[]}
const classNames={door:'门',fridge:'冰箱',refrigerator:'冰箱',cabinet:'柜子',closet:'衣柜',wardrobe:'衣柜',drawer:'抽屉',drawer_cabinet:'抽屉柜'};
function classZh(value){const key=text(value).toLowerCase().replace(/[ -]+/g,'_');return classNames[key]||text(value)||'目标物体'}
function eventObjectClass(e){let id=text(e?.object_id||e?.target_kind||e?.context?.semantic_class||'');if(id.startsWith('physical_')){const parts=id.slice(9).split('_');while(parts.length>1&&/^-?\\d+$/.test(parts[parts.length-1]))parts.pop();id=parts.join('_')}return classZh(id)}
function isRoomM1(e){if(e?.m1_input_image_key||e?.object_id||e?.m1_input_label)return false;let raw=e?.raw_text??e?.response?.raw_text??e?.payload?.result??'';try{const o=typeof raw==='string'?JSON.parse(raw):raw;return !!(o&&('room_attribute' in o||'room_id' in o))}catch(_){return false}}
function m1DisplayName(e){if(isRoomM1(e))return '房间属性';const label=text(e?.m1_input_label||eventObjectClass(e)).replaceAll('_',' '),raw=text(e?.m1_input_object_id??e?.m1_input_track_id??e?.object_id),match=raw.match(/(\\d+)$/);return label+(match?' #'+Number(match[1]):'')}
function m1InputCanvas(e,index){const kind=text(e?.target_kind||e?.context?.target_kind||'').toLowerCase();if(isRoomM1(e)||kind==='room'||kind==='room_attribute'||kind.includes('room'))return null;const canvas=make('canvas','thumb');canvas.width=320;canvas.height=180;const key=text(e?.m1_input_image_key);if(!key)return null;const image=new Image();image.onload=()=>{const c=canvas.getContext('2d'),sx=canvas.width/image.width,sy=canvas.height/image.height;c.drawImage(image,0,0,canvas.width,canvas.height);const b=e?.m1_input_bbox||[];if(b.length>=4){c.strokeStyle='#ffe04b';c.lineWidth=3;c.strokeRect(Number(b[0])*sx,Number(b[1])*sy,(Number(b[2])-Number(b[0]))*sx,(Number(b[3])-Number(b[1]))*sy);const name=m1DisplayName(e),x=Math.max(0,Number(b[0])*sx),y=Math.max(15,Number(b[1])*sy-4);c.font='bold 14px sans-serif';const w=c.measureText(name).width+8;c.fillStyle='#ffe04b';c.fillRect(x,y-15,w,18);c.fillStyle='#101418';c.fillText(name,x+4,y)}};image.src='/api/m1-input-image?key='+encodeURIComponent(key)+'&v='+(e.timestamp||index);return canvas}
function simplifiedQuestion(e,stage){if(stage==='M1'){const k=eventObjectClass(e),raw=text(e?.m1_input_label||e?.context?.semantic_class||'').toLowerCase();if(raw==='locker'||k==='locker')return '判断图中标注的 locker 是否为冰箱，给出冰箱置信度，并识别其开合状态。';return `判断图中的${k}是否可交互，并识别它的开合状态。`;}if(stage==='M2')return '结合历史和候选目标，选择下一个导航子目标。';return `判断画面中的${eventObjectClass(e)}是否已经打开，并给出置信度。`}
function m1Prompt(e){const kind=text(e?.target_kind||e?.context?.target_kind||'').toLowerCase();if(isRoomM1(e)||kind==='room'||kind==='room_attribute'||kind.includes('room'))return '根据当前空间观测判断房间属性与语义类别，并给出房间类别置信度。';return e?.instruction||e?.prompt||e?.request?.instruction||e?.request?.prompt||simplifiedQuestion(e,'M1')}
function m1Answer(e){let raw=e?.raw_text??e?.response?.raw_text??e?.payload?.result??e?.result??'';if(typeof raw==='object')raw=JSON.stringify(raw);if(!raw)return '（无 M1 回答）';try{const o=JSON.parse(raw);if(o.room_attribute||o.room_id){return [o.room_attribute||'unknown',o.confidence!=null?'置信度 '+Number(o.confidence).toFixed(2):''].filter(Boolean).join(' · ')}const name=o.observed_object_name||o.semantic_name||o.object_name||o.label||o.interaction_class||o.semantic_class||'';const state=o.coarse_state||o.state||o.status||'';const conf=o.confidence??o.score;const fridgeConf=o.fridge_confidence??o.refrigerator_confidence??o.is_refrigerator_confidence;const category=o.interaction_class||o.semantic_class||'';const confidence=fridgeConf??conf;return [name,category,state,confidence!=null?'置信度 '+Number(confidence).toFixed(2):''].filter(Boolean).join(' · ')}catch(_){return raw.replace(/[{}\[\]"]/g,'').replace(/[:,]/g,' · ').slice(0,180)}}
function outputSummary(e){if(e?.error)return '调用失败：'+clip(e.error,120);let raw=e?.raw_text??e?.response?.raw_text??e?.payload?.result??'';if(typeof raw==='object')raw=JSON.stringify(raw);try{const o=JSON.parse(raw),choice=o.ranked_ids||o.candidate_id||o.label||o.interaction_class||o.coarse_state||o.state||'',state=o.coarse_state&&o.coarse_state!==choice?o.coarse_state:'';return clip([Array.isArray(choice)?choice.join(' → '):choice,state,o.reason,o.confidence!=null?'置信度 '+o.confidence:''].filter(Boolean).join(' · '),150)}catch(_){return clip(raw||'已完成（无文本结果）',150)}}
function addChips(parent,items,max=5){const wrap=make('div','chips');items.slice(0,max).forEach(v=>wrap.append(make('span','chip',clip(v,28))));if(items.length>max)wrap.append(make('span','chip','+'+(items.length-max)));parent.append(wrap)}
function m2EventsWithLiveState(s){
  const events=[...(s?.mllm?.M2||[])],nav=s?.navigation||{},exec=nav.execution_state||{},selection=nav.selection||{};
  const state=text(exec.state||''),behavior=text(exec.behavior_type||selection.behavior_type||'').toUpperCase();
  if(behavior!=='INTERACT'&&!state.toUpperCase().includes('INTERACTION'))return events;
  const id=text(exec.candidate_id||selection.candidate_id||selection.model_selected_candidate_id||''),target=text(selection.target_name||selection.target_id||id||'交互目标');
  events.push({event_type:'live_execution_state',stage:'M2',timestamp:Number(exec.timestamp||selection.selected_at||Date.now()/1000),role:'live_execution',model:'M2 → executor',target_id:selection.target_id||'',target_name:target,context:{mission:{mode:'等待人工交互'},candidates:id?[{id,target_name:target}]:[]},raw_text:JSON.stringify({candidate_id:id,reason:'M2 已完成选择；当前 '+(state||'等待交互'),confidence:'live'})});
  return events
}
const eventSignatures={};
function renderEvents(id,events,stage){const box=q(id),recent=(events||[]).slice(-8).reverse(),signature=recent.map(e=>[e.timestamp,e.request_sequence,e.object_id,e.target_id,e.raw_text,e.error].join('|')).join('||');if(eventSignatures[stage]===signature)return;eventSignatures[stage]=signature;box.replaceChildren();if(!recent.length){box.append(make('div','empty','等待 '+stage+' 调用'));return}recent.forEach((e,index)=>{const row=make('article','event'),left=make('div','event-col'),mid=make('div','event-col'),right=make('div','event-col');left.append(make('div','label',stage==='M1'?'真实 M1 输入':'候选 subgoal'));if(stage==='M1'){const thumb=m1InputCanvas(e,index);if(thumb)left.append(thumb);left.append(make('div','chip',m1DisplayName(e)))}else{const cs=contextCandidates(e);addChips(left,cs.map(c=>c.id||c.candidate_id||c.target_name||'candidate'))}mid.append(make('div','label',stage==='M2'?'历史 / 目标 / 简化请求':'M1 实际调用文本'));if(stage==='M2'){const hist=e?.context?.recent_decisions||e?.payload?.recent_decisions||[],histCount=Number(e?.context?.recent_decision_count??(Array.isArray(hist)?hist.length:0)),mission=e?.context?.mission||e?.payload?.mission||{};addChips(mid,['历史 '+histCount,'候选 '+contextCandidates(e).length,'目标 '+(mission.target_name||mission.target||mission.mode||'探索')])}if(stage==='M1')mid.append(make('div','prompt',clip(m1Prompt(e),1200)));else mid.append(make('div','prompt',simplifiedQuestion(e,stage)));if(stage==='M1'){right.append(make('div','label','M1 完整回答'));right.append(make('div','result',clip(m1Answer(e),1600)))}else{right.append(make('div','label','MLLM 输出'));right.append(make('div','result',outputSummary(e)))}const meta=make('div','meta',`${e.timestamp?new Date(e.timestamp*1000).toLocaleTimeString():'--:--:--'} · ${e.latency_s!=null?Number(e.latency_s).toFixed(2)+' s':'等待耗时'} · ${e.model||e.role||''}`);row.append(left,mid,right,meta);box.append(row)})}
function firstObj(...values){return values.find(v=>v&&typeof v==='object')||{}}
function renderM3(m3){
  const box=q('#m3');box.replaceChildren();
  const events=Array.isArray(m3?.events)?m3.events:[],latest=events[events.length-1]||{},sample=firstObj(latest.result),fb=firstObj(m3?.behavior_feedback,m3?.interaction_result),detail=firstObj(fb.detail,fb.result),exec=firstObj(m3?.execution_state),humanAssist=String(exec.state||'').toUpperCase()==='INTERACTING';
  let status=humanAssist?'HUMAN ASSIST':text(latest.phase||sample.status||fb.status||detail.status||sample.state||'WAITING').toUpperCase(),success=humanAssist?null:(fb.success??detail.success);
  const kind=success===true||/PASS|SUCCESS|COMPLETE|OPEN/.test(status)?'good':success===false||/FAIL|ERROR|BLOCK/.test(status)?'bad':'warn';
  const targetKind=classZh(latest.target_kind||fb.target_kind||detail.target_kind||'暂无交互对象');
  const targetRef=text(latest.target_id||fb.object_id||fb.candidate_id||latest.target_name||fb.source_object_name||exec.candidate_id||'');
  const head=make('div','m3head');head.append(make('div','m3status',status),make('span','badge '+kind,humanAssist?'等待人工操作':success===true?'验证通过':success===false?'验证未通过':events.length?'模型观察中':'等待交互 policy'));box.append(head);
  const period=Number(latest.sample_period_s||m3?.sample_period_s||1),next=Number(latest.next_call_at||m3?.next_call_at||0),remaining=next>0?Math.max(0,next-Date.now()/1000):null,evaluating=String(latest.phase||'').toUpperCase()==='EVALUATING',schedule=make('div','m3schedule');schedule.append(make('span','badge','评价周期 '+period.toFixed(1)+' s'),make('span','badge '+(humanAssist||evaluating?'warn':remaining===null?'':remaining<=0?'good':'warn'),humanAssist?'当前：语音请求 / 人工操作':evaluating?'当前：正在评价':remaining===null?'下次：等待交互触发':remaining<=0?'下次：立即评价':'下次：'+remaining.toFixed(1)+' s'));box.append(schedule);
  const grid=make('div','m3details');
  [['评价对象类别',targetKind],['实例 ID（跟踪/回写）',targetRef||'等待具体目标'],['结果原因',humanAssist?'已到达交互位姿，等待语音播放与人工操作':sample.reason||detail.reason||fb.reason||detail.failure_stage||'等待交互反馈'],['状态变化',detail.pre_state&&detail.post_state?detail.pre_state+' → '+detail.post_state:(sample.state||'未产生')]].forEach(([n,v])=>{const d=make('div','mini');d.append(make('span','',n),make('b','',clip(v,60)));grid.append(d)});
  const question=make('div','mini question');question.append(make('span','','简化问题'),make('b','',`判断画面中的${targetKind}是否已经打开，并给出置信度。`));grid.append(question);
  box.append(grid,make('div','m3history-title','近期交互评价历史'));
  const history=make('div','m3history');if(!events.length)history.append(make('div','empty','尚无 M3 调用；到达交互目标后开始评价'));else events.slice(-10).reverse().forEach(event=>{const result=firstObj(event.result),row=make('div','m3event');row.append(make('span','object',classZh(event.target_kind||event.target_name||'目标物体')),make('span','state',String(result.state||result.status||'unknown').toUpperCase()),make('span','reason',clip(result.reason||'无补充说明',80)),make('time','',event.timestamp?new Date(event.timestamp*1000).toLocaleTimeString():'--:--:--'));history.append(row)});box.append(history)
}
function num(v,d=1){const n=Number(v);return Number.isFinite(n)?n.toFixed(d):'--'}
function speed(t){const v=t?.velocity||t?.linear_velocity||[];if(Array.isArray(v))return Math.hypot(...v.slice(0,3).map(Number));if(v&&typeof v==='object')return Math.hypot(Number(v.x||0),Number(v.y||0),Number(v.z||0));return Number(t?.speed||0)}
function renderGo2(s){const t=s.telemetry||{},b=t.battery||{},link=s.link||{},grid=make('div','state-grid');const metrics=[['🔋',num(b.soc??t.battery_soc,0)+' %','电量'],['↗',num(speed(t),2)+' m/s','速度'],['⟳',num((Number(t.yaw||0)*180/Math.PI),1)+'°','航向'],['◉',text(t.mode||'站立'),'动作模式'],['↕',num(t.body_height,2)+' m','机身高度'],['⚠',String(t.error_code??0),'错误码']];metrics.forEach(([i,v,n])=>{const d=make('div','metric');d.append(make('div','icon',i),make('div','value',v),make('div','name',n));grid.append(d)});const line=make('div','statusline');line.append(make('span','badge '+(link.connected===false?'bad':'good'),link.connected===false?'相机断开':'相机在线'),make('span','badge good','控制输出阻断'),make('span','badge','图节点 '+(s.graph?.node_count??0)),make('span','badge','候选 '+((s.navigation?.candidates?.candidates||[]).length)));const box=q('#go2');box.replaceChildren(grid,line);q('#stamp').textContent='导航步 '+(s.navigation_step??'--')+' · 相机帧 '+(s.frame_seq??'--')+' · '+new Date().toLocaleTimeString()}
async function refresh(){try{const r=await fetch('/api/state-summary?ts='+Date.now(),{cache:'no-store'}),s=await r.json();renderEvents('#m1',s.mllm?.M1,'M1');renderEvents('#m2',m2EventsWithLiveState(s),'M2');renderM3(s.m3||{});renderGo2(s)}catch(e){q('#stamp').textContent='刷新失败';q('#go2').replaceChildren(make('div','empty',clip(e,100)))}}
function refreshStill(id,path){
  if(document.hidden)return;
  const image=q(id);
  // Do not replace an in-flight image request.  Reassigning ``src`` every
  // 100 ms used to cancel a slower LAN response and caused a BrokenPipe /
  // reconnect storm on the gateway.  A slow client now drops display ticks
  // locally and immediately resumes at the newest cached frame.
  if(!image||image.dataset.loading==='1')return;
  image.dataset.loading='1';
  const done=()=>{image.dataset.loading='0'};
  image.onload=done;image.onerror=done;
  image.src=path+'?ts='+Date.now();
}
function resumeDashboard(){if(document.hidden)return;refresh();refreshStill('#overview','/snapshot.jpg');refreshStill('#overview-camera','/camera-box-overlay.jpg')}
setupResizableLayout();const rateLabel=document.querySelector('.ratebar span:nth-child(3)');if(rateLabel)rateLabel.textContent='● 感知图 10 Hz · 六面板 5 Hz';setInterval(()=>{if(!document.hidden)refresh()},1000);setInterval(()=>refreshStill('#overview','/snapshot.jpg'),200);setInterval(()=>refreshStill('#overview-camera','/camera-box-overlay.jpg'),100);document.addEventListener('visibilitychange',resumeDashboard);resumeDashboard();
</script></html>"""


_SHOWCASE_HTML = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>具身智能交互导航展示平台</title>
<style>
:root{color-scheme:dark;--bg:#07101d;--surface:#0d1929;--surface2:#111f32;--line:#263b55;--cyan:#49d5ff;--blue:#6d8dff;--green:#4ce6a6;--gold:#f5c66b;--text:#f4f8ff;--muted:#91a6bf;--danger:#ff7676}*{box-sizing:border-box}body{margin:0;min-height:100vh;color:var(--text);background:radial-gradient(circle at 15% -10%,#17365b 0,transparent 34%),radial-gradient(circle at 95% 18%,#142d49 0,transparent 28%),var(--bg);font-family:Inter,"Noto Sans SC","Microsoft YaHei",system-ui,sans-serif}.shell{max-width:1920px;margin:auto;padding:20px 24px 24px}.topbar{display:flex;align-items:center;justify-content:space-between;gap:22px;margin-bottom:16px}.brand{display:flex;align-items:center;gap:15px}.brandmark{width:48px;height:48px;border:1px solid #59ccff80;border-radius:14px;display:grid;place-items:center;background:linear-gradient(145deg,#173d63,#0b1b2e);box-shadow:0 0 28px #39bfff25}.brandmark svg{width:29px;height:29px}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.16em;text-transform:uppercase}.brand h1{font-size:26px;letter-spacing:.02em;margin:3px 0 0}.topmeta{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}.tag{border:1px solid #35506d;border-radius:999px;padding:7px 11px;color:#c9d8e9;background:#0d1a2aab;font-size:12px}.tag.live{border-color:#287b61;color:#75efb8}.tag.live:before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green);margin-right:7px}.layout{display:grid;grid-template-columns:minmax(680px,1.62fr) minmax(390px,.78fr);gap:14px}.stage,.sidecard{border:1px solid var(--line);background:linear-gradient(145deg,#0e1b2d,#0a1422);border-radius:16px;box-shadow:0 14px 38px #0005}.stage{padding:12px;display:grid;gap:10px}.visual-primary{position:relative;overflow:hidden;border-radius:12px;background:#080d14;aspect-ratio:16/9}.visual-primary canvas,.visual-small canvas{display:block;width:100%;height:100%;object-fit:contain}.visual-primary:after,.visual-small:after{content:"";position:absolute;inset:0;pointer-events:none;border:1px solid #7897b62b;border-radius:inherit}.caption{position:absolute;left:12px;right:12px;top:10px;display:flex;justify-content:space-between;align-items:flex-start;pointer-events:none;text-shadow:0 2px 6px #000}.caption strong{font-size:15px}.caption span{font-size:11px;color:#d5e1ef;background:#07111ecc;padding:4px 8px;border:1px solid #31465f;border-radius:999px}.visual-row{display:grid;grid-template-columns:1fr 1fr;gap:10px}.visual-small{position:relative;overflow:hidden;border-radius:11px;background:#080d14;aspect-ratio:16/9}.visual-small .caption strong{font-size:13px}.rail{display:grid;grid-template-rows:auto minmax(0,1fr);gap:14px}.sidecard{padding:14px;min-width:0}.sectionhead{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:11px}.sectionhead h2{font-size:16px;margin:0}.sectionhead .sub{font-size:11px;color:var(--muted)}.health{display:flex;align-items:center;gap:7px;color:var(--green);font-size:12px}.health i{width:7px;height:7px;border-radius:50%;background:currentColor;box-shadow:0 0 9px currentColor}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.metric{border:1px solid #293d54;border-radius:10px;background:#091522;padding:9px 7px;min-width:0}.metric .name{color:var(--muted);font-size:10px}.metric .value{font-size:15px;font-weight:750;margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.statusline{margin-top:9px;display:flex;align-items:center;gap:6px;flex-wrap:wrap}.pill{font-size:10px;border:1px solid #314861;padding:4px 7px;border-radius:999px;color:#bbcee2}.pill.safe{border-color:#286c57;color:#66e6ad}.agent{display:grid;grid-template-rows:auto minmax(180px,.85fr) minmax(230px,1.15fr);min-height:0}.agenttitle{display:flex;align-items:center;gap:9px}.agentorb{width:27px;height:27px;border-radius:9px;background:linear-gradient(145deg,var(--cyan),var(--blue));box-shadow:0 0 18px #4fb8ff50;display:grid;place-items:center;color:#06101e;font-weight:900}.agentpart{border-top:1px solid #263b55;padding-top:12px;min-height:0}.parthead{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}.parthead h3{margin:0;font-size:13px}.parthead span{font-size:10px;color:var(--muted)}.calls{display:grid;gap:7px;max-height:210px;overflow:auto;padding-right:3px}.call{display:grid;grid-template-columns:42px 1fr auto;align-items:center;gap:8px;border:1px solid #273c53;background:#091522;border-radius:10px;padding:8px}.stagebadge{height:34px;border-radius:8px;display:grid;place-items:center;background:#172b47;color:#79d9ff;font-size:12px;font-weight:800}.callmain{min-width:0}.callmain b,.callmain span{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.callmain b{font-size:11px}.callmain span{font-size:10px;color:var(--muted);margin-top:3px}.latency{font-size:10px;color:var(--green)}.dialogue{display:flex;flex-direction:column;gap:8px;max-height:285px;overflow:auto;padding-right:4px}.message{display:grid;grid-template-columns:25px 1fr;gap:8px}.avatar{width:25px;height:25px;border-radius:8px;display:grid;place-items:center;background:#172b47;color:#7fdcff;font-size:10px;font-weight:800}.bubble{border:1px solid #2a4058;background:#0a1725;border-radius:4px 11px 11px 11px;padding:8px 9px}.bubble.command{border-color:#5a4d2e;background:#211c12}.bubble.success{border-color:#235d4b;background:#0c201a}.bubble b{display:block;font-size:11px;margin-bottom:3px}.bubble p{font-size:11px;line-height:1.45;color:#bdd0e4;margin:0;word-break:break-word}.bubble time{display:block;color:#667d96;font-size:9px;margin-top:5px}.empty{border:1px dashed #31465e;border-radius:10px;color:#7189a3;padding:18px;text-align:center;font-size:11px}.foot{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-top:12px;color:#6f849c;font-size:10px}.flow{display:flex;align-items:center;gap:7px}.flow b{color:#b8cbe0;font-weight:600}.flow i{width:16px;height:1px;background:#39536e}.debuglink{color:#708ba8;text-decoration:none}.debuglink:hover{color:var(--cyan)}@media(max-width:1200px){.shell{padding:14px}.layout{grid-template-columns:1fr}.rail{grid-template-columns:1fr 1.5fr;grid-template-rows:auto}.agent{min-height:590px}}@media(max-width:760px){.shell{padding:10px}.topbar{align-items:flex-start}.brand h1{font-size:19px}.topmeta{display:none}.visual-row,.rail{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}.agent{min-height:620px}.caption strong{font-size:11px}.caption span{display:none}.foot{display:block}.flow{margin-bottom:7px}}
</style>
<div class='shell'><header class='topbar'><div class='brand'><div class='brandmark' aria-hidden='true'><svg viewBox='0 0 32 32' fill='none'><path d='M5 24V10l11-6 11 6v12l-11 6-7-4' stroke='#6fe2ff' stroke-width='2'/><circle cx='16' cy='16' r='4' fill='#68e8b1'/><path d='m7 11 9 5 9-5M16 16v10' stroke='#718dff' stroke-width='1.5'/></svg></div><div><div class='eyebrow'>Embodied Intelligence · Interactive Navigation</div><h1>具身智能交互导航展示平台</h1></div></div><div class='topmeta'><span class='tag live' id='live'>系统在线</span><span class='tag'>开放词汇感知</span><span class='tag'>分层交互图</span><span class='tag'>模型驱动决策</span></div></header>
<main class='layout'><section class='stage' aria-label='实时交互导航可视化'><div class='visual-primary'><canvas id='view1' width='480' height='270'></canvas><div class='caption'><strong>01 · 实时语义感知</strong><span id='perception-meta'>RGB · YOLOE Box</span></div></div><div class='visual-row'><div class='visual-small'><canvas id='view3' width='480' height='270'></canvas><div class='caption'><strong>03 · 空间理解与交互目标</strong><span>Occupancy · Room · Interaction</span></div></div><div class='visual-small'><canvas id='view6' width='480' height='270'></canvas><div class='caption'><strong>06 · 分层交互语义图</strong><span>Room → Portal → Container → Object</span></div></div></section>
<aside class='rail'><section class='sidecard'><div class='sectionhead'><h2>Go2 实时状态</h2><div class='health'><i></i><span id='stamp'>连接中</span></div></div><div id='metrics' class='metrics'></div><div id='statusline' class='statusline'></div></section><section class='sidecard agent'><div class='sectionhead'><div class='agenttitle'><div class='agentorb'>AI</div><div><h2>交互导航 Agent</h2><div class='sub'>感知 · 建图 · 推理 · 行为闭环</div></div></div><span class='pill safe'>动作安全阻断</span></div><div class='agentpart'><div class='parthead'><h3>汇总的 MLLM 调用</h3><span id='call-count'>0 次记录</span></div><div id='calls' class='calls'></div></div><div class='agentpart'><div class='parthead'><h3>Agent 行为对话</h3><span>含当前行为与交互命令</span></div><div id='dialogue' class='dialogue'></div></div></section></aside></main>
<footer class='foot'><div class='flow'><b>真实环境输入</b><i></i><b>开放词汇感知</b><i></i><b>语义交互图</b><i></i><b>MLLM 决策</b><i></i><b>状态验证</b></div><div><span id='step'>导航步 --</span> · <a class='debuglink' href='/'>进入调试页面</a></div></footer></div>
<script>
const el=id=>document.getElementById(id), clean=v=>String(v??'').replace(/\\s+/g,' ').trim(), short=(v,n=66)=>{v=clean(v);return v.length>n?v.slice(0,n)+'…':v};
function node(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e}
function number(v,d=1){const n=Number(v);return Number.isFinite(n)?n.toFixed(d):'--'}
function velocity(t){const v=t?.velocity||t?.linear_velocity||[];return Array.isArray(v)?Math.hypot(...v.slice(0,3).map(Number)):Number(t?.speed||0)}
function resultText(event){if(!event)return '等待调用';if(event.error)return '调用异常：'+short(event.error);let raw=event.raw_text??event.response?.raw_text??event.payload?.result??'';if(typeof raw==='object')raw=JSON.stringify(raw);try{const o=JSON.parse(raw),rank=o.ranked_ids||o.candidate_id||o.label||o.state||'';return short([Array.isArray(rank)?rank.join(' → '):rank,o.reason].filter(Boolean).join(' · ')||'调用完成')}catch(_){return short(raw||'调用完成')}}
function renderMetrics(s){const t=s.telemetry||{},b=t.battery||{},yaw=Number(t.yaw),items=[['电量',number(b.soc??t.battery_soc,0)+' %'],['移动速度',number(velocity(t),2)+' m/s'],['航向',Number.isFinite(yaw)?number(yaw*180/Math.PI,1)+'°':'--'],['机器人动作',t.mode===undefined?'--':'模式 '+t.mode]];const box=el('metrics');box.replaceChildren();items.forEach(([name,value])=>{const m=node('div','metric');m.append(node('div','name',name),node('div','value',value));box.append(m)});const line=el('statusline');line.replaceChildren(node('span','pill '+(s.link?.connected===false?'':'safe'),s.link?.connected===false?'相机离线':'D435i 在线'),node('span','pill safe','实物动作阻断'),node('span','pill','语义节点 '+(s.graph?.node_count??0)),node('span','pill','感知目标 '+(s.detections?.length??0)));el('stamp').textContent=s.link?.connected===false?'连接异常':'实时连接';el('step').textContent='导航步 '+(s.navigation_step??'--')+' · 相机帧 '+(s.frame_seq??'--');el('perception-meta').textContent='RGB · YOLOE Box · '+(s.detections?.length??0)+' targets'}
function renderCalls(s){const all=[...(s.mllm?.M1||[]),...(s.mllm?.M2||[])].sort((a,b)=>(b.timestamp||0)-(a.timestamp||0));el('call-count').textContent=all.length+' 次近期记录';const box=el('calls');box.replaceChildren();if(!all.length){box.append(node('div','empty','等待真实 MLLM 调用'));return}all.slice(0,6).forEach(e=>{const stage=e.stage||'MLLM',row=node('div','call'),badge=node('div','stagebadge',stage),main=node('div','callmain'),purpose=stage==='M1'?'识别交互属性与状态':'选择下一语义子目标';main.append(node('b','',purpose),node('span','',resultText(e)));row.append(badge,main,node('div','latency',e.latency_s!=null?number(e.latency_s,2)+' s':'完成'));box.append(row)})}
const dialogueLast={};
function dialogueItem(role,title,body,kind='',stamp=''){const sig=[title,body].join('|');if(dialogueLast[role]===sig)return null;dialogueLast[role]=sig;const row=node('div','message'),avatar=node('div','avatar',role),bubble=node('div','bubble '+kind);bubble.append(node('b','',title),node('p','',body));if(stamp)bubble.append(node('time','',new Date(Number(stamp)*1000).toLocaleTimeString()));row.append(avatar,bubble);return row}
function firstText(o,keys){for(const k of keys){const v=o?.[k];if(v!==undefined&&v!==null&&v!=='')return clean(typeof v==='object'?JSON.stringify(v):v)}return ''}
function renderDialogue(s){const n=s.navigation||{},trace=n.decision_trace||{},exec=n.execution_state||{},feedback=n.behavior_feedback||{},interaction=n.interaction_result||{},box=el('dialogue'),rows=[];const candidate=firstText(trace,['executed_candidate_id','model_selected_candidate_id','active_candidate_id'])||firstText(exec,['candidate_id']);const reason=firstText(trace,['model_reason','selection_override_reason','model_result_source','model_error']);rows.push(dialogueItem('A','Agent 决策',candidate?'选择候选 '+candidate+(reason?'；'+short(reason,80):''):'当前没有可执行候选，继续更新感知与交互图','',trace.timestamp));const state=firstText(exec,['state'])||'IDLE',behavior=firstText(exec,['behavior_type']);rows.push(dialogueItem('→','当前行为',behavior?state+' · '+behavior+(candidate?' · '+candidate:''):state==='IDLE'?'保持待机，等待有效子目标':state,'command',exec.timestamp));if(Object.keys(feedback).length){const status=firstText(feedback,['status'])||'反馈更新',target=firstText(feedback,['target_name','target_id','candidate_id']);rows.push(dialogueItem('R','行为反馈',status+(target?' · '+target:''),/SUCCESS|COMPLETE|PASS/i.test(status)?'success':'',feedback.timestamp))}if(Object.keys(interaction).length){const status=firstText(interaction,['status','success'])||'结果已回传',target=firstText(interaction,['source_object_name','object_id','instance_id']);rows.push(dialogueItem('M3','交互结果验证',status+(target?' · '+target:''),'success',interaction.timestamp))}const previous=[...box.children];rows.filter(Boolean).forEach(row=>box.prepend(row));if(!box.children.length&&!previous.length)box.append(node('div','empty','等待 Agent 行为事件'));while(box.children.length>9)box.lastElementChild.remove()}
let videoBusy=false;
async function refreshVideo(){if(videoBusy)return;videoBusy=true;try{const r=await fetch('/snapshot.jpg?ts='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error(r.status);const bmp=await createImageBitmap(await r.blob());[["view1",0,0],["view3",960,0],["view6",960,270]].forEach(([id,x,y])=>el(id).getContext('2d').drawImage(bmp,x,y,480,270,0,0,480,270));bmp.close()}catch(_){el('live').textContent='画面重连中'}finally{videoBusy=false}}
async function refreshState(){try{const r=await fetch('/api/state-summary?ts='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error(r.status);const s=await r.json();renderMetrics(s);renderCalls(s);renderDialogue(s);el('live').textContent='系统在线'}catch(_){el('live').textContent='状态重连中'}}
setInterval(refreshVideo,200);setInterval(refreshState,1000);refreshVideo();refreshState();
</script></html>"""


def _compact_qwen_context(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Keep sparse masks out of the text prompt sent to the remote Qwen."""
    compact_detections: list[dict[str, Any]] = []
    detections = [item for item in snapshot.get("detections", []) if isinstance(item, dict)]
    detections.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    for detection in detections[:12]:
        item = {
            key: detection.get(key)
            for key in (
                "semantic_class", "confidence", "bbox", "mask_area", "depth_median_m",
                "world_position", "aabb_center", "aabb_size", "map_transform_status",
            )
            if key in detection
        }
        mask = detection.get("mask")
        if isinstance(mask, dict):
            item["mask_summary"] = {
                "area": mask.get("area", mask.get("mask_area", 0)),
                "rows": len(mask.get("rows", [])),
                "cols": len(mask.get("cols", [])),
            }
        compact_detections.append(item)

    graph = snapshot.get("graph")
    compact_graph: dict[str, Any] = {}
    if isinstance(graph, dict):
        compact_graph = {
            key: graph.get(key)
            for key in ("scene_id", "episode_id", "source_mode", "graph_revision", "timestamp", "module1_mode")
            if key in graph
        }
        compact_graph["nodes"] = []
        nodes = [item for item in graph.get("nodes", []) if isinstance(item, dict)]
        nodes.sort(key=lambda item: (bool(item.get("is_currently_visible", True)), float(item.get("last_seen") or 0.0)), reverse=True)
        for node in nodes[:16]:
            item = {
                key: node.get(key)
                for key in ("id", "type", "label", "centroid", "aabb_center", "aabb_size", "room_id")
                if key in node
            }
            interaction = node.get("interaction")
            if isinstance(interaction, dict):
                item["interaction"] = {
                    key: interaction.get(key)
                    for key in ("is_interactable", "interaction_mode", "capability", "state", "cost", "requires_interaction", "traversable")
                    if key in interaction
                }
            compact_graph["nodes"].append(item)
        compact_graph["edges"] = graph.get("edges", [])

    return {
        "frame_seq": snapshot.get("frame_seq", -1),
        "navigation_step": snapshot.get("navigation_step", -1),
        "telemetry": _compact_telemetry(snapshot.get("telemetry", {})),
        "detections": compact_detections,
        "graph": compact_graph,
        "consistency": _compact_consistency(snapshot.get("consistency", {})),
    }


def _compact_consistency(consistency: Any) -> dict[str, Any]:
    if not isinstance(consistency, dict):
        return {}
    result = {
        key: consistency.get(key)
        for key in ("status", "counts", "detection_count", "graph_revision", "projection")
        if key in consistency
    }
    result["detections"] = []
    for item in consistency.get("detections", [])[:12]:
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        result["detections"].append({
            "object_id": item.get("object_id", ""),
            "status": item.get("status", ""),
            "metrics": {
                key: metrics.get(key)
                for key in ("bbox_iou", "bbox_center_px", "map_projected_depth_abs_m", "map_distance_m", "map_z_abs_m", "rgbd_depth_lift_abs_m")
                if key in metrics
            },
            "reasons": item.get("reasons", []),
        })
    return result


def _compact_telemetry(telemetry: Any) -> dict[str, Any]:
    if not isinstance(telemetry, dict):
        return {}
    result = {
        key: telemetry.get(key)
        for key in ("position", "velocity", "yaw", "yaw_speed", "mode", "error_code", "body_height")
        if key in telemetry
    }
    battery = telemetry.get("battery")
    if isinstance(battery, dict):
        result["battery"] = {key: battery.get(key) for key in ("soc", "voltage", "current", "power") if key in battery}
    return result


class PhysicalGateway:
    def __init__(self, host: str, port: int, qwen_url: str = "", qwen_model: str = "qwen3.6-35b-a3b-fp8", camera_parent: str = "tf_frame_base_link", camera_x: float = 0.03, camera_y: float = 0.0, camera_z: float = 0.62, camera_roll: float = 0.0, camera_pitch: float = 0.0, camera_yaw: float = 0.0, qwen_auto_interval: float = 0.0, record_dir: str = DEFAULT_RECORD_DIR, record_mode: str = "raw_plus_panels", record_queue_size: int = 4096, record_on_start: bool = False) -> None:
        self.host, self.port = host, port
        self.state, self.gate = RuntimeState(), ReadOnlySafetyGate()
        self.renderer = SixPanelRenderer(self.state)
        self.qwen = QwenClient(qwen_url, qwen_model) if qwen_url else None
        self.recorder = PhysicalRawRecorder(
            record_dir,
            default_mode=record_mode,
            queue_size=record_queue_size,
            panel_fps=5.0,
        )
        self.camera_parent, self.camera_translation, self.camera_rpy = camera_parent, (camera_x, camera_y, camera_z), (camera_roll, camera_pitch, camera_yaw)
        self.state.set_calibration(
            parent_frame=camera_parent,
            translation_m=[camera_x, camera_y, camera_z],
            rpy_rad=[camera_roll, camera_pitch, camera_yaw],
            source="PHYSICAL_NAV_CAMERA_X/Y/Z/ROLL/PITCH/YAW",
            calibrated=any(abs(value) > 1e-12 for value in (camera_x, camera_y, camera_z, camera_roll, camera_pitch, camera_yaw)),
        )
        self.qwen_auto_interval = qwen_auto_interval
        self.http: ThreadingHTTPServer | None = None
        self.https: ThreadingHTTPServer | None = None
        self.https_port = 0
        self.tls_cert = ""
        self.tls_key = ""
        self.phone_stream_url = ""
        # Keep WebSocket ingestion independent from JPEG/PNG decoding. Under
        # transient CPU load only the newest sensor receipt is useful; queuing
        # every old frame creates backpressure on the Go2 and eventually makes
        # websocket-client hit its send timeout.
        self._sensor_lock = threading.Lock()
        self._sensor_event = threading.Event()
        self._latest_sensor_packet: dict[str, Any] | None = None
        self._latest_sensor_stamp = float("-inf")
        self._last_record_snapshot_mono = 0.0
        threading.Thread(target=self._sensor_decode_loop, daemon=True).start()
        if record_on_start:
            self.recorder.start(
                mode=record_mode,
                label="gateway_start",
                metadata={
                    "page": "showcase-dark",
                    "sections": {
                        "perception": "panel1/camera",
                        "spatial": "panel3/room",
                        "graph": "panel6/topology",
                        "right_rail": "state/right_panel",
                    },
                    "audio_source": "phone",
                },
            )

    def _process_sensor_frame(self, packet: dict[str, Any]) -> None:
        if packet.get("camera_imu") and int(packet.get("seq", 0)) % 50 == 0:
            print(f"camera_imu received seq={packet.get('seq')} keys={list(packet['camera_imu'])}", flush=True)
        rgb = _decode(packet["rgb"]["data"], "jpeg")
        depth = _decode(packet["depth"]["data"], "png16")
        self.state.update_frame(
            rgb=rgb,
            depth=depth,
            rgb_b64=packet["rgb"]["data"],
            depth_b64=packet["depth"]["data"],
            depth_scale=float(packet.get("depth_scale", .001)),
            intrinsics=packet.get("intrinsics", {}),
            rgb_intrinsics=packet.get("rgb_intrinsics", packet.get("intrinsics", {})),
            depth_intrinsics=packet.get("depth_intrinsics", packet.get("intrinsics", {})),
            depth_to_color_extrinsics=packet.get("depth_to_color_extrinsics", {}),
            camera_frame=packet.get("camera_frame", ""),
            depth_frame=packet.get("depth_frame", packet.get("camera_frame", "")),
            camera_imu=packet.get("camera_imu", {}),
            frame_seq=int(packet["seq"]),
            frame_stamp=float(packet["stamp"]),
            sync_ms=packet.get("color_depth_sync_ms"),
        )

    def queue_sensor_frame(self, packet: dict[str, Any]) -> None:
        """Validate and retain only the newest not-yet-decoded RGB-D frame."""
        validate_packet(packet)
        # Persist the encoded receipt before the latest-only decode queue can
        # replace it under load.  The recorder is asynchronous and is a no-op
        # when no session is active.
        self.recorder.record_sensor_packet(packet)
        self.state.link_packet("sensor_frame")
        stamp = float(packet.get("stamp", 0.0))
        with self._sensor_lock:
            pending_stamp = (
                float(self._latest_sensor_packet.get("stamp", 0.0))
                if self._latest_sensor_packet is not None
                else float("-inf")
            )
            # A timed-out SSH forwarding channel can finish delivering after
            # its replacement is live. Never let that delayed session move the
            # shared camera receipt backwards or replace a newer queued frame.
            if stamp <= max(self._latest_sensor_stamp, pending_stamp):
                with self.state._lock:
                    self.state.counters["dropped"] += 1
                return
            if self._latest_sensor_packet is not None:
                with self.state._lock:
                    self.state.counters["dropped"] += 1
            self._latest_sensor_packet = packet
            self._sensor_event.set()

    def _sensor_decode_loop(self) -> None:
        while True:
            self._sensor_event.wait()
            with self._sensor_lock:
                packet = self._latest_sensor_packet
                self._latest_sensor_packet = None
                self._sensor_event.clear()
            if packet is None:
                continue
            # Claim before decoding so a delayed packet from another socket
            # cannot enter the queue while this newer receipt is in flight.
            packet_stamp = float(packet.get("stamp", 0.0))
            with self._sensor_lock:
                self._latest_sensor_stamp = max(
                    self._latest_sensor_stamp,
                    packet_stamp,
                )
            try:
                self._process_sensor_frame(packet)
            except Exception as exc:
                self.state.last_error = f"sensor decode: {exc}"

    def receive(self, packet: dict[str, Any]) -> dict[str, Any]:
        validate_packet(packet)
        self.state.link_packet(str(packet.get("type", "")))
        if packet.get("type") == "hello":
            self.state.link_connected(packet)
            return {"type": "ack", "v": 1, "accepted": True, "read_only": True, "capabilities": ["rgb", "depth", "camera_info", "pose", "telemetry"]}
        if packet.get("type") in {"control", "cmd", "lidar", "posture", "speak", "teleop_intent"}:
            # Defensive boundary: even if an old policy client connects to this
            # port, no command is forwarded to Go2 or any local actuator.
            return self.gate.handle_intent(packet)
        if packet["type"] == "sensor_frame":
            # Never decode/process RGB-D inline in the websocket receive
            # thread.  The decoder owns a latest-only queue (capacity=1), so
            # a slow YOLO/depth pipeline cannot make the socket retain old
            # frames and report seconds-old perception results.
            self.queue_sensor_frame(packet)
            # ROS publication is handled by physical_ros_gateway.py using the
            # /api/raw-frame endpoint. Keeping this process ROS-free avoids a
            # Python 3.13/ROS Noetic runtime conflict.
        elif packet["type"] == "telemetry":
            self.recorder.record_telemetry(packet.get("telemetry", {}), packet=packet)
            self.state.update_topic("telemetry", packet.get("telemetry", {}))
        return {"type": "ack", "v": 1, "seq": packet.get("seq", -1), "accepted": True, "read_only": True}

    def start_http(self) -> None:
        handler = type("PhysicalWebHandler", (_WebHandler,), {})
        handler.state, handler.renderer, handler.gate = self.state, self.renderer, self.gate
        # Keep the recorder on the handler class so every HTTP worker (including
        # the phone publisher and recording controls) writes into the same
        # session owned by this gateway.  Omitting this assignment silently
        # disables all HTTP-side recording while sensor receipts still work.
        handler.recorder = self.recorder
        handler.phone_stream_url = self.phone_stream_url
        def submit_qwen(payload: dict[str, Any]) -> dict[str, Any]:
            if self.qwen is None: return {"accepted": False, "error": "Qwen client is disabled"}
            user_prompt = str(payload.get("prompt", "请分析当前全局语义图与感知一致性"))
            snapshot = self.state.snapshot()
            context = _compact_qwen_context(snapshot)
            prompt = user_prompt + "\n\n当前实物平台状态(JSON)：\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
            request_id = f"qwen-{int(time.time() * 1000)}-{secrets.token_hex(3)}"
            request = {"request_id": request_id, "prompt": user_prompt, "context": context, "requested_at": time.time(), "model": self.qwen.model}
            self.state.add_qwen(request)
            self.recorder.record_qwen_event({"request_id": request_id, "kind": "request", **request}, source="web_api")
            def worker() -> None:
                result = self.qwen.chat(prompt, max_tokens=int(payload.get("max_tokens", 256)))
                response = {"request_id": request_id, "kind": "result", "completed_at": time.time(), "prompt": user_prompt, "result": result}
                self.state.add_qwen({}, response)
                self.recorder.record_qwen_event(response, source="web_api")
            threading.Thread(target=worker, daemon=True).start()
            return {"accepted": True, "queued_at": request["requested_at"]}
        # Store as a static callback; otherwise BaseHTTPRequestHandler binds
        # this closure as an instance method and adds an unwanted ``self``.
        handler.qwen_submit = staticmethod(submit_qwen)
        self.http = _ReusableHTTPServer((self.host, self.port), handler)
        camera_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="camera-overlay")
        record_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="physical-recording")
        def camera_overlay_loop() -> None:
            """Publish source-resolution camera overlays independently.

            The six-panel map/graph renderer is intentionally heavier than a
            camera overlay.  Keeping these two paths in one loop made the
            10-Hz perception endpoint inherit map rendering latency.  This
            small loop owns the two cached camera products used by the debug
            page and the enlarged right-hand image; it never touches robot
            control and always works from the newest RuntimeState frame.
            """
            period = 0.1  # 10 Hz perception display target
            while True:
                started = time.monotonic()
                try:
                    # Generate each representation once per camera tick.  The
                    # right-side image intentionally keeps segmentation,
                    # while the presentation six-panel remains box-only.
                    # One source-resolution overlay is sufficient for both
                    # endpoints. Rendering masked and box-only JPEGs in
                    # parallel doubled the CPU cost of the 10-Hz camera
                    # path and starved the actual perception/ROS callbacks.
                    box_future = camera_pool.submit(
                        self.renderer.render_camera_overlay, include_masks=False
                    )
                    camera_box = box_future.result()
                    camera_seg = camera_box
                    with handler.frame_lock:
                        handler.latest_camera_jpeg = camera_seg
                        handler.latest_camera_box_jpeg = camera_box
                except Exception as exc:
                    self.state.last_error = f"camera overlay: {exc}"
                time.sleep(max(0.0, period - (time.monotonic() - started)))

        # Camera perception is a separate 10-Hz product; starting it before
        # the heavier composite loop prevents the right-side image from
        # waiting behind OCC/graph/costmap drawing.
        threading.Thread(target=camera_overlay_loop, name="physical-camera-overlay-10hz", daemon=True).start()

        def render_loop() -> None:
            while True:
                render_started = time.monotonic()
                try:
                    self.renderer.set_capture_panel_streams(self.recorder.is_active())
                    frame = self.renderer.render()
                    with handler.frame_lock:
                        camera_frame = handler.latest_camera_jpeg
                        camera_box_frame = handler.latest_camera_box_jpeg
                    original_panels = self.renderer.latest_original_panels
                    with handler.frame_lock:
                        handler.latest_jpeg = frame
                        handler.latest_camera_jpeg = camera_frame
                        handler.latest_camera_box_jpeg = camera_box_frame
                        handler.latest_panel3_jpeg = original_panels.get(3, b"")
                        handler.latest_panel6_jpeg = original_panels.get(6, b"")
                    if self.recorder.is_active():
                        # The dark showcase's four durable sections are the
                        # perception image (1), OCC/costmap evidence (2–4),
                        # spatial understanding (3), and interaction graph (6)
                        # plus the right-rail JSON snapshot.  Keep 1–4 for a
                        # direct raw chain and panel 6 for exact showcase
                        # replay; all are source rasters, not browser crops.
                        record_panels = dict(self.renderer.latest_panel_streams)
                        topology_bytes = self.renderer.latest_original_panels.get(6, b"")
                        record_stamp = self.state.frame_stamp
                        record_seq = self.state.frame_seq
                        record_step = self.state.navigation_step
                        # JPEG copies and recorder queue submission are kept
                        # out of the render/navigation loop. The recorder has
                        # its own bounded writer queue; this extra single
                        # producer prevents panel bookkeeping from competing
                        # with YOLO, mapping, and explore callbacks.
                        def submit_recording() -> None:
                            self.recorder.record_panel(1, camera_box_frame, stamp=record_stamp, frame_seq=record_seq)
                            for panel_index, panel_bytes in record_panels.items():
                                if panel_index != 1:
                                    self.recorder.record_panel(panel_index, panel_bytes, stamp=record_stamp, frame_seq=record_seq)
                            if topology_bytes:
                                self.recorder.record_panel(6, topology_bytes, stamp=record_stamp, frame_seq=record_seq)
                            self.recorder.record_step_boundary(step_index=record_step, stamp=record_stamp, frame_seq=record_seq)
                        record_pool.submit(submit_recording)
                        now_mono = time.monotonic()
                        if now_mono - self._last_record_snapshot_mono >= 1.0:
                            self._last_record_snapshot_mono = now_mono
                            self.recorder.record_state_snapshot(self.state.snapshot())
                except Exception as exc:
                    self.state.last_error = str(exc)
                # The camera perception overlay is kept at 10 Hz above.  The
                # composite contains four full map rasters plus topology and
                # several JPEG encodes; on the physical CPU it is the heavy
                # product, and rebuilding it at 5 Hz starves ROS callbacks.
                # Refresh it at 0.5 Hz while the browser continues polling at
                # 5 Hz, so the browser always gets a cached frame and never
                # causes a render or JPEG encode in its request path.  The
                # live camera overlay remains 10 Hz; map/topology panels are
                # deliberately lower-rate diagnostics.
                time.sleep(max(0.0, 2.0 - (time.monotonic() - render_started)))
        threading.Thread(target=render_loop, daemon=True).start()
        if self.qwen is not None and self.qwen_auto_interval > 0:
            def qwen_loop() -> None:
                while True:
                    time.sleep(self.qwen_auto_interval)
                    snapshot = self.state.snapshot()
                    compact = _compact_qwen_context(snapshot)
                    prompt = "请根据当前全局语义图、检测结果和一致性诊断，简要报告空间关系异常和需要人工确认的对象。\n" + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
                    request_id = f"qwen-auto-{int(time.time() * 1000)}-{secrets.token_hex(3)}"
                    request = {"request_id": request_id, "prompt": prompt, "requested_at": time.time(), "model": self.qwen.model, "source": "auto_graph_review"}
                    self.state.add_qwen(request)
                    self.recorder.record_qwen_event({"request_id": request_id, "kind": "request", **request}, source="auto_graph_review")
                    result = self.qwen.chat(prompt, max_tokens=256)
                    response = {"request_id": request_id, "kind": "result", "completed_at": time.time(), "source": "auto_graph_review", "result": result}
                    self.state.add_qwen({}, response)
                    self.recorder.record_qwen_event(response, source="auto_graph_review")
            threading.Thread(target=qwen_loop, daemon=True).start()
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        print(f"physical six-panel web: http://{self.host}:{self.port}/", flush=True)
        if self.https_port:
            if not self.tls_cert or not self.tls_key:
                raise ValueError("HTTPS requires both TLS certificate and key")
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(self.tls_cert, self.tls_key)
            self.https = _ReusableHTTPServer((self.host, self.https_port), handler)
            self.https.socket = context.wrap_socket(self.https.socket, server_side=True)
            threading.Thread(target=self.https.serve_forever, daemon=True).start()
            print(f"physical phone HTTPS: https://{self.host}:{self.https_port}/phone-stream", flush=True)


async def run_gateway(args: argparse.Namespace) -> None:
    import websockets
    gateway = PhysicalGateway(
        args.http_host,
        args.http_port,
        args.qwen_url,
        args.qwen_model,
        args.camera_parent,
        args.camera_x,
        args.camera_y,
        args.camera_z,
        args.camera_roll,
        args.camera_pitch,
        args.camera_yaw,
        args.qwen_auto_interval,
        args.record_dir,
        args.record_mode,
        args.record_queue_size,
        args.record_on_start,
    )
    gateway.https_port, gateway.tls_cert, gateway.tls_key = args.https_port, args.tls_cert, args.tls_key
    gateway.phone_stream_url = args.phone_stream_url
    gateway.start_http()
    async def handler(websocket: Any) -> None:
        try:
            async for raw in websocket:
                try:
                    packet = decode_wire_packet(raw)
                    if packet.get("type") == "sensor_frame":
                        gateway.queue_sensor_frame(packet)
                        # Sensor packets are fire-and-forget. Avoid one ACK per
                        # RGB-D frame so the Go2 never has an ACK backlog to
                        # drain before it can publish the next capture.
                        continue
                    reply = gateway.receive(packet)
                    if packet.get("type") == "telemetry":
                        continue
                except Exception as exc:
                    reply = {"type": "error", "accepted": False, "error": str(exc)}
                await websocket.send(json.dumps(reply, separators=(",", ":")))
        except websockets.exceptions.ConnectionClosed:
            # A browser/Go2 reconnect or a normal process shutdown can close
            # without a WebSocket close frame; it is not a sensor error.
            return
        finally:
            gateway.state.link_disconnected()
    # The Go2 endpoint is intentionally send-only and never enters a command
    # receive loop. Disable protocol pings (which require the client to read
    # and answer them); 5 Hz telemetry already provides application-level
    # liveness and avoids a deterministic ping-timeout reconnect every 40 s.
    try:
        async with websockets.serve(
            handler,
            args.ws_host,
            args.ws_port,
            max_size=args.max_message_mb * 1024 * 1024,
            ping_interval=None,
        ):
            print(f"physical sensor WebSocket: ws://{args.ws_host}:{args.ws_port}", flush=True)
            await __import__("asyncio").Future()
    finally:
        # Explicit shutdown drains a background recording session.  A hard
        # process kill still leaves session.json marked as open/incomplete,
        # which the session listing exposes for recovery.
        gateway.recorder.stop(reason="gateway_shutdown")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ws-host", default="0.0.0.0"); p.add_argument("--ws-port", type=int, default=12334)
    p.add_argument("--http-host", default="0.0.0.0"); p.add_argument("--http-port", type=int, default=8765)
    p.add_argument("--https-port", type=int, default=8767, help="HTTPS port for phone camera capture; 0 disables it")
    p.add_argument("--tls-cert", default="/tmp/molmospaces-phone-tls.crt"); p.add_argument("--tls-key", default="/tmp/molmospaces-phone-tls.key")
    p.add_argument("--phone-stream-url", default="https://10.100.5.3:8767/phone-stream", help="HTTPS URL encoded in the phone QR code")
    p.add_argument("--max-message-mb", type=int, default=16)
    p.add_argument("--qwen-url", default="", help="e.g. http://127.0.0.1:18080/v1 after SSH forwarding")
    p.add_argument("--qwen-model", default="qwen3.6-35b-a3b-fp8")
    p.add_argument("--qwen-auto-interval", type=float, default=0.0, help="seconds; 0 disables periodic graph review")
    p.add_argument("--record-dir", default=os.environ.get("PHYSICAL_NAV_RECORD_DIR", DEFAULT_RECORD_DIR), help="local root for physical recording sessions")
    p.add_argument("--record-mode", choices=sorted(PhysicalRawRecorder.MODES), default=os.environ.get("PHYSICAL_NAV_RECORD_MODE", "raw_plus_panels"))
    p.add_argument("--record-queue-size", type=int, default=int(os.environ.get("PHYSICAL_NAV_RECORD_QUEUE_SIZE", "4096")))
    p.add_argument("--record-on-start", action="store_true", default=os.environ.get("PHYSICAL_NAV_RECORD_ON_START", "0") == "1", help="start a raw recording as soon as the gateway starts")
    p.add_argument("--camera-parent", default="tf_frame_base_link")
    p.add_argument("--camera-x", type=float, default=0.03); p.add_argument("--camera-y", type=float, default=0.0); p.add_argument("--camera-z", type=float, default=0.62); p.add_argument("--camera-roll", type=float, default=0.0); p.add_argument("--camera-pitch", type=float, default=0.0); p.add_argument("--camera-yaw", type=float, default=0.0)
    args = p.parse_args()
    if args.https_port:
        _ensure_phone_tls_certificate(args.tls_cert, args.tls_key)
    try:
        import asyncio; asyncio.run(run_gateway(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__": main()
