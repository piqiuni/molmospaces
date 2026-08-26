#!/usr/bin/env python3
"""One-shot Go2 TTS, reply capture, transcription, and translation."""

from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import types
from typing import Any, Optional
import wave


SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_PYTHON = SCRIPT_DIR / "vendor" / "python"
WHISPER_ROOT = SCRIPT_DIR / "vendor" / "whisper.cpp"
WHISPER_CLI = WHISPER_ROOT / "build" / "bin" / "whisper-cli"
WHISPER_MODEL = WHISPER_ROOT / "models" / "ggml-base.bin"
LOCAL_TTS_MODEL_ROOT = SCRIPT_DIR / "vendor" / "local_tts" / "models"
DEFAULT_MATCHA_MODEL_DIR = LOCAL_TTS_MODEL_ROOT / "matcha-icefall-zh-baker"
DEFAULT_MATCHA_VOCODER = LOCAL_TTS_MODEL_ROOT / "vocos-22khz-univ.onnx"

if VENDOR_PYTHON.is_dir():
    sys.path.insert(0, str(VENDOR_PYTHON))

# The WebRTC driver imports sounddevice although this program never opens the
# Jetson ALSA device. Go2 body audio is carried by the WebRTC media track.
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))

import av  # type: ignore  # noqa: E402
import edge_tts  # type: ignore  # noqa: E402
import numpy as np  # type: ignore  # noqa: E402
from unitree_webrtc_connect import (  # type: ignore  # noqa: E402
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from unitree_webrtc_connect.constants import AUDIO_API, RTC_TOPIC  # type: ignore  # noqa: E402


ROBOT_CONTROLLER_IP = "192.168.123.161"
CAPTURE_RATE = 48000
CAPTURE_CHANNELS = 1
CALIBRATION_FRAMES = 75
START_FRAMES = 3
END_SILENCE_FRAMES = 50
PREROLL_FRAMES = 25
AUDIOHUB_CHUNK_SIZE = 4096


def response_status(response: object) -> object:
    if not isinstance(response, dict):
        return None
    return (
        response.get("data", {})
        .get("header", {})
        .get("status", {})
        .get("code")
    )


def response_data(response: object) -> object:
    if not isinstance(response, dict):
        return None
    raw = response.get("data", {}).get("data")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


async def rpc_request(
    conn: UnitreeWebRTCConnection,
    topic: str,
    api_id: int,
    parameter: Optional[object] = None,
    timeout: float = 12.0,
) -> object:
    options: dict[str, object] = {"api_id": api_id}
    if parameter is not None:
        options["parameter"] = parameter
    response = await asyncio.wait_for(
        conn.datachannel.pub_sub.publish_request_new(topic, options),
        timeout=timeout,
    )
    return response


def require_success(operation: str, response: object) -> None:
    code = response_status(response)
    if code != 0:
        raise RuntimeError(f"{operation} failed with status {code}: {response!r}")


def convert_audio(source: Path, target: Path, sample_rate: int) -> None:
    """Decode with PyAV and write signed 16-bit mono PCM WAV."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    pcm_parts: list[bytes] = []
    with av.open(str(source)) as container:
        audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio_stream is None:
            raise RuntimeError(f"no audio stream in {source}")
        for frame in container.decode(audio_stream):
            for converted in resampler.resample(frame):
                pcm_parts.append(converted.to_ndarray().astype(np.int16, copy=False).tobytes())
        for converted in resampler.resample(None):
            pcm_parts.append(converted.to_ndarray().astype(np.int16, copy=False).tobytes())

    if not pcm_parts:
        raise RuntimeError(f"audio conversion returned no PCM samples: {source}")
    with wave.open(str(target), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(pcm_parts))


async def synthesize_text(text: str, voice: str, mp3_path: Path, wav_path: Path) -> None:
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate="-5%",
        volume="+0%",
        pitch="+0Hz",
    )
    await communicate.save(str(mp3_path))
    convert_audio(mp3_path, wav_path, sample_rate=44100)


class MatchaTtsSynthesizer:
    """Persistent sherpa-onnx Matcha model used by the bridge speech worker."""

    def __init__(
        self,
        model_dir: Path = DEFAULT_MATCHA_MODEL_DIR,
        vocoder: Path = DEFAULT_MATCHA_VOCODER,
        num_threads: int = 4,
    ) -> None:
        if num_threads < 1:
            raise ValueError("Matcha thread count must be positive")
        self.model_dir = Path(model_dir)
        self.vocoder = Path(vocoder)
        required = [
            self.model_dir / "model-steps-3.onnx",
            self.model_dir / "lexicon.txt",
            self.model_dir / "tokens.txt",
            self.model_dir / "phone.fst",
            self.model_dir / "date.fst",
            self.model_dir / "number.fst",
            self.vocoder,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing Matcha files: " + ", ".join(missing))

        import sherpa_onnx  # type: ignore

        self.sherpa_onnx = sherpa_onnx
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=str(self.model_dir / "model-steps-3.onnx"),
                vocoder=str(self.vocoder),
                lexicon=str(self.model_dir / "lexicon.txt"),
                tokens=str(self.model_dir / "tokens.txt"),
            ),
            provider="cpu",
            debug=False,
            num_threads=num_threads,
        )
        config = sherpa_onnx.OfflineTtsConfig(
            model=model_config,
            rule_fsts=",".join(
                str(self.model_dir / name)
                for name in ("phone.fst", "date.fst", "number.fst")
            ),
            max_num_sentences=1,
        )
        if not config.validate():
            raise ValueError(f"invalid Matcha TTS config in {self.model_dir}")
        started = time.perf_counter()
        self.tts = sherpa_onnx.OfflineTts(config)
        self.load_seconds = time.perf_counter() - started

    def synthesize(self, text: str, wav_path: Path) -> dict[str, object]:
        generation = self.sherpa_onnx.GenerationConfig()
        generation.sid = 0
        generation.speed = 1.0
        generation.silence_scale = 0.2
        started = time.perf_counter()
        audio = self.tts.generate(text, generation)
        elapsed = time.perf_counter() - started
        if len(audio.samples) == 0:
            raise RuntimeError("Matcha returned zero audio samples")

        source_wav = wav_path.with_name(f"{wav_path.stem}_matcha_22050.wav")
        self.sherpa_onnx.write_wave(
            str(source_wav), audio.samples, audio.sample_rate
        )
        convert_audio(source_wav, wav_path, sample_rate=44100)
        duration = len(audio.samples) / audio.sample_rate
        return {
            "synthesis_backend": "matcha",
            "synthesis_seconds": elapsed,
            "audio_duration_seconds": duration,
            "source_sample_rate": audio.sample_rate,
            "model_load_seconds": self.load_seconds,
        }


class AudioHubSession:
    def __init__(self, conn: UnitreeWebRTCConnection) -> None:
        self.conn = conn
        self.states: list[dict[str, Any]] = []
        self.play_started = asyncio.Event()
        self.play_finished = asyncio.Event()
        self.expected_id: Optional[str] = None
        self.saw_playing = False
        conn.datachannel.pub_sub.subscribe(
            RTC_TOPIC["AUDIO_HUB_PLAY_STATE"], self._on_state
        )

    def _on_state(self, message: object) -> None:
        if not isinstance(message, dict):
            return
        raw = message.get("data")
        try:
            state = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            return
        if not isinstance(state, dict):
            return
        self.states.append(state)
        unique_id = state.get("current_audio_unique_id")
        is_playing = bool(state.get("is_playing"))
        if self.expected_id and unique_id == self.expected_id and is_playing:
            self.saw_playing = True
            self.play_started.set()
        elif self.saw_playing and not is_playing:
            self.play_finished.set()

    async def upload(self, wav_path: Path, custom_name: str) -> str:
        audio_data = wav_path.read_bytes()
        encoded = base64.b64encode(audio_data).decode("ascii")
        chunks = [
            encoded[index : index + AUDIOHUB_CHUNK_SIZE]
            for index in range(0, len(encoded), AUDIOHUB_CHUNK_SIZE)
        ]
        file_md5 = hashlib.md5(audio_data).hexdigest()
        create_time = int(time.time() * 1000)
        for index, chunk in enumerate(chunks, start=1):
            response = await rpc_request(
                self.conn,
                RTC_TOPIC["AUDIO_HUB_REQ"],
                AUDIO_API["UPLOAD_AUDIO_FILE"],
                json.dumps(
                    {
                        "file_name": custom_name,
                        "file_type": "wav",
                        "file_size": len(audio_data),
                        "current_block_index": index,
                        "total_block_number": len(chunks),
                        "block_content": chunk,
                        "current_block_size": len(chunk),
                        "file_md5": file_md5,
                        "create_time": create_time,
                    },
                    ensure_ascii=True,
                ),
            )
            require_success(f"AudioHub upload block {index}/{len(chunks)}", response)
            await asyncio.sleep(0.05)

        for _ in range(10):
            response = await rpc_request(
                self.conn,
                RTC_TOPIC["AUDIO_HUB_REQ"],
                AUDIO_API["GET_AUDIO_LIST"],
                json.dumps({}),
            )
            require_success("AudioHub list", response)
            data = response_data(response)
            audio_list = data.get("audio_list", []) if isinstance(data, dict) else []
            record = next(
                (
                    item
                    for item in audio_list
                    if isinstance(item, dict) and item.get("CUSTOM_NAME") == custom_name
                ),
                None,
            )
            if record and record.get("UNIQUE_ID"):
                return str(record["UNIQUE_ID"])
            await asyncio.sleep(0.5)
        raise RuntimeError("uploaded TTS audio did not appear in AudioHub list")

    async def play(self, unique_id: str) -> None:
        self.expected_id = unique_id
        self.saw_playing = False
        self.play_started.clear()
        self.play_finished.clear()
        response = await rpc_request(
            self.conn,
            RTC_TOPIC["AUDIO_HUB_REQ"],
            AUDIO_API["SELECT_START_PLAY"],
            json.dumps({"unique_id": unique_id}),
        )
        require_success("AudioHub play", response)
        await asyncio.wait_for(self.play_started.wait(), timeout=8.0)
        print("TTS_PLAYBACK_STARTED", flush=True)
        await asyncio.wait_for(self.play_finished.wait(), timeout=30.0)
        print("TTS_PLAYBACK_FINISHED", flush=True)

    async def delete(self, unique_id: str) -> None:
        response = await rpc_request(
            self.conn,
            RTC_TOPIC["AUDIO_HUB_REQ"],
            AUDIO_API["SELECT_DELETE"],
            json.dumps({"unique_id": unique_id}),
        )
        require_success("AudioHub delete", response)


async def speak_text_once(
    text: str,
    *,
    voice: str = "zh-CN-XiaoxiaoNeural",
    robot_ip: str = ROBOT_CONTROLLER_IP,
    volume: Optional[int] = None,
    connect_attempts: int = 2,
    synthesis_backend: str = "edge",
    fallback_backend: str = "disabled",
    matcha_synthesizer: Optional[MatchaTtsSynthesizer] = None,
) -> dict[str, object]:
    """Synthesize and play one text without enabling the microphone."""
    if not text.strip():
        raise ValueError("speech text must not be empty")
    if volume is not None and not 0 <= volume <= 10:
        raise ValueError("volume must be between 0 and 10")
    if synthesis_backend not in {"matcha", "edge"}:
        raise ValueError(f"unsupported synthesis backend: {synthesis_backend}")
    if fallback_backend not in {"edge", "disabled"}:
        raise ValueError(f"unsupported fallback backend: {fallback_backend}")

    summary: dict[str, object] = {
        "text": text,
        "voice": voice,
        "robot_ip": robot_ip,
        "requested_synthesis_backend": synthesis_backend,
        "fallback_backend": fallback_backend,
    }
    with tempfile.TemporaryDirectory(prefix="go2_speech_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        mp3_path = temp_dir / "speech.mp3"
        wav_path = temp_dir / "speech.wav"
        synthesis_started = time.perf_counter()
        try:
            if synthesis_backend == "matcha":
                synthesizer = matcha_synthesizer or MatchaTtsSynthesizer()
                summary.update(synthesizer.synthesize(text, wav_path))
            else:
                await synthesize_text(text, voice, mp3_path, wav_path)
                summary["synthesis_backend"] = "edge"
        except Exception as primary_error:
            if synthesis_backend == "edge" or fallback_backend != "edge":
                raise
            summary["fallback_from"] = synthesis_backend
            summary["fallback_reason"] = f"{type(primary_error).__name__}: {primary_error}"
            await synthesize_text(text, voice, mp3_path, wav_path)
            summary["synthesis_backend"] = "edge"
        summary["total_synthesis_seconds"] = time.perf_counter() - synthesis_started
        summary["wav_bytes"] = wav_path.stat().st_size

        conn: Optional[UnitreeWebRTCConnection] = None
        last_connect_error: Optional[Exception] = None
        for attempt in range(1, max(1, connect_attempts) + 1):
            candidate = UnitreeWebRTCConnection(
                WebRTCConnectionMethod.LocalSTA,
                ip=robot_ip,
            )
            try:
                await asyncio.wait_for(candidate.connect(), timeout=25.0)
                conn = candidate
                summary["connect_attempt"] = attempt
                break
            except Exception as exc:
                last_connect_error = exc
                with contextlib.suppress(Exception):
                    await candidate.disconnect()
                if attempt < connect_attempts:
                    await asyncio.sleep(1.0)
        if conn is None:
            raise RuntimeError(
                f"cannot connect to Go2 AudioHub after {connect_attempts} attempts"
            ) from last_connect_error

        audio_hub = AudioHubSession(conn)
        uploaded_id: Optional[str] = None
        original_volume: Optional[int] = None
        try:
            volume_response = await rpc_request(conn, RTC_TOPIC["VUI"], 1004)
            require_success("VUI GetVolume", volume_response)
            volume_data = response_data(volume_response)
            if isinstance(volume_data, dict) and isinstance(volume_data.get("volume"), int):
                original_volume = int(volume_data["volume"])
            summary["original_volume"] = original_volume
            if volume is not None:
                response = await rpc_request(
                    conn,
                    RTC_TOPIC["VUI"],
                    1003,
                    {"volume": volume},
                )
                require_success("VUI SetVolume", response)
                summary["playback_volume"] = volume

            custom_name = f"policy_tts_{int(time.time() * 1000)}"
            uploaded_id = await audio_hub.upload(wav_path, custom_name)
            summary["audiohub_unique_id"] = uploaded_id
            await audio_hub.play(uploaded_id)
            summary["status"] = "completed"
            return summary
        finally:
            if uploaded_id:
                try:
                    await audio_hub.delete(uploaded_id)
                    summary["temporary_audio_deleted"] = True
                except Exception as exc:
                    summary["temporary_audio_delete_error"] = repr(exc)
            if original_volume is not None and volume is not None:
                try:
                    response = await rpc_request(
                        conn,
                        RTC_TOPIC["VUI"],
                        1003,
                        {"volume": original_volume},
                    )
                    require_success("VUI restore volume", response)
                    summary["volume_restored"] = original_volume
                except Exception as exc:
                    summary["volume_restore_error"] = repr(exc)
            await conn.disconnect()


class OneUtteranceCapture:
    def __init__(self, max_seconds: float) -> None:
        self.max_frames = max(1, round(max_seconds * 50.0))
        self.phase = "calibrating"
        self.noise_levels: list[float] = []
        self.noise_median = 0.0
        self.start_threshold = 0.0
        self.end_threshold = 0.0
        self.start_run = 0
        self.silence_run = 0
        self.recording_frames = 0
        self.pre_roll: collections.deque[bytes] = collections.deque(maxlen=PREROLL_FRAMES)
        self.captured: list[bytes] = []
        self.calibrated = asyncio.Event()
        self.speech_started = asyncio.Event()
        self.utterance_done = asyncio.Event()
        self.peak_rms = 0.0
        self.fixed_mode = False
        self.frame_shape: Optional[tuple[int, ...]] = None
        self.frame_sample_rate: Optional[int] = None

    async def on_frame(self, frame: object) -> None:
        array = frame.to_ndarray().astype(np.int16, copy=False)
        if self.frame_shape is None:
            self.frame_shape = tuple(array.shape)
            self.frame_sample_rate = int(frame.sample_rate)
        interleaved = array.reshape(-1)
        if interleaved.size % 2 == 0:
            left = interleaved[0::2].astype(np.int32)
            right = interleaved[1::2].astype(np.int32)
            mono = ((left + right) // 2).astype(np.int16)
        else:
            mono = interleaved
        raw = mono.tobytes()
        samples = mono.astype(np.float64)
        rms = math.sqrt(float(np.dot(samples, samples)) / max(1, samples.size)) / 32768.0
        self.peak_rms = max(self.peak_rms, rms)

        if self.phase == "calibrating":
            self.noise_levels.append(rms)
            if len(self.noise_levels) >= CALIBRATION_FRAMES:
                self.noise_median = statistics.median(self.noise_levels)
                self.start_threshold = max(
                    0.0065,
                    self.noise_median * 1.25,
                    self.noise_median + 0.0015,
                )
                self.end_threshold = max(
                    self.noise_median * 1.1,
                    self.start_threshold * 0.8,
                )
                self.phase = "calibrated"
                self.calibrated.set()
            return

        if self.phase == "waiting":
            self.pre_roll.append(raw)
            if rms >= self.start_threshold:
                self.start_run += 1
            else:
                self.start_run = 0
            if self.start_run >= START_FRAMES:
                self.captured.extend(self.pre_roll)
                self.pre_roll.clear()
                self.phase = "recording"
                self.speech_started.set()
            return

        if self.phase != "recording":
            return
        self.captured.append(raw)
        self.recording_frames += 1
        if self.fixed_mode:
            if self.recording_frames >= self.max_frames:
                self.phase = "done"
                self.utterance_done.set()
            return
        if rms < self.end_threshold:
            self.silence_run += 1
        else:
            self.silence_run = 0
        if (
            self.silence_run >= END_SILENCE_FRAMES
            or self.recording_frames >= self.max_frames
        ):
            self.phase = "done"
            self.utterance_done.set()

    def begin_waiting(self) -> None:
        if not self.calibrated.is_set():
            raise RuntimeError("capture noise calibration has not completed")
        self.phase = "waiting"
        self.start_run = 0
        self.silence_run = 0
        self.recording_frames = 0
        self.pre_roll.clear()
        self.captured.clear()
        self.speech_started.clear()
        self.utterance_done.clear()
        self.peak_rms = 0.0
        self.fixed_mode = False

    def begin_fixed(self, seconds: float) -> None:
        if not self.calibrated.is_set():
            raise RuntimeError("capture noise calibration has not completed")
        self.phase = "recording"
        self.max_frames = max(1, round(seconds * 50.0))
        self.start_run = 0
        self.silence_run = 0
        self.recording_frames = 0
        self.pre_roll.clear()
        self.captured.clear()
        self.speech_started.set()
        self.utterance_done.clear()
        self.peak_rms = 0.0
        self.fixed_mode = True

    def write_capture(self, path: Path) -> float:
        if not self.captured:
            raise RuntimeError("microphone capture contains no samples")
        payload = b"".join(self.captured)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(CAPTURE_CHANNELS)
            wav_file.setsampwidth(2)
            wav_file.setframerate(CAPTURE_RATE)
            wav_file.writeframes(payload)
        return len(payload) / (CAPTURE_RATE * CAPTURE_CHANNELS * 2)


def run_whisper(audio_path: Path, language: str, translate: bool) -> dict[str, object]:
    if not WHISPER_CLI.is_file() or not WHISPER_MODEL.is_file():
        raise RuntimeError("whisper.cpp CLI or ggml-base model is missing")
    command = [
        str(WHISPER_CLI),
        "--model",
        str(WHISPER_MODEL),
        "--file",
        str(audio_path),
        "--language",
        language,
        "--threads",
        str(min(6, max(1, os.cpu_count() or 1))),
        "--beam-size",
        "5",
        "--best-of",
        "5",
        "--no-gpu",
        "--no-timestamps",
        "--no-prints",
    ]
    if translate:
        command.append("--translate")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    text = result.stdout.strip()
    if result.returncode != 0:
        raise RuntimeError(
            f"whisper-cli failed ({result.returncode}): {result.stderr[-2000:]}"
        )
    return {
        "text": text,
        "returncode": result.returncode,
        "stderr_tail": result.stderr[-1000:].strip(),
    }


def normalize_wav_in_place(path: Path, max_gain: float = 50.0) -> float:
    with wave.open(str(path), "rb") as wav_file:
        params = wav_file.getparams()
        payload = wav_file.readframes(params.nframes)
    samples = np.frombuffer(payload, dtype=np.int16)
    if samples.size == 0:
        return 1.0
    peak = int(np.max(np.abs(samples.astype(np.int32))))
    if peak == 0:
        return 1.0
    gain = min(max_gain, 29490.0 / peak)
    if gain <= 1.0:
        return 1.0
    amplified = np.clip(samples.astype(np.float64) * gain, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setparams(params)
        wav_file.writeframes(amplified.tobytes())
    return gain


async def execute(args: argparse.Namespace) -> dict[str, object]:
    summary: dict[str, object] = {
        "text_to_speak": args.text,
        "voice": args.voice,
        "robot_ip": args.robot_ip,
        "raw_audio_retained": False,
    }
    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip=args.robot_ip,
    )
    uploaded_id: Optional[str] = None
    original_volume: Optional[int] = None

    with tempfile.TemporaryDirectory(prefix="go2_voice_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        tts_mp3 = temp_dir / "tts.mp3"
        tts_wav = temp_dir / "tts.wav"
        reply_raw = temp_dir / "reply_raw.wav"
        reply_16k = temp_dir / "reply_16k.wav"

        if not args.listen_only:
            print("TTS_SYNTHESIS_STARTED", flush=True)
            await synthesize_text(args.text, args.voice, tts_mp3, tts_wav)
            summary["tts_wav_bytes"] = tts_wav.stat().st_size
            print("TTS_SYNTHESIS_FINISHED", flush=True)

        try:
            await asyncio.wait_for(conn.connect(), timeout=25.0)
            summary["webrtc_connected"] = bool(conn.isConnected)

            volume_response = await rpc_request(conn, RTC_TOPIC["VUI"], 1004)
            require_success("VUI GetVolume", volume_response)
            volume_data = response_data(volume_response)
            if isinstance(volume_data, dict) and isinstance(volume_data.get("volume"), int):
                original_volume = int(volume_data["volume"])
            summary["original_volume"] = original_volume
            if args.volume is not None:
                response = await rpc_request(
                    conn,
                    RTC_TOPIC["VUI"],
                    1003,
                    {"volume": args.volume},
                )
                require_success("VUI SetVolume", response)
                summary["playback_volume"] = args.volume

            capture = OneUtteranceCapture(args.max_utterance_seconds)
            conn.audio.add_track_callback(capture.on_frame)
            conn.audio.switchAudioChannel(True)
            print("MICROPHONE_CALIBRATION_STARTED", flush=True)
            await asyncio.wait_for(capture.calibrated.wait(), timeout=8.0)
            summary["vad"] = {
                "noise_median": capture.noise_median,
                "start_threshold": capture.start_threshold,
                "end_threshold": capture.end_threshold,
                "frame_shape": capture.frame_shape,
                "frame_sample_rate": capture.frame_sample_rate,
            }
            print("MICROPHONE_CALIBRATION_FINISHED", flush=True)

            if not args.listen_only:
                audio_hub = AudioHubSession(conn)
                custom_name = f"go2_tts_{int(time.time())}"
                print("TTS_UPLOAD_STARTED", flush=True)
                uploaded_id = await audio_hub.upload(tts_wav, custom_name)
                summary["audiohub_unique_id"] = uploaded_id
                print("TTS_UPLOAD_FINISHED", flush=True)
                await audio_hub.play(uploaded_id)

            if args.fixed_record_seconds is not None:
                capture.begin_fixed(args.fixed_record_seconds)
            else:
                capture.begin_waiting()
            print("READY_FOR_REPLY", flush=True)
            if args.fixed_record_seconds is None:
                try:
                    await asyncio.wait_for(
                        capture.speech_started.wait(), timeout=args.listen_timeout
                    )
                except asyncio.TimeoutError:
                    summary["reply_status"] = "no_speech_timeout"
                    summary["reply_wait_peak_rms"] = capture.peak_rms
                    return summary
            print("REPLY_SPEECH_STARTED", flush=True)
            await asyncio.wait_for(
                capture.utterance_done.wait(),
                timeout=(args.fixed_record_seconds or args.max_utterance_seconds) + 5.0,
            )
            conn.audio.switchAudioChannel(False)
            await asyncio.sleep(0.2)
            duration = capture.write_capture(reply_raw)
            if args.save_reply:
                retained_path = Path(args.save_reply).expanduser().resolve()
                retained_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(reply_raw, retained_path)
                summary["raw_audio_retained"] = True
                summary["raw_audio_path"] = str(retained_path)
            summary["reply_capture"] = {
                "duration_seconds": duration,
                "peak_rms": capture.peak_rms,
                "stopped_by": (
                    "fixed_window"
                    if args.fixed_record_seconds is not None
                    else "max_duration"
                    if capture.recording_frames >= capture.max_frames
                    else "end_silence"
                ),
            }
            print("REPLY_CAPTURE_FINISHED", flush=True)

            convert_audio(reply_raw, reply_16k, sample_rate=16000)
            summary["reply_normalization_gain"] = normalize_wav_in_place(reply_16k)
            print("LOCAL_TRANSCRIPTION_STARTED", flush=True)
            loop = asyncio.get_running_loop()
            transcript = await loop.run_in_executor(
                None, run_whisper, reply_16k, args.language, False
            )
            summary["transcript"] = transcript["text"]
            if not args.skip_translation:
                translation = await loop.run_in_executor(
                    None, run_whisper, reply_16k, args.language, True
                )
                summary["english_translation"] = translation["text"]
            print("LOCAL_TRANSCRIPTION_FINISHED", flush=True)
            summary["reply_status"] = "recognized"
            return summary
        finally:
            datachannel = getattr(conn, "datachannel", None)
            if datachannel and datachannel.data_channel_opened:
                conn.audio.switchAudioChannel(False)
                if uploaded_id:
                    try:
                        await AudioHubSession(conn).delete(uploaded_id)
                        summary["temporary_audiohub_record_deleted"] = True
                    except Exception as exc:
                        summary["temporary_audiohub_record_delete_error"] = repr(exc)
                if original_volume is not None and args.volume is not None:
                    try:
                        response = await rpc_request(
                            conn,
                            RTC_TOPIC["VUI"],
                            1003,
                            {"volume": original_volume},
                        )
                        require_success("VUI restore volume", response)
                        summary["volume_restored"] = original_volume
                    except Exception as exc:
                        summary["volume_restore_error"] = repr(exc)
                await asyncio.sleep(0.2)
            await conn.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="我到门口了，请帮我开门")
    parser.add_argument("--voice", default="zh-CN-XiaoxiaoNeural")
    parser.add_argument("--robot-ip", default=ROBOT_CONTROLLER_IP)
    parser.add_argument("--volume", type=int, default=4)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--listen-timeout", type=float, default=45.0)
    parser.add_argument("--max-utterance-seconds", type=float, default=12.0)
    parser.add_argument(
        "--listen-only",
        action="store_true",
        help="skip TTS generation and playback; only capture and recognize one reply",
    )
    parser.add_argument(
        "--fixed-record-seconds",
        type=float,
        help="record this many seconds without VAD (recommended for low mic gain)",
    )
    parser.add_argument(
        "--save-reply",
        help="retain the raw 48 kHz mono WAV at this path (default: delete it)",
    )
    parser.add_argument("--skip-translation", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.volume <= 10:
        parser.error("--volume must be between 0 and 10")
    if not 5.0 <= args.listen_timeout <= 180.0:
        parser.error("--listen-timeout must be between 5 and 180 seconds")
    if not 2.0 <= args.max_utterance_seconds <= 30.0:
        parser.error("--max-utterance-seconds must be between 2 and 30 seconds")
    if args.fixed_record_seconds is not None and not 3.0 <= args.fixed_record_seconds <= 30.0:
        parser.error("--fixed-record-seconds must be between 3 and 30 seconds")
    return args


def main() -> None:
    args = parse_args()
    try:
        result = asyncio.run(execute(args))
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "error": repr(exc)},
                ensure_ascii=False,
                indent=2,
            )
        )
        raise


if __name__ == "__main__":
    main()
