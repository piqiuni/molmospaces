#!/usr/bin/env python3
"""Minimal, privacy-preserving microphone and low-volume speaker checks."""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import wave
from typing import Any


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def pulse_defaults() -> tuple[str, str]:
    info = run(["pactl", "info"])
    sink_match = re.search(r"^Default Sink:\s*(.+)$", info, re.MULTILINE)
    source_match = re.search(r"^Default Source:\s*(.+)$", info, re.MULTILINE)
    if sink_match is None or source_match is None:
        raise RuntimeError("PulseAudio default sink/source could not be resolved")
    return sink_match.group(1).strip(), source_match.group(1).strip()


def microphone_test(
    source: str,
    seconds: float,
    sample_rate: int,
    channels: int,
) -> dict[str, Any]:
    command = [
        "parec",
        f"--device={source}",
        "--format=s16le",
        f"--rate={sample_rate}",
        f"--channels={channels}",
        "--raw",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        raw, stderr = process.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        process.terminate()
        raw, stderr = process.communicate(timeout=3.0)
    samples = array.array("h")
    samples.frombytes(raw[: len(raw) - len(raw) % 2])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise RuntimeError(f"microphone returned no samples: {stderr.decode(errors='replace')}")
    square_mean = sum(float(value) * float(value) for value in samples) / len(samples)
    rms = math.sqrt(square_mean)
    peak = max(abs(value) for value in samples)
    dbfs = 20.0 * math.log10(max(rms, 1.0) / 32768.0)
    return {
        "source": source,
        "requested_seconds": seconds,
        "captured_seconds": len(samples) / float(sample_rate * channels),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_count": len(samples),
        "rms": round(rms, 2),
        "peak": int(peak),
        "rms_dbfs": round(dbfs, 2),
        "samples_saved": False,
    }


def make_tone(path: str, seconds: float, frequency: float, sample_rate: int) -> None:
    frame_count = int(seconds * sample_rate)
    # Keep the waveform itself quiet (-24 dBFS), in addition to sink volume limiting.
    amplitude = int(32767 * 0.063)
    samples = array.array(
        "h",
        (
            int(amplitude * math.sin(2.0 * math.pi * frequency * index / sample_rate))
            for index in range(frame_count)
        ),
    )
    if sys.byteorder != "little":
        samples.byteswap()
    with wave.open(path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def make_gentle_music(path: str, seconds: float, sample_rate: int) -> None:
    notes = [
        (261.63, 0.60),
        (329.63, 0.60),
        (392.00, 0.80),
        (329.63, 0.60),
        (293.66, 0.60),
        (261.63, 0.80),
    ]
    scale = seconds / sum(duration for _, duration in notes)
    amplitude = 0.045 * 32767
    samples = array.array("h")
    phase = 0.0
    for frequency, base_duration in notes:
        frame_count = max(1, int(base_duration * scale * sample_rate))
        fade_count = min(int(0.04 * sample_rate), frame_count // 3)
        for index in range(frame_count):
            envelope = 1.0
            if fade_count:
                envelope = min(
                    1.0,
                    index / fade_count,
                    (frame_count - 1 - index) / fade_count,
                )
            fundamental = math.sin(phase)
            soft_harmonic = 0.18 * math.sin(phase * 2.0)
            samples.append(int(amplitude * envelope * (fundamental + soft_harmonic)))
            phase += 2.0 * math.pi * frequency / sample_rate
    if sys.byteorder != "little":
        samples.byteswap()
    with wave.open(path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def speaker_test(
    sink: str,
    volume_percent: int,
    seconds: float,
    frequency: float,
    pattern: str,
) -> dict[str, Any]:
    sink_output = run(["pactl", "list", "sinks"])
    sink_block = next(
        (
            block
            for block in re.split(r"(?m)^Sink #\d+\s*$", sink_output)
            if re.search(rf"(?m)^\s*Name:\s*{re.escape(sink)}\s*$", block)
        ),
        None,
    )
    if sink_block is None:
        raise RuntimeError(f"sink not found in pactl output: {sink}")
    volume_match = re.search(r"(?m)^\s*Volume:.*?/\s*(\d+)%", sink_block)
    if volume_match is None:
        raise RuntimeError(f"could not parse current sink volume for {sink}")
    previous_volume = int(volume_match.group(1))
    mute_match = re.search(r"(?m)^\s*Mute:\s*(yes|no)\s*$", sink_block)
    if mute_match is None:
        raise RuntimeError(f"could not parse current mute state for {sink}")
    previous_muted = mute_match.group(1) == "yes"

    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(prefix="go2_quiet_tone_", suffix=".wav", delete=False) as handle:
            temporary_path = handle.name
        if pattern == "gentle":
            make_gentle_music(temporary_path, seconds, 16000)
        else:
            make_tone(temporary_path, seconds, frequency, 16000)
        run(["pactl", "set-sink-volume", sink, f"{volume_percent}%"])
        run(["pactl", "set-sink-mute", sink, "0"])
        run(["paplay", f"--device={sink}", temporary_path])
    finally:
        try:
            run(["pactl", "set-sink-volume", sink, f"{previous_volume}%"])
            run(["pactl", "set-sink-mute", sink, "1" if previous_muted else "0"])
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)
    return {
        "sink": sink,
        "test_volume_percent": volume_percent,
        "tone_seconds": seconds,
        "tone_frequency_hz": frequency,
        "pattern": pattern,
        "previous_volume_percent_restored": previous_volume,
        "previous_mute_restored": previous_muted,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microphone", action="store_true")
    parser.add_argument("--speaker", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--sink")
    parser.add_argument("--microphone-seconds", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, choices=(1, 2), default=1)
    parser.add_argument("--speaker-volume-percent", type=int, default=1)
    parser.add_argument("--speaker-seconds", type=float, default=0.20)
    parser.add_argument("--speaker-frequency", type=float, default=440.0)
    parser.add_argument("--speaker-pattern", choices=("tone", "gentle"), default="tone")
    args = parser.parse_args()
    if not args.microphone and not args.speaker:
        parser.error("select --microphone and/or --speaker")
    if not 0.2 <= args.microphone_seconds <= 5.0:
        parser.error("--microphone-seconds must be between 0.2 and 5.0")
    if not 0 <= args.speaker_volume_percent <= 30:
        parser.error("--speaker-volume-percent must be between 0 and 30")
    if not 0.05 <= args.speaker_seconds <= 10.0:
        parser.error("--speaker-seconds must be between 0.05 and 10.0")
    return args


def main() -> None:
    args = parse_args()
    default_sink, default_source = pulse_defaults()
    results: dict[str, Any] = {
        "default_sink": default_sink,
        "default_source": default_source,
    }
    if args.microphone:
        results["microphone"] = microphone_test(
            args.source or default_source,
            args.microphone_seconds,
            args.sample_rate,
            args.channels,
        )
    if args.speaker:
        results["speaker"] = speaker_test(
            args.sink or default_sink,
            args.speaker_volume_percent,
            args.speaker_seconds,
            args.speaker_frequency,
            args.speaker_pattern,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
