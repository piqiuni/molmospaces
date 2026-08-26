#!/usr/bin/env python3
"""Benchmark Go2-local sherpa-onnx TTS and online edge-tts without playback.

The script only synthesizes audio in memory or into the benchmark output
directory.  It never imports or calls the Go2 AudioHub speaker interface.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_ROOT = Path("/home/unitree/uni_control")
TEXTS = {
    "short": "我到门口了，请帮我开门。",
    "medium": (
        "你好，我是导航机器人。我已经到达目标位置，正在等待下一步指令。"
        "如果需要帮助，请直接告诉我。"
    ),
    "long": (
        "你好，我是导航机器人。我已经完成当前区域的巡检，并到达目标位置。"
        "前方的门目前处于关闭状态，因此暂时无法继续通行。请帮我打开这扇门，"
        "或者告诉我新的导航目标。我会保持静止并等待下一步指令，收到指令后再继续移动，"
        "感谢你的帮助。"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--backends",
        default="aishell3,matcha,melo,edge",
        help="Comma-separated subset of aishell3,matcha,melo,edge",
    )
    parser.add_argument("--edge-voice", default="zh-CN-XiaoxiaoNeural")
    return parser.parse_args()


def add_vendor_path(root: Path) -> None:
    vendor = str(root / "vendor" / "python")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


def model_specs(root: Path) -> Dict[str, Dict[str, Any]]:
    model_root = root / "vendor" / "local_tts" / "models"
    aishell = model_root / "vits-icefall-zh-aishell3"
    matcha = model_root / "matcha-icefall-zh-baker"
    melo = model_root / "vits-melo-tts-zh_en"
    return {
        "aishell3": {
            "kind": "vits",
            "dir": aishell,
            "model": aishell / "model.onnx",
            "tokens": aishell / "tokens.txt",
            "lexicon": aishell / "lexicon.txt",
            "rule_fsts": [aishell / "phone.fst", aishell / "date.fst", aishell / "number.fst"],
            "sid": 66,
        },
        "matcha": {
            "kind": "matcha",
            "dir": matcha,
            "acoustic_model": matcha / "model-steps-3.onnx",
            "vocoder": model_root / "vocos-22khz-univ.onnx",
            "tokens": matcha / "tokens.txt",
            "lexicon": matcha / "lexicon.txt",
            "rule_fsts": [matcha / "phone.fst", matcha / "date.fst", matcha / "number.fst"],
            "sid": 0,
        },
        "melo": {
            "kind": "vits",
            "dir": melo,
            "model": melo / "model.onnx",
            "tokens": melo / "tokens.txt",
            "lexicon": melo / "lexicon.txt",
            "rule_fsts": [melo / "phone.fst", melo / "date.fst", melo / "number.fst"],
            "sid": 0,
        },
    }


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing))


def build_local_tts(sherpa_onnx: Any, spec: Dict[str, Any], threads: int) -> Any:
    common = {
        "provider": "cpu",
        "debug": False,
        "num_threads": threads,
    }
    if spec["kind"] == "vits":
        require_files([spec["model"], spec["tokens"], spec["lexicon"], *spec["rule_fsts"]])
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(spec["model"]),
                lexicon=str(spec["lexicon"]),
                tokens=str(spec["tokens"]),
            ),
            **common,
        )
    else:
        require_files(
            [
                spec["acoustic_model"],
                spec["vocoder"],
                spec["tokens"],
                spec["lexicon"],
                *spec["rule_fsts"],
            ]
        )
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=str(spec["acoustic_model"]),
                vocoder=str(spec["vocoder"]),
                lexicon=str(spec["lexicon"]),
                tokens=str(spec["tokens"]),
            ),
            **common,
        )

    config = sherpa_onnx.OfflineTtsConfig(
        model=model_config,
        rule_fsts=",".join(str(path) for path in spec["rule_fsts"]),
        max_num_sentences=1,
    )
    if not config.validate():
        raise ValueError(f"Invalid sherpa-onnx config for {spec['dir']}")
    return sherpa_onnx.OfflineTts(config)


def synthesize_local(sherpa_onnx: Any, tts: Any, text: str, sid: int) -> Dict[str, Any]:
    generation = sherpa_onnx.GenerationConfig()
    generation.sid = sid
    generation.speed = 1.0
    generation.silence_scale = 0.2
    started = time.perf_counter()
    audio = tts.generate(text, generation)
    elapsed = time.perf_counter() - started
    if len(audio.samples) == 0:
        raise RuntimeError("TTS returned zero samples")
    duration = len(audio.samples) / audio.sample_rate
    return {
        "elapsed_seconds": elapsed,
        "audio_duration_seconds": duration,
        "rtf": elapsed / duration,
        "sample_rate": audio.sample_rate,
        "audio": audio,
    }


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    elapsed = [run["elapsed_seconds"] for run in runs]
    rtfs = [run["rtf"] for run in runs]
    return {
        "runs": [
            {key: value for key, value in run.items() if key != "audio"}
            for run in runs
        ],
        "elapsed_mean_seconds": statistics.mean(elapsed),
        "elapsed_min_seconds": min(elapsed),
        "rtf_mean": statistics.mean(rtfs),
        "audio_duration_mean_seconds": statistics.mean(
            run["audio_duration_seconds"] for run in runs
        ),
    }


def save_wave(sherpa_onnx: Any, path: Path, run: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = run["audio"]
    sherpa_onnx.write_wave(str(path), audio.samples, audio.sample_rate)


def benchmark_local(
    name: str,
    spec: Dict[str, Any],
    repeats: int,
    threads: int,
    sample_dir: Path,
) -> Dict[str, Any]:
    import sherpa_onnx  # type: ignore

    load_started = time.perf_counter()
    tts = build_local_tts(sherpa_onnx, spec, threads)
    load_seconds = time.perf_counter() - load_started

    first = synthesize_local(sherpa_onnx, tts, TEXTS["short"], spec["sid"])
    result: Dict[str, Any] = {
        "mode": "offline_local",
        "model_load_seconds": load_seconds,
        "first_generation": {
            key: value for key, value in first.items() if key != "audio"
        },
        "texts": {},
    }
    del first["audio"]

    for length_name, text in TEXTS.items():
        runs = [
            synthesize_local(sherpa_onnx, tts, text, spec["sid"])
            for _ in range(repeats)
        ]
        if length_name == "medium":
            save_wave(sherpa_onnx, sample_dir / f"{name}_medium.wav", runs[0])
        result["texts"][length_name] = {
            "characters": len(text),
            "text": text,
            **summarize_runs(runs),
        }
        del runs
        gc.collect()
    del tts
    gc.collect()
    return result


def media_duration(path: Path) -> float:
    import av  # type: ignore

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        samples = sum(frame.samples for frame in container.decode(audio=0))
        return samples / float(stream.rate)


async def benchmark_edge(
    root: Path,
    repeats: int,
    voice: str,
    sample_dir: Path,
) -> Dict[str, Any]:
    import edge_tts  # type: ignore

    result: Dict[str, Any] = {
        "mode": "online_network",
        "voice": voice,
        "model_load_seconds": None,
        "texts": {},
    }
    for length_name, text in TEXTS.items():
        runs: List[Dict[str, Any]] = []
        for repeat in range(repeats):
            keep = length_name == "medium" and repeat == 0
            if keep:
                output = sample_dir / "edge_medium.mp3"
                output.parent.mkdir(parents=True, exist_ok=True)
            else:
                handle = tempfile.NamedTemporaryFile(
                    prefix="tts-edge-", suffix=".mp3", dir=str(root / "benchmarks"), delete=False
                )
                output = Path(handle.name)
                handle.close()
            try:
                started = time.perf_counter()
                await edge_tts.Communicate(text=text, voice=voice).save(str(output))
                elapsed = time.perf_counter() - started
                duration = media_duration(output)
                runs.append(
                    {
                        "elapsed_seconds": elapsed,
                        "audio_duration_seconds": duration,
                        "rtf": elapsed / duration,
                        "sample_rate": None,
                        "file_bytes": output.stat().st_size,
                    }
                )
            finally:
                if not keep:
                    output.unlink(missing_ok=True)
        result["texts"][length_name] = {
            "characters": len(text),
            "text": text,
            **summarize_runs(runs),
        }
    return result


def save_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    requested = [item.strip() for item in args.backends.split(",") if item.strip()]
    unknown = sorted(set(requested) - {"aishell3", "matcha", "melo", "edge"})
    if unknown:
        raise ValueError(f"Unknown backends: {', '.join(unknown)}")

    add_vendor_path(args.root)
    output_dir = args.root / "benchmarks"
    sample_dir = output_dir / "samples"
    report_path = output_dir / "tts_benchmark.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "playback_performed": False,
        "host": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "threads": args.threads,
        },
        "repeats": args.repeats,
        "backends": {},
    }
    specs = model_specs(args.root)

    for name in requested:
        print(f"[benchmark] starting {name}", flush=True)
        started = time.perf_counter()
        try:
            if name == "edge":
                backend = asyncio.run(
                    benchmark_edge(args.root, args.repeats, args.edge_voice, sample_dir)
                )
            else:
                backend = benchmark_local(
                    name, specs[name], args.repeats, args.threads, sample_dir
                )
            backend["benchmark_wall_seconds"] = time.perf_counter() - started
            report["backends"][name] = backend
            print(f"[benchmark] completed {name}", flush=True)
        except Exception as exc:
            report["backends"][name] = {
                "error": f"{type(exc).__name__}: {exc}",
                "benchmark_wall_seconds": time.perf_counter() - started,
            }
            print(f"[benchmark] failed {name}: {exc}", file=sys.stderr, flush=True)
        save_report(report_path, report)

    print(f"[benchmark] report: {report_path}")
    print(f"[benchmark] samples: {sample_dir}")
    print("[benchmark] playback performed: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
