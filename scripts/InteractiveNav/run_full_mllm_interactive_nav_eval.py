#!/usr/bin/env python3
"""Run the full-MLLM InteractiveNav evaluator over three V3 domains.

The maintained V3 ROS evaluator is a *single episode* launcher: one process
owns one ROS master and one output tree.  This wrapper is intentionally thin
and only schedules those launchers.  It keeps a logical worker on one ROS
master port, distributes each domain independently across workers, and writes
an aggregate SR while episodes finish.

The default mode consumes the frozen ``channel.json``, ``container.json`` and
``mixed.json`` files and invokes ``run_interactive_nav_v3_ros_eval_test.zsh``
with ``METHOD=full_mllm_object_goal``.  ``--runner-mode raw`` is available for
the older ``run_house7_semantic_exploration_ros_test.zsh`` entry point, but it
does not provide the formal V3 episode scorer and is therefore not the default.

All temporary/cache paths are placed below the requested output directory (or
the explicit ``/home/ldl`` paths supplied by the caller).  No benchmark or
attempt evidence is overwritten: a task directory is unique, while shallow
image/video aliases are atomically refreshed to expose the final attempt and
``--resume`` only reuses a completed task summary from the same output round.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Mapping

try:  # tqdm is optional for lightweight/static environments.
    from tqdm import tqdm
except Exception:  # pragma: no cover - exercised only without tqdm installed.
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_ROOT = Path(
    "/home/ldl/molmospaces/scripts/InteractiveNav/output/"
    "interactive_nav_v3_procthor10k_val_release_v1_1/benchmark"
)
DEFAULT_V3_RUNNER = SCRIPT_DIR / "run_interactive_nav_v3_ros_eval_test.zsh"
DEFAULT_RAW_RUNNER = SCRIPT_DIR / "run_house7_semantic_exploration_ros_test.zsh"
DOMAINS = ("channel", "container", "mixed")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def sha256_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def publish_file(source: Path, destination: Path) -> None:
    """Atomically publish an immutable artifact at a shallow stable path.

    Episode attempts retain their complete evidence trees.  The outer scene
    directory gets a regular-file alias for the final user-facing artifacts;
    a hard link avoids duplicating large videos and ``copy2`` is the portable
    fallback when source and destination are on different filesystems.
    """

    source = Path(source)
    destination = Path(destination)
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"Cannot publish missing or empty artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if (
            not destination.is_symlink()
            and destination.is_file()
            and os.path.samefile(source, destination)
        ):
            return
    except OSError:
        pass

    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def publish_task_artifacts(
    task_dir: Path,
    row: Mapping[str, Any],
    *,
    recording_required: bool,
    topdown_required: bool,
) -> dict[str, Any]:
    """Expose final image/video artifacts directly below one scene directory."""

    published = dict(row)
    specs = (
        (
            "topdown_path",
            "source_topdown_path",
            "published_topdown_path",
            task_dir / "episode_topdown.png",
        ),
        (
            "topdown_metadata_path",
            "source_topdown_metadata_path",
            "published_topdown_metadata_path",
            task_dir / "episode_topdown.json",
        ),
        (
            "six_panel_video",
            "source_six_panel_video",
            "published_six_panel_video",
            task_dir / "overview_6panel.mp4",
        ),
    )
    errors: list[str] = []
    for active_key, source_key, published_key, destination in specs:
        source_value = published.get(source_key) or published.get(active_key)
        published[source_key] = source_value
        published[published_key] = None
        if not source_value:
            continue
        source = Path(str(source_value))
        if not source.is_file() or source.stat().st_size <= 0:
            continue
        try:
            publish_file(source, destination)
        except OSError as exc:
            errors.append(f"{source} -> {destination}: {type(exc).__name__}: {exc}")
            continue
        published[active_key] = str(destination)
        published[published_key] = str(destination)

    published["artifact_publish_dir"] = str(task_dir)
    published["artifact_publish_errors"] = errors
    published_topdown = published.get("published_topdown_path")
    published_topdown_metadata = published.get("published_topdown_metadata_path")
    published_video = published.get("published_six_panel_video")
    published["topdown_exists"] = bool(
        published_topdown and Path(str(published_topdown)).is_file()
    )
    published_topdown_valid = bool(
        published_topdown
        and published_topdown_metadata
        and Path(str(published_topdown)).is_file()
        and Path(str(published_topdown_metadata)).is_file()
    )
    published_video_valid = bool(
        published_video and Path(str(published_video)).is_file()
    )
    published["topdown_artifact_valid"] = (not topdown_required) or published_topdown_valid
    published["recording_artifact_valid"] = (not recording_required) or published_video_valid
    published["artifact_publish_valid"] = (
        not errors
        and ((not topdown_required) or published_topdown_valid)
        and ((not recording_required) or published_video_valid)
    )
    return published


def _copy_file_atomic(source: Path, destination: Path) -> None:
    """Copy one gallery artifact without changing its authoritative source."""

    source = Path(source)
    destination = Path(destination)
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"Cannot copy missing or empty artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if (
            not destination.is_symlink()
            and destination.is_file()
            and os.path.samefile(source, destination)
        ):
            return
    except OSError:
        pass
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _relative_symlink_atomic(source: Path, destination: Path) -> str:
    """Publish a relative symlink and return its stored target."""

    source = Path(source)
    destination = Path(destination)
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"Cannot link missing or empty artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(source.absolute(), start=destination.parent.absolute())
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        temporary.symlink_to(relative_target)
        temporary.replace(destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return relative_target


def _first_existing_artifact(row: Mapping[str, Any], *keys: str) -> Path | None:
    for key in keys:
        value = row.get(key)
        if not value:
            continue
        candidate = Path(str(value))
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        except OSError:
            continue
    return None


def _unlink_generated_alias(path: Path) -> None:
    """Remove only a known gallery file/link, including a broken symlink."""

    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
    except FileNotFoundError:
        pass


def _gallery_stem(task: "EpisodeTask") -> str:
    """Return a compact name unique within one immutable batch plan."""

    return (
        f"{task.ordinal:03d}_{slug(task.domain, max_length=20)}_"
        f"ep{task.local_index:04d}_h{task.house_index:04d}"
    )


def _render_topdown_contact_sheet(
    output_path: Path,
    images: list[tuple[str, Path]],
) -> tuple[Path | None, list[str]]:
    """Render compact previews while leaving full-resolution gallery copies intact."""

    if not images:
        return None, []
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except ImportError as exc:
        return None, [f"Pillow unavailable: {exc}"]

    tile_width = 480
    tile_height = 382
    label_height = 34
    loaded: list[tuple[str, Any]] = []
    errors: list[str] = []
    for label, image_path in images:
        try:
            with Image.open(image_path) as source:
                loaded.append((label, source.convert("RGB")))
        except (OSError, ValueError) as exc:
            errors.append(f"{image_path}: {type(exc).__name__}: {exc}")
    if not loaded:
        return None, errors

    columns = min(5, len(loaded))
    rows = (len(loaded) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (tile_width * columns, tile_height * rows),
        (242, 242, 242),
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    try:
        for index, (label, source) in enumerate(loaded):
            thumbnail = ImageOps.contain(
                source,
                (tile_width - 12, tile_height - label_height - 12),
                method=Image.Resampling.LANCZOS,
            )
            column = index % columns
            row = index // columns
            x0 = column * tile_width
            y0 = row * tile_height
            image_x = x0 + (tile_width - thumbnail.width) // 2
            image_y = y0 + label_height + (tile_height - label_height - thumbnail.height) // 2
            sheet.paste(thumbnail, (image_x, image_y))
            draw.rectangle(
                (x0, y0, x0 + tile_width - 1, y0 + label_height - 1),
                fill=(255, 255, 255),
            )
            draw.text((x0 + 8, y0 + 7), label, fill=(20, 20, 20), font=font)
            thumbnail.close()
    finally:
        for _label, source in loaded:
            source.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.png"
    )
    try:
        sheet.save(temporary, format="PNG", optimize=True)
        temporary.replace(output_path)
    finally:
        sheet.close()
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return output_path, errors


def publish_batch_galleries(
    output_dir: Path,
    tasks: list["EpisodeTask"],
    rows: list[dict[str, Any]],
    *,
    recording_enabled: bool,
    topdown_enabled: bool,
) -> dict[str, Any]:
    """Collect final episode artifacts into stable batch-level galleries.

    Top-down reports are copied so the gallery remains directly portable. Videos
    are exposed as relative symlinks to avoid duplicating potentially large MP4s.
    The combined contact sheet intentionally lives in ``output_dir`` rather than
    inside ``topdown_gallery``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    topdown_dir = output_dir / "topdown_gallery"
    video_dir = output_dir / "video_gallery"
    by_episode = {
        (str(row.get("domain")), int(row.get("local_index", -1))): row
        for row in rows
    }
    topdown_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    contact_images: list[tuple[str, Path]] = []
    stems: set[str] = set()

    for task in sorted(tasks, key=lambda item: item.ordinal):
        stem = _gallery_stem(task)
        if stem in stems:
            raise ValueError(f"Duplicate gallery artifact name: {stem}")
        stems.add(stem)
        row = by_episode.get((task.domain, task.local_index), {})
        result_complete = bool(row.get("result_complete"))

        if topdown_enabled:
            gallery_png = topdown_dir / f"{stem}_topdown.png"
            gallery_json = topdown_dir / f"{stem}_topdown.json"
            source_png = _first_existing_artifact(
                row,
                "source_topdown_path",
                "published_topdown_path",
                "topdown_path",
            )
            source_json = _first_existing_artifact(
                row,
                "source_topdown_metadata_path",
                "published_topdown_metadata_path",
                "topdown_metadata_path",
            )
            if result_complete and source_png is None:
                errors.append(
                    f"topdown {task.domain}:{task.local_index}: missing PNG"
                )
            if result_complete and source_json is None:
                errors.append(
                    f"topdown {task.domain}:{task.local_index}: missing JSON sidecar"
                )
            if source_png is None:
                _unlink_generated_alias(gallery_png)
                _unlink_generated_alias(gallery_json)
            elif source_json is None:
                _unlink_generated_alias(gallery_json)
            if source_png is not None:
                try:
                    if source_json is not None:
                        _copy_file_atomic(source_json, gallery_json)
                    _copy_file_atomic(source_png, gallery_png)
                    metadata = read_json(source_json) if source_json is not None else None
                    coverage = metadata.get("coverage", {}) if isinstance(metadata, dict) else {}
                    scene_background = (
                        metadata.get("scene_background", {}) if isinstance(metadata, dict) else {}
                    )
                    coverage_ratio = coverage.get("exploration_coverage_ratio")
                    topdown_rows.append(
                        {
                            "global_index": task.ordinal,
                            "domain": task.domain,
                            "episode_index": task.local_index,
                            "house_index": task.house_index,
                            "case_id": task.case_id,
                            "coverage": coverage_ratio,
                            "mapped_free": coverage.get("mapped_free_coverage_ratio"),
                            "false_occupied": coverage.get(
                                "mapped_occupied_on_gt_free_ratio"
                            ),
                            "map_source": scene_background.get("map_source"),
                            "scene_mode": scene_background.get("mode"),
                            "source_png": str(source_png),
                            "source_json": None if source_json is None else str(source_json),
                            "gallery_png": gallery_png.name,
                            "gallery_json": gallery_json.name if source_json is not None else None,
                            "png_bytes": gallery_png.stat().st_size,
                        }
                    )
                    contact_images.append(
                        (
                            f"{task.ordinal:03d} {task.domain} "
                            f"ep{task.local_index:04d} h{task.house_index:04d} "
                            + (
                                f"cov {100.0 * float(coverage_ratio):.1f}%"
                                if isinstance(coverage_ratio, (int, float))
                                else "cov n/a"
                            ),
                            gallery_png,
                        )
                    )
                except OSError as exc:
                    _unlink_generated_alias(gallery_png)
                    _unlink_generated_alias(gallery_json)
                    errors.append(
                        f"topdown {task.domain}:{task.local_index}: "
                        f"{type(exc).__name__}: {exc}"
                    )

        if recording_enabled:
            link_path = video_dir / f"{stem}_overview_6panel.mp4"
            source_video = _first_existing_artifact(
                row,
                "source_six_panel_video",
                "published_six_panel_video",
                "six_panel_video",
            )
            if result_complete and source_video is None:
                errors.append(
                    f"video {task.domain}:{task.local_index}: missing overview_6panel.mp4"
                )
            if source_video is None:
                _unlink_generated_alias(link_path)
            if source_video is not None:
                try:
                    relative_target = _relative_symlink_atomic(source_video, link_path)
                    video_rows.append(
                        {
                            "global_index": task.ordinal,
                            "domain": task.domain,
                            "episode_index": task.local_index,
                            "house_index": task.house_index,
                            "case_id": task.case_id,
                            "source_video": str(source_video),
                            "link_name": link_path.name,
                            "relative_target": relative_target,
                            "source_bytes": source_video.stat().st_size,
                            "is_symlink": link_path.is_symlink(),
                            "link_resolves_to_expected_source": (
                                link_path.resolve() == source_video.resolve()
                            ),
                            "link_readable": link_path.is_file(),
                        }
                    )
                except OSError as exc:
                    _unlink_generated_alias(link_path)
                    errors.append(
                        f"video {task.domain}:{task.local_index}: "
                        f"{type(exc).__name__}: {exc}"
                    )

    contact_sheet: Path | None = None
    if topdown_enabled:
        topdown_dir.mkdir(parents=True, exist_ok=True)
        try:
            contact_sheet, contact_errors = _render_topdown_contact_sheet(
                output_dir / "contact_sheet_all.png",
                contact_images,
            )
            errors.extend(f"contact sheet: {error}" for error in contact_errors)
        except OSError as exc:
            errors.append(f"contact sheet: {type(exc).__name__}: {exc}")
        if contact_sheet is None:
            _unlink_generated_alias(output_dir / "contact_sheet_all.png")
        atomic_json(
            topdown_dir / "index.json",
            {
                "schema_version": "interactive_nav_topdown_gallery_v1",
                "created_at": utc_now(),
                "source_batch": str(output_dir),
                "copy_policy": (
                    "PNG and JSON sidecars copied; original files retained unchanged"
                ),
                "count": len(topdown_rows),
                "contact_sheet": (
                    None if contact_sheet is None else f"../{contact_sheet.name}"
                ),
                "rows": topdown_rows,
                "errors": errors,
            },
        )
    if recording_enabled:
        video_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            video_dir / "index.json",
            {
                "schema_version": "interactive_nav_video_gallery_v1",
                "created_at": utc_now(),
                "source_batch": str(output_dir),
                "link_policy": (
                    "Relative symbolic links; source videos retained unchanged; "
                    "no video bytes copied"
                ),
                "count": len(video_rows),
                "symlink_count": sum(bool(row["is_symlink"]) for row in video_rows),
                "total_source_bytes": sum(int(row["source_bytes"]) for row in video_rows),
                "rows": video_rows,
                "errors": errors,
            },
        )

    summary = {
        "topdown_gallery": str(topdown_dir) if topdown_enabled else None,
        "topdown_count": len(topdown_rows),
        "video_gallery": str(video_dir) if recording_enabled else None,
        "video_count": len(video_rows),
        "contact_sheet": None if contact_sheet is None else str(contact_sheet),
        "error_count": len(errors),
        "errors": errors,
    }
    atomic_json(output_dir / "artifact_gallery_summary.json", summary)
    return summary


def as_episode_list(payload: Any, path: Path) -> list[dict[str, Any]]:
    """Normalize both the released list schema and an ``episodes`` wrapper."""

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        rows = payload["episodes"]
    else:
        raise ValueError(f"Benchmark must be a JSON list (or episodes list): {path}")
    result = [row for row in rows if isinstance(row, dict)]
    if len(result) != len(rows):
        raise ValueError(f"Benchmark contains non-object episode entries: {path}")
    return result


def slug(value: str, max_length: int = 100) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in value)
    safe = safe.strip("-") or "episode"
    return safe[:max_length]


@dataclass(frozen=True)
class EpisodeTask:
    """One independently launchable benchmark episode."""

    ordinal: int
    domain: str
    local_index: int
    case_id: str
    house_index: int
    benchmark: str
    worker_id: int
    master_port: int
    task_dir: str

    @property
    def ros_master_uri(self) -> str:
        return f"http://127.0.0.1:{self.master_port}"


def task_key(task: EpisodeTask) -> str:
    return f"{task.domain}:{task.local_index}:{task.case_id}"


def benchmark_case_id(episode: Mapping[str, Any], domain: str, index: int) -> str:
    interactive = episode.get("interactive_nav")
    if isinstance(interactive, Mapping) and interactive.get("case_id"):
        return str(interactive["case_id"])
    if episode.get("case_id"):
        return str(episode["case_id"])
    return f"{domain}_episode_{index:04d}"


def choose_episodes(
    path: Path,
    domain: str,
    count: int,
    *,
    offset: int = 0,
    shuffle_seed: int | None = None,
    allow_short: bool = False,
) -> list[tuple[int, dict[str, Any]]]:
    payload = read_json(path)
    episodes = as_episode_list(payload, path)
    if offset < 0:
        raise ValueError("--episode-offset must be non-negative")
    if count < 1:
        raise ValueError("--episodes-per-domain must be positive")
    available = len(episodes) - offset
    if available < count and not allow_short:
        raise ValueError(
            f"{domain} has only {max(0, available)} episodes after offset {offset}; "
            f"requested {count}"
        )
    # Keep the frozen file index paired with each episode.  Shuffling only the
    # payload while regenerating ``local_index`` would launch the wrong episode
    # in the V3 runner, whose second positional argument is the file index.
    selected = list(enumerate(episodes[offset : offset + count], start=offset))
    if shuffle_seed is not None:
        import random

        random.Random(int(shuffle_seed) + DOMAINS.index(domain)).shuffle(selected)
    return selected


def build_tasks(
    benchmark_paths: Mapping[str, Path],
    *,
    output_dir: Path,
    episodes_per_domain: int,
    episode_offset: int,
    shuffle_seed: int | None,
    allow_short_domain: bool,
    workers: int,
    base_master_port: int,
) -> list[EpisodeTask]:
    """Build a balanced worker assignment.

    Episodes are distributed independently within each domain (worker
    ``i % workers``).  Thus, with 10 workers and 10 episodes per domain, every
    worker receives exactly one channel, one container and one mixed episode;
    unequal domain lengths differ by at most one episode per worker.
    """

    selected_by_domain: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for domain in DOMAINS:
        selected_by_domain[domain] = choose_episodes(
            benchmark_paths[domain],
            domain,
            episodes_per_domain,
            offset=episode_offset,
            shuffle_seed=shuffle_seed,
            allow_short=allow_short_domain,
        )

    tasks: list[EpisodeTask] = []
    ordinal = 0
    for domain in DOMAINS:
        for local_index, episode in selected_by_domain[domain]:
            case_id = benchmark_case_id(episode, domain, local_index)
            try:
                house_index = int(episode["house_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{domain}[{local_index}] has no integer house_index") from exc
            worker_id = (local_index - episode_offset) % workers
            task_dir = output_dir / domain / f"episode_{local_index:04d}_{slug(case_id, 72)}"
            tasks.append(
                EpisodeTask(
                    ordinal=ordinal,
                    domain=domain,
                    local_index=local_index,
                    case_id=case_id,
                    house_index=house_index,
                    benchmark=str(benchmark_paths[domain]),
                    worker_id=worker_id,
                    master_port=base_master_port + worker_id,
                    task_dir=str(task_dir),
                )
            )
            ordinal += 1
    return tasks


def preflight_master_ports(base_master_port: int, workers: int) -> None:
    """Fail early when a requested localhost master port is already bound.

    This check is deliberately read-only: it never kills or reconfigures an
    existing ROS master.  There is still a small TOCTOU window before roscore
    starts, so the runner's own startup check remains authoritative.
    """

    for worker_id in range(workers):
        port = int(base_master_port) + worker_id
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(
                f"ROS master port {port} (worker {worker_id}) is unavailable; "
                "choose --base-master-port or stop the owning process explicitly"
            ) from exc
        finally:
            probe.close()


def terminate_group(process: subprocess.Popen[Any], grace_s: float = 30.0) -> None:
    """Terminate a runner process and all children without a global pkill."""

    if process.poll() is not None:
        return
    for signum, wait_s in (
        (signal.SIGINT, grace_s),
        (signal.SIGTERM, 10.0),
        (signal.SIGKILL, 2.0),
    ):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


def _attempt_process_records(attempt_dir: Path) -> list[tuple[int, int, int, str]]:
    """Find processes whose command line belongs to one episode attempt."""

    marker = str(attempt_dir)
    records: list[tuple[int, int, int, str]] = []
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return records
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\\0", b" ").decode(
                errors="ignore"
            )
            fields = (entry / "stat").read_text(encoding="utf-8").split()
            ppid, pgid = int(fields[3]), int(fields[4])
        except (OSError, ValueError, IndexError):
            continue
        if marker in cmdline:
            records.append((pid, ppid, pgid, cmdline))
    return records


def cleanup_attempt_processes(attempt_dir: Path, grace_s: float = 5.0) -> None:
    """Stop an incomplete attempt's ROS tree without a global process kill."""

    if not str(attempt_dir):
        return

    def matching() -> list[tuple[int, int, int, str]]:
        return _attempt_process_records(attempt_dir)

    records = matching()
    if not records:
        return
    # A detached roslaunch/roscore supervisor may not include the attempt path
    # in its own argv. Walk only through known ROS/evaluator ancestors of a
    # matching node, then terminate their dedicated process groups.
    targets = {pid for pid, _ppid, _pgid, _cmd in records}
    frontier = list(targets)
    while frontier:
        child = frontier.pop()
        try:
            fields = Path(f"/proc/{child}/stat").read_text(encoding="utf-8").split()
            parent = int(fields[3])
            parent_cmd = (
                Path(f"/proc/{parent}/cmdline")
                .read_bytes()
                .replace(b"\\0", b" ")
                .decode(errors="ignore")
            )
        except (OSError, ValueError, IndexError):
            continue
        if parent > 1 and any(
            token in parent_cmd
            for token in ("roslaunch", "roscore", "rosmaster", "run_interactive_nav_")
        ) and parent not in targets:
            targets.add(parent)
            frontier.append(parent)

    own_group = os.getpgrp()
    for signum, wait_s in (
        (signal.SIGINT, min(float(grace_s), 2.0)),
        (signal.SIGTERM, float(grace_s)),
        (signal.SIGKILL, 1.0),
    ):
        current = matching()
        groups = {
            pgid for _pid, _ppid, pgid, _cmd in current if pgid > 1 and pgid != own_group
        }
        pids = {pid for pid, _ppid, _pgid, _cmd in current}
        pids.update(pid for pid in targets if Path(f"/proc/{pid}").exists())
        for pid in sorted(pids):
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, signum)
            except (ProcessLookupError, PermissionError):
                pass
        for pgid in sorted(groups):
            try:
                os.killpg(pgid, signum)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if not matching():
                return
            time.sleep(0.1)


def assigned_endpoint(worker_id: int, endpoints: list[str] | None) -> str | None:
    if not endpoints:
        return None
    return str(endpoints[worker_id % len(endpoints)])


def derive_model_env(task_dir: Path, source: Path, endpoint: str) -> Path:
    text = source.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text += "\n"
    # dotenv accepts a quoted value; shlex.quote also protects '#' and spaces.
    text += (
        "\n# Generated by run_full_mllm_interactive_nav_eval.py.\n"
        f"SEMANTIC_MODEL_ENDPOINT={shlex.quote(endpoint)}\n"
    )
    target = task_dir / "semantic_model.env"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    return target


def make_environment(
    task: EpisodeTask,
    args: argparse.Namespace,
    *,
    attempt_dir: Path,
) -> dict[str, str]:
    """Construct an isolated environment for one V3 or raw runner."""

    cache_root = args.output_dir / "_runtime_cache"
    tmp_dir = attempt_dir / "tmp"
    ros_home = attempt_dir / "ros_home"
    ros_log_dir = ros_home / "log"
    mpl_dir = attempt_dir / "mplconfig"
    nltk_dir = cache_root / "nltk"
    for path in (cache_root, tmp_dir, ros_home, ros_log_dir, mpl_dir, nltk_dir):
        path.mkdir(parents=True, exist_ok=True)
    cache_hf = cache_root / "hf"
    cache_torch = cache_root / "torch"
    cache_cuda = cache_root / "cuda"
    for path in (cache_hf, cache_torch, cache_cuda):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "ROS_MASTER_URI": task.ros_master_uri,
            "ROS_IP": env.get("ROS_IP", "127.0.0.1"),
            "ROS_HOSTNAME": env.get("ROS_HOSTNAME", "127.0.0.1"),
            "ROS_HOME": str(ros_home),
            "ROS_LOG_DIR": str(ros_log_dir),
            "TMPDIR": str(tmp_dir),
            "XDG_CACHE_HOME": str(cache_root),
            "HF_HOME": str(cache_hf),
            "TORCH_HOME": str(cache_torch),
            "CUDA_CACHE_PATH": str(cache_cuda),
            "MPLCONFIGDIR": str(mpl_dir),
            "NLTK_DATA": str(nltk_dir),
            "PYTHONUNBUFFERED": "1",
            # container_scene_probe otherwise falls back to /tmp; keep every
            # writable scene mirror inside this episode's large-volume cache.
            "INTERACTIVE_NAV_SCENE_MIRROR": str(attempt_dir / "scene_mirror"),
            "INTERACTIVE_NAV_DEFAULT_SCENE_MIRROR": str(cache_root / "scene_mirror"),
            "BENCHMARK": task.benchmark,
            # The raw legacy runner has no formal V3 scorer, but it can still
            # annotate its static top-down report with the selected benchmark
            # episode when one is available.  These are consumed only by the
            # post-run renderer and never exposed to the policy.
            "INTERACTIVE_NAV_TOPDOWN_BENCHMARK": task.benchmark,
            "INTERACTIVE_NAV_TOPDOWN_EPISODE_INDEX": str(task.local_index),
            "INTERACTIVE_NAV_TOPDOWN_CASE_ID": task.case_id,
            "MAX_STEPS": str(args.max_steps),
            "STEP_BUDGET_MODE": str(args.step_budget_mode),
            "MIN_STEPS": str(args.min_steps),
            "FAST_EVAL": "false" if args.recording else "true",
            "METHOD": "full_mllm_object_goal" if args.runner_mode == "v3" else "full_mllm_exploration",
            "POLICY": "ros_object_goal_rule",
        }
    )
    ros_overrides = {
        "ROS_ACTION_TIMEOUT_S": args.ros_action_timeout_s,
        "ROS_COMMAND_STARVATION_TIMEOUT_S": args.ros_command_starvation_timeout_s,
        "ROS_OBSERVATION_TURN_MULTIPLIER": args.ros_observation_turn_multiplier,
    }
    for key, value in ros_overrides.items():
        if value is not None:
            env[key] = str(value)
    if args.runner_mode == "raw":
        env.update(
            {
                "HOUSE_IND": str(task.house_index),
                "SCENE_SEED": str(task.house_index),
                "ROUTE_ID": task.case_id,
                "USE_FIXED_ROUTE": "false",
                "TASK_HORIZON": str(args.max_steps),
                "SIM_TIMEOUT_S": str(args.scene_timeout_s),
                "ENABLE_RECORDING": "true" if args.recording else "false",
                "INTERACTIVE_NAV_RENDER_TOPDOWN": "true" if args.render_topdown else "false",
                "RAW_DATA_SPLIT": str(args.raw_data_split),
                "FORCE_CLOSE_CONTAINERS": "true",
                "CLEAN_INTERMEDIATE": "false",
            }
        )
    if args.conda_env:
        env["CONDA_ENV"] = str(args.conda_env)
    if args.python_bin:
        env["PYTHON_BIN"] = str(args.python_bin)
    if args.ros_setup:
        env["ROS_SETUP"] = str(args.ros_setup)
    if args.semantic_model_env_file:
        endpoint = assigned_endpoint(task.worker_id, args.model_endpoints)
        if endpoint:
            env["SEMANTIC_MODEL_ENV_FILE"] = str(
                derive_model_env(attempt_dir, args.semantic_model_env_file, endpoint)
            )
        else:
            env["SEMANTIC_MODEL_ENV_FILE"] = str(args.semantic_model_env_file)
    if args.mujoco_egl_devices:
        env["MUJOCO_EGL_DEVICE_ID"] = str(
            args.mujoco_egl_devices[task.worker_id % len(args.mujoco_egl_devices)]
        )
    return env


def result_paths(task: EpisodeTask, args: argparse.Namespace) -> list[Path]:
    root = Path(task.task_dir)
    if args.runner_mode == "v3":
        return sorted(
            (root / "eval" / "episodes").glob(
                f"{task.local_index:04d}_*/episode_result.json"
            )
        )
    candidate = root / "semantic_exploration_result.json"
    return [candidate] if candidate.is_file() else []


def extract_timing_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    timing = result.get("timing_summary") or result.get("step_timing") or {}
    if not isinstance(timing, Mapping):
        return {"timing_summary": timing}
    phases = timing.get("phases")
    if isinstance(phases, Mapping):
        def phase_mean(name: str) -> float | None:
            value = phases.get(name)
            return float(value.get("mean")) if isinstance(value, Mapping) and isinstance(value.get("mean"), (int, float)) else None
        return {
            "timing_summary": dict(timing),
            "step_timing_loop_ms_avg": phase_mean("step_total"),
            "step_timing_policy_ms_avg": phase_mean("policy_act"),
            "step_timing_task_ms_avg": phase_mean("base_action"),
        }
    return {
        "timing_summary": dict(timing),
        "step_timing_loop_ms_avg": timing.get("loop_ms_avg"),
        "step_timing_policy_ms_avg": timing.get("policy_ms_avg"),
        "step_timing_task_ms_avg": timing.get("task_ms_avg"),
    }


def parse_episode_result(task: EpisodeTask, args: argparse.Namespace) -> dict[str, Any]:
    paths = result_paths(task, args)
    result_path = paths[0] if len(paths) == 1 else None
    document = read_json(result_path) if result_path else {}
    if not isinstance(document, dict):
        document = {}
    if args.runner_mode == "v3":
        result = document.get("result")
        if not isinstance(result, dict):
            result = document
        complete = document.get("status") == "complete" and result.get("status") == "complete"
        success = bool(result.get("success", False)) if complete else False
        row = {
            "case_id": result.get("case_id") or task.case_id,
            "house_index": result.get("house_index", task.house_index),
            "interaction_requirement": result.get("interaction_requirement"),
            "terminal_reason": result.get("terminal_reason"),
            "success": success,
            "task_success": bool(result.get("task_success", False)) if complete else False,
            "interaction_conditioned_success": bool(
                result.get("interaction_conditioned_success", success)
            )
            if complete
            else False,
            "nav_success": bool(result.get("nav_success", False)) if complete else False,
            "required_interaction_success": result.get("required_interaction_success"),
            "sequence_success": result.get("sequence_success"),
            "interaction_action_count": result.get("interaction_action_count"),
            "correct_interaction_action_count": result.get("correct_interaction_action_count"),
            "invalid_interaction_action_count": result.get("invalid_interaction_action_count"),
            "step_count": result.get("step_count"),
            "applied_action_step_count": result.get("applied_action_step_count"),
            "no_fresh_action_count": result.get("no_fresh_action_count"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "spl": result.get("spl"),
            "result_status": result.get("status"),
        }
    else:
        result = document
        complete = bool(result)
        success = bool(
            result.get("overall_success")
            or result.get("target_goal_success")
            or result.get("success")
        )
        row = {
            "case_id": task.case_id,
            "house_index": task.house_index,
            "interaction_requirement": None,
            "terminal_reason": result.get("completion_reason"),
            "success": success,
            "task_success": success,
            "interaction_conditioned_success": success,
            "nav_success": bool(result.get("target_object_visible_navigation_success", False)),
            "required_interaction_success": bool(
                result.get("target_container_interaction_success", False)
            ),
            "sequence_success": None,
            "interaction_action_count": result.get("interaction_count"),
            "correct_interaction_action_count": result.get("physical_interaction_success_count"),
            "invalid_interaction_action_count": result.get("physical_interaction_failure_count"),
            "step_count": result.get("sim_step_frames"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "spl": None,
            "result_status": "complete" if complete else None,
        }
    video_path = Path(task.task_dir) / "videos" / "overview_6panel.mp4"
    if args.runner_mode == "v3" and result_path is not None:
        topdown_path = result_path.with_name("episode_topdown.png")
        topdown_metadata_path = topdown_path.with_suffix(".json")
    else:
        topdown_path = Path(task.task_dir) / "topdown.png"
        topdown_metadata_path = Path(task.task_dir) / "topdown.json"
    topdown_document = read_json(topdown_metadata_path)
    row.update(
        {
            "recording_artifact_valid": (not args.recording) or (video_path.is_file() and video_path.stat().st_size > 0),
            "topdown_path": str(topdown_path),
            "topdown_exists": topdown_path.is_file() and topdown_path.stat().st_size > 0,
            "topdown_metadata_path": str(topdown_metadata_path),
            "topdown_artifact_valid": topdown_path.is_file() and topdown_path.stat().st_size > 0 and topdown_metadata_path.is_file(),
            "topdown_status": (
                topdown_document.get("topdown_status")
                if isinstance(topdown_document, dict) and "topdown_status" in topdown_document
                else (0 if topdown_path.is_file() else None)
            ),
            "topdown_warnings": topdown_document.get("warnings", []) if isinstance(topdown_document, dict) else [],
            "domain": task.domain,
            "local_index": task.local_index,
            "episode_index": task.local_index,
            "worker_id": task.worker_id,
            "ros_master_uri": task.ros_master_uri,
            "output_dir": str(task.task_dir),
            "episode_result_path": None if result_path is None else str(result_path),
            "result_complete": bool(complete),
            "six_panel_video": str(Path(task.task_dir) / "videos" / "overview_6panel.mp4"),
            "force_interaction_events": str(
                Path(task.task_dir) / "force_interaction_events.json"
            ),
            "run_signature": document.get("run_signature"),
            "scoring_eligible": result.get("scoring_eligible"),
            "early_stop": result.get("early_stop"),
            **extract_timing_fields(result),
        }
    )
    return row


def run_task(
    task: EpisodeTask,
    args: argparse.Namespace,
    telemetry: ResourceTelemetry | None = None,
) -> dict[str, Any]:
    task_dir = Path(task.task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    summary_path = task_dir / "batch_task_summary.json"
    if args.resume:
        previous = read_json(summary_path)
        previous_video = (
            previous.get("source_six_panel_video") or previous.get("six_panel_video")
            if isinstance(previous, dict)
            else None
        )
        if (
            isinstance(previous, dict)
            and previous.get("result_complete")
            and bool(previous.get("recording_enabled", args.recording)) == bool(args.recording)
            and (
                not args.recording
                or Path(str(previous_video or "")).is_file()
            )
        ):
            previous = publish_task_artifacts(
                task_dir,
                previous,
                recording_required=bool(args.recording),
                topdown_required=bool(args.render_topdown),
            )
            previous["resumed"] = True
            atomic_json(summary_path, previous)
            return previous

    attempt_dir = task_dir / "attempt_001"
    if attempt_dir.exists():
        # A retry gets a monotonic attempt directory; never overwrite an old
        # evaluator tree or its recorder/log evidence.
        attempt_number = 2
        while (task_dir / f"attempt_{attempt_number:03d}").exists():
            attempt_number += 1
        attempt_dir = task_dir / f"attempt_{attempt_number:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    env = make_environment(task, args, attempt_dir=attempt_dir)
    runner = Path(args.runner).resolve()
    # V3's positional argument is the frozen JSON index; the legacy house7
    # launcher interprets its second argument as a route identifier instead.
    runner_episode_arg = str(task.local_index) if args.runner_mode == "v3" else task.case_id
    command = [args.runner_shell, str(runner), str(attempt_dir), runner_episode_arg]
    command_repr = shlex.join(command)
    task_log = task_dir / "batch_task.log"
    runner_log = attempt_dir / "runner.log"
    started = time.monotonic()
    started_at = utc_now()
    with task_log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{started_at}] domain={task.domain} local_index={task.local_index} "
            f"worker={task.worker_id} ROS_MASTER_URI={task.ros_master_uri}\n"
            f"start command={command_repr}\n"
        )
    if args.dry_run:
        row = {
            **asdict(task),
            "ros_master_uri": task.ros_master_uri,
            "command": command,
            "runner": str(runner),
            "attempt_dir": str(attempt_dir),
            "output_dir": str(attempt_dir),
            "completed": False,
            "result_complete": False,
            "recording_artifact_valid": not args.recording,
            "dry_run": True,
            "recording_enabled": bool(args.recording),
            "model_endpoint": assigned_endpoint(task.worker_id, args.model_endpoints),
        }
        atomic_json(summary_path, row)
        return row

    exception_text: str | None = None
    timed_out = False
    try:
        with runner_log.open("wb") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if telemetry is not None:
                telemetry.mark_started(task, process.pid)
            try:
                runner_exit_code = process.wait(timeout=float(args.scene_timeout_s))
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_group(process)
                runner_exit_code = 124
            finally:
                if telemetry is not None:
                    telemetry.mark_finished(task)
    except (OSError, subprocess.SubprocessError) as exc:
        runner_exit_code = 127
        exception_text = f"{type(exc).__name__}: {exc}"

    # The V3 runner may return a non-zero shell status after writing a complete
    # evaluator result (for example a post-processing failure).  Preserve both
    # facts: scoring eligibility comes from the signed episode result, while
    # ``runner_exit_code`` remains visible for debugging.
    row = parse_episode_result(
        EpisodeTask(**{**asdict(task), "task_dir": str(attempt_dir)}),
        args,
    )
    row = publish_task_artifacts(
        task_dir,
        row,
        recording_required=bool(args.recording),
        topdown_required=bool(args.render_topdown),
    )
    # A startup/timeout failure can leave roslaunch detached from the runner
    # shell.  Clean only this incomplete attempt before the worker reuses its
    # dedicated master port; completed episodes retain recorder post-processing.
    if timed_out or not row.get("result_complete"):
        cleanup_attempt_processes(attempt_dir)
    row.update(
        {
            "attempt_dir": str(attempt_dir),
            "scene_output_dir": str(task_dir),
            "runner": str(runner),
            "runner_log": str(runner_log),
            "command": command,
            "started_at": started_at,
            "finished_at": utc_now(),
            "elapsed_sec": time.monotonic() - started,
            "runner_exit_code": runner_exit_code,
            "exit_code": runner_exit_code,
            "timed_out": timed_out,
            "recording_enabled": bool(args.recording),
            "model_endpoint": assigned_endpoint(task.worker_id, args.model_endpoints),
            "resumed": False,
            "error": exception_text,
        }
    )
    # Keep a stable output_dir pointing at the attempt containing artifacts.
    row["output_dir"] = str(attempt_dir)
    atomic_json(attempt_dir / "wrapper_summary.json", row)
    atomic_json(summary_path, row)
    with task_log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{row['finished_at']}] exit_code={runner_exit_code} "
            f"result_complete={row.get('result_complete')} success={row.get('success')}\n"
        )
    return row


def _host_cpu_counters() -> tuple[float, float] | None:
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
        if not fields or fields[0] != "cpu":
            return None
        values = [float(value) for value in fields[1:]]
    except (OSError, IndexError, ValueError):
        return None
    if not values:
        return None
    idle = values[3] if len(values) > 3 else 0.0
    iowait = values[4] if len(values) > 4 else 0.0
    return sum(values), idle + iowait


def _host_memory_usage() -> dict[str, float]:
    values: dict[str, float] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        key, separator, raw = line.partition(":")
        if not separator:
            continue
        fields = raw.strip().split()
        if fields:
            try:
                values[key] = float(fields[0]) / 1024.0
            except ValueError:
                pass
    result: dict[str, float] = {}
    if "MemTotal" in values and "MemAvailable" in values:
        result["host_mem_used_mb"] = values["MemTotal"] - values["MemAvailable"]
        result["host_mem_available_mb"] = values["MemAvailable"]
    if "SwapTotal" in values and "SwapFree" in values:
        result["host_swap_used_mb"] = values["SwapTotal"] - values["SwapFree"]
    return result


def _gpu_usage() -> list[dict[str, float | int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    rows: list[dict[str, float | int]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            rows.append(
                {
                    "index": int(fields[0]),
                    "memory_used_mb": float(fields[1]),
                    "memory_total_mb": float(fields[2]),
                    "utilization_gpu_percent": float(fields[3]),
                }
            )
        except ValueError:
            continue
    return rows


def _process_rss_mb(pid: int) -> tuple[int, float]:
    """Return descendant process count/RSS for one launched process group."""

    records: dict[int, tuple[int, float]] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return 0, 0.0
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            remainder = stat[stat.rfind(")") + 2 :].split()
            parent = int(remainder[1])
            status = (entry / "status").read_text(encoding="utf-8")
            rss_kb = 0.0
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    rss_kb = float(line.split()[1])
                    break
            records[int(entry.name)] = (parent, rss_kb / 1024.0)
        except (OSError, ValueError, IndexError):
            continue
    descendants = {int(pid)}
    changed = True
    while changed:
        changed = False
        for child, (parent, _rss) in records.items():
            if parent in descendants and child not in descendants:
                descendants.add(child)
                changed = True
    return len(descendants), sum(records.get(child, (0, 0.0))[1] for child in descendants)


class ResourceTelemetry:
    """Batch-wide host/GPU/process sampler kept off the root filesystem."""

    fieldnames = (
        "wall_time",
        "elapsed_sec",
        "active_worker_count",
        "active_task_keys",
        "host_cpu_percent",
        "host_mem_used_mb",
        "host_mem_available_mb",
        "host_swap_used_mb",
        "process_count",
        "process_rss_mb",
        "process_rss_by_task_json",
        "gpu_metrics_json",
    )

    def __init__(self, output_dir: Path, interval_s: float) -> None:
        self.output_dir = output_dir
        self.interval_s = max(0.25, float(interval_s))
        self.path = output_dir / "resource_telemetry.csv"
        self._started = time.monotonic()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._active: dict[str, int] = {}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="interactive-nav-resource", daemon=True)
        self._thread.start()

    def mark_started(self, task: EpisodeTask, pid: int) -> None:
        with self._lock:
            self._active[task_key(task)] = int(pid)

    def mark_finished(self, task: EpisodeTask) -> None:
        with self._lock:
            self._active.pop(task_key(task), None)

    def _run(self) -> None:
        previous = _host_cpu_counters()
        append_existing = False
        try:
            append_existing = self.path.is_file() and self.path.stat().st_size > 0
        except OSError:
            append_existing = False
        with self.path.open("a" if append_existing else "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if not append_existing:
                writer.writeheader()
            while True:
                current = _host_cpu_counters()
                cpu_percent: float | None = None
                if previous is not None and current is not None:
                    total_delta = current[0] - previous[0]
                    idle_delta = current[1] - previous[1]
                    if total_delta > 0:
                        cpu_percent = 100.0 * max(0.0, min(1.0, 1.0 - idle_delta / total_delta))
                previous = current
                with self._lock:
                    active = dict(self._active)
                process_count = 0
                process_rss = 0.0
                process_rss_by_task: dict[str, float] = {}
                for task_key_value, pid in active.items():
                    count, rss = _process_rss_mb(pid)
                    process_count += count
                    process_rss += rss
                    process_rss_by_task[task_key_value] = rss
                row: dict[str, Any] = {
                    "wall_time": time.time(),
                    "elapsed_sec": time.monotonic() - self._started,
                    "active_worker_count": len(active),
                    "active_task_keys": json.dumps(sorted(active)),
                    "host_cpu_percent": cpu_percent,
                    "process_count": process_count,
                    "process_rss_mb": process_rss,
                    "process_rss_by_task_json": json.dumps(process_rss_by_task, separators=(",", ":")),
                    "gpu_metrics_json": json.dumps(_gpu_usage(), separators=(",", ":")),
                    **_host_memory_usage(),
                }
                writer.writerow(row)
                handle.flush()
                if self._stop.wait(self.interval_s):
                    break

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_s + 2.0))
        return self.summary()

    def summary(self) -> dict[str, Any]:
        try:
            rows = list(csv.DictReader(self.path.open(encoding="utf-8")))
        except OSError:
            rows = []
        if not rows:
            return {"path": str(self.path), "sample_count": 0}

        def numbers(key: str) -> list[float]:
            values: list[float] = []
            for row in rows:
                raw = row.get(key)
                if raw in {None, ""}:
                    continue
                try:
                    values.append(float(raw))
                except ValueError:
                    pass
            return values

        gpu_peaks: dict[int, dict[str, float | int]] = {}
        for row in rows:
            try:
                metrics = json.loads(row.get("gpu_metrics_json") or "[]")
            except ValueError:
                metrics = []
            if not isinstance(metrics, list):
                continue
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue
                try:
                    index = int(metric["index"])
                    used = float(metric["memory_used_mb"])
                    total = float(metric["memory_total_mb"])
                    util = float(metric["utilization_gpu_percent"])
                except (KeyError, TypeError, ValueError):
                    continue
                peak = gpu_peaks.setdefault(
                    index,
                    {"index": index, "peak_memory_used_mb": 0.0, "memory_total_mb": total, "peak_utilization_gpu_percent": 0.0},
                )
                peak["peak_memory_used_mb"] = max(float(peak["peak_memory_used_mb"]), used)
                peak["memory_total_mb"] = total
                peak["peak_utilization_gpu_percent"] = max(float(peak["peak_utilization_gpu_percent"]), util)
        cpu = numbers("host_cpu_percent")
        rss = numbers("process_rss_mb")
        mem = numbers("host_mem_used_mb")
        avail = numbers("host_mem_available_mb")
        active = numbers("active_worker_count")
        return {
            "path": str(self.path),
            "sample_count": len(rows),
            "peak_active_worker_count": max(active) if active else 0,
            "mean_host_cpu_percent": sum(cpu) / len(cpu) if cpu else None,
            "peak_host_cpu_percent": max(cpu) if cpu else None,
            "peak_process_rss_mb": max(rss) if rss else None,
            "peak_host_mem_used_mb": max(mem) if mem else None,
            "min_host_mem_available_mb": min(avail) if avail else None,
            "gpu_peaks": [gpu_peaks[index] for index in sorted(gpu_peaks)],
        }


class ProgressReporter:
    """Terminal + durable progress bars with cumulative success rate."""

    def __init__(self, output_dir: Path, total: int) -> None:
        self.output_dir = output_dir
        self.total = int(total)
        self.done = 0
        self.completed = 0
        self.successes = 0
        self.lock = threading.Lock()
        self.log_path = output_dir / "progress.log"
        self.jsonl_path = output_dir / "progress.jsonl"
        self.log_handle = self.log_path.open("a", encoding="utf-8")
        self.terminal = (
            tqdm(total=total, desc="interactive-nav", unit="ep", dynamic_ncols=True)
            if tqdm is not None
            else None
        )
        self.log_bar = (
            tqdm(
                total=total,
                desc="interactive-nav",
                unit="ep",
                dynamic_ncols=False,
                file=self.log_handle,
                disable=False,
            )
            if tqdm is not None
            else None
        )
        self.log_handle.write(f"[{utc_now()}] total={total}\n")
        self.log_handle.flush()

    def update(self, row: Mapping[str, Any]) -> None:
        with self.lock:
            self.done += 1
            if bool(row.get("result_complete")):
                self.completed += 1
            if bool(row.get("success")):
                self.successes += 1
            denominator = self.done
            sr = self.successes / denominator if denominator else 0.0
            planned_sr = self.successes / self.total if self.total else 0.0
            postfix = (
                f"SR={self.successes}/{denominator} ({sr:.1%}) "
                f"planned={self.successes}/{self.total} ({planned_sr:.1%})"
            )
            if self.terminal is not None:
                self.terminal.update(1)
                self.terminal.set_postfix_str(postfix)
            if self.log_bar is not None:
                self.log_bar.update(1)
                self.log_bar.set_postfix_str(postfix)
            line = (
                f"[{utc_now()}] {row.get('domain')}[{row.get('local_index')}] "
                f"worker={row.get('worker_id')} complete={row.get('result_complete')} "
                f"success={row.get('success')} {postfix}\n"
            )
            self.log_handle.write(line)
            self.log_handle.flush()
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "timestamp": utc_now(),
                            "done": self.done,
                            "completed": self.completed,
                            "successes": self.successes,
                            "sr": sr,
                            "planned_sr": planned_sr,
                            "domain": row.get("domain"),
                            "local_index": row.get("local_index"),
                            "success": bool(row.get("success")),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def close(self) -> None:
        if self.terminal is not None:
            self.terminal.close()
        if self.log_bar is not None:
            self.log_bar.close()
        self.log_handle.close()


_RETRYABLE_STARTUP_MARKERS = (
    "lmdb.error",
    "lmdb error",
    ".lmdb/scenes/",
    "no such file or directory",
    "refusing to reuse an existing ros master",
    "expected one completed episode result",
    "roscore exited before becoming ready",
    "ros master did not become ready",
    "may be occupied",
    "address already in use",
)


def _retryable_runner_failure(row: Mapping[str, Any]) -> bool:
    """Identify infrastructure startup failures, not ordinary algorithm failures."""

    if bool(row.get("result_complete")):
        return False
    chunks = [str(row.get("error") or ""), str(row.get("terminal_reason") or "")]
    for key in ("runner_log", "attempt_dir", "output_dir"):
        value = row.get(key)
        if not value:
            continue
        path = Path(str(value))
        if key == "attempt_dir":
            candidates = (path / "runner.log", path / "roslaunch.log", path / "eval.log")
        elif path.is_file():
            candidates = (path,)
        else:
            candidates = ()
        for candidate in candidates:
            try:
                chunks.append(candidate.read_text(encoding="utf-8", errors="replace")[-30000:])
            except OSError:
                pass
    text = "\\n".join(chunks).lower()
    return any(marker in text for marker in _RETRYABLE_STARTUP_MARKERS)


def run_worker_group(
    group: list[EpisodeTask],
    args: argparse.Namespace,
    reporter: ProgressReporter | None = None,
    telemetry: ResourceTelemetry | None = None,
) -> list[dict[str, Any]]:
    """Run one logical worker's shard serially on its dedicated ROS port."""

    rows: list[dict[str, Any]] = []
    for task in group:
        try:
            row = run_task(task, args, telemetry)
        except Exception as exc:  # keep later episodes in this worker alive
            row = {
                **asdict(task),
                "domain": task.domain,
                "local_index": task.local_index,
                "episode_index": task.local_index,
                "case_id": task.case_id,
                "house_index": task.house_index,
                "worker_id": task.worker_id,
                "ros_master_uri": task.ros_master_uri,
                "result_complete": False,
                "success": False,
                "task_success": False,
                "nav_success": False,
                "error": f"worker task exception: {type(exc).__name__}: {exc}",
            }
            atomic_json(Path(task.task_dir) / "batch_task_summary.json", row)
        retry_count = 0
        while retry_count < int(args.resource_init_retries) and _retryable_runner_failure(row):
            retry_count += 1
            time.sleep(float(args.resource_retry_backoff_s))
            retry_row = run_task(task, args, telemetry)
            retry_row["retry_count"] = retry_count
            retry_row["retried_after_attempt"] = row.get("attempt_dir")
            row = retry_row
            atomic_json(Path(task.task_dir) / "batch_task_summary.json", row)
        rows.append(row)
        if reporter is not None:
            reporter.update(row)
    return rows


def reconcile_rows_from_disk(
    tasks: list[EpisodeTask],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Recover durable episode results when a worker callback was lost.

    A ROS launcher can write ``episode_result.json`` and then spend time
    draining recorder/post-processing processes.  If a worker callback or its
    progress update fails in that window, final aggregation must inspect the
    durable task summaries and evaluator result instead of counting the task as
    incomplete.
    """

    reported: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row.get('domain')}:{row.get('local_index')}"
        reported[key] = row

    reconciled: list[dict[str, Any]] = []
    for task in tasks:
        key = f"{task.domain}:{task.local_index}"
        current = reported.get(key)
        persisted = read_json(Path(task.task_dir) / "batch_task_summary.json")
        chosen = persisted if isinstance(persisted, dict) else current

        # Inspect the latest attempt only after all worker futures have joined,
        # so a result cannot be mistaken for a still-running episode.
        attempts = sorted(Path(task.task_dir).glob("attempt_*"), key=lambda p: p.name)
        for attempt_dir in reversed(attempts):
            if not attempt_dir.is_dir():
                continue
            parsed = parse_episode_result(
                EpisodeTask(**{**asdict(task), "task_dir": str(attempt_dir)}),
                args,
            )
            if not parsed.get("result_complete"):
                continue
            wrapper = read_json(attempt_dir / "wrapper_summary.json")
            retry_metadata = {}
            if isinstance(persisted, dict):
                retry_metadata = {
                    key: persisted[key]
                    for key in ("retry_count", "retried_after_attempt")
                    if key in persisted
                }
            chosen = (
                {**retry_metadata, **wrapper, **parsed}
                if isinstance(wrapper, dict)
                else {**retry_metadata, **parsed}
            )
            break

        if not isinstance(chosen, dict):
            chosen = {
                **asdict(task),
                "ros_master_uri": task.ros_master_uri,
                "domain": task.domain,
                "local_index": task.local_index,
                "episode_index": task.local_index,
                "case_id": task.case_id,
                "house_index": task.house_index,
                "result_complete": False,
                "success": False,
                "task_success": False,
                "nav_success": False,
                "terminal_reason": "missing_worker_result",
                "error": "worker did not report a result",
                "output_dir": task.task_dir,
            }
        else:
            # Keep plan identity authoritative even if a stale summary exists.
            chosen.update(
                {
                    "ordinal": task.ordinal,
                    "domain": task.domain,
                    "local_index": task.local_index,
                    "episode_index": task.local_index,
                    "case_id": task.case_id,
                    "house_index": task.house_index,
                    "benchmark": task.benchmark,
                    "worker_id": task.worker_id,
                    "master_port": task.master_port,
                    "ros_master_uri": task.ros_master_uri,
                }
            )
        chosen = publish_task_artifacts(
            Path(task.task_dir),
            chosen,
            recording_required=bool(args.recording),
            topdown_required=bool(args.render_topdown),
        )
        chosen["scene_output_dir"] = task.task_dir
        atomic_json(Path(task.task_dir) / "batch_task_summary.json", chosen)
        reconciled.append(chosen)
    return reconciled


def numeric_mean(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return sum(values) / len(values) if values else None


def aggregate_results(tasks: list[EpisodeTask], rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    # Work on a copy: intermediate summary writes happen while other workers
    # are still running.  Appending temporary missing rows to the shared list
    # would create duplicates when the real worker result arrives later.  Also
    # collapse any duplicate worker recovery row by logical episode key so the
    # formal denominator can never exceed the planned episode count.
    by_episode: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row.get('domain')}:{row.get('local_index')}"
        by_episode[key] = row
    summary_rows = list(by_episode.values())
    # Preserve one row per plan; worker exceptions or missing reports are not
    # allowed to disappear from the denominator.
    present = {
        f"{row.get('domain')}:{row.get('local_index')}"
        for row in summary_rows
    }
    for task in tasks:
        key = f"{task.domain}:{task.local_index}"
        if key in present:
            continue
        summary_rows.append(
            {
                **asdict(task),
                "ros_master_uri": task.ros_master_uri,
                "domain": task.domain,
                "local_index": task.local_index,
                "episode_index": task.local_index,
                "result_complete": False,
                "success": False,
                "task_success": False,
                "nav_success": False,
                "terminal_reason": "missing_worker_result",
                "error": "worker did not report a result",
                "output_dir": task.task_dir,
            }
        )
    summary_rows.sort(key=lambda row: (str(row.get("domain")), int(row.get("local_index", -1))))
    planned = len(tasks)
    complete = sum(bool(row.get("result_complete")) for row in summary_rows)
    successes = sum(bool(row.get("success")) and bool(row.get("result_complete")) for row in summary_rows)
    task_successes = sum(bool(row.get("task_success")) and bool(row.get("result_complete")) for row in summary_rows)
    nav_successes = sum(bool(row.get("nav_success")) and bool(row.get("result_complete")) for row in summary_rows)
    interaction_successes = sum(
        bool(row.get("required_interaction_success")) and bool(row.get("result_complete"))
        for row in summary_rows
    )
    aggregate: dict[str, Any] = {
        "planned_episode_count": planned,
        "reported_episode_count": len(summary_rows),
        "completed_result_count": complete,
        "missing_or_incomplete_count": planned - complete,
        "success_count": successes,
        "task_success_count": task_successes,
        "task_success_rate": task_successes / planned if planned else 0.0,
        "nav_success_count": nav_successes,
        "nav_success_rate": nav_successes / planned if planned else 0.0,
        "interaction_success_count": interaction_successes,
        "interaction_success_rate": interaction_successes / planned if planned else 0.0,
        "interaction_action_count": sum(
            int(row.get("interaction_action_count", 0) or 0) for row in summary_rows
            if isinstance(row.get("interaction_action_count", 0), (int, float))
        ),
        "correct_interaction_action_count": sum(
            int(row.get("correct_interaction_action_count", 0) or 0) for row in summary_rows
            if isinstance(row.get("correct_interaction_action_count", 0), (int, float))
        ),
        "invalid_interaction_action_count": sum(
            int(row.get("invalid_interaction_action_count", 0) or 0) for row in summary_rows
            if isinstance(row.get("invalid_interaction_action_count", 0), (int, float))
        ),
        "applied_action_step_count": sum(
            int(row.get("applied_action_step_count", 0) or 0) for row in summary_rows
            if isinstance(row.get("applied_action_step_count", 0), (int, float))
        ),
        "no_fresh_action_count": sum(
            int(row.get("no_fresh_action_count", 0) or 0) for row in summary_rows
            if isinstance(row.get("no_fresh_action_count", 0), (int, float))
        ),
        # ``success_rate`` uses all planned episodes so timeouts cannot improve
        # SR.  The completed-only rate is exposed for diagnosis as well.
        "success_rate": successes / planned if planned else 0.0,
        "completed_success_rate": successes / complete if complete else 0.0,
        "recording_enabled": bool(args.recording),
        "recording_failure_count": sum(
            bool(args.recording) and bool(row.get("result_complete"))
            and row.get("recording_artifact_valid") is False
            for row in summary_rows
        ),
        "topdown_artifact_count": sum(bool(row.get("topdown_exists")) for row in summary_rows),
        "topdown_failure_count": sum(
            bool(args.render_topdown)
            and bool(row.get("result_complete"))
            and not bool(row.get("topdown_exists"))
            for row in summary_rows
        ),
        "mean_elapsed_sec": numeric_mean(summary_rows, "elapsed_sec"),
        "mean_step_count": numeric_mean(summary_rows, "step_count"),
        "mean_step_timing_loop_ms": numeric_mean(summary_rows, "step_timing_loop_ms_avg"),
        "mean_step_timing_policy_ms": numeric_mean(summary_rows, "step_timing_policy_ms_avg"),
        "mean_step_timing_task_ms": numeric_mean(summary_rows, "step_timing_task_ms_avg"),
        "domains": {},
        "failure_breakdown": {},
    }
    failure_reasons = Counter(
        str(row.get("terminal_reason") or row.get("error") or "incomplete")
        for row in summary_rows
        if not bool(row.get("success"))
    )
    aggregate["failure_breakdown"] = dict(failure_reasons)
    aggregate["interaction_failure_count"] = sum(
        bool(row.get("result_complete"))
        and row.get("required_interaction_success") is False
        for row in summary_rows
    )
    aggregate["navigation_failure_count"] = sum(
        bool(row.get("result_complete"))
        and row.get("nav_success") is False
        for row in summary_rows
    )
    aggregate["timeout_count"] = sum(bool(row.get("timed_out")) for row in summary_rows)
    for domain in DOMAINS:
        domain_rows = [row for row in summary_rows if row.get("domain") == domain]
        domain_successes = sum(
            bool(row.get("success")) and bool(row.get("result_complete"))
            for row in domain_rows
        )
        domain_complete = sum(bool(row.get("result_complete")) for row in domain_rows)
        domain_task_successes = sum(
            bool(row.get("task_success")) and bool(row.get("result_complete"))
            for row in domain_rows
        )
        domain_nav_successes = sum(
            bool(row.get("nav_success")) and bool(row.get("result_complete"))
            for row in domain_rows
        )
        domain_interaction_successes = sum(
            bool(row.get("required_interaction_success")) and bool(row.get("result_complete"))
            for row in domain_rows
        )
        aggregate["domains"][domain] = {
            "planned_episode_count": len(domain_rows),
            "completed_result_count": domain_complete,
            "success_count": domain_successes,
            "success_rate": domain_successes / len(domain_rows) if domain_rows else 0.0,
            "completed_success_rate": domain_successes / domain_complete if domain_complete else 0.0,
            "task_success_count": domain_task_successes,
            "task_success_rate": domain_task_successes / len(domain_rows) if domain_rows else 0.0,
            "nav_success_count": domain_nav_successes,
            "nav_success_rate": domain_nav_successes / len(domain_rows) if domain_rows else 0.0,
            "interaction_success_count": domain_interaction_successes,
            "interaction_success_rate": domain_interaction_successes / len(domain_rows) if domain_rows else 0.0,
            "recording_failure_count": sum(
                bool(args.recording) and row.get("recording_artifact_valid") is False
                for row in domain_rows
            ),
            "topdown_artifact_count": sum(bool(row.get("topdown_exists")) for row in domain_rows),
            "topdown_failure_count": sum(
                bool(args.render_topdown)
                and bool(row.get("result_complete"))
                and not bool(row.get("topdown_exists"))
                for row in domain_rows
            ),
        }
    return aggregate


def write_summary(
    output_dir: Path,
    tasks: list[EpisodeTask],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    resource_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    aggregate = aggregate_results(tasks, rows, args)
    if resource_summary is not None:
        aggregate["resource_telemetry"] = dict(resource_summary)
    # Include explicit incomplete rows in persisted summaries without mutating
    # the live worker-result list used by subsequent progress updates.
    display_by_episode: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row.get('domain')}:{row.get('local_index')}"
        display_by_episode[key] = row
    display_rows = list(display_by_episode.values())
    display_keys = set(display_by_episode)
    for task in tasks:
        if f"{task.domain}:{task.local_index}" in display_keys:
            continue
        display_rows.append(
            {
                **asdict(task),
                "domain": task.domain,
                "local_index": task.local_index,
                "episode_index": task.local_index,
                "case_id": task.case_id,
                "house_index": task.house_index,
                "worker_id": task.worker_id,
                "ros_master_uri": task.ros_master_uri,
                "result_complete": False,
                "success": False,
                "task_success": False,
                "nav_success": False,
                "terminal_reason": "missing_worker_result",
                "error": "worker did not report a result",
                "output_dir": task.task_dir,
            }
        )
    display_rows.sort(key=lambda row: (str(row.get('domain')), int(row.get('local_index', -1))))
    config = {
        "runner_mode": args.runner_mode,
        "raw_data_split": args.raw_data_split,
        "runner": str(args.runner),
        "workers": args.workers,
        "base_master_port": args.base_master_port,
        "episodes_per_domain": args.episodes_per_domain,
        "episode_offset": args.episode_offset,
        "max_steps": args.max_steps,
        "step_budget_mode": args.step_budget_mode,
        "min_steps": args.min_steps,
        "scene_timeout_s": args.scene_timeout_s,
        "ros_action_timeout_s": args.ros_action_timeout_s,
        "ros_command_starvation_timeout_s": args.ros_command_starvation_timeout_s,
        "ros_observation_turn_multiplier": args.ros_observation_turn_multiplier,
        "recording": bool(args.recording),
        "render_topdown": bool(args.render_topdown),
        "benchmark_files": {domain: str(args.benchmarks[domain]) for domain in DOMAINS},
        "benchmark_sha256": dict(
            getattr(args, "benchmark_sha256", {})
            or {domain: sha256_file(args.benchmarks[domain]) for domain in DOMAINS}
        ),
        "model_endpoints": args.model_endpoints,
        "mujoco_egl_devices": args.mujoco_egl_devices,
        "resource_telemetry": bool(args.resource_telemetry),
        "resource_sample_interval_s": args.resource_sample_interval_s,
        "resource_init_retries": args.resource_init_retries,
        "resource_retry_backoff_s": args.resource_retry_backoff_s,
        "generated_at": utc_now(),
    }
    summary = {
        "schema_version": "full_mllm_interactive_nav_eval_v1",
        "config": config,
        "aggregate": aggregate,
        "plans": [asdict(task) | {"ros_master_uri": task.ros_master_uri} for task in tasks],
        "episodes": display_rows,
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "aggregate_metrics.json", aggregate)
    fields = [
        "domain",
        "local_index",
        "case_id",
        "house_index",
        "worker_id",
        "ros_master_uri",
        "result_complete",
        "recording_artifact_valid",
        "topdown_exists",
        "topdown_artifact_valid",
        "topdown_path",
        "topdown_metadata_path",
        "topdown_status",
        "success",
        "task_success",
        "nav_success",
        "required_interaction_success",
        "interaction_action_count",
        "correct_interaction_action_count",
        "invalid_interaction_action_count",
        "terminal_reason",
        "step_count",
        "applied_action_step_count",
        "no_fresh_action_count",
        "elapsed_sec",
        "step_timing_loop_ms_avg",
        "step_timing_policy_ms_avg",
        "step_timing_task_ms_avg",
        "runner_exit_code",
        "retry_count",
        "retried_after_attempt",
        "timed_out",
        "episode_result_path",
        "six_panel_video",
        "runner_log",
        "output_dir",
        "error",
    ]
    temporary = output_dir / f".summary.csv.{os.getpid()}.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in display_rows:
            writer.writerow(row)
    temporary.replace(output_dir / "summary.csv")
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    for domain in DOMAINS:
        parser.add_argument(
            f"--{domain}-benchmark",
            type=Path,
            help=f"Override the frozen {domain}.json path under --benchmark-root.",
        )
    parser.add_argument(
        "--episodes-per-domain",
        "--episodes-per-category",
        dest="episodes_per_domain",
        type=int,
        default=10,
    )
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--allow-short-domain", action="store_true")
    parser.add_argument("--workers", "--threads", dest="workers", type=int, default=10)
    parser.add_argument("--base-master-port", type=int, default=13500)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--step-budget-mode", choices=("dynamic", "fixed"), default="dynamic")
    parser.add_argument("--min-steps", type=int, default=300)
    parser.add_argument("--scene-timeout-s", type=float, default=7200.0)
    parser.add_argument(
        "--ros-action-timeout-s",
        type=float,
        default=None,
        help="Override the ROS policy action freshness timeout (seconds); unset keeps runner default.",
    )
    parser.add_argument(
        "--ros-command-starvation-timeout-s",
        type=float,
        default=None,
        help="Override the evaluator no-fresh-command starvation timeout (seconds).",
    )
    parser.add_argument(
        "--ros-observation-turn-multiplier",
        type=float,
        default=None,
        help="Override the ROS observation turn multiplier used by starvation detection.",
    )
    parser.add_argument(
        "--runner-mode",
        choices=("v3", "raw"),
        default="v3",
        help="v3 uses the formal episode scorer; raw invokes the legacy exploration shell.",
    )
    parser.add_argument(
        "--raw-data-split",
        choices=("train", "val", "test"),
        default=None,
        help="Scene split for --runner-mode raw (default: val for benchmark alignment).",
    )
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--runner-shell", choices=("bash", "zsh"), default="bash")
    parser.add_argument(
        "--recording",
        "--enable-recording",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep recorder/video artifacts (default: enabled; --no-recording sets FAST_EVAL).",
    )
    parser.add_argument(
        "--render-topdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate a start/GT/trajectory/interaction top-down report (default: enabled).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--conda-env")
    parser.add_argument("--python-bin", type=Path)
    parser.add_argument("--ros-setup", type=Path)
    parser.add_argument("--semantic-model-env-file", type=Path)
    parser.add_argument("--model-endpoints", nargs="+")
    parser.add_argument("--mujoco-egl-devices", nargs="+")
    parser.add_argument(
        "--resource-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample host CPU/RAM/GPU/process RSS into resource_telemetry.csv (default: enabled).",
    )
    parser.add_argument("--resource-sample-interval-s", type=float, default=2.0)
    parser.add_argument(
        "--resource-init-retries",
        type=int,
        default=1,
        help="Retry an episode once when ROS/LMDB startup fails before a result is written.",
    )
    parser.add_argument(
        "--resource-retry-backoff-s",
        type=float,
        default=2.0,
        help="Seconds to wait before a startup retry so the failed ROS master can release its port.",
    )
    parser.add_argument("--port-preflight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many rounds without SR improvement; 0 disables early stop.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.expanduser().resolve()
    args.benchmark_root = args.benchmark_root.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.raw_data_split is None:
        args.raw_data_split = "val" if args.runner_mode == "raw" else "train"
    args.benchmarks = {}
    for domain in DOMAINS:
        override = getattr(args, f"{domain}_benchmark")
        path = (override or (args.benchmark_root / f"{domain}.json")).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{domain} benchmark not found: {path}")
        args.benchmarks[domain] = path
    # Benchmark files are large frozen JSON artifacts; hash them once during
    # validation rather than rereading all three files on every progress write.
    args.benchmark_sha256 = {
        domain: sha256_file(args.benchmarks[domain]) for domain in DOMAINS
    }
    if args.runner is None:
        args.runner = DEFAULT_V3_RUNNER if args.runner_mode == "v3" else DEFAULT_RAW_RUNNER
    args.runner = args.runner.expanduser().resolve()
    if not args.runner.is_file():
        raise FileNotFoundError(args.runner)
    if args.python_bin is not None:
        args.python_bin = args.python_bin.expanduser().resolve()
        if not args.python_bin.is_file():
            raise FileNotFoundError(args.python_bin)
    if args.ros_setup is not None:
        args.ros_setup = args.ros_setup.expanduser().resolve()
    if args.semantic_model_env_file is not None:
        args.semantic_model_env_file = args.semantic_model_env_file.expanduser().resolve()
        if not args.semantic_model_env_file.is_file():
            raise FileNotFoundError(args.semantic_model_env_file)
    elif args.model_endpoints:
        inherited = os.environ.get("SEMANTIC_MODEL_ENV_FILE")
        args.semantic_model_env_file = (
            Path(inherited).expanduser().resolve() if inherited else (REPO_ROOT / ".env").resolve()
        )
        if not args.semantic_model_env_file.is_file():
            raise FileNotFoundError(
                "--model-endpoints requires --semantic-model-env-file (or a readable .env)"
            )
    if args.model_endpoints:
        args.model_endpoints = [str(item).strip() for item in args.model_endpoints]
        if any(not item for item in args.model_endpoints):
            raise ValueError("--model-endpoints must not contain empty URLs")
    if args.mujoco_egl_devices:
        args.mujoco_egl_devices = [str(item).strip() for item in args.mujoco_egl_devices]
        if any(not item.isdigit() for item in args.mujoco_egl_devices):
            raise ValueError("--mujoco-egl-devices must contain integer IDs")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.episodes_per_domain < 1:
        raise ValueError("--episodes-per-domain must be positive")
    if args.max_steps < 1 or args.min_steps < 0:
        raise ValueError("--max-steps must be positive and --min-steps non-negative")
    if args.scene_timeout_s <= 0:
        raise ValueError("--scene-timeout-s must be positive")
    for option_name in (
        "ros_action_timeout_s",
        "ros_command_starvation_timeout_s",
        "ros_observation_turn_multiplier",
    ):
        option_value = getattr(args, option_name)
        if option_value is not None and option_value <= 0:
            raise ValueError(f"--{option_name.replace('_', '-')} must be positive")
    if args.resource_sample_interval_s <= 0:
        raise ValueError("--resource-sample-interval-s must be positive")
    if args.resource_init_retries < 0:
        raise ValueError("--resource-init-retries must be non-negative")
    if args.resource_retry_backoff_s < 0:
        raise ValueError("--resource-retry-backoff-s must be non-negative")
    if args.rounds < 1:
        raise ValueError("--rounds must be positive")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be non-negative")
    if not 1 <= args.base_master_port <= 65535:
        raise ValueError("--base-master-port must be in [1, 65535]")
    if args.base_master_port + args.workers - 1 > 65535:
        raise ValueError("ROS master port range exceeds 65535")


def interleave_worker_group(group: list[EpisodeTask], worker_id: int) -> list[EpisodeTask]:
    """Cycle domains per worker so early completions stay near 1:1:1."""

    queues: dict[str, list[EpisodeTask]] = {domain: [] for domain in DOMAINS}
    for task in group:
        queues[task.domain].append(task)
    ordered: list[EpisodeTask] = []
    while any(queues.values()):
        for offset in range(len(DOMAINS)):
            domain = DOMAINS[(worker_id + offset) % len(DOMAINS)]
            if queues[domain]:
                ordered.append(queues[domain].pop(0))
    return ordered


def _round_child_argv(args: argparse.Namespace, round_dir: Path) -> list[str]:
    """Rebuild CLI args for a one-round child without duplicating round flags."""

    original = list(sys.argv[1:])
    output: list[str] = []
    skip_next = False
    options_with_value = {"--rounds", "--early-stop-patience", "--output-dir"}
    for token in original:
        if skip_next:
            skip_next = False
            continue
        if token in options_with_value:
            skip_next = True
            continue
        if any(token.startswith(option + "=") for option in options_with_value):
            continue
        output.append(token)
    output.extend(["--rounds", "1", "--output-dir", str(round_dir)])
    return output


def run_rounds(args: argparse.Namespace) -> int:
    """Run independent round directories and stop when SR plateaus."""

    root_output = args.output_dir
    if not args.resume:
        existing_round_artifacts = [
            root_output / "round_summary.json",
            root_output / "analysis.json",
            root_output / "batch_manifest.json",
        ]
        existing_round_dirs = sorted(root_output.glob("round_[0-9][0-9]"))
        present = [str(path) for path in existing_round_artifacts if path.exists()]
        present.extend(str(path) for path in existing_round_dirs)
        if present:
            raise RuntimeError(
                "output directory already contains round/evaluation artifacts: "
                + ", ".join(present)
                + "; use --resume or choose a new empty output directory"
            )
    round_rows: list[dict[str, Any]] = []
    best_sr = -1.0
    stale_rounds = 0
    final_code = 0
    for round_index in range(1, args.rounds + 1):
        round_dir = root_output / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), *_round_child_argv(args, round_dir)]
        started = time.monotonic()
        completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
        final_code = max(final_code, int(completed.returncode))
        aggregate = read_json(round_dir / "aggregate_metrics.json")
        if not isinstance(aggregate, dict):
            aggregate = {
                "success_rate": 0.0,
                "success_count": 0,
                "planned_episode_count": 0,
                "error": "round did not produce aggregate_metrics.json",
            }
        sr = float(aggregate.get("success_rate", 0.0) or 0.0)
        if sr > best_sr + 1e-12:
            best_sr = sr
            stale_rounds = 0
        else:
            stale_rounds += 1
        previous_sr = round_rows[-1]["success_rate"] if round_rows else None
        round_rows.append(
            {
                "round": round_index,
                "output_dir": str(round_dir),
                "return_code": int(completed.returncode),
                "elapsed_sec": time.monotonic() - started,
                "success_rate": sr,
                "delta_success_rate": None if previous_sr is None else sr - float(previous_sr),
                "success_count": aggregate.get("success_count", 0),
                "planned_episode_count": aggregate.get("planned_episode_count", 0),
                "aggregate": aggregate,
            }
        )
        atomic_json(
            root_output / "round_summary.json",
            {
                "schema_version": "full_mllm_interactive_nav_rounds_v1",
                "rounds_requested": args.rounds,
                "early_stop_patience": args.early_stop_patience,
                "best_success_rate": best_sr,
                "completed_rounds": round_rows,
            },
        )
        if args.early_stop_patience > 0 and stale_rounds >= args.early_stop_patience:
            break
    atomic_json(
        root_output / "analysis.json",
        {
            "best_success_rate": best_sr if best_sr >= 0 else None,
            "completed_round_count": len(round_rows),
            "rounds": round_rows,
            "early_stopped": len(round_rows) < args.rounds,
        },
    )
    return final_code if args.allow_failures else (0 if all(row["return_code"] == 0 for row in round_rows) else 1)


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.rounds > 1:
        return run_rounds(args)
    tasks = build_tasks(
        args.benchmarks,
        output_dir=args.output_dir,
        episodes_per_domain=args.episodes_per_domain,
        episode_offset=args.episode_offset,
        shuffle_seed=args.shuffle_seed,
        allow_short_domain=args.allow_short_domain,
        workers=args.workers,
        base_master_port=args.base_master_port,
    )
    manifest_path = args.output_dir / "batch_manifest.json"
    manifest_config = {
        "runner_mode": args.runner_mode,
        "raw_data_split": args.raw_data_split,
        "runner": str(args.runner),
        "runner_shell": args.runner_shell,
        "workers": args.workers,
        "base_master_port": args.base_master_port,
        "episodes_per_domain": args.episodes_per_domain,
        "episode_offset": args.episode_offset,
        "shuffle_seed": args.shuffle_seed,
        "allow_short_domain": bool(args.allow_short_domain),
        "max_steps": args.max_steps,
        "step_budget_mode": args.step_budget_mode,
        "min_steps": args.min_steps,
        "scene_timeout_s": args.scene_timeout_s,
        "ros_action_timeout_s": args.ros_action_timeout_s,
        "ros_command_starvation_timeout_s": args.ros_command_starvation_timeout_s,
        "ros_observation_turn_multiplier": args.ros_observation_turn_multiplier,
        "recording": bool(args.recording),
        "render_topdown": bool(args.render_topdown),
        "resource_telemetry": bool(args.resource_telemetry),
        "resource_sample_interval_s": args.resource_sample_interval_s,
        "resource_init_retries": args.resource_init_retries,
        "resource_retry_backoff_s": args.resource_retry_backoff_s,
        "benchmarks": {
            domain: {
                "path": str(args.benchmarks[domain]),
                "sha256": getattr(args, "benchmark_sha256", {}).get(domain)
                or sha256_file(args.benchmarks[domain]),
            }
            for domain in DOMAINS
        },
    }
    manifest_payload = {
        "schema_version": "full_mllm_interactive_nav_eval_v1",
        "created_at": utc_now(),
        "config": manifest_config,
        "plans": [asdict(task) | {"ros_master_uri": task.ros_master_uri} for task in tasks],
    }
    if manifest_path.exists():
        if not args.resume:
            raise RuntimeError(
                f"output directory already contains batch_manifest.json: {args.output_dir}; "
                "use --resume or choose a new empty output directory"
            )
        existing_manifest = read_json(manifest_path)
        if not isinstance(existing_manifest, dict):
            raise RuntimeError(
                f"--resume refused: unreadable batch_manifest.json: {manifest_path}"
            )
        existing_config = existing_manifest.get("config")
        if not isinstance(existing_config, dict):
            raise RuntimeError(
                f"--resume refused: manifest has no compatible config: {manifest_path}"
            )
        # Resuming is intentionally append-only.  A changed benchmark, worker
        # assignment, recorder mode, or runner would make old episode summaries
        # incomparable, so require an explicitly new output directory instead
        # of silently replacing the manifest.
        immutable_keys = (
            "runner_mode",
            "raw_data_split",
            "runner",
            "runner_shell",
            "workers",
            "base_master_port",
            "episodes_per_domain",
            "episode_offset",
            "shuffle_seed",
            "allow_short_domain",
            "max_steps",
            "step_budget_mode",
            "min_steps",
            "scene_timeout_s",
            "resource_init_retries",
            "resource_retry_backoff_s",
            "ros_action_timeout_s",
            "ros_command_starvation_timeout_s",
            "ros_observation_turn_multiplier",
            "recording",
            "render_topdown",
            "benchmarks",
        )
        mismatches = [
            key for key in immutable_keys if existing_config.get(key) != manifest_config.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                "--resume refused: batch manifest configuration differs for "
                + ", ".join(mismatches)
                + "; choose a new output directory"
            )
        # Never rewrite the existing manifest (including its original timestamp
        # and plan). Episode attempts and summaries are appended below it.
    else:
        atomic_json(manifest_path, manifest_payload)
    # A dry run never starts roscore, so an occupied port is irrelevant and
    # should not prevent plan/command inspection.  Keep preflight strict for
    # real subprocess launches.
    if args.port_preflight and not args.dry_run:
        preflight_master_ports(args.base_master_port, args.workers)
    if args.dry_run:
        rows = [run_task(task, args) for task in tasks]
        aggregate = write_summary(args.output_dir, tasks, rows, args)
        print(json.dumps({"dry_run": True, "aggregate": aggregate}, ensure_ascii=False, indent=2))
        return 0

    reporter = ProgressReporter(args.output_dir, len(tasks))
    telemetry = (
        ResourceTelemetry(args.output_dir, args.resource_sample_interval_s)
        if args.resource_telemetry
        else None
    )
    if telemetry is not None:
        telemetry.start()
    rows: list[dict[str, Any]] = []
    groups: list[list[EpisodeTask]] = [[] for _ in range(args.workers)]
    for task in tasks:
        groups[task.worker_id].append(task)
    groups = [interleave_worker_group(group, worker_id) for worker_id, group in enumerate(groups)]
    resource_summary: dict[str, Any] | None = None
    try:
        with ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(tasks)))) as executor:
            futures = {
                executor.submit(run_worker_group, group, args, reporter, telemetry): worker_id
                for worker_id, group in enumerate(groups)
                if group
            }
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    group_rows = future.result()
                except Exception as exc:
                    # A worker-level failure (for example an unexpected
                    # executor/reporting exception outside run_task) must not
                    # abort the other ROS masters. Recover any task summaries
                    # already persisted by that worker, then mark the rest as
                    # explicit incomplete failures.
                    group_rows = []
                    for task in groups[worker_id]:
                        persisted = read_json(Path(task.task_dir) / "batch_task_summary.json")
                        if isinstance(persisted, dict) and persisted.get("domain") == task.domain:
                            group_rows.append(persisted)
                            continue
                        group_rows.append(
                            {
                                **asdict(task),
                                "domain": task.domain,
                                "local_index": task.local_index,
                                "episode_index": task.local_index,
                                "case_id": task.case_id,
                                "house_index": task.house_index,
                                "worker_id": task.worker_id,
                                "ros_master_uri": task.ros_master_uri,
                                "result_complete": False,
                                "success": False,
                                "task_success": False,
                                "nav_success": False,
                                "terminal_reason": "worker_exception",
                                "error": (
                                    f"worker {worker_id} exception: "
                                    f"{type(exc).__name__}: {exc}"
                                ),
                            }
                        )
                rows.extend(group_rows)
                # Keep summaries useful if the process is interrupted later.
                write_summary(args.output_dir, tasks, rows, args)
    finally:
        resource_summary = telemetry.stop() if telemetry is not None else None
        reporter.close()
    # Reconcile durable evaluator artifacts after all worker callbacks have
    # joined.  This closes the recorder/report-queue race without changing the
    # live tqdm semantics during execution.
    rows = reconcile_rows_from_disk(tasks, rows, args)
    gallery_summary = publish_batch_galleries(
        args.output_dir,
        tasks,
        rows,
        recording_enabled=bool(args.recording),
        topdown_enabled=bool(args.render_topdown),
    )
    aggregate = write_summary(args.output_dir, tasks, rows, args, resource_summary)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "episodes": len(tasks),
                "success_count": aggregate["success_count"],
                "success_rate": aggregate["success_rate"],
                "aggregate_metrics": str(args.output_dir / "aggregate_metrics.json"),
                "artifact_galleries": gallery_summary,
                "resource_telemetry": resource_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    failures = int(aggregate["missing_or_incomplete_count"])
    return 0 if args.allow_failures or failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
