"""Freeze a runtime-qualified InteractiveNav V3 benchmark release.

The command never overwrites the source collection.  It verifies that a
quality-gate run exactly covers the candidate benchmark, keeps only
``scoring_eligible`` episodes in the formal benchmark, sanitizes machine-local
paths, and writes release fingerprints, checksums, QC artifacts, and provenance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
from scripts.InteractiveNav import interactive_nav_v3


RELEASE_SCHEMA = "interactive_nav_v3_release_manifest_v1"
RESOURCE_BENCHMARK_URI = (
    "molmospaces://benchmarks/molmospaces-bench-v2/20240407/"
    "procthor-10k/NavToObjDataGenConfig/"
    "NavToObjProcthor10kBench_20260112_json_benchmark/benchmark.json"
)
CODE_FILES = (
    "scripts/InteractiveNav/collect_interactive_nav.py",
    "scripts/InteractiveNav/build_door_interaction_benchmark.py",
    "scripts/InteractiveNav/build_container_interaction_benchmark.py",
    "scripts/InteractiveNav/build_mixed_interaction_benchmark.py",
    "scripts/InteractiveNav/interactive_nav_v3.py",
    "scripts/InteractiveNav/repair_v3_channel_target_identity.py",
    "scripts/InteractiveNav/evaluation/benchmark_runner.py",
    "scripts/InteractiveNav/evaluation/benchmark_metrics.py",
    "scripts/InteractiveNav/analyze_interactive_nav_dataset.py",
    "scripts/InteractiveNav/freeze_interactive_nav_v3_release.py",
)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark_file(path: Path) -> Path:
    return path / "benchmark.json" if path.is_dir() else path


def episode_domain(episode: dict[str, Any]) -> str:
    domains = episode["interactive_nav"]["interaction_domains"]
    if domains == ["channel"]:
        return "channel"
    if domains == ["container"]:
        return "container"
    if domains == ["channel", "container"]:
        return "mixed"
    raise ValueError(f"Unsupported interaction domains: {domains!r}")


def sanitize_local_string(value: str) -> str:
    if not value.startswith("/"):
        return value
    if "/bench/" in value:
        return "molmospaces://trajectory-data/" + value.split("/bench/", 1)[1]
    cache_marker = "/.cache/molmo-spaces-resources/"
    if cache_marker in value:
        relative = value.split(cache_marker, 1)[1]
        return "molmospaces://resources/" + relative
    try:
        relative = Path(value).resolve().relative_to(REPO_ROOT)
    except (ValueError, OSError):
        relative = None
    if relative is not None:
        return "repo://" + relative.as_posix()
    if value.endswith("benchmark.json"):
        return RESOURCE_BENCHMARK_URI
    return "local-path-redacted://" + Path(value).name


def sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, str):
        return sanitize_local_string(value)
    return value


def sanitize_analysis_manifest(analysis_dir: Path) -> None:
    """Redact machine-local inputs recorded by the analysis command."""
    manifest_path = analysis_dir / "analysis_manifest.json"
    if manifest_path.is_file():
        write_json(manifest_path, sanitize_payload(read_json(manifest_path)))


def load_scoring_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    commit = run("git", "rev-parse", "HEAD")
    dirty_paths = [line for line in run("git", "status", "--short").splitlines() if line]
    return {
        "commit": commit,
        "worktree_clean": not dirty_paths,
        "dirty_paths": dirty_paths,
    }


def code_fingerprints() -> list[dict[str, Any]]:
    rows = []
    for relative in CODE_FILES:
        path = REPO_ROOT / relative
        if path.is_file():
            rows.append({"path": relative, "sha256": sha256(path)})
    return rows


def resource_versions() -> dict[str, Any]:
    registry = Path.home() / ".cache/molmo-spaces-resources/mjthor_data_type_to_source_to_versions.json"
    if not registry.is_file():
        return {
            "robot": {"rby1": "unknown"},
            "scene": {"procthor-10k-val": "unknown"},
            "objects": {"thor": "unknown"},
            "benchmark": {"molmospaces-bench-v2": "20240407"},
        }
    payload = read_json(registry)
    return {
        "robot": {"rby1": payload.get("robots", {}).get("rby1", [])},
        "scene": {
            "procthor-10k-val": payload.get("scenes", {}).get("procthor-10k-val", [])
        },
        "objects": {"thor": payload.get("objects", {}).get("thor", [])},
        "benchmark": {
            "molmospaces-bench-v2": payload.get("benchmarks", {}).get(
                "molmospaces-bench-v2", []
            )
        },
    }


def validate_quality_gate(
    episodes: list[dict[str, Any]],
    scoring_dir: Path,
    source_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    run_manifest = read_json(scoring_dir / "run_manifest.json")
    summary = read_json(scoring_dir / "summary.json")
    scoring_rows = load_scoring_rows(scoring_dir / "scoring_manifest.jsonl")
    if run_manifest.get("benchmark_sha256") != source_sha256:
        raise ValueError("Quality-gate benchmark hash does not match the candidate benchmark")
    if run_manifest.get("evaluation_config", {}).get("quality_gate_only") is not True:
        raise ValueError("Scoring run was not executed with quality_gate_only=true")
    indices = [int(row["episode_index"]) for row in scoring_rows]
    if sorted(indices) != list(range(len(episodes))):
        raise ValueError("Scoring manifest does not exactly cover all benchmark indices")
    if len(indices) != len(set(indices)):
        raise ValueError("Scoring manifest contains duplicate episode indices")
    by_index = {int(row["episode_index"]): row for row in scoring_rows}
    for index, episode in enumerate(episodes):
        row = by_index[index]
        case_id = str(episode["interactive_nav"]["case_id"])
        if row.get("case_id") != case_id:
            raise ValueError(f"Scoring case_id mismatch at episode_index={index}")
        runtime = row.get("runtime_consistency")
        if bool(row.get("scoring_eligible")):
            if row.get("status") != "complete":
                raise ValueError(f"Eligible row is not complete at episode_index={index}")
            if not isinstance(runtime, dict) or runtime.get("eligible") is not True:
                raise ValueError(f"Eligible row lacks passing runtime evidence at episode_index={index}")
    return scoring_rows, run_manifest, summary


def validate_release_episode(episode: dict[str, Any]) -> None:
    domains = list(episode["interactive_nav"]["interaction_domains"])
    interactive_nav_v3.validate_interactive_nav_v3_episode(
        episode, expected_domains=domains
    )
    EpisodeSpec.model_validate(episode)


def write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    atomic_text(root / "checksums.sha256", "\n".join(rows) + "\n")


def release_readme(manifest: dict[str, Any]) -> str:
    counts = manifest["formal_domain_counts"]
    candidate_count = manifest["candidate_episode_count"]
    formal_count = manifest["formal_episode_count"]
    excluded_count = manifest["excluded_episode_count"]
    return "\n".join([
        "# InteractiveNav V3 frozen benchmark release",
        "",
        f"Release ID: `{manifest['release_id']}`",
        "",
        f"The source candidate contains {candidate_count} episodes. The formal scoring denominator is "
        f"{formal_count}; {excluded_count} runtime-ineligible episodes are retained only in the scoring audit.",
        "",
        "Formal runtime-qualified composition:",
        "",
        f"- Channel: {counts.get('channel', 0)}",
        f"- Container: {counts.get('container', 0)}",
        f"- Mixed: {counts.get('mixed', 0)}",
        "",
        "`benchmark/benchmark.json` is the only formal evaluation input. "
        "`scoring/scoring_manifest.jsonl` records the quality-gate decision for every candidate episode.",
        "",
        "`scoring/summary.json` describes the zero-action quality-gate run and is not a policy "
        "benchmark result; policy metrics must be produced separately on the formal benchmark.",
        "",
        "Machine-local source paths were replaced with logical `molmospaces://`, "
        "`repo://`, or redacted provenance URIs before the release hash was computed.",
        "",
        "Verify the package from this directory with:",
        "",
        "```bash",
        "sha256sum -c checksums.sha256",
        "```",
        "",
        "The resolved builder configuration and random seed are in "
        "`provenance/config.resolved.yaml`; code, Git, resource-version and benchmark "
        "fingerprints are in `release_manifest.json`. Publish or archive the complete "
        "directory so the checksum file and audit evidence remain coupled to the benchmark.",
        "",
        "Known coverage limitations are recorded in `analysis/dataset_qc_report.md`: "
        "unnecessary-interaction controls occur only in Channel, beneficial interactions are absent, "
        "and target/house/path distributions are not uniform.",
        "",
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--scoring-run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--repair-manifest", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--release-id", default="interactive-nav-v3-procthor10k-val-release-v1")
    parser.add_argument(
        "--require-all-eligible",
        action="store_true",
        help="Fail instead of freezing an eligible-only subset when any candidate is ineligible.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_benchmark = benchmark_file(args.benchmark.resolve())
    scoring_dir = args.scoring_run_dir.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    candidate = read_json(source_benchmark)
    if isinstance(candidate, dict):
        candidate = candidate.get("episodes", [])
    if not isinstance(candidate, list):
        raise ValueError("Candidate benchmark must contain a JSON episode list")
    source_hash = sha256(source_benchmark)
    scoring_rows, run_manifest, scoring_summary = validate_quality_gate(
        candidate, scoring_dir, source_hash
    )
    eligible_indices = sorted(
        int(row["episode_index"])
        for row in scoring_rows
        if bool(row.get("scoring_eligible"))
    )
    if args.require_all_eligible and len(eligible_indices) != len(candidate):
        raise ValueError(
            f"Only {len(eligible_indices)}/{len(candidate)} candidates are scoring-eligible"
        )
    formal = [sanitize_payload(candidate[index]) for index in eligible_indices]
    for episode in formal:
        validate_release_episode(episode)
    domains = {
        domain: [episode for episode in formal if episode_domain(episode) == domain]
        for domain in ("channel", "container", "mixed")
    }

    benchmark_dir = output_root / "benchmark"
    write_json(benchmark_dir / "benchmark.json", formal)
    for domain, episodes in domains.items():
        write_json(benchmark_dir / f"{domain}.json", episodes)

    sanitized_scoring = [sanitize_payload(row) for row in scoring_rows]
    write_jsonl(output_root / "scoring" / "scoring_manifest.jsonl", sanitized_scoring)
    write_json(
        output_root / "scoring" / "excluded_episodes.json",
        [row for row in sanitized_scoring if not bool(row.get("scoring_eligible"))],
    )
    write_json(output_root / "scoring" / "run_manifest.json", sanitize_payload(run_manifest))
    write_json(output_root / "scoring" / "summary.json", sanitize_payload(scoring_summary))

    provenance_dir = output_root / "provenance"
    if args.source_config:
        config = yaml.safe_load(args.source_config.resolve().read_text(encoding="utf-8"))
        atomic_text(
            provenance_dir / "config.resolved.yaml",
            yaml.safe_dump(sanitize_payload(config), sort_keys=False, allow_unicode=True),
        )
    if args.repair_manifest:
        write_json(
            provenance_dir / "repair_manifest.json",
            sanitize_payload(read_json(args.repair_manifest.resolve())),
        )
    if args.analysis_dir:
        source_analysis = args.analysis_dir.resolve()
        if not source_analysis.is_dir():
            raise FileNotFoundError(source_analysis)
        release_analysis = output_root / "analysis"
        shutil.copytree(source_analysis, release_analysis, dirs_exist_ok=True)
        sanitize_analysis_manifest(release_analysis)

    domain_counts = Counter(episode_domain(episode) for episode in formal)
    exclusion_counts = Counter(
        str(reason)
        for row in scoring_rows
        if not bool(row.get("scoring_eligible"))
        for reason in row.get("scoring_exclusion_reasons", row.get("exclusion_reasons", []))
    )
    manifest = {
        "schema_version": RELEASE_SCHEMA,
        "release_id": args.release_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_candidate_benchmark_sha256": source_hash,
        "formal_benchmark_relative_path": "benchmark/benchmark.json",
        "formal_benchmark_sha256": sha256(benchmark_dir / "benchmark.json"),
        "candidate_episode_count": len(candidate),
        "formal_episode_count": len(formal),
        "excluded_episode_count": len(candidate) - len(formal),
        "formal_domain_counts": dict(domain_counts),
        "exclusion_reason_counts": dict(exclusion_counts),
        "quality_gate": {
            "protocol_version": run_manifest.get("protocol_version"),
            "run_signature": run_manifest.get("run_signature"),
            "candidate_benchmark_sha256": run_manifest.get("benchmark_sha256"),
            "summary_schema_version": scoring_summary.get("schema_version"),
        },
        "resources": {
            "scene_dataset": "procthor-10k",
            "data_split": "val",
            "source_nav_benchmark": RESOURCE_BENCHMARK_URI,
            "versions": resource_versions(),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "distribution": {
            "status": "frozen_local_package",
            "download_uri": None,
            "checksum_file": "checksums.sha256",
            "note": "Assign download_uri only when this complete directory is uploaded unchanged.",
        },
        "git": git_provenance(),
        "code_fingerprints": code_fingerprints(),
    }
    write_json(output_root / "release_manifest.json", manifest)
    atomic_text(output_root / "README.md", release_readme(manifest))
    write_checksums(output_root)
    print(json.dumps({
        "output_root": str(output_root),
        "formal_episode_count": len(formal),
        "excluded_episode_count": len(candidate) - len(formal),
        "formal_domain_counts": dict(domain_counts),
        "formal_benchmark_sha256": manifest["formal_benchmark_sha256"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
