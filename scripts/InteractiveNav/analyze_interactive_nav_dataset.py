"""Health, distribution, and paper-figure analysis for InteractiveNav V3 data.

The collector produces nested benchmark episodes, which are convenient for the
simulator but cumbersome for analysis.  This script writes a stable flat index
plus machine-readable reports and publication-ready figures without modifying
the collected data.

Example:
    python scripts/InteractiveNav/analyze_interactive_nav_dataset.py \
        --dataset-root scripts/InteractiveNav/output/interactive_nav_v3_nav_benchmark_val_light_3000_manifest_v3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DOMAINS = ("channel", "container", "mixed")
DOMAIN_COLORS = {
    "channel": "#4C78A8",
    "container": "#F58518",
    "mixed": "#54A24B",
    "all": "#1F1F1F",
}
PATH_BOUNDS = (0.0, 3.0, 5.0, 8.0, 12.0, 20.0, math.inf)


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def scalar(value: Any) -> Any:
    """Keep CSV cells readable while preserving lists in the JSONL index."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalar(row.get(key)) for key in fieldnames})


def text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def path_bin(length_m: float | None) -> str:
    if length_m is None:
        return "missing"
    for lower, upper in zip(PATH_BOUNDS[:-1], PATH_BOUNDS[1:]):
        if lower <= length_m < upper:
            upper_text = "∞" if math.isinf(upper) else f"{upper:g}"
            return f"[{lower:g}, {upper_text})"
    return "missing"


def gt_path_length(validation: dict[str, Any]) -> tuple[float | None, str]:
    """Extract the comparable oracle-feasible navigation length.

    Channel and Mixed use the fully opened/oracle-restored path for the task's
    navigation GT.  Container tasks do not always expose that field and record
    their validated start-to-interaction path as ``path_length_m``.
    """
    for key in (
        "all_open_path_length_m",
        "oracle_restored_path_length_m",
        "path_length_m",
        "initial_state_path_length_m",
    ):
        result = number(validation.get(key))
        if result is not None:
            return result, key
    return None, "missing"


def visibility_gain(validation_rows: list[dict[str, Any]]) -> tuple[float | None, int | None]:
    gains: list[float] = []
    pixel_gains: list[int] = []
    for row in validation_rows:
        before = number(row.get("visibility_fraction_before"))
        after = number(row.get("visibility_fraction_after"))
        if before is not None and after is not None:
            gains.append(after - before)
        before_px = row.get("visible_pixels_before")
        after_px = row.get("visible_pixels_after")
        if isinstance(before_px, (int, float)) and isinstance(after_px, (int, float)):
            pixel_gains.append(int(after_px - before_px))
    return (max(gains) if gains else None, max(pixel_gains) if pixel_gains else None)


@dataclass
class FlatEpisode:
    domain: str
    case_id: str
    house_index: int
    schema_version: str
    interaction_requirement: str
    interaction_domains: list[str]
    legacy_case_type: str | None
    source_episode_index: int | None
    target_category: str
    target_instance: str
    container_category: str | None
    container_name: str | None
    interaction_count: int
    channel_interaction_count: int
    container_interaction_count: int
    interaction_types: list[str]
    interaction_joint_types: list[str]
    interaction_signature: list[str]
    oracle_plan_count: int
    gt_path_length_m: float | None
    gt_path_length_source: str
    gt_path_bin: str
    initial_path_found: bool | None
    all_open_path_found: bool | None
    all_closed_path_found: bool | None
    minimal_plan_status: str | None
    visibility_gain_fraction: float | None
    visibility_gain_pixels: int | None
    has_generation_validation: bool
    has_interaction_validations: bool


def flatten_episode(domain: str, episode: dict[str, Any]) -> FlatEpisode:
    interactive = episode.get("interactive_nav", {})
    target = interactive.get("target", {})
    validation = interactive.get("generation_validation", {})
    navigation = validation.get("navigation_validation", {})
    interactions = interactive.get("interactions", [])
    interaction_validations = validation.get("interaction_validations", [])
    if not isinstance(interaction_validations, list):
        interaction_validations = []
    path_length, source = gt_path_length(navigation if isinstance(navigation, dict) else {})
    visibility_fraction, visibility_pixels = visibility_gain(interaction_validations)
    interaction_domains = interactive.get("interaction_domains", [])
    if not isinstance(interaction_domains, list):
        interaction_domains = []
    interaction_types = [text(row.get("type")) for row in interactions if isinstance(row, dict)]
    interaction_signature = sorted(
        "|".join(
            (
                text(row.get("interaction_id")),
                text(row.get("type")),
                text(row.get("object_name")),
                text(row.get("joint_name")),
                text(row.get("target_state", {}).get("semantic_state")),
            )
        )
        for row in interactions
        if isinstance(row, dict)
    )
    interaction_joint_types = [
        text(row.get("joint_type"), "unknown")
        for row in interaction_validations
        if isinstance(row, dict)
    ]
    parent_index = interactive.get("parent_benchmark_episode_index")
    if parent_index is None:
        parent_index = episode.get("seed_generation", {}).get("source_episode_index")
    try:
        parent_index = int(parent_index) if parent_index is not None else None
    except (TypeError, ValueError):
        parent_index = None
    channel_count = sum(
        1
        for row in interactions
        if isinstance(row, dict) and str(row.get("interaction_id", "")).startswith("channel::")
    )
    container_count = sum(
        1
        for row in interactions
        if isinstance(row, dict) and str(row.get("interaction_id", "")).startswith("container::")
    )
    return FlatEpisode(
        domain=domain,
        case_id=text(interactive.get("case_id")),
        house_index=int(episode.get("house_index", -1)),
        schema_version=text(interactive.get("schema_version")),
        interaction_requirement=text(interactive.get("interaction_requirement")),
        interaction_domains=[text(value) for value in interaction_domains],
        legacy_case_type=(
            text(interactive.get("legacy_case_type"), "") or None
        ),
        source_episode_index=parent_index,
        target_category=text(target.get("category")),
        target_instance=text(target.get("selected_instance")),
        container_category=(
            text(target.get("container_category"), "") or None
        ),
        container_name=text(target.get("container_name"), "") or None,
        interaction_count=len(interactions),
        channel_interaction_count=channel_count,
        container_interaction_count=container_count,
        interaction_types=interaction_types,
        interaction_joint_types=interaction_joint_types,
        interaction_signature=interaction_signature,
        oracle_plan_count=len(interactive.get("oracle_plans", [])),
        gt_path_length_m=path_length,
        gt_path_length_source=source,
        gt_path_bin=path_bin(path_length),
        initial_path_found=(
            navigation.get("initial_state_path_found")
            if isinstance(navigation, dict)
            else None
        ),
        all_open_path_found=(
            navigation.get("all_open_path_found")
            if isinstance(navigation, dict)
            else None
        ),
        all_closed_path_found=(
            navigation.get("all_closed_path_found")
            if isinstance(navigation, dict)
            else None
        ),
        minimal_plan_status=(
            text(validation.get("minimal_plan_validation", {}).get("status"), "")
            if isinstance(validation.get("minimal_plan_validation"), dict)
            else None
        )
        or None,
        visibility_gain_fraction=visibility_fraction,
        visibility_gain_pixels=visibility_pixels,
        has_generation_validation=bool(validation),
        has_interaction_validations=bool(interaction_validations),
    )


def load_rows(dataset_root: Path) -> tuple[list[FlatEpisode], list[dict[str, Any]]]:
    rows: list[FlatEpisode] = []
    raw: list[dict[str, Any]] = []
    for domain in DOMAINS:
        path = dataset_root / "raw" / domain / "benchmark.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing benchmark data: {path}")
        episodes = read_json(path)
        if not isinstance(episodes, list):
            raise ValueError(f"Expected a JSON list in {path}")
        for episode in episodes:
            rows.append(flatten_episode(domain, episode))
            raw.append(episode)
    return rows, raw


def count_by(rows: Iterable[FlatEpisode], key: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for row in rows:
        value = getattr(row, key)
        result[text(value)] += 1
    return result


def numeric_summary(values: Iterable[float | None]) -> dict[str, float | int | None]:
    array = sorted(value for value in values if value is not None)
    if not array:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "mean": None, "max": None}

    def quantile(q: float) -> float:
        index = (len(array) - 1) * q
        low, high = math.floor(index), math.ceil(index)
        if low == high:
            return array[low]
        return array[low] + (array[high] - array[low]) * (index - low)

    return {
        "count": len(array),
        "min": array[0],
        "p25": quantile(0.25),
        "median": quantile(0.50),
        "p75": quantile(0.75),
        "mean": sum(array) / len(array),
        "max": array[-1],
    }


def health_report(rows: list[FlatEpisode]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    case_ids: Counter[str] = Counter(row.case_id for row in rows)
    semantic_checks: Counter[str] = Counter()
    for row in rows:
        reasons: list[str] = []
        if row.schema_version != "interactive_nav_v3":
            reasons.append("schema_version_not_v3")
        if row.case_id == "unknown":
            reasons.append("missing_case_id")
        if not row.has_generation_validation:
            reasons.append("missing_generation_validation")
        if row.oracle_plan_count < 1:
            reasons.append("missing_oracle_plan")
        if row.gt_path_length_m is None:
            reasons.append("missing_gt_path_length")
        if case_ids[row.case_id] > 1:
            reasons.append("duplicate_case_id")
        domains = set(row.interaction_domains)
        if row.domain == "channel":
            if domains != {"channel"}:
                reasons.append("channel_domain_mismatch")
            # Zero interaction is valid for counterfactual/distractor episodes.
            if row.legacy_case_type is None:
                reasons.append("missing_channel_recipe")
        elif row.domain == "container":
            if domains != {"container"}:
                reasons.append("container_domain_mismatch")
            if row.interaction_requirement != "required":
                reasons.append("container_not_required")
            if row.container_interaction_count < 1:
                reasons.append("container_interaction_missing")
            if row.container_name is None:
                reasons.append("container_target_missing")
        else:
            if domains != {"channel", "container"}:
                reasons.append("mixed_domain_mismatch")
            if row.interaction_requirement != "required":
                reasons.append("mixed_not_required")
            if row.channel_interaction_count < 1:
                reasons.append("mixed_channel_interaction_missing")
            if row.container_interaction_count < 1:
                reasons.append("mixed_container_interaction_missing")
        for reason in reasons:
            semantic_checks[reason] += 1
        if reasons:
            issues.append({
                "case_id": row.case_id,
                "domain": row.domain,
                "house_index": row.house_index,
                "issues": reasons,
            })

    identity_groups: Counter[tuple[Any, ...]] = Counter(
        (
            row.domain,
            row.house_index,
            row.target_instance,
            row.container_name,
            row.source_episode_index,
            row.legacy_case_type,
            tuple(row.interaction_signature),
        )
        for row in rows
    )
    near_duplicate_groups = [
        {
            "domain": key[0],
            "house_index": key[1],
            "target_instance": key[2],
            "container_name": key[3],
            "source_episode_index": key[4],
            "legacy_case_type": key[5],
            "interaction_signature": list(key[6]),
            "count": count,
        }
        for key, count in identity_groups.items()
        if count > 1
    ]
    report = {
        "episode_count": len(rows),
        "domain_counts": dict(count_by(rows, "domain")),
        "schema_v3_count": sum(row.schema_version == "interactive_nav_v3" for row in rows),
        "unique_case_id_count": len(case_ids),
        "duplicate_case_id_count": sum(count - 1 for count in case_ids.values() if count > 1),
        "semantic_issue_count": len(issues),
        "semantic_issue_counts": dict(semantic_checks),
        "near_duplicate_group_count": len(near_duplicate_groups),
        "near_duplicate_extra_episode_count": sum(
            row["count"] - 1 for row in near_duplicate_groups
        ),
        "missing_gt_path_length_count": sum(
            row.gt_path_length_m is None for row in rows
        ),
        "missing_generation_validation_count": sum(
            not row.has_generation_validation for row in rows
        ),
    }
    return report, issues + [
        {"near_duplicate": row} for row in near_duplicate_groups
    ]


def make_interaction_distribution(rows: list[FlatEpisode]) -> list[dict[str, Any]]:
    maximum = max((row.interaction_count for row in rows), default=0)
    output = []
    for domain in (*DOMAINS, "all"):
        subset = rows if domain == "all" else [row for row in rows if row.domain == domain]
        counts = Counter(row.interaction_count for row in subset)
        for count in range(maximum + 1):
            output.append({
                "domain": domain,
                "interaction_count": count,
                "episode_count": counts[count],
                "proportion": counts[count] / len(subset) if subset else 0.0,
            })
    return output


def make_path_distribution(rows: list[FlatEpisode]) -> list[dict[str, Any]]:
    labels = [path_bin(lower + 1e-8) for lower in PATH_BOUNDS[:-1]]
    output = []
    for domain in (*DOMAINS, "all"):
        subset = rows if domain == "all" else [row for row in rows if row.domain == domain]
        counts = Counter(row.gt_path_bin for row in subset)
        for label in labels:
            output.append({
                "domain": domain,
                "path_bin": label,
                "episode_count": counts[label],
                "proportion": counts[label] / len(subset) if subset else 0.0,
            })
    return output


def proportions(rows: list[FlatEpisode], field: str) -> list[dict[str, Any]]:
    counts = count_by(rows, field)
    total = len(rows)
    return [
        {"category": category, "episode_count": count, "proportion": count / total}
        for category, count in counts.most_common()
    ]


def setup_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(figure: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(figures_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_interaction_distribution(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 3.8))
    for domain in (*DOMAINS, "all"):
        values = [row for row in rows if row["domain"] == domain]
        axis.plot(
            [row["interaction_count"] for row in values],
            [row["episode_count"] for row in values],
            marker="o",
            linewidth=2.2 if domain == "all" else 1.8,
            color=DOMAIN_COLORS[domain],
            label=domain.capitalize(),
        )
    axis.set_xlabel("Required interaction count")
    axis.set_ylabel("Episodes")
    axis.set_xticks(sorted({row["interaction_count"] for row in rows}))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    save_figure(figure, figures_dir, "interaction_count_distribution")


def plot_path_distribution(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    labels = [path_bin(lower + 1e-8) for lower in PATH_BOUNDS[:-1]]
    positions = list(range(len(labels)))
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    for domain in (*DOMAINS, "all"):
        values_by_label = {
            row["path_bin"]: row["episode_count"]
            for row in rows
            if row["domain"] == domain
        }
        axis.plot(
            positions,
            [values_by_label.get(label, 0) for label in labels],
            marker="o",
            linewidth=2.2 if domain == "all" else 1.8,
            color=DOMAIN_COLORS[domain],
            label=domain.capitalize(),
        )
    axis.set_xlabel("GT navigation path length (m)")
    axis.set_ylabel("Episodes")
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    save_figure(figure, figures_dir, "gt_path_length_distribution")


def plot_pie(
    rows: list[dict[str, Any]],
    figures_dir: Path,
    stem: str,
    title: str,
    *,
    max_categories: int | None = None,
) -> None:
    values = rows[:]
    if max_categories is not None and len(values) > max_categories:
        head = values[:max_categories]
        tail_count = sum(row["episode_count"] for row in values[max_categories:])
        values = head + [{"category": "Other", "episode_count": tail_count}]
    figure, axis = plt.subplots(figsize=(5.4, 4.1))
    palette = plt.get_cmap("tab20")
    wedges, _ = axis.pie(
        [row["episode_count"] for row in values],
        startangle=90,
        colors=[palette(index % 20) for index in range(len(values))],
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    labels = [
        f"{row['category']} ({row['episode_count']}, {row['episode_count'] / sum(item['episode_count'] for item in values):.1%})"
        for row in values
    ]
    axis.legend(wedges, labels, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False)
    axis.set_title(title)
    save_figure(figure, figures_dir, stem)


def summary_by_domain(rows: list[FlatEpisode]) -> list[dict[str, Any]]:
    output = []
    for domain in (*DOMAINS, "all"):
        subset = rows if domain == "all" else [row for row in rows if row.domain == domain]
        output.append({
            "domain": domain,
            "episode_count": len(subset),
            "house_count": len({row.house_index for row in subset}),
            "target_category_count": len({row.target_category for row in subset}),
            "container_category_count": len({row.container_category for row in subset if row.container_category}),
            "interaction_count": numeric_summary([float(row.interaction_count) for row in subset]),
            "gt_path_length_m": numeric_summary([row.gt_path_length_m for row in subset]),
            "visibility_gain_fraction": numeric_summary([row.visibility_gain_fraction for row in subset]),
        })
    return output


def house_distribution(rows: list[FlatEpisode]) -> list[dict[str, Any]]:
    output = []
    for domain in DOMAINS:
        counts = Counter(row.house_index for row in rows if row.domain == domain)
        for house_index, count in sorted(counts.items()):
            output.append({
                "domain": domain,
                "house_index": house_index,
                "episode_count": count,
            })
    return output


def audit_samples(rows: list[FlatEpisode], issues: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: dict[str, FlatEpisode] = {}
    for domain in DOMAINS:
        subset = [row for row in rows if row.domain == domain]
        for row in rng.sample(subset, min(20, len(subset))):
            selected[row.case_id] = row
    issue_case_ids = {row.get("case_id") for row in issues if row.get("case_id")}
    for row in rows:
        if row.case_id in issue_case_ids:
            selected[row.case_id] = row
    return [asdict(row) for row in sorted(selected.values(), key=lambda row: (row.domain, row.case_id))]


def report_markdown(
    health: dict[str, Any],
    domain_summary: list[dict[str, Any]],
    domain_proportions: list[dict[str, Any]],
) -> str:
    lines = [
        "# InteractiveNav V3 dataset analysis",
        "",
        "## Dataset health",
        "",
        f"- Episodes: {health['episode_count']}",
        f"- Schema V3: {health['schema_v3_count']} / {health['episode_count']}",
        f"- Unique case IDs: {health['unique_case_id_count']}",
        f"- Exact duplicate case IDs: {health['duplicate_case_id_count']}",
        f"- Semantic issues: {health['semantic_issue_count']}",
        f"- Missing comparable GT path length: {health['missing_gt_path_length_count']}",
        "",
        "## Domain proportions",
        "",
        "| Domain | Episodes | Proportion |",
        "|---|---:|---:|",
    ]
    for row in domain_proportions:
        lines.append(f"| {row['category']} | {row['episode_count']} | {row['proportion']:.2%} |")
    lines.extend(["", "## Per-domain summary", "", "| Domain | Episodes | Houses | Target categories | Path median (m) | Interaction median |", "|---|---:|---:|---:|---:|---:|"] )
    for row in domain_summary:
        path_median = row["gt_path_length_m"]["median"]
        interaction_median = row["interaction_count"]["median"]
        path_text = f"{path_median:.3f}" if path_median is not None else "NA"
        interaction_text = (
            f"{interaction_median:.1f}"
            if interaction_median is not None
            else "NA"
        )
        lines.append(
            f"| {row['domain']} | {row['episode_count']} | {row['house_count']} | "
            f"{row['target_category_count']} | "
            f"{path_text} | {interaction_text} |"
        )
    lines.extend(["", "## Paper figures", "", "- `figures/interaction_count_distribution.pdf`: 0-to-n required interaction count distribution.", "- `figures/gt_path_length_distribution.pdf`: comparable GT path-length bin distribution.", "- `figures/domain_proportions.pdf`: Channel / Container / Mixed task share.", "- `figures/target_category_proportions.pdf`: top target-category share (remaining categories grouped as Other).", "- `figures/container_category_proportions.pdf`: Container/Mixed container-type share.", "", "All figure source tables are under `paper_data/`.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Collection output root containing raw/channel|container|mixed/benchmark.json.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        help="Default: <dataset-root>/analysis",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    analysis_dir = (args.analysis_dir or dataset_root / "analysis").resolve()
    figures_dir = analysis_dir / "figures"
    paper_data_dir = analysis_dir / "paper_data"
    setup_plot_style()

    rows, _ = load_rows(dataset_root)
    flat_rows = [asdict(row) for row in rows]
    write_jsonl(analysis_dir / "episodes_flat.jsonl", flat_rows)
    write_csv(analysis_dir / "episodes_flat.csv", flat_rows)

    health, issues = health_report(rows)
    domain_summary = summary_by_domain(rows)
    houses = house_distribution(rows)
    interaction_rows = make_interaction_distribution(rows)
    path_rows = make_path_distribution(rows)
    domain_rows = proportions(rows, "domain")
    target_rows = proportions(rows, "target_category")
    container_rows = proportions(
        [row for row in rows if row.container_category is not None], "container_category"
    )

    write_json(analysis_dir / "dataset_health_report.json", health)
    write_json(analysis_dir / "distribution_summary.json", {
        "domain_summary": domain_summary,
        "domain_proportions": domain_rows,
        "target_category_proportions": target_rows,
        "container_category_proportions": container_rows,
        "interaction_count_distribution": interaction_rows,
        "gt_path_length_distribution": path_rows,
    })
    write_jsonl(analysis_dir / "anomalies_and_duplicates.jsonl", issues)
    write_jsonl(analysis_dir / "audit_samples.jsonl", audit_samples(rows, issues, args.seed))
    write_csv(analysis_dir / "house_distribution.csv", houses)
    write_csv(paper_data_dir / "interaction_count_distribution.csv", interaction_rows)
    write_csv(paper_data_dir / "gt_path_length_distribution.csv", path_rows)
    write_csv(paper_data_dir / "gt_path_length_summary.csv", [
        {
            "domain": row["domain"],
            **{f"gt_path_length_{key}": value for key, value in row["gt_path_length_m"].items()},
        }
        for row in domain_summary
    ])
    write_csv(paper_data_dir / "domain_proportions.csv", domain_rows)
    write_csv(paper_data_dir / "target_category_proportions.csv", target_rows)
    write_csv(paper_data_dir / "container_category_proportions.csv", container_rows)

    plot_interaction_distribution(interaction_rows, figures_dir)
    plot_path_distribution(path_rows, figures_dir)
    plot_pie(domain_rows, figures_dir, "domain_proportions", "Task-domain proportions")
    plot_pie(target_rows, figures_dir, "target_category_proportions", "Target-category proportions (Top 10)", max_categories=10)
    plot_pie(container_rows, figures_dir, "container_category_proportions", "Container-category proportions")

    (analysis_dir / "distribution_report.md").write_text(
        report_markdown(health, domain_summary, domain_rows), encoding="utf-8"
    )
    print(json.dumps({
        "analysis_dir": str(analysis_dir),
        "episode_count": len(rows),
        "semantic_issue_count": health["semantic_issue_count"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
