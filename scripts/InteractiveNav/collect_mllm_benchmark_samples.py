#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description="Bind scene frames to the MLLM question bank.")
    parser.add_argument("--question-bank", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--bind",
        action="append",
        default=[],
        metavar="CASE_ID=PATH[,PATH]",
        help="Bind one case to explicit source image paths.",
    )
    args = parser.parse_args()

    bank_path = Path(args.question_bank).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in source_dir.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not images:
        raise SystemExit(f"no images found under {source_dir}")
    images = images[: max(1, args.limit)]
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bindings = {}
    for item in args.bind:
        case_id, separator, values = item.partition("=")
        if not separator:
            raise SystemExit(f"invalid --bind value: {item}")
        bindings[case_id] = [Path(value).expanduser().resolve() for value in values.split(",")]
    cursor = 0
    for index, case in enumerate(bank.get("cases") or []):
        if not bool(case.get("requires_image", False)):
            case["image_paths"] = []
            continue
        selected = bindings.get(case["id"])
        if selected is None:
            image_count = max(1, int(case.get("image_count", 1)))
            selected = [images[(cursor + offset) % len(images)] for offset in range(image_count)]
            cursor += image_count
        copied = []
        for image_index, source in enumerate(selected):
            target = output_dir / (
                f"{case['id']}_{image_index:02d}{source.suffix.lower()}"
            )
            shutil.copy2(source, target)
            copied.append(str(target))
        case["image_paths"] = copied
    output_bank = output_dir / "question_bank_bound.json"
    output_bank.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_bank)


if __name__ == "__main__":
    main()
