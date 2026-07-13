#!/usr/bin/env python3

from pathlib import Path

from ultralytics.utils.downloads import attempt_download_asset


WEIGHTS = [
    "yoloe-26m-seg-pf.pt",
    "yoloe-26l-seg-pf.pt",
    "yoloe-26x-seg-pf.pt",
]


def main():
    base = Path("/home/user/ldl/molmospaces/detection_models/yoloe/weights")
    base.mkdir(parents=True, exist_ok=True)

    for name in WEIGHTS:
        path = attempt_download_asset(str(base / name), repo="ultralytics/assets", release="v8.4.0")
        print(path)


if __name__ == "__main__":
    main()
