#!/usr/bin/env python3
"""Capture a showcase page with Chromium's native HTML/CSS renderer."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://127.0.0.1:8767/showcase-dark")
    parser.add_argument("--output", type=Path, default=Path("/tmp/showcase-browser.png"))
    parser.add_argument("--wait-ms", type=int, default=4000)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path="/usr/bin/chromium-browser",
            args=["--no-sandbox"],
        )
        page = browser.new_page(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=1,
            ignore_https_errors=True,
        )
        page.goto(args.url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_timeout(max(0, args.wait_ms))
        page.screenshot(path=str(args.output), full_page=False)
        browser.close()
    print(args.output)


if __name__ == "__main__":
    main()
