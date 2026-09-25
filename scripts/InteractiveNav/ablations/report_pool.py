"""Compatibility entry point for the current paired V4 ablation report."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ablations.report_large_pool import summarize


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    report = summarize(parser.parse_args().output_dir)
    print(json.dumps({"complete": report["complete"], "groups": report["groups"]}, ensure_ascii=False))
