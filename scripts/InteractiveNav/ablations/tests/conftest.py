from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts/InteractiveNav"))
from ablations import add_source_paths

add_source_paths(REPO)
