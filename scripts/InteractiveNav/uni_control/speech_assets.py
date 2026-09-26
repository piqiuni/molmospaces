"""Fixed English interaction prompts, generated ahead of robot operation."""
from __future__ import annotations

from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parent / "audio"
INTERACTION_PROMPTS = {
    "Please open the door in front of me. Thank you.": "open_door_en.wav",
    "Please open the refrigerator in front of me. Thank you.": "open_refrigerator_en.wav",
    "Please open the drawer in front of me. Thank you.": "open_drawer_en.wav",
}


def prerecorded_audio(text: str) -> Path | None:
    name = INTERACTION_PROMPTS.get(text.strip())
    return ASSET_DIR / name if name else None


def is_english_text(text: str) -> bool:
    return text.isascii() and any(c.isalpha() for c in text)
