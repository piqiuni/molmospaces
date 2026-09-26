#!/usr/bin/env python3
"""Generate fixed WAV prompts offline using the installed FFmpeg/libflite."""
import subprocess
import wave

from speech_assets import ASSET_DIR, INTERACTION_PROMPTS


def main():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for text, name in INTERACTION_PROMPTS.items():
        path = ASSET_DIR / name
        if not path.exists():
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
                            "-f", "lavfi", "-i", f"flite=text='{text}':voice=slt",
                            "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", str(path)], check=True)
        with wave.open(str(path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
            if duration < 1.5 or audio.getsampwidth() != 2 or audio.getnchannels() != 1:
                raise ValueError(f"Invalid speech asset: {path}")
        print(f"{path}: {duration:.2f} s")


if __name__ == "__main__":
    main()
