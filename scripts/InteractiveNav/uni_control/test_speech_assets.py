"""Offline fixed-prompt tests; no network, playback, or robot dependency."""
import ast
import asyncio
import hashlib
from pathlib import Path
import shutil
import time
import wave

from speech_assets import INTERACTION_PROMPTS, prerecorded_audio, is_english_text


def session_class():
    source = Path(__file__).with_name("go2_voice_intercom.py").read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "PersistentSpeechSession")
    cls.body = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in {"_cache_name", "_synthesize"}]
    scope = dict(globals())
    exec("from __future__ import annotations\n" + ast.unparse(cls), scope)
    return scope[cls.name]


def test_fixed_files_are_nonempty_english_pcm():
    for text in INTERACTION_PROMPTS:
        assert is_english_text(text)
        with wave.open(str(prerecorded_audio(text)), "rb") as f:
            assert f.getnchannels() == 1 and f.getsampwidth() == 2
            assert f.getframerate() == 44100
            assert 2.0 < f.getnframes() / f.getframerate() < 10.0
            assert any(f.readframes(f.getnframes()))


def test_fixed_audio_bypasses_chinese_model_and_old_cache(tmp_path):
    session = session_class()()
    session.synthesis_backend = "matcha"
    session.fallback_backend = "edge"
    session.matcha_synthesizer = None
    text = next(iter(INTERACTION_PROMPTS))
    output = tmp_path / "speech.wav"
    result = asyncio.run(session._synthesize(text, "zh-CN-XiaoxiaoNeural", output))
    assert result["synthesis_backend"] == "prerecorded"
    assert output.read_bytes() == prerecorded_audio(text).read_bytes()
    old = "policy_tts_cache_" + hashlib.sha256(f"matcha:1.000:0.200\0zh-CN-XiaoxiaoNeural\0{text}".encode()).hexdigest()[:20]
    assert session._cache_name(text, "zh-CN-XiaoxiaoNeural") != old
