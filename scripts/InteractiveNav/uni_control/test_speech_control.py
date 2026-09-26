#!/usr/bin/env python3
"""Small dependency-light tests for policy-to-Go2 speech control."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import types
import unittest
from unittest import mock

from control_protocol import (
    SpeechCommand,
    make_speech_message,
    parse_speech_message,
)
from go2_control_bridge import SpeakerWorker
from policy_control_server import PolicyControlServer


def test_execution_enabled_requires_fresh_connected_robot_telemetry():
    server = PolicyControlServer(ack_timeout=1, telemetry_print_period=10)
    assert not server.execution_enabled(0)
    server.connected.set()
    server.latest_telemetry_at = 10
    server.latest_telemetry = {"motion_enabled": True}
    assert server.execution_enabled(11)
    assert not server.execution_enabled(13)
    server.latest_telemetry["motion_enabled"] = False
    assert not server.execution_enabled(11)
    server.latest_telemetry["motion_enabled"] = True
    server.connected.clear()
    assert not server.execution_enabled(11)


class SpeechProtocolTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        payload = make_speech_message(
            7,
            "  我到门口了，请帮我开门  ",
            volume=4,
            ttl_ms=45000,
        )
        command = parse_speech_message(
            payload,
            last_seq=6,
            max_text_chars=100,
            max_ttl_ms=120000,
        )
        self.assertEqual(command.seq, 7)
        self.assertEqual(command.text, "我到门口了，请帮我开门")
        self.assertEqual(command.volume, 4)
        self.assertEqual(command.ttl_ms, 45000)

    def test_rejects_invalid_volume_and_oversized_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "volume"):
            parse_speech_message(
                make_speech_message(1, "你好", volume=11),
                last_seq=0,
                max_text_chars=100,
                max_ttl_ms=120000,
            )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            parse_speech_message(
                make_speech_message(1, "太长了"),
                last_seq=0,
                max_text_chars=2,
                max_ttl_ms=120000,
            )


class SpeakerWorkerTest(unittest.TestCase):
    def test_simulated_worker_finishes_without_blocking_submit(self) -> None:
        events = []
        args = argparse.Namespace(
            speaker_queue_size=2,
            speaker_backend="simulated",
            robot_controller_ip="192.168.123.161",
        )
        worker = SpeakerWorker(args, events.append)
        worker.start()
        started = time.monotonic()
        depth = worker.submit(
            SpeechCommand(
                seq=3,
                text="测试",
                voice="zh-CN-XiaoxiaoNeural",
                volume=4,
                ttl_ms=30000,
            )
        )
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertIn(depth, (0, 1))
        deadline = time.monotonic() + 1.0
        while not events and time.monotonic() < deadline:
            time.sleep(0.01)
        worker.close()
        self.assertEqual(events[0]["state"], "completed")
        self.assertEqual(events[0]["command_type"], "speech")

    def test_matcha_preload_failure_selects_edge_fallback(self) -> None:
        class BrokenMatcha:
            def __init__(self, **_kwargs) -> None:
                raise RuntimeError("model unavailable")

        fake_voice_module = types.SimpleNamespace(
            MatchaTtsSynthesizer=BrokenMatcha,
        )
        args = argparse.Namespace(
            speaker_queue_size=2,
            speaker_backend="go2",
            robot_controller_ip="192.168.123.161",
            speech_primary_backend="matcha",
            speech_fallback_backend="edge",
            matcha_model_dir="/missing/matcha",
            matcha_vocoder="/missing/vocoder.onnx",
            tts_threads=4,
        )
        worker = SpeakerWorker(args, lambda _payload: None)
        with mock.patch.dict(sys.modules, {"go2_voice_intercom": fake_voice_module}):
            worker._initialize_synthesis_backend()
        self.assertEqual(worker.active_synthesis_backend, "edge")
        self.assertIn("model unavailable", worker.backend_error)


class PolicySpeechTest(unittest.IsolatedAsyncioTestCase):
    async def test_publish_speech_waits_for_final_status(self) -> None:
        server = PolicyControlServer(ack_timeout=1.0, telemetry_print_period=1.0)

        async def fake_send(payload):
            self.assertEqual(payload["type"], "speak")
            seq = payload["seq"]
            server._handle_bridge_message(
                {"type": "ack", "seq": seq, "accepted": True, "state": "queued"}
            )
            asyncio.get_running_loop().call_soon(
                server._handle_bridge_message,
                {
                    "type": "status",
                    "seq": seq,
                    "command_type": "speech",
                    "state": "completed",
                },
            )

        server._send = fake_send
        result = await server.publish_speech("请帮我开门", volume=4)
        self.assertEqual(result["status"]["state"], "completed")


if __name__ == "__main__":
    unittest.main()
