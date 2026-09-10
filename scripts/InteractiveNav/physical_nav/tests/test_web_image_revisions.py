"""Regression tests for latest-only browser JPEG delivery."""

from __future__ import annotations

import io
import json
import pathlib
import shutil
import subprocess
import sys
import threading

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physical_six_panel_server import _HTML, _WebHandler  # noqa: E402
from showcase_pages import _SHOWCASE_CAMERA_10HZ_SCRIPT  # noqa: E402


class _FakeJpegHandler:
    image_cache_epoch = "test"

    def __init__(self, path: str, headers: dict[str, str] | None = None) -> None:
        self.path = path
        self.headers = headers or {}
        self.status: int | None = None
        self.response_headers: dict[str, str] = {}
        self.wfile = io.BytesIO()
        self.json_error: tuple[object, int] | None = None

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, key: str, value: object) -> None:
        self.response_headers[str(key)] = str(value)

    def end_headers(self) -> None:
        pass

    def _json(self, value: object, status: int = 200) -> None:
        self.status = status
        self.json_error = (value, status)


def _serve(
    path: str,
    *,
    revision: int = 7,
    headers: dict[str, str] | None = None,
    frame: bytes = b"jpeg-data",
) -> _FakeJpegHandler:
    handler = _FakeJpegHandler(path, headers)
    _WebHandler._revisioned_jpeg(
        handler,
        frame,
        revision,
        scope="snapshot",
        not_ready_error="frame not ready",
    )
    return handler


def test_revisioned_jpeg_sends_body_once_then_empty_cursor_response() -> None:
    fresh = _serve("/snapshot.jpg")
    assert fresh.status == 200
    assert fresh.wfile.getvalue() == b"jpeg-data"
    assert fresh.response_headers["X-Physical-Image-Revision"] == "7"
    assert fresh.response_headers["ETag"] == '"physical-test-snapshot-7"'
    assert "no-store" not in fresh.response_headers["Cache-Control"]

    unchanged = _serve("/snapshot.jpg?after=7")
    assert unchanged.status == 204
    assert unchanged.wfile.getvalue() == b""
    assert unchanged.response_headers["Content-Length"] == "0"
    assert unchanged.response_headers["X-Physical-Image-Revision"] == "7"

    # A cursor from a previous server epoch must receive the current image,
    # even if its integer happens to be larger than the restarted revision.
    restarted = _serve("/snapshot.jpg?after=99")
    assert restarted.status == 200
    assert restarted.wfile.getvalue() == b"jpeg-data"


def test_revisioned_jpeg_supports_standard_etag_and_not_ready_errors() -> None:
    unchanged = _serve(
        "/snapshot.jpg",
        headers={"If-None-Match": '"physical-test-snapshot-7"'},
    )
    assert unchanged.status == 304
    assert unchanged.wfile.getvalue() == b""
    assert unchanged.response_headers["Content-Length"] == "0"

    unavailable = _serve("/snapshot.jpg", frame=b"")
    assert unavailable.status == 503
    assert unavailable.json_error == (
        {"ok": False, "error": "frame not ready"},
        503,
    )


def test_all_live_jpeg_routes_use_their_independent_revision() -> None:
    class RouteProbe:
        frame_lock = threading.Lock()
        latest_jpeg = b"snapshot"
        latest_snapshot_revision = 3
        latest_camera_jpeg = b"camera"
        latest_camera_revision = 5
        latest_camera_box_jpeg = b"box"
        latest_camera_box_revision = 8

        def __init__(self, path: str) -> None:
            self.path = path
            self.calls: list[tuple[bytes, int, str]] = []

        def _revisioned_jpeg(
            self,
            frame: bytes,
            revision: int,
            *,
            scope: str,
            not_ready_error: str,
        ) -> None:
            del not_ready_error
            self.calls.append((frame, revision, scope))

    expected = {
        "/snapshot.jpg": (b"snapshot", 3, "snapshot"),
        "/camera-overlay.jpg": (b"camera", 5, "camera-overlay"),
        "/camera-box-overlay.jpg": (b"box", 8, "camera-box-overlay"),
    }
    for path, call in expected.items():
        probe = RouteProbe(path)
        _WebHandler.do_GET(probe)
        assert probe.calls == [call]


def test_debug_dashboard_uses_revision_cursor_before_reading_jpeg_body() -> None:
    start = _HTML.index("async function refreshStill")
    end = _HTML.index("function resumeDashboard", start)
    refresh = _HTML[start:end]
    assert "Date.now()" not in refresh
    assert "?after=" in refresh
    assert "X-Physical-Image-Revision" in refresh
    assert refresh.index("response.status===204") < refresh.index("response.blob()")
    assert "URL.revokeObjectURL(previousUrl)" in refresh


def test_showcase_skips_blob_and_decode_when_camera_revision_is_unchanged() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute browser polling")
    harness = r"""
const vm=require('node:vm'),assert=require('node:assert/strict');
const script=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const requests=[],timers=[],draws=[];let blobs=0,decodes=0,closes=0;
const revisions={'/camera-overlay.jpg':7,'/snapshot.jpg':11};
const canvas={width:640,height:360,getContext:()=>({drawImage:()=>draws.push(1)})};
const context=vm.createContext({
  document:{hidden:false,getElementById:id=>id==='live'?{textContent:''}:canvas},
  encodeURIComponent,setInterval:fn=>timers.push(fn),
  fetch:async url=>{
    const parsed=new URL(url,'http://localhost'),revision=revisions[parsed.pathname];
    const after=parsed.searchParams.get('after'),status=Number(after)===revision?204:200;
    requests.push([parsed.pathname,after,status]);
    return {status,ok:true,headers:{get:name=>name==='X-Physical-Image-Revision'?String(revision):null},blob:async()=>{
      assert.equal(status,200,'unchanged responses must not read a JPEG body');blobs++;return {};
    }};
  },
  createImageBitmap:async()=>{decodes++;return {width:640,height:360,close:()=>closes++}}
});
(async()=>{
  vm.runInContext(script,context);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(timers.length,1);
  assert.deepEqual(requests,[['/camera-overlay.jpg',null,200]]);
  await timers[0]();
  assert.deepEqual(requests[1],['/camera-overlay.jpg','7',204]);
  await context.video();
  await context.video();
  assert.deepEqual(requests.slice(-2),[
    ['/snapshot.jpg',null,200],['/snapshot.jpg','11',204]
  ]);
  assert.equal(blobs,2);
  assert.equal(decodes,2);
  assert.equal(draws.length,3);
  assert.equal(closes,2);
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run(
        [node, "-e", harness],
        input=json.dumps(_SHOWCASE_CAMERA_10HZ_SCRIPT),
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
