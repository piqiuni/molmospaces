"""Local launcher lifecycle for the optional recording side channel."""
import argparse
import json
import urllib.request


def control(action, port):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path, payload=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/recording/{path}",
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=40) as response:
            return json.load(response)
    status = request("status")
    if action == "start":
        status = request("start", {"mode": "raw_plus_panels", "label": "launcher"})
        if not status.get("active") or status.get("mode") != "raw_plus_panels":
            raise RuntimeError("raw_plus_panels recording did not start")
    elif status.get("active"):
        status = request("stop", {"session_id": status["session_id"], "reason": "launcher_stop"})
        if status.get("active") or status.get("degraded"):
            raise RuntimeError("recording stop/drain incomplete; inspect session statistics")
    print(f"Recording {action}: {status.get('record_dir') or 'inactive'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    control(args.action, args.port)
