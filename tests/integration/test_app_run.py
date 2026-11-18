"""The app keeps a model loaded for chat apps (spec 4c item 1).

  python tests/integration/test_app_run.py --kera build/bin/kera --models DIR

Starts `kera app`, adds DIR as a model folder, runs one of its models, waits until it is ready, checks that
chat apps can reach it, stops it, then quits and restarts the app: the same model must load again by
itself, on the same address.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.request

APP = 8361


def req(path, body=None, port=APP, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                               headers={"X-Kera": "1", "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


def start_app(kera, home):
    env = dict(os.environ, KERA_HOME=home)
    p = subprocess.Popen([kera, "app", "--no-browser", "--quick", "--port", str(APP)], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            req("/api/status")
            return p
        except Exception:
            time.sleep(0.5)
    p.kill()
    sys.exit("the app didn't start")


def wait_ready():
    s = {}
    for _ in range(120):
        s = req("/api/run")
        if s["state"] in ("ready", "failed"):
            break
        time.sleep(0.5)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True)
    ap.add_argument("--models", required=True)
    args = ap.parse_args()
    home = tempfile.mkdtemp(prefix="kera-app-")
    app = start_app(args.kera, home)
    try:
        for _ in range(240):  # first launch measures the computer (quick setup)
            if not req("/api/status")["setup"]["running"]:
                break
            time.sleep(0.5)
        assert req("/api/models/folder", {"folder": str(pathlib.Path(args.models).resolve())})["added"]
        model = [m["path"] for m in req("/api/models")["models"] if "nemotron_h_moe" in m["path"]][0]
        assert req("/api/run", {"path": model})["started"]
        s = wait_ready()
        assert s["state"] == "ready", s
        url = s["url"]
        print(f"ready after {s['seconds']:.1f} s at {url}")
        with urllib.request.urlopen(url + "/models", timeout=10) as r:
            assert json.loads(r.read())["data"], "the server lists no model"
        req("/api/run/stop", {})
        assert req("/api/run")["state"] == "stopped"
    finally:
        try:
            req("/api/quit", {})
        except Exception:
            pass
        app.wait(timeout=60)
    app = start_app(args.kera, home)
    try:
        s = wait_ready()
        assert s["state"] == "ready" and s["url"] == url, s
        print(f"after a restart the same model loaded again by itself, at {s['url']}")
    finally:
        req("/api/quit", {})
        app.wait(timeout=60)
    print("PASS")


if __name__ == "__main__":
    main()
