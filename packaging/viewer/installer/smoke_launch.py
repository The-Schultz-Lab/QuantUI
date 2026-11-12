"""CI smoke test: start the installed viewer the way a shortcut does and
check that it serves the app.

    python smoke_launch.py PATH/TO/quantui[.exe]

Run from a shell where the installed environment is NOT activated, so this
catches anything that only works when its folders are on PATH (the first
Windows build failed exactly that way: "Voila is not installed or not on
PATH" from a Start-menu shortcut).

The page is fetched with the per-run token that ``quantui view`` prints, and
must contain a rendered widget. Without the token Voila answers 200 with its
login page, so a bare HTTP 200 proved only that a server was listening.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request

# nbconvert's lab template (which Voila renders with) emits every widget
# output as <script type="application/vnd.jupyter.widget-view+json">.
WIDGET_MARKER = "application/vnd.jupyter.widget-view+json"
_URL_RE = re.compile(r"Starting QuantUI Viewer at (http://\S+)")
# Loopback only: never route the request through a CI proxy.
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _echo_and_find_url(stream, found: list) -> None:
    """Pass the launcher's stdout through (CI log) and note the app URL.

    Drains the pipe for the whole run so Voila never blocks on a full pipe.
    """
    for line in stream:
        try:
            sys.stdout.write(line)
        except UnicodeError:  # a console that cannot show it must not stop us
            pass
        m = _URL_RE.search(line)
        if m and not found:
            found.append(m.group(1))


def main(quantui: str) -> int:
    port = _free_port()  # never a server left over from an earlier run
    env = dict(os.environ, PYTHONUNBUFFERED="1")  # the URL line arrives promptly
    proc = subprocess.Popen(
        [quantui, "view", "--no-browser", "--port", str(port)],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    url: list = []
    threading.Thread(
        target=_echo_and_find_url, args=(proc.stdout, url), daemon=True
    ).start()
    deadline = time.monotonic() + 180
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                print(f"quantui view exited early with code {proc.returncode}")
                return 1
            if not url or not _port_open(port):
                time.sleep(2)
                continue
            try:
                # Voila executes the notebook before the page completes, so
                # allow the app's start-up time on the read.
                with _NO_PROXY.open(url[0], timeout=120) as r:
                    status = r.status
                    body = r.read().decode("utf-8", "replace")
            except OSError as exc:
                print(f"fetch failed ({exc}); retrying")
                time.sleep(2)
                continue
            if status == 200 and WIDGET_MARKER in body:
                print("viewer served the app (HTTP 200 with a rendered widget)")
                break
            print(f"HTTP {status} without a rendered widget; page starts:")
            print(body[:2000])
            return 1
        else:
            print("viewer did not serve the app within 180 s")
            return 1
    finally:
        if os.name == "nt":
            # TerminateProcess runs no handlers, so Voila (a child) would
            # outlive it; kill the whole tree. Real Windows use is covered
            # differently: closing the console window stops every process
            # attached to it, and the app's Exit button stops Voila itself.
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
            )
        else:
            proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    if os.name == "nt":
        return 0
    # macOS/Linux: SIGTERM to quantui must stop Voila too (the launcher
    # forwards it), not leave it serving.
    time.sleep(3)
    if _port_open(port):
        print(f"Voila still serving on port {port} after quantui stopped")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
