#!/usr/bin/env python3
"""
Start all three Connect4 robot services in a single terminal.
Each service's output is prefixed with a colored tag and tee'd to a log file.

Usage (from anywhere):
    python connect4_robot/start_services.py

Override the Python interpreter:
    PYTHON=/path/to/python python connect4_robot/start_services.py

Logs are written to connect4_robot/logs/<service>.log (plain text, no ANSI).
"""
import datetime
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

REPO_DIR   = pathlib.Path(__file__).resolve().parent       # connect4_robot/
PARENT_DIR = str(REPO_DIR.parent)                          # repo root (cwd for subprocesses)
LOG_DIR    = REPO_DIR / "logs"
PYTHON     = os.environ.get("PYTHON", "/home/aft/dsr-motion/api/python/venv/bin/python3")

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

SERVICES = [
    {
        "name":  "vision",
        "port":  8001,
        "color": "\033[96m",   # cyan
        "cmd":   [PYTHON, "-m", "connect4_robot.vision.service"],
    },
    {
        "name":  "orchestrator",
        "port":  8000,
        "color": "\033[93m",   # yellow
        "cmd":   [PYTHON, "-m", "connect4_robot.orchestrator_service"],
    },
    {
        "name":  "web",
        "port":  8003,
        "color": "\033[92m",   # green
        "cmd":   [PYTHON, "-m", "connect4_robot.web_service"],
    },
]


def _stream(proc: subprocess.Popen, tag: str, log_path: pathlib.Path) -> None:
    """Read lines from proc.stdout, print with colored tag, and write to log file."""
    with log_path.open("a", buffering=1) as log_f:
        for line in proc.stdout:
            ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"{tag} {line}", end="", flush=True)
            log_f.write(f"[{ts}] {line}")


def main() -> int:
    LOG_DIR.mkdir(exist_ok=True)
    print(f"\n{BOLD}Connect4 Robot — starting services{RESET}\n")

    processes: list[tuple[str, subprocess.Popen]] = []
    for svc in SERVICES:
        log_path = LOG_DIR / f"{svc['name']}.log"
        # Write a start-of-session marker so log files are easy to scan
        with log_path.open("a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"  Session started {datetime.datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"{'='*60}\n")

        proc = subprocess.Popen(
            svc["cmd"],
            cwd=PARENT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        tag = f"{svc['color']}[{svc['name']:<12}]{RESET}"
        threading.Thread(target=_stream, args=(proc, tag, log_path), daemon=True).start()
        processes.append((svc["name"], proc))
        print(f"  {svc['color']}[{svc['name']}]{RESET} port {svc['port']}  pid {proc.pid}  log → logs/{svc['name']}.log")
        time.sleep(0.4)   # stagger startups so logs don't collide

    print(f"\n{BOLD}{'─'*52}{RESET}")
    print(f"  Web dashboard  →  http://localhost:8003")
    print(f"  Vision API     →  http://localhost:8001")
    print(f"  Orchestrator   →  http://localhost:8000")
    print(f"  Logs           →  {LOG_DIR}")
    print(f"\n  {DIM}Ctrl+C to stop all services{RESET}")
    print(f"{BOLD}{'─'*52}{RESET}\n")

    def _shutdown(sig=None, frame=None):
        print(f"\n{BOLD}Stopping services…{RESET}")
        for name, proc in processes:
            proc.terminate()
        for name, proc in processes:
            try:
                proc.wait(timeout=5)
                print(f"  ✓ {name}")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"  ✗ {name} (killed)")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Watch for unexpected process death
    while True:
        time.sleep(1)
        for name, proc in processes:
            if proc.poll() is not None:
                print(f"\n{BOLD}⚠  {name} exited (code {proc.returncode}){RESET}")
                _shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
