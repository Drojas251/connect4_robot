#!/usr/bin/env python3
"""
Start all three Connect4 robot services in a single terminal.
Each service's output is prefixed with a colored tag and tee'd to a log file.

Usage:
    python start_services.py            # real hardware (camera + motors)
    python start_services.py --sim      # simulated robot + simulated vision

Override the Python interpreter:
    PYTHON=/path/to/python python start_services.py

By default the launcher uses the project's local virtualenv at
`./.venv/bin/python` (created via `python -m venv .venv && .venv/bin/pip install -e .`).

Logs are written to logs/<service>.log (plain text, no ANSI).
"""
import argparse
import datetime
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

REPO_DIR   = pathlib.Path(__file__).resolve().parent       # connect4_robot/
LOG_DIR    = REPO_DIR / "logs"
DEFAULT_VENV_PY = REPO_DIR / ".venv" / "bin" / "python"
PYTHON     = os.environ.get("PYTHON", str(DEFAULT_VENV_PY) if DEFAULT_VENV_PY.exists() else sys.executable)

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"


def build_services(sim: bool):
    vision_module = "connect4_robot.vision.sim_service" if sim else "connect4_robot.vision.service"
    return [
        {
            "name":  "vision" + ("-sim" if sim else ""),
            "port":  8001,
            "color": "\033[96m",   # cyan
            "cmd":   [PYTHON, "-m", vision_module],
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
    parser = argparse.ArgumentParser(description="Start Connect4 robot services.")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Run in simulation mode: fake gantry + fake vision (no hardware needed).",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    services = build_services(args.sim)

    mode = "SIM" if args.sim else "HARDWARE"
    print(f"\n{BOLD}Connect4 Robot — starting services [{mode}]{RESET}")
    print(f"  python  →  {PYTHON}\n")

    # Child-process env: tell the orchestrator to use the sim gantry.
    env = os.environ.copy()
    if args.sim:
        env["CONNECT4_SIM"] = "1"

    processes: list[tuple[str, subprocess.Popen]] = []
    for svc in services:
        log_path = LOG_DIR / f"{svc['name']}.log"
        # Write a start-of-session marker so log files are easy to scan
        with log_path.open("a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"  Session started {datetime.datetime.now().isoformat(timespec='seconds')} [{mode}]\n")
            f.write(f"{'='*60}\n")

        proc = subprocess.Popen(
            svc["cmd"],
            cwd=str(REPO_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        tag = f"{svc['color']}[{svc['name']:<14}]{RESET}"
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
