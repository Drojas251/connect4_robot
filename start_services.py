#!/usr/bin/env python3
"""
Simple script to start all three Connect4 robot services in separate processes.
Easier than tmux for simple use cases.

Usage:
    python start_services.py
"""
import subprocess
import sys
import time
import signal
import os
import pathlib

# Run from the parent of the connect4_robot package so imports resolve correctly
PARENT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)

# Python interpreter — override with PYTHON env var if needed
PYTHON = os.environ.get(
    "PYTHON",
    "/home/aft/dsr-motion/api/python/venv/bin/python3",
)


def signal_handler(sig, frame):
    print("\n\nStopping all services...")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)

    services = [
        {
            "name": "Vision Service",
            "port": 8001,
            "cmd": [PYTHON, "-m", "connect4_robot.vision.service"],
        },
        {
            "name": "Orchestrator Service",
            "port": 8000,
            "cmd": [PYTHON, "-m", "connect4_robot.orchestrator_service"],
        },
        {
            "name": "Web UI Service",
            "port": 8003,
            "cmd": [PYTHON, "-m", "connect4_robot.web_service"],
        },
    ]

    print("Starting Connect4 Robot Services\n")

    processes = []
    try:
        for service in services:
            print(f"Starting {service['name']} on port {service['port']}...")
            proc = subprocess.Popen(
                service['cmd'],
                cwd=PARENT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid if sys.platform != "win32" else None,
            )
            processes.append((service['name'], proc))
            time.sleep(0.5)

        print("\n" + "=" * 60)
        print("All services started!")
        print("=" * 60)
        print(f"\n  Web UI:        http://localhost:8003")
        print(f"  Vision:        http://localhost:8001")
        print(f"  Orchestrator:  http://localhost:8000")
        print(f"\nPress Ctrl+C to stop all services\n")
        print("=" * 60 + "\n")
        
        # Wait for all processes
        while True:
            time.sleep(1)
            # Check if any process died
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"\n⚠️  {name} died with exit code {proc.returncode}")
                    return 1
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping all services...")
    
    finally:
        print("Killing processes...")
        for name, proc in processes:
            try:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
                proc.wait(timeout=5)
                print(f"  ✓ {name} stopped")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                try:
                    proc.kill()
                except:
                    pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
