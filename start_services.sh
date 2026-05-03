#!/bin/bash
# Start all three Connect4 robot services in separate tmux panes

set -e

# The connect4_robot package lives here; Python needs the *parent* on sys.path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Python interpreter — the venv that has fastapi/uvicorn/pydantic installed
PYTHON="${PYTHON:-/home/aft/dsr-motion/api/python/venv/bin/python3}"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install it first:"
    echo "  Ubuntu/Debian: sudo apt-get install tmux"
    exit 1
fi

SESSION="connect4"

# Kill existing session if it exists
tmux kill-session -t $SESSION 2>/dev/null || true

# Create new session (one window, one pane) from the parent directory
tmux new-session -d -s $SESSION -c "$PARENT_DIR" -x 220 -y 55

# Pane 0.0 already exists — Vision Service
tmux send-keys -t ${SESSION}:0.0 "$PYTHON -m connect4_robot.vision.service" Enter

# Pane 0.1 — split right, Orchestrator Service
tmux split-window -t ${SESSION}:0.0 -h -c "$PARENT_DIR"
tmux send-keys -t ${SESSION}:0.1 "$PYTHON -m connect4_robot.orchestrator_service" Enter

# Pane 0.2 — split pane 0.0 vertically, Web UI Service
tmux split-window -t ${SESSION}:0.0 -v -c "$PARENT_DIR"
tmux send-keys -t ${SESSION}:0.2 "$PYTHON -m connect4_robot.web_service" Enter

# Even out the layout
tmux select-layout -t ${SESSION}:0 main-vertical

# Focus the vision pane
tmux select-pane -t ${SESSION}:0.0

echo "Connect4 Robot services started in tmux session: $SESSION"
echo ""
echo "  Web UI:        http://localhost:8003"
echo "  Vision:        http://localhost:8001"
echo "  Orchestrator:  http://localhost:8000"
echo ""
echo "Commands:"
echo "  Attach:  tmux attach-session -t $SESSION"
echo "  Detach:  Ctrl+B then D"
echo "  Kill:    tmux kill-session -t $SESSION"
echo ""
echo "Pane navigation:"
echo "  Ctrl+B then q       - show pane numbers"
echo "  Ctrl+B then 0/1/2   - switch pane"
echo "  Ctrl+B then [       - scroll mode (ESC to exit)"
