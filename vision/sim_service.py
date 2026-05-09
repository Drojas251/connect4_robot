from __future__ import annotations

import random
import time

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from connect4_robot.config import BOARD, SERVICES, VISION_SIM
from connect4_robot.game_engine.board import Cell
from connect4_robot.vision.sim_world import VisionState, SimBoard

app = FastAPI(title="Connect4 Vision Sim Service", version="1.0.0")
state = VisionState()


class HumanMoveRequest(BaseModel):
    column: int | None = Field(default=None, ge=0, le=BOARD.cols - 1)


def append_history(msg: str):
    state.move_history.append(msg)
    state.move_history = state.move_history[-40:]


def get_orchestrator_status():
    resp = requests.get(f"{SERVICES.orchestrator_url}/status", timeout=10)
    resp.raise_for_status()
    return resp.json()


def send_board_if_changed(reason: str):
    board_top_down = state.board.to_strings_top_down()
    if state.last_sent_board == board_top_down:
        return {"sent": False, "reason": "no_change"}
    payload = {"board": board_top_down, "source": "vision_sim_service"}
    resp = requests.post(f"{SERVICES.orchestrator_url}/vision/update", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    state.last_sent_board = board_top_down
    append_history(f"sensed change -> sent to orchestrator ({reason})")
    return data


def wait_for_robot_motion_complete(timeout_s: float = VISION_SIM.motion_wait_timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        orch = get_orchestrator_status()
        motor_state = orch.get("motor_state", {})
        moving = bool(motor_state.get("moving", False))
        awaiting = bool(orch.get("awaiting_robot_confirmation", False))
        target_col = orch.get("robot_target_col", None)

        if awaiting and target_col is not None and not moving:
            return orch

        time.sleep(0.1)
    raise TimeoutError("Timed out waiting for robot motion to complete.")


@app.get("/health")
def health():
    return {"ok": True, "service": "connect4-vision-sim"}


@app.get("/status")
def get_status():
    orch = None
    orch_error = None

    try:
        orch = get_orchestrator_status()
    except Exception as e:
        orch_error = str(e)

    return {
        "vision_board_top_down": state.board.to_strings_top_down(),
        "last_sent_board": state.last_sent_board,
        "move_history": state.move_history,
        "orchestrator": orch,
        "orchestrator_error": orch_error,
    }


@app.post("/reset")
def reset():
    state.board = SimBoard()
    state.last_sent_board = None
    state.move_history.clear()
    r = requests.post(f"{SERVICES.orchestrator_url}/reset", timeout=10)
    r.raise_for_status()
    send_board_if_changed("reset-empty-board")
    return {"ok": True}


@app.post("/simulate/human_move")
def simulate_human_move(req: HumanMoveRequest):
    legal = state.board.legal_moves()
    if not legal:
        raise HTTPException(status_code=400, detail="No legal human moves.")
    col = req.column if req.column is not None else random.choice(legal)
    if col not in legal:
        raise HTTPException(status_code=400, detail=f"Column {col} is not legal.")

    state.board.apply_move(col, Cell.HUMAN)
    append_history(f"human sim -> col {col}")
    robot_plan = send_board_if_changed(f"human col {col}")

    orch = wait_for_robot_motion_complete()
    target_col = orch.get("robot_target_col")
    if target_col is None:
        target_col = robot_plan.get("chosen_column")

    if target_col is None:
        raise HTTPException(status_code=500, detail="Could not determine robot target column.")

    if target_col not in state.board.legal_moves():
        raise HTTPException(
            status_code=500,
            detail=f"Robot target column {target_col} is not legal in vision world state."
        )

    state.board.apply_move(target_col, Cell.ROBOT)
    append_history(f"vision sensed robot -> col {target_col}")
    send_board_if_changed(f"robot confirmation col {target_col}")

    return {"ok": True, "human_col": col, "robot_col": target_col}


@app.get("/", response_class=HTMLResponse)
def ui():
    return HTMLResponse(
        f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Connect4 Vision Sim Service</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #111; color: #eee; }}
    .controls {{ margin: 16px 0; display: flex; gap: 8px; flex-wrap: wrap; }}
    button {{ padding: 10px 14px; border: none; border-radius: 8px; cursor: pointer; }}
    .board {{ display: inline-grid; grid-template-columns: repeat({BOARD.cols}, 54px); gap: 8px; padding: 12px; background: #1f4db8; border-radius: 16px; }}
    .cell {{ width: 54px; height: 54px; border-radius: 50%; background: #1b1b1b; display: inline-block; }}
    .human {{ background: #fdd835; }}   /* human = yellow */
    .robot {{ background: #e53935; }}   /* robot = red   */
    .panel {{ margin-top: 16px; display: grid; grid-template-columns: repeat(3, minmax(260px, 420px)); gap: 16px; }}
    .card {{ background: #1c1c1c; padding: 14px; border-radius: 12px; }}
    .mono {{ font-family: monospace; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Connect 4 Vision Sim Service</h1>
  <div>This page is the simulated camera-side world state. Human and robot pieces both appear here only when the vision service "senses" them.</div>

  <div class="controls">
    <button onclick="simulateHumanMove()">Simulate random human move</button>
    {''.join(f'<button onclick="simulateHumanMove({c})">Col {c}</button>' for c in range(BOARD.cols))}
    <button onclick="resetAll()">Reset</button>
  </div>

  <div id="board" class="board"></div>

  <div class="panel">
    <div class="card"><h3>Vision state</h3><div id="visionStatus" class="mono"></div></div>
    <div class="card"><h3>Orchestrator state</h3><div id="orchStatus" class="mono"></div></div>
    <div class="card"><h3>Vision history</h3><div id="history" class="mono"></div></div>
  </div>

  <script>
    function renderBoard(rowsTopDown) {{
      const board = document.getElementById('board');
      board.innerHTML = '';
      for (const row of rowsTopDown) {{
        for (const val of row) {{
          const d = document.createElement('div');
          d.className = 'cell';
          if (val === 'H') d.classList.add('human');
          if (val === 'R') d.classList.add('robot');
          board.appendChild(d);
        }}
      }}
    }}

    async function fetchStatus() {{
      try {{
        const resp = await fetch('/status');

        if (!resp.ok) {{
          console.error("Status request failed");
          return;
        }}

        const data = await resp.json();

        renderBoard(data.vision_board_top_down);

        document.getElementById('visionStatus').textContent =
          'last_sent_board: ' + JSON.stringify(data.last_sent_board);

        const o = data.orchestrator;

        if (!o) {{
          document.getElementById('orchStatus').textContent =
            'orchestrator unavailable\\n' +
            'error: ' + (data.orchestrator_error || 'unknown');
        }} else {{
          document.getElementById('orchStatus').textContent =
            'status: ' + o.status + '\\n' +
            'winner: ' + o.winner + '\\n' +
            'board_version: ' + o.board_version + '\\n' +
            'last_human_col: ' + o.last_human_col + '\\n' +
            'last_robot_col: ' + o.last_robot_col + '\\n' +
            'robot_target_col: ' + o.robot_target_col + '\\n' +
            'awaiting_robot_confirmation: ' + o.awaiting_robot_confirmation + '\\n\\n' +
            o.pretty;
        }}

        document.getElementById('history').textContent =
          (data.move_history || []).join('\\n');

      }} catch (err) {{
        console.error("fetchStatus error:", err);
      }}
    }}

    async function simulateHumanMove(col = null) {{
      await fetch('/simulate/human_move', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{column: col}})
      }});
      await fetchStatus();
    }}

    async function resetAll() {{
      await fetch('/reset', {{method: 'POST'}});
      await fetchStatus();
    }}

    fetchStatus();
    setInterval(fetchStatus, {int(VISION_SIM.poll_period_s*1000)});
  </script>
</body>
</html>
        """
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "connect4_robot.vision.sim_service:app",
        host=SERVICES.vision_host,
        port=SERVICES.vision_port,
        reload=False,
    )