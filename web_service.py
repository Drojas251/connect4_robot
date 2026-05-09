from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from connect4_robot.config import BOARD, SERVICES

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class _WebState:
    def __init__(self):
        self.last_board_top_down: Optional[list] = None
        self.last_vision_update: float = 0.0
        self.orchestrator_data: dict = {}
        self.orch_ok: bool = False
        self.vision_svc_data: dict = {}
        self.vision_svc_ok: bool = False
        self._lock = threading.Lock()

    def push_board(self, board_top_down: Optional[list]):
        with self._lock:
            self.last_board_top_down = board_top_down
            self.last_vision_update = time.time()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "orchestrator": dict(self.orchestrator_data),
                "orch_ok": self.orch_ok,
                "vision_board": self.last_board_top_down,
                "last_vision_update": self.last_vision_update,
                "vision_svc": dict(self.vision_svc_data),
                "vision_svc_ok": self.vision_svc_ok,
            }


_state = _WebState()


def _poll_loop(url: str, interval: float, ok_attr: str, data_attr: str):
    while True:
        try:
            resp = requests.get(f"{url}/status", timeout=1.5)
            with _state._lock:
                setattr(_state, data_attr, resp.json())
                setattr(_state, ok_attr, True)
        except Exception as e:
            with _state._lock:
                setattr(_state, data_attr, {"error": str(e)})
                setattr(_state, ok_attr, False)
        time.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for url, interval, ok_attr, data_attr in [
        (SERVICES.orchestrator_url, 1.0, "orch_ok",       "orchestrator_data"),
        (SERVICES.vision_url,       1.0, "vision_svc_ok", "vision_svc_data"),
    ]:
        t = threading.Thread(
            target=_poll_loop, args=(url, interval, ok_attr, data_attr), daemon=True
        )
        t.start()
    yield


app = FastAPI(title="Connect4 Robot Web UI", version="1.0.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

class BoardUpdatePayload(BaseModel):
    board: list
    source: str = "vision_service"


@app.get("/health")
def health():
    return {"ok": True, "service": "connect4-web"}


@app.post("/api/board_update")
def board_update(payload: BoardUpdatePayload):
    _state.push_board(payload.board)
    return {"ok": True}


@app.get("/api/status")
def api_status():
    return _state.snapshot()


@app.post("/api/pause")
def api_pause():
    results = {}
    for name, url in [("orchestrator", SERVICES.orchestrator_url), ("vision", SERVICES.vision_url)]:
        try:
            results[name] = requests.post(f"{url}/pause", timeout=3).json()
        except Exception as e:
            results[name] = {"error": str(e)}
    return {"ok": True, "results": results}


@app.post("/api/resume")
def api_resume():
    results = {}
    for name, url in [("orchestrator", SERVICES.orchestrator_url), ("vision", SERVICES.vision_url)]:
        try:
            results[name] = requests.post(f"{url}/resume", timeout=3).json()
        except Exception as e:
            results[name] = {"error": str(e)}
    return {"ok": True, "results": results}


@app.post("/api/retry_detection")
def retry_detection():
    """Restart vision detection only — does NOT home the robot."""
    try:
        r = requests.post(f"{SERVICES.vision_url}/reset", timeout=3)
        return {"ok": True, "vision": r.json()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/vision_image")
def vision_image():
    """Proxy the detection image from the vision service."""
    try:
        resp = requests.get(f"{SERVICES.vision_url}/api/detection_image", timeout=3)
        resp.raise_for_status()
        return Response(content=resp.content, media_type="image/jpeg")
    except Exception:
        raise HTTPException(status_code=404, detail="Detection image not available")


@app.post("/api/reset")
def api_reset():
    results = {}
    # Orchestrator reset homes the robot — must match the motor homing timeout (60s)
    try:
        results["orchestrator"] = requests.post(
            f"{SERVICES.orchestrator_url}/reset", timeout=75
        ).json()
    except Exception as e:
        results["orchestrator"] = {"error": str(e)}
    # Vision reset is instant (just sets restart_requested flag)
    try:
        results["vision"] = requests.post(
            f"{SERVICES.vision_url}/reset", timeout=5
        ).json()
    except Exception as e:
        results["vision"] = {"error": str(e)}
    _state.push_board(None)
    return {"ok": True, "results": results}


@app.get("/api/ai")
def api_get_ai():
    try:
        return requests.get(f"{SERVICES.orchestrator_url}/ai", timeout=3).json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


class AISelectPayload(BaseModel):
    name: str
    difficulty: Optional[str] = None


@app.post("/api/ai")
def api_set_ai(payload: AISelectPayload):
    try:
        r = requests.post(
            f"{SERVICES.orchestrator_url}/ai",
            json=payload.model_dump(exclude_none=True),
            timeout=3,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_COLS = BOARD.cols
_ROWS = BOARD.rows
_COL_MM_JS   = str(BOARD.column_centers_mm)   # e.g. [20, 58, 96, ...]
_HOME_MM      = BOARD.home_mm
_STAGING_MM   = BOARD.staging_mm

_HTML = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Connect4 Robot Dashboard</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Arial, sans-serif; background: #0d0d0d; color: #e0e0e0;
            padding: 18px; min-height: 100vh; }}

    /* ---- Header ---- */
    .header {{ display: flex; align-items: center; justify-content: space-between;
               margin-bottom: 12px; flex-wrap: wrap; gap: 10px; }}
    h1 {{ font-size: 1.3rem; color: #fff; }}
    .services {{ display: flex; gap: 8px; }}
    .svc {{ padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: bold;
             letter-spacing: 0.4px; background: #333; color: #999; transition: all 0.3s; }}
    .svc.ok  {{ background: #1b5e20; color: #a5d6a7; }}
    .svc.err {{ background: #4e1010; color: #ef9a9a; }}

    /* ---- Action banner ---- */
    .action-banner {{
      padding: 10px 16px; border-radius: 10px; margin-bottom: 12px;
      font-size: 1rem; font-weight: bold; background: #1c1c1c;
      border-left: 4px solid #607d8b; transition: border-color 0.3s, background 0.3s;
    }}

    /* ---- Controls ---- */
    .controls {{ display: flex; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; align-items: center; }}
    button {{ padding: 8px 18px; border: none; border-radius: 8px;
              cursor: pointer; font-size: 0.88rem; font-weight: bold; transition: opacity 0.2s; }}
    button:hover {{ opacity: 0.82; }}
    .btn-pause  {{ background: #7b1fa2; color: #fff; }}
    .btn-resume {{ background: #2e7d32; color: #fff; }}
    .btn-reset  {{ background: #b71c1c; color: #fff; }}

    /* ---- Difficulty selector ---- */
    .difficulty-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }}
    .difficulty-label {{ font-size: 0.8rem; color: #78909c; margin-right: 4px; }}
    .diff-btn {{
      padding: 6px 16px; border: 2px solid transparent; border-radius: 20px;
      cursor: pointer; font-size: 0.82rem; font-weight: bold;
      background: #1e1e1e; color: #777; transition: all 0.2s;
    }}
    .diff-btn:hover {{ color: #ccc; border-color: #555; }}
    .diff-btn.easy.active   {{ background: #1b5e20; color: #a5d6a7; border-color: #4caf50; }}
    .diff-btn.medium.active {{ background: #e65100; color: #ffe0b2; border-color: #ff9800; }}
    .diff-btn.hard.active   {{ background: #b71c1c; color: #ffcdd2; border-color: #f44336; }}
    .ai-badge {{
      font-size: 0.75rem; color: #546e7a; background: #1a1a1a;
      border: 1px solid #2a2a2a; border-radius: 12px; padding: 3px 10px;
      margin-left: 6px;
    }}

    /* ---- Main layout ---- */
    .main-grid {{ display: grid; grid-template-columns: auto 1fr; gap: 20px; align-items: start; }}

    /* ---- Detection panel ---- */
    .det-panel {{
      border-radius: 12px; padding: 14px 16px; margin-bottom: 16px;
      border: 1px solid #333; background: #111;
      transition: border-color 0.3s, background 0.3s;
    }}
    .det-panel.searching   {{ border-color: #ffa000; }}
    .det-panel.checking    {{ border-color: #ffa000; }}
    .det-panel.detection_failed {{ border-color: #c62828; background: #1a0000; }}
    .det-panel.board_not_empty  {{ border-color: #e65100; background: #1a0d00; }}
    .det-panel.ready       {{ border-color: #2e7d32; background: #071207; }}
    .det-panel.hidden      {{ display: none; }}
    .det-header  {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
    .det-icon    {{ font-size: 1.4rem; width: 28px; text-align: center; flex-shrink: 0; }}
    .det-text    {{ flex: 1; min-width: 0; }}
    .det-title   {{ font-weight: bold; font-size: 0.95rem; color: #fff; }}
    .det-msg     {{ font-size: 0.82rem; color: #999; margin-top: 3px; }}
    .det-img     {{ display: block; margin-top: 12px; max-width: 100%;
                    border-radius: 8px; border: 1px solid #333; }}
    .btn-retry   {{ background: #bf360c; color: #fff; padding: 7px 16px;
                    border: none; border-radius: 7px; cursor: pointer;
                    font-weight: bold; font-size: 0.85rem; }}
    .btn-dismiss {{ background: #1b5e20; color: #fff; padding: 7px 16px;
                    border: none; border-radius: 7px; cursor: pointer;
                    font-weight: bold; font-size: 0.85rem; }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    .spin {{ display: inline-block; animation: spin 1s linear infinite; }}

    /* ---- Board section ---- */
    .board-section {{ display: flex; flex-direction: column; gap: 0; }}
    .legend {{ display: flex; gap: 14px; margin-bottom: 8px; }}
    .legend-item {{ display: flex; align-items: center; gap: 6px;
                    font-size: 0.82rem; color: #bbb; }}
    .legend-dot {{ width: 16px; height: 16px; border-radius: 50%; flex-shrink: 0; }}
    .board-wrap {{
      background: #1a3a8f; border-radius: 14px; padding: 12px;
      display: inline-grid; grid-template-columns: repeat({_COLS}, 54px); gap: 7px;
    }}
    .cell {{
      width: 54px; height: 54px; border-radius: 50%;
      background: #0a1a4a; transition: background 0.15s, box-shadow 0.2s;
    }}
    .human {{ background: #fdd835; }}
    .robot {{ background: #e53935; }}
    .last-human {{ box-shadow: 0 0 0 3px #fff59d, 0 0 12px 4px #fdd83599; }}
    .last-robot {{ box-shadow: 0 0 0 3px #ffcdd2, 0 0 12px 4px #e5393599; }}

    /* column labels */
    .col-labels {{
      display: grid; grid-template-columns: repeat({_COLS}, 54px); gap: 7px;
      padding: 5px 12px 0; font-size: 0.78rem; color: #555; text-align: center;
    }}

    /* ---- Gantry track ---- */
    .gantry-section {{ margin-top: 10px; padding: 0 12px; }}
    .gantry-label {{ font-size: 0.72rem; color: #555; margin-bottom: 4px; }}
    .gantry-track {{
      position: relative; height: 20px;
      background: #1c1c2e; border-radius: 10px; overflow: visible;
    }}
    .gantry-col-tick {{
      position: absolute; top: 0; bottom: 0; width: 2px;
      background: #2a2a50; border-radius: 1px;
    }}
    .gantry-marker {{
      position: absolute; top: 2px; width: 16px; height: 16px;
      background: #fff; border-radius: 50%;
      transition: left 0.4s ease; transform: translateX(-50%);
      box-shadow: 0 0 6px #fff8;
    }}
    .gantry-marker.moving {{ background: #ffd740; box-shadow: 0 0 8px #ffd74099; }}

    /* ---- Info panel ---- */
    .info-panel {{ display: flex; flex-direction: column; gap: 12px; }}
    .card {{ background: #161616; border-radius: 12px; padding: 13px; }}
    .card h3 {{ font-size: 0.88rem; color: #90caf9; border-bottom: 1px solid #2a2a2a;
                padding-bottom: 6px; margin-bottom: 10px; }}
    .kv {{ display: grid; grid-template-columns: max-content 1fr;
           gap: 3px 12px; font-size: 0.8rem; }}
    .kv .k {{ color: #78909c; }}
    .kv .v {{ color: #e0e0e0; font-family: monospace; }}

    /* ---- Activity log ---- */
    .activity-log {{ display: flex; flex-direction: column; gap: 4px;
                     max-height: 300px; overflow-y: auto; }}
    .log-entry {{
      font-size: 0.78rem; font-family: monospace; padding: 4px 8px;
      border-radius: 5px; border-left: 3px solid #333; color: #bbb;
      background: #111;
    }}
    .log-entry.human  {{ border-color: #fdd835; color: #fff59d; }}
    .log-entry.robot  {{ border-color: #e53935; color: #ffcdd2; }}
    .log-entry.winner {{ border-color: #ffd700; background: #1a1500;
                         color: #ffe57f; font-weight: bold; }}
    .log-entry.system {{ border-color: #546e7a; color: #90a4ae; }}
    .log-entry.error  {{ border-color: #c62828; background: #1a0000; color: #ef9a9a; }}

    .ts {{ color: #444; font-size: 0.72rem; margin-top: 8px; }}
    @media (max-width: 760px) {{ .main-grid {{ grid-template-columns: 1fr; }} }}

    /* ---- Tabs ---- */
    .tab-bar {{ display: flex; gap: 4px; margin-bottom: 14px; border-bottom: 1px solid #2a2a2a; padding-bottom: 0; }}
    .tab-btn {{
      padding: 8px 22px; border: none; border-radius: 8px 8px 0 0;
      cursor: pointer; font-size: 0.88rem; font-weight: bold;
      background: #1a1a1a; color: #666; transition: all 0.15s;
      border-bottom: 2px solid transparent; margin-bottom: -1px;
    }}
    .tab-btn:hover {{ color: #bbb; background: #222; }}
    .tab-btn.active {{ background: #0d0d0d; color: #fff; border-bottom-color: #2196f3; }}
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; }}

    /* ---- Debug tab ---- */
    .debug-layout {{ display: grid; grid-template-columns: 1fr 300px; gap: 20px; align-items: start; }}
    .debug-feed img {{ width: 100%; border-radius: 10px; border: 1px solid #2a2a2a;
                       background: #111; display: block; }}
    .debug-feed .no-feed {{ padding: 60px 0; text-align: center; color: #444;
                            font-size: 0.9rem; background: #111; border-radius: 10px;
                            border: 1px solid #2a2a2a; }}
    @media (max-width: 760px) {{ .debug-layout {{ grid-template-columns: 1fr; }} }}

    /* ---- Begin-game modal ---- */
    .modal-overlay {{
      position: fixed; inset: 0; z-index: 1000;
      background: rgba(0,0,0,0.88);
      backdrop-filter: blur(4px);
      display: flex; align-items: center; justify-content: center;
    }}
    .modal-overlay.hidden {{ display: none; }}
    .modal-card {{
      background: #12122a; border: 1px solid #2a2a50;
      border-radius: 20px; padding: 36px 40px;
      max-width: 460px; width: 92%; text-align: center;
      box-shadow: 0 24px 80px #0009;
    }}
    .modal-logo {{ font-size: 2.6rem; margin-bottom: 4px; }}
    .modal-title {{ font-size: 1.6rem; font-weight: bold; color: #fff; }}
    .modal-sub {{ font-size: 0.88rem; color: #546e7a; margin: 6px 0 28px; }}
    .modal-diff-row {{ display: flex; gap: 12px; justify-content: center; margin-bottom: 28px; }}
    .modal-diff-card {{
      flex: 1; max-width: 120px; padding: 14px 10px 12px;
      border-radius: 12px; border: 2px solid #2a2a2a;
      background: #1a1a2e; cursor: pointer;
      transition: border-color 0.2s, background 0.2s, transform 0.15s;
    }}
    .modal-diff-card:hover {{ transform: translateY(-2px); }}
    .modal-diff-card .diff-name {{
      font-size: 1rem; font-weight: bold; color: #bbb; margin-bottom: 5px;
    }}
    .modal-diff-card .diff-desc {{
      font-size: 0.72rem; color: #546e7a; line-height: 1.35;
    }}
    .modal-diff-card.easy.active  {{ border-color: #4caf50; background: #0d2010; }}
    .modal-diff-card.easy.active  .diff-name {{ color: #a5d6a7; }}
    .modal-diff-card.medium.active {{ border-color: #ff9800; background: #1f1000; }}
    .modal-diff-card.medium.active .diff-name {{ color: #ffcc80; }}
    .modal-diff-card.hard.active  {{ border-color: #f44336; background: #200808; }}
    .modal-diff-card.hard.active  .diff-name {{ color: #ffcdd2; }}
    .btn-begin {{
      width: 100%; padding: 14px; font-size: 1.1rem; font-weight: bold;
      border: none; border-radius: 10px; cursor: pointer;
      background: #1565c0; color: #fff;
      transition: background 0.2s, transform 0.1s;
    }}
    .btn-begin:hover {{ background: #1976d2; transform: translateY(-1px); }}
    .btn-begin:active {{ transform: translateY(0); }}
  </style>
</head>
<body>

  <!-- Begin-game modal -->
  <div id="beginModal" class="modal-overlay">
    <div class="modal-card">
      <div class="modal-logo">🤖</div>
      <div class="modal-title">Connect4 Robot</div>
      <div class="modal-sub" id="modalSub">Choose a difficulty to begin</div>
      <div class="modal-diff-row">
        <div class="modal-diff-card easy" data-diff="easy" onclick="selectModalDiff('easy')">
          <div class="diff-name">Easy</div>
          <div class="diff-desc">Occasional blunders — good for learning</div>
        </div>
        <div class="modal-diff-card medium active" data-diff="medium" onclick="selectModalDiff('medium')">
          <div class="diff-name">Medium</div>
          <div class="diff-desc">Strategic play — solid challenge</div>
        </div>
        <div class="modal-diff-card hard" data-diff="hard" onclick="selectModalDiff('hard')">
          <div class="diff-name">Hard</div>
          <div class="diff-desc">Near-perfect — good luck!</div>
        </div>
      </div>
      <button class="btn-begin" onclick="startGame()">Begin Game</button>
    </div>
  </div>

  <div class="header">
    <h1>Connect4 Robot</h1>
    <div class="services">
      <span class="svc" id="svcVision">Vision</span>
      <span class="svc" id="svcOrch">Orchestrator</span>
      <span class="svc" id="svcWeb" style="background:#1b5e20;color:#a5d6a7">Web</span>
    </div>
  </div>

  <div class="tab-bar">
    <button class="tab-btn active" id="tab-btn-game"  onclick="showTab('game')">Game</button>
    <button class="tab-btn"        id="tab-btn-debug" onclick="showTab('debug')">Debug</button>
  </div>

  <div id="tab-game" class="tab-content active">

  <div id="actionBanner" class="action-banner">Connecting…</div>

  <div class="controls">
    <button class="btn-pause"  onclick="doAction('pause')">Pause</button>
    <button class="btn-resume" onclick="doAction('resume')">Resume</button>
    <button class="btn-reset"  onclick="showBeginModal(true)">Reset</button>
  </div>

  <div class="difficulty-row">
    <span class="difficulty-label">Robot difficulty:</span>
    <button class="diff-btn easy"   onclick="setDifficulty('easy')">Easy</button>
    <button class="diff-btn medium" onclick="setDifficulty('medium')">Medium</button>
    <button class="diff-btn hard"   onclick="setDifficulty('hard')">Hard</button>
    <span class="ai-badge" id="aiBadge">minimax / medium</span>
  </div>

  <!-- Detection panel -->
  <div id="detPanel" class="det-panel searching">
    <div class="det-header">
      <div class="det-icon" id="detIcon"><span class="spin">⟳</span></div>
      <div class="det-text">
        <div class="det-title" id="detTitle">Detecting Board</div>
        <div class="det-msg"   id="detMsg">Searching for board corners…</div>
      </div>
      <button class="btn-retry"   id="btnRetry"   onclick="retryDetection()" style="display:none">Retry Detection</button>
      <button class="btn-dismiss" id="btnDismiss" onclick="dismissDetection()" style="display:none">Dismiss</button>
    </div>
    <img id="detImg" class="det-img" style="display:none" alt="Board detection overlay">
  </div>

  <div class="main-grid">

    <!-- Board column -->
    <div class="board-section">
      <div class="legend">
        <div class="legend-item">
          <div class="legend-dot" style="background:#fdd835"></div>Human (yellow)
        </div>
        <div class="legend-item">
          <div class="legend-dot" style="background:#e53935"></div>Robot (red)
        </div>
      </div>

      <div class="board-wrap" id="board"></div>

      <div class="col-labels" id="colLabels"></div>

      <div class="gantry-section">
        <div class="gantry-label">Gantry position</div>
        <div class="gantry-track" id="gantryTrack">
          <div class="gantry-marker" id="gantryMarker" style="left:0%"></div>
        </div>
      </div>

      <div id="lastUpdate" class="ts" style="margin-top:6px"></div>
    </div>

    <!-- Info panel -->
    <div class="info-panel">

      <div class="card">
        <h3>Game</h3>
        <div id="gameStats" class="kv"></div>
      </div>

      <div class="card">
        <h3>Gantry &amp; End Effector</h3>
        <div id="motorState" class="kv"></div>
      </div>

      <div class="card">
        <h3>Activity Log</h3>
        <div id="activityLog" class="activity-log"></div>
      </div>

    </div>
  </div>

  </div> <!-- /tab-game -->

  <!-- Debug tab -->
  <div id="tab-debug" class="tab-content">
    <div class="debug-layout">

      <div class="debug-feed">
        <img id="debugImg" alt="Vision feed" style="display:none">
        <div id="debugNoFeed" class="no-feed">No image yet — switch to Game tab to start detection</div>
      </div>

      <div class="info-panel">
        <div class="card">
          <h3>Vision</h3>
          <div id="debugVisionStats" class="kv"></div>
        </div>
        <div class="card">
          <h3>Board State</h3>
          <div id="debugBoardState" class="kv"></div>
        </div>
      </div>

    </div>
  </div> <!-- /tab-debug -->

  <script>
    const ROWS = {_ROWS}, COLS = {_COLS};
    const COL_MM = {_COL_MM_JS};
    const MM_MIN = {_HOME_MM};
    const MM_MAX = COL_MM[COL_MM.length - 1] + 20;

    const STATUS_COLORS = {{
      waiting_for_board: {{ bg: '#263238', border: '#546e7a', text: 'Waiting for board detection…' }},
      human_turn:        {{ bg: '#1b3a1b', border: '#4caf50', text: "Human's turn — waiting for move" }},
      robot_deciding:    {{ bg: '#2d2000', border: '#ff9800', text: 'Robot deciding next move…' }},
      robot_moving:      {{ bg: '#0d1f3c', border: '#2196f3', text: null }},  // text set dynamically
      game_over:         {{ bg: '#2d0e1a', border: '#e91e63', text: null }},
      paused:            {{ bg: '#1e0a2e', border: '#9c27b0', text: 'System paused' }},
      error:             {{ bg: '#1a0000', border: '#f44336', text: null }},
    }};

    // --- Build board cells once ---
    const boardEl = document.getElementById('board');
    const cells = [];
    for (let i = 0; i < ROWS * COLS; i++) {{
      const d = document.createElement('div');
      d.className = 'cell';
      boardEl.appendChild(d);
      cells.push(d);
    }}

    // Column labels
    const colLabels = document.getElementById('colLabels');
    for (let c = 0; c < COLS; c++) {{
      const s = document.createElement('span');
      s.textContent = c;
      colLabels.appendChild(s);
    }}

    // Column ticks on gantry track
    const track = document.getElementById('gantryTrack');
    COL_MM.forEach(mm => {{
      const tick = document.createElement('div');
      tick.className = 'gantry-col-tick';
      tick.style.left = mmToPct(mm) + '%';
      track.appendChild(tick);
    }});

    function mmToPct(mm) {{
      return Math.max(0, Math.min(100, (mm - MM_MIN) / (MM_MAX - MM_MIN) * 100));
    }}

    function renderBoard(rows, lastHumanCol, lastRobotCol) {{
      if (!rows) return;
      let i = 0;
      for (const row of rows) {{
        for (const v of row) {{
          cells[i].className = 'cell' + (v === 'H' ? ' human' : v === 'R' ? ' robot' : '');
          i++;
        }}
      }}
      // Highlight last move in each column (topmost non-empty = last placed)
      [['last-human', lastHumanCol], ['last-robot', lastRobotCol]].forEach(([cls, col]) => {{
        if (col == null) return;
        for (let r = 0; r < ROWS; r++) {{
          const idx = r * COLS + col;
          if (cells[idx].classList.contains('human') || cells[idx].classList.contains('robot')) {{
            cells[idx].classList.add(cls);
            break;
          }}
        }}
      }});
    }}

    function updateGantry(m) {{
      const marker = document.getElementById('gantryMarker');
      if (m.pos_mm != null) {{
        marker.style.left = mmToPct(m.pos_mm) + '%';
        marker.classList.toggle('moving', !!m.moving);
      }}
    }}

    function updateBanner(o) {{
      const banner = document.getElementById('actionBanner');
      const s = o.status;
      const cfg = STATUS_COLORS[s] || {{ bg: '#1c1c1c', border: '#607d8b', text: s }};

      let text = cfg.text;
      if (s === 'robot_moving') {{
        const col = o.robot_target_col;
        text = col != null ? `Robot moving to column ${{col}}` : 'Robot homing to start position…';
      }} else if (s === 'game_over') {{
        text = o.winner ? `Game over — ${{o.winner}} wins!` : 'Game over — Draw';
      }} else if (s === 'error') {{
        text = `Error: ${{o.last_error || 'unknown'}}`;
      }}

      banner.textContent = text || s;
      banner.style.background = cfg.bg;
      banner.style.borderLeftColor = cfg.border;
    }}

    function kv(pairs) {{
      return pairs
        .filter(([, v]) => v != null && v !== false && v !== '')
        .map(([k, v]) => `<span class="k">${{k}}</span><span class="v">${{v}}</span>`)
        .join('');
    }}

    function updateGameStats(o, visionTs) {{
      const board = o.board_top_down || [];
      let human = 0, robot = 0;
      board.forEach(row => row.forEach(v => {{ if (v==='H') human++; else if (v==='R') robot++; }}));
      const move = (o.board_version ?? 0);
      const aiLabel = o.ai_name
        ? (o.ai_difficulty ? `${{o.ai_name}} / ${{o.ai_difficulty}}` : o.ai_name)
        : null;
      document.getElementById('gameStats').innerHTML = kv([
        ['status',         o.status?.replace(/_/g,' ')],
        ['move #',         move],
        ['human pieces',   human],
        ['robot pieces',   robot],
        ['last human col', o.last_human_col],
        ['last robot col', o.last_robot_col],
        ['robot target',   o.robot_target_col],
        ['awaiting confirm', o.awaiting_robot_confirmation ? 'yes' : null],
        ['winner',         o.winner],
        ['robot ai',       aiLabel],
      ]);
    }}

    function updateMotorState(m) {{
      if (!m) return;
      const ee = m.end_effector || {{}};
      document.getElementById('motorState').innerHTML = kv([
        ['pos',       m.pos_mm   != null ? m.pos_mm.toFixed(1)   + ' mm' : null],
        ['vel',       m.vel_mm_s != null ? m.vel_mm_s.toFixed(1) + ' mm/s' : null],
        ['moving',    m.moving   ? 'yes' : null],
        ['ready',     m.ready    ? 'yes' : null],
        ['mode',      m.mode],
        ['last ack',  m.last_ack],
        ['EE state',  ee.state],
        ['last error',m.last_error],
      ]);
    }}

    function entryClass(msg) {{
      const m = msg.toLowerCase();
      if (m.includes('winner')) return 'winner';
      if (m.includes('error'))  return 'error';
      if (m.includes('human'))  return 'human';
      if (m.includes('robot'))  return 'robot';
      if (m.includes('reset') || m.includes('homed') || m.includes('paused') ||
          m.includes('resumed') || m.includes('startup')) return 'system';
      return '';
    }}

    function updateActivityLog(history) {{
      const log = document.getElementById('activityLog');
      const entries = (history || []).slice().reverse().slice(0, 40);
      log.innerHTML = entries
        .map(msg => `<div class="log-entry ${{entryClass(msg)}}">${{msg}}</div>`)
        .join('');
    }}

    function setSvc(id, ok) {{
      const el = document.getElementById(id);
      el.className = 'svc ' + (ok ? 'ok' : 'err');
    }}

    // ── Tab switching ────────────────────────────────────────────────
    let _activeTab = 'game';
    function showTab(name) {{
      _activeTab = name;
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById('tab-' + name).classList.add('active');
      document.getElementById('tab-btn-' + name).classList.add('active');
    }}

    // ── Debug tab update ─────────────────────────────────────────────
    function updateDebugTab(vsvc, visionOk) {{
      if (_activeTab !== 'debug') return;
      const imgEl    = document.getElementById('debugImg');
      const noFeedEl = document.getElementById('debugNoFeed');

      if (visionOk && vsvc.detection_image_available) {{
        imgEl.src = '/api/vision_image?t=' + Date.now();
        imgEl.style.display = '';
        noFeedEl.style.display = 'none';
      }} else {{
        imgEl.style.display = 'none';
        noFeedEl.style.display = '';
      }}

      document.getElementById('debugVisionStats').innerHTML = kv([
        ['phase',      vsvc.detection_phase],
        ['message',    vsvc.detection_message],
        ['classifier', vsvc.classifier_name],
        ['frames',     vsvc.frame_count],
        ['holes',      vsvc.detected_holes_count],
        ['paused',     vsvc.is_paused ? 'yes' : null],
        ['last error', vsvc.last_error],
      ]);
    }}

    async function fetchStatus() {{
      try {{
        const resp = await fetch('/api/status');
        const data = await resp.json();
        const o = data.orchestrator || {{}};

        setSvc('svcVision', data.vision_svc_ok);
        setSvc('svcOrch',   data.orch_ok);
        updateDetectionPanel(data.vision_svc || {{}});
        updateDebugTab(data.vision_svc || {{}}, data.vision_svc_ok);

        const board = data.vision_board || o.board_top_down || null;
        renderBoard(board, o.last_human_col, o.last_robot_col);

        if (data.orch_ok && o.status) updateBanner(o);
        if (data.orch_ok && o.ai_name) updateAIBadge(o);

        updateGameStats(o, data.last_vision_update);
        updateMotorState(o.motor_state);
        updateActivityLog(o.move_history);
        updateGantry(o.motor_state || {{}});

        const ts = data.last_vision_update;
        document.getElementById('lastUpdate').textContent =
          ts ? 'vision: ' + new Date(ts * 1000).toLocaleTimeString() : 'no vision data yet';

      }} catch (err) {{
        document.getElementById('actionBanner').textContent = 'Web service error: ' + err;
      }}
    }}

    // ── Detection panel ──────────────────────────────────────────────
    let _detReady = false;
    let _detDismissTimer = null;

    const DET_CFG = {{
      searching:        {{ icon: '<span class="spin">⟳</span>', title: 'Detecting Board',       retry: false, dismiss: false, img: false }},
      checking:         {{ icon: '<span class="spin">⟳</span>', title: 'Checking Board',        retry: false, dismiss: false, img: true  }},
      detection_failed: {{ icon: '✗',                           title: 'Board Not Detected',    retry: true,  dismiss: false, img: true  }},
      board_not_empty:  {{ icon: '⚠',                           title: 'Board Not Empty',       retry: true,  dismiss: false, img: true  }},
      ready:            {{ icon: '✓',                           title: 'Board Ready',           retry: false, dismiss: true,  img: true  }},
    }};

    function updateDetectionPanel(vsvc) {{
      if (_isHoming) return;  // don't overwrite the homing display
      const phase = vsvc && vsvc.detection_phase;
      if (!phase) return;

      const panel    = document.getElementById('detPanel');
      const cfg      = DET_CFG[phase] || DET_CFG.searching;

      // If already dismissed and still ready, leave hidden
      if (_detReady && phase === 'ready' && panel.classList.contains('hidden')) return;

      // Re-show panel whenever we're not ready
      if (phase !== 'ready') {{
        _detReady = false;
        panel.classList.remove('hidden');
        if (_detDismissTimer) {{ clearTimeout(_detDismissTimer); _detDismissTimer = null; }}
      }}

      panel.className = 'det-panel ' + phase;
      document.getElementById('detIcon').innerHTML  = cfg.icon;
      document.getElementById('detTitle').textContent = cfg.title;
      document.getElementById('detMsg').textContent   = vsvc.detection_message || '';
      document.getElementById('btnRetry').style.display   = cfg.retry   ? '' : 'none';
      document.getElementById('btnDismiss').style.display = cfg.dismiss ? '' : 'none';

      const img = document.getElementById('detImg');
      if (cfg.img && vsvc.detection_image_available) {{
        img.src = '/api/vision_image?t=' + Date.now();
        img.style.display = '';
      }} else {{
        img.style.display = 'none';
      }}

      // Auto-dismiss after 6 s when ready
      if (phase === 'ready' && !_detReady) {{
        _detReady = true;
        _detDismissTimer = setTimeout(() => panel.classList.add('hidden'), 6000);
      }}
    }}

    function retryDetection() {{
      _detReady = false;
      if (_detDismissTimer) {{ clearTimeout(_detDismissTimer); _detDismissTimer = null; }}
      const panel = document.getElementById('detPanel');
      panel.className = 'det-panel searching';
      panel.classList.remove('hidden');
      const cfg = DET_CFG.searching;
      document.getElementById('detIcon').innerHTML    = cfg.icon;
      document.getElementById('detTitle').textContent = cfg.title;
      document.getElementById('detMsg').textContent   = 'Restarting detection…';
      document.getElementById('btnRetry').style.display   = 'none';
      document.getElementById('btnDismiss').style.display = 'none';
      document.getElementById('detImg').style.display     = 'none';
      fetch('/api/retry_detection', {{method: 'POST'}}).catch(e => console.warn('retry failed:', e));
    }}

    function dismissDetection() {{
      document.getElementById('detPanel').classList.add('hidden');
    }}

    async function doAction(action) {{
      try {{
        await fetch('/api/' + action, {{method: 'POST'}});
      }} catch(e) {{ alert('Failed: ' + e); }}
    }}

    // ── Difficulty selector ──────────────────────────────────────────
    let _currentDifficulty = 'medium';

    function updateDifficultyUI(difficulty) {{
      _currentDifficulty = difficulty || 'medium';
      document.querySelectorAll('.diff-btn').forEach(btn => {{
        btn.classList.toggle('active', btn.classList.contains(_currentDifficulty));
      }});
    }}

    async function setDifficulty(level) {{
      updateDifficultyUI(level);  // instant feedback
      try {{
        const resp = await fetch('/api/ai', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{name: 'minimax', difficulty: level}}),
        }});
        if (!resp.ok) console.warn('setDifficulty failed:', await resp.text());
      }} catch(e) {{ console.warn('setDifficulty error:', e); }}
    }}

    function updateAIBadge(o) {{
      const name = o.ai_name || '';
      const diff = o.ai_difficulty || '';
      const label = diff ? `${{name}} / ${{diff}}` : name;
      document.getElementById('aiBadge').textContent = label;
      if (diff && diff !== _currentDifficulty) updateDifficultyUI(diff);
    }}

    // ── Begin-game modal ────────────────────────────────────────────
    let _modalIsReset = false;
    let _modalDiff = 'medium';

    function showBeginModal(isReset) {{
      _modalIsReset = isReset;
      document.getElementById('modalSub').textContent = isReset
        ? 'Select difficulty — robot will home and board will reset'
        : 'Choose a difficulty to begin';
      document.getElementById('beginModal').classList.remove('hidden');
      selectModalDiff(_currentDifficulty || _modalDiff);
    }}

    function selectModalDiff(diff) {{
      _modalDiff = diff;
      document.querySelectorAll('.modal-diff-card').forEach(card => {{
        card.classList.toggle('active', card.dataset.diff === diff);
      }});
    }}

    let _isHoming = false;

    function _showDetectionPanel(phase, title, msg) {{
      _detReady = false;
      if (_detDismissTimer) {{ clearTimeout(_detDismissTimer); _detDismissTimer = null; }}
      const panel = document.getElementById('detPanel');
      panel.className = 'det-panel ' + phase;
      panel.classList.remove('hidden');
      document.getElementById('detIcon').innerHTML    = DET_CFG[phase]?.icon ?? '<span class="spin">⟳</span>';
      document.getElementById('detTitle').textContent = title;
      document.getElementById('detMsg').textContent   = msg;
      document.getElementById('btnRetry').style.display   = 'none';
      document.getElementById('btnDismiss').style.display = 'none';
      document.getElementById('detImg').style.display     = 'none';
    }}

    async function startGame() {{
      document.getElementById('beginModal').classList.add('hidden');

      // Set AI difficulty (fire-and-forget — fast)
      fetch('/api/ai', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{name: 'minimax', difficulty: _modalDiff}}),
      }}).catch(e => console.warn('setAI failed:', e));
      updateDifficultyUI(_modalDiff);

      // Show homing panel and block vision updates until homing is done
      _isHoming = true;
      _showDetectionPanel('searching', 'Homing Robot', 'Robot is moving to home position…');

      try {{
        await fetch('/api/reset', {{method: 'POST'}});
      }} catch(e) {{ console.warn('reset failed:', e); }}

      // Homing done — hand off to vision detection
      _isHoming = false;
      _showDetectionPanel('searching', 'Detecting Board', 'Searching for board corners…');
    }}

    // Show modal on first load
    showBeginModal(false);

    async function poll() {{
      await fetchStatus();
      setTimeout(poll, 500);
    }}
    poll();
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_HTML)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVICES.web_host, port=SERVICES.web_port, reload=False)
