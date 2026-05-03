from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from connect4_robot.config import SERVICES
from connect4_robot.game_engine import (
    build_orchestrator,
    build_ai,
    list_ais,
    ControllerStatus,
    VisionBoardUpdate,
    StatusResponse,
    RobotMoveResponse,
)

orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    orchestrator = build_orchestrator()
    try:
        yield
    finally:
        if orchestrator is not None:
            orchestrator.shutdown()


app = FastAPI(
    title="Connect4 Orchestrator Service",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True, "service": "connect4-orchestrator"}


@app.get("/status", response_model=StatusResponse)
def get_status():
    return orchestrator.get_status()


@app.post("/reset")
def reset():
    orchestrator.reset()
    return {"ok": True}


@app.post("/pause")
def pause():
    with orchestrator._lock:
        orchestrator.state.status = ControllerStatus.PAUSED
        orchestrator._append_history("orchestrator paused")
    return {"ok": True, "status": "paused"}


@app.post("/resume")
def resume():
    with orchestrator._lock:
        restored = (
            ControllerStatus.HUMAN_TURN
            if orchestrator.state.board_version > 0
            else ControllerStatus.WAITING_FOR_BOARD
        )
        orchestrator.state.status = restored
        orchestrator._append_history("orchestrator resumed")
    return {"ok": True, "status": restored.value}


@app.post("/vision/update", response_model=RobotMoveResponse)
def vision_update(payload: VisionBoardUpdate):
    try:
        return orchestrator.handle_vision_board_update(payload)
    except ValueError as e:
        with orchestrator._lock:
            orchestrator.state.status = ControllerStatus.ERROR
            orchestrator.state.last_error = str(e)
            orchestrator._append_history(f"error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        with orchestrator._lock:
            orchestrator.state.status = ControllerStatus.ERROR
            orchestrator.state.last_error = str(e)
            orchestrator._append_history(f"error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# AI management
# ---------------------------------------------------------------------------

class AISelectRequest(BaseModel):
    name: str
    difficulty: Optional[str] = None   # only meaningful for minimax


@app.get("/ai")
def get_ai():
    """Return the current AI config and the list of all registered AIs."""
    status = orchestrator.get_status()
    return {
        "current": {
            "name": status.ai_name,
            "difficulty": status.ai_difficulty or None,
        },
        "available": list_ais(),
        "difficulties": ["easy", "medium", "hard"],
    }


@app.post("/ai")
def set_ai(req: AISelectRequest):
    """Hot-swap the robot AI (and optionally its difficulty) without restarting."""
    params = {}
    if req.difficulty:
        params["difficulty"] = req.difficulty

    try:
        ai = build_ai(req.name, **params)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    orchestrator.set_policy(ai)
    return {
        "ok": True,
        "ai": req.name,
        "difficulty": getattr(ai, "difficulty", None),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=SERVICES.orchestrator_host,
        port=SERVICES.orchestrator_port,
        reload=False,
    )
