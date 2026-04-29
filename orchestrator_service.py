from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from connect_4.config import SERVICES
from connect_4.game_engine import (
    build_orchestrator,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=SERVICES.orchestrator_host,
        port=SERVICES.orchestrator_port,
        reload=False,
    )