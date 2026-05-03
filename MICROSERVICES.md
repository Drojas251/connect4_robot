# Connect4 Robot - Microservices Architecture

This is a three-service architecture for the Connect4 robot system:

1. **Vision Service** - Detects board state from camera feed
2. **Orchestrator Service** - Controls robot actions and game logic
3. **Web UI Service** - Displays game state and provides user controls

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│           Web UI Service (Port 8003)        │
│  - Shows game board                         │
│  - Reset/Pause buttons                      │
│  - Real-time status display                 │
└────────────────────┬────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼───────────┐  ┌────────▼──────────┐
│ Vision Service    │  │ Orchestrator      │
│ (Port 8001)       │  │ (Port 8002)       │
│                   │  │                   │
│ - Camera feed     │  │ - Game logic      │
│ - Board detection │  │ - Robot control   │
│ - State updates   │  │ - Decision making │
└───────┬───────────┘  └────────┬──────────┘
        │                       │
        └───────────┬───────────┘
                    │
            ┌───────▼────────┐
            │  Robot Motors  │
            │  (Arduino)     │
            └────────────────┘
```

## Services

### Vision Service (`vision/service.py`)

Runs real-time camera-based board state detection:

- Uses ArUco tag grid detection to find board location
- Classifies pieces (empty, human, robot) using machine learning
- Publishes board updates when state changes
- Responds to pause/resume commands
- Provides `/status` endpoint with frame count and detection info

**Endpoints:**
- `GET /health` - Health check
- `GET /status` - Current vision state
- `POST /pause` - Pause vision polling
- `POST /resume` - Resume vision polling
- `POST /reset` - Reset vision state

### Orchestrator Service (`orchestrator_service.py`)

Controls game logic and robot actions:

- Receives board state updates from vision service
- Executes AI decision-making
- Commands robot motor to move to columns
- Confirms actions align with vision feedback
- Maintains game state and history

**Endpoints:**
- `GET /health` - Health check
- `GET /status` - Game state and robot motor status
- `POST /reset` - Reset game and home robot
- `POST /pause` - Pause game
- `POST /resume` - Resume game
- `POST /vision/update` - Receive board state from vision service

### Web UI Service (`web_service.py`)

Provides a real-time dashboard:

- Displays current game board (7x6 Connect4)
- Shows vision service status (frames processed, paused state)
- Shows orchestrator status (game state, decision)
- Shows robot status (moving/idle)
- **Reset button** - Homes robot and clears board
- **Pause button** - Pauses vision and orchestrator
- Polls all services every 500ms for updates

**Endpoints:**
- `GET /health` - Health check
- `GET /api/status` - Aggregated system status
- `POST /api/board_update` - Receives board updates from vision
- `POST /api/reset` - Reset game
- `POST /api/pause` - Pause system
- `POST /api/resume` - Resume system
- `GET /` - Serve HTML UI

## Setup and Configuration

### 1. Update `connect4_robot/config.py`

Ensure the configuration has the correct service URLs and ports:

```python
from dataclasses import dataclass


@dataclass
class ServiceConfig:
    orchestrator_url: str = "http://localhost:8002"
    orchestrator_host: str = "localhost"
    orchestrator_port: int = 8002
    
    vision_url: str = "http://localhost:8001"
    vision_host: str = "localhost"
    vision_port: int = 8001
    
    web_host: str = "localhost"
    web_port: int = 8003


@dataclass
class BoardConfig:
    rows: int = 6
    cols: int = 7


@dataclass
class VisionConfig:
    camera_index: int = 1
    refresh_seconds: float = 5
    poll_period_s: float = 0.1


SERVICES = ServiceConfig()
BOARD = BoardConfig()
VISION = VisionConfig()
```

### 2. Start the Services

In separate terminals:

```bash
# Terminal 1: Vision Service
python -m connect4_robot.vision.service

# Terminal 2: Orchestrator Service
python -m connect4_robot.orchestrator_service

# Terminal 3: Web UI Service
python -m connect4_robot.web_service
```

### 3. Access the Web UI

Open your browser to:
```
http://localhost:8003
```

## Data Flow

1. **Vision Detection Loop:**
   - Vision service polls camera at ~100ms intervals
   - Detects board state using computer vision
   - Publishes updates to orchestrator and web UI every 5 seconds (or when board changes)

2. **Orchestrator Decision:**
   - Receives board update from vision
   - Runs game AI to decide next move
   - Sends motor commands to robot
   - Waits for vision to confirm piece placement

3. **Web UI Updates:**
   - Polls orchestrator and vision every 500ms
   - Displays aggregated status
   - Routes control commands (reset/pause) to both services

## Message Protocol

### Board Update (Vision → Orchestrator + Web)

```json
POST /vision/update
{
  "board": [
    [".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", "."],
    [".", ".", "H", ".", ".", ".", "."],
    [".", "R", "H", "R", ".", ".", "."],
    ["H", "R", "H", "R", ".", ".", "."],
    ["R", "H", "R", "H", "R", ".", "."]
  ],
  "source": "vision_service"
}
```

Board representation:
- `.` = Empty cell
- `H` = Human piece (red)
- `R` = Robot piece (yellow)

## Troubleshooting

### Vision service fails to find grid
- Check camera connection
- Ensure good lighting on the board
- Verify ArUco tag grid is visible
- Check `CAMERA_INDEX` in config (usually 0 or 1)

### Services can't communicate
- Verify all three services started successfully
- Check that URLs in config match running services
- Check firewall/network settings

### Web UI shows "checking..." indefinitely
- Verify all three services are running and healthy
- Check `/health` endpoint on each service
- Look at service console output for errors

## Service States

**Vision Service:**
- Running: Polling camera
- Paused: Not polling, but can resume
- Error: Failed to initialize or access camera

**Orchestrator Service:**
- IDLE: Waiting for board update
- ERROR: Invalid move or internal error

**Robot Motor:**
- MOVING: Executing command
- IDLE: Waiting for next command

## Development Notes

### Adding New Features

1. **New AI Strategy:** Edit `game_engine/orchestrator.py`
2. **Vision Tuning:** Adjust HSV ranges in `vision/piece_color_classifier.py`
3. **UI Updates:** Modify `web_service.py` HTML template
4. **Motor Calibration:** Update motor control parameters

### Testing Individual Services

```bash
# Test vision service health
curl http://localhost:8001/health

# Test orchestrator health
curl http://localhost:8002/health

# Test web service health
curl http://localhost:8003/health

# Get full system status
curl http://localhost:8003/api/status
```

### Logs

Each service prints logs to console with `[service_name]` prefix:
- `[vision]` - Vision service
- `[orchestrator]` - Orchestrator service  
- `[web]` - Web service

Monitor these for debugging.
