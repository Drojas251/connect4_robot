# Connect4 Robot - Microservices Migration Summary

## What Changed

You've moved from a simulated vision system to a real three-service microservices architecture with live camera-based board detection.

### Old System (Sim-Based)
```
Web UI ←→ Orchestrator ←→ Simulated Vision
  │                           │
  └──────────────────────────┘
```
- Single vision service simulated board state
- Limited to pre-defined scenarios
- UI included human move buttons

### New System (Real Vision + Microservices)
```
┌──────────────────────┐
│  Web UI (Port 8003)  │ ← Shows board & controls
└──────────────────────┘
         │
    ┌────┴─────┐
    ▼          ▼
┌────────────────┐  ┌─────────────────┐
│ Vision Service │  │  Orchestrator   │
│  (Port 8001)   │  │  (Port 8002)    │
│ Real Camera    │  │  Game Logic     │
└────────────────┘  └─────────────────┘
         │                  │
         └──────────┬───────┘
                    │
            ┌───────▼────────┐
            │  Robot Motors  │
            │   (Arduino)    │
            └────────────────┘
```

## Files Created/Modified

### New Files
1. **`connect4_robot/vision/service.py`**
   - Real vision service using camera feed
   - Detects board state with computer vision
   - Publishes updates via HTTP
   - Supports pause/resume

2. **`connect4_robot/web_service.py`**
   - Beautiful web UI dashboard
   - Real-time status display
   - Reset and Pause buttons
   - No human move buttons (physical game now)

3. **`start_services.sh`**
   - Bash script to start all 3 services in tmux
   - Easy terminal management

4. **`start_services.py`**
   - Python script to start all 3 services
   - Simpler than tmux, works cross-platform

5. **`MICROSERVICES.md`**
   - Complete architecture documentation
   - Setup instructions
   - Troubleshooting guide
   - API reference

### Modified Files
1. **`connect4_robot/orchestrator_service.py`**
   - Added `/pause` endpoint
   - Added `/resume` endpoint
   - Can now be paused by web UI

2. **`sim_service.py`** (deprecated)
   - No longer needed
   - You can delete it

## Key Differences

| Aspect | Old (Sim) | New (Real) |
|--------|-----------|-----------|
| **Vision** | Simulated board moves | Real camera detection |
| **Board Updates** | Manual buttons | Automatic from camera |
| **Human Moves** | UI buttons | Physical game board |
| **Control** | Single app | 3 microservices |
| **Scalability** | Limited | Easily extensible |

## Services Overview

### Vision Service (`vision/service.py`)
**What it does:**
- Continuously polls camera
- Detects ArUco tag grid
- Classifies pieces (empty/human/robot)
- Publishes board updates every 5 seconds (or on change)

**Endpoints:**
```
GET  /health          - Health check
GET  /status          - Current state (frame count, paused status)
POST /pause           - Pause vision polling
POST /resume          - Resume vision polling
POST /reset           - Clear board state
```

### Orchestrator Service (`orchestrator_service.py`)
**What it does:**
- Receives board updates from vision
- Runs game AI to decide next move
- Commands robot motor
- Validates moves with vision feedback

**Endpoints:**
```
GET  /health          - Health check
GET  /status          - Game state + motor status
POST /reset           - Reset game + home robot
POST /pause           - Pause orchestrator
POST /resume          - Resume orchestrator
POST /vision/update   - Receive board updates from vision
```

### Web UI Service (`web_service.py`)
**What it does:**
- Polls all services every 500ms
- Aggregates status information
- Displays board state
- Routes reset/pause commands

**Endpoints:**
```
GET  /                - Main UI (HTML page)
GET  /health          - Health check
GET  /api/status      - Aggregated system status
POST /api/reset       - Reset all services
POST /api/pause       - Pause all services
POST /api/resume      - Resume all services
POST /api/board_update - Receive updates from vision
```

**Web UI at:** http://localhost:8003

## Quick Start

### Option 1: Using Python (Easiest)
```bash
cd /home/aft/dsr-motion/api/python
python start_services.py
```

### Option 2: Using Bash/Tmux
```bash
cd /home/aft/dsr-motion/api/python
./start_services.sh
```

### Option 3: Manual (3 terminals)
```bash
# Terminal 1
python -m connect4_robot.vision.service

# Terminal 2
python -m connect4_robot.orchestrator_service

# Terminal 3
python -m connect4_robot.web_service
```

Then open browser to: **http://localhost:8003**

## Configuration

All settings are in `connect4_robot/config.py`:

```python
@dataclass
class ServiceConfig:
    orchestrator_url = "http://localhost:8002"
    vision_url = "http://localhost:8001"
    web_host = "localhost"
    web_port = 8003

@dataclass
class VisionConfig:
    camera_index = 1  # Change if using different camera
    refresh_seconds = 5  # Board update frequency
    poll_period_s = 0.1  # Vision polling rate
```

## Testing Services

```bash
# Test vision
curl http://localhost:8001/health
curl http://localhost:8001/status

# Test orchestrator
curl http://localhost:8002/health
curl http://localhost:8002/status

# Test web UI
curl http://localhost:8003/health
```

## Troubleshooting

### Vision can't find grid
```bash
# Check camera is accessible
ls -la /dev/video*

# Verify correct camera_index in config
# Try index 0, 1, 2, etc.
```

### Services won't start
```bash
# Check ports aren't in use
lsof -i :8001
lsof -i :8002
lsof -i :8003

# Kill process using port if needed
kill -9 <PID>
```

### Web UI shows "checking..."
- Verify all 3 services are running
- Check service health endpoints
- Look at console output for errors

## Migration from Sim System

1. ✅ Delete or archive `sim_service.py`
2. ✅ New vision polling code is in `vision/service.py`
3. ✅ Orchestrator mostly unchanged, just added pause/resume
4. ✅ Web UI completely redesigned - no human buttons
5. ✅ Configuration centralized in `config.py`

## Next Steps

1. **Test the setup** - Start all services and verify board detection
2. **Tune vision** - Adjust HSV ranges for your lighting/pieces
3. **Test game flow** - Make manual moves and verify robot responds
4. **Monitor logs** - Watch console output for any issues
5. **Add features** - Extend AI, add metrics, etc.

## Architecture Benefits

✅ **Modularity** - Change vision without touching orchestrator  
✅ **Scalability** - Add more services easily  
✅ **Testability** - Test each service independently  
✅ **Robustness** - Failure in one service doesn't crash all  
✅ **Real Vision** - Uses actual camera, not simulation  
✅ **HTTP APIs** - Easy to integrate, monitor, extend  

## Support

If you have issues:

1. Check `MICROSERVICES.md` for full documentation
2. Look at console logs with `[service_name]` prefix
3. Use curl to test individual endpoints
4. Check connectivity between services
5. Verify camera/Arduino connections

Good luck! 🎮🤖
