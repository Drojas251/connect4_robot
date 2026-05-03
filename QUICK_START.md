# Quick Start Guide - Connect4 Robot

## System Overview

You now have a complete microservices setup:
- **Vision Service** (Port 8001) - Detects board state from camera
- **Orchestrator Service** (Port 8002) - Controls game logic & robot
- **Web UI Service** (Port 8003) - Shows game board & status

## Starting the System

### Option 1: Python (Recommended - Cross-Platform)
```bash
cd /home/aft/dsr-motion/api/python
python start_services.py
```

### Option 2: Bash with Tmux (Linux/Mac)
```bash
cd /home/aft/dsr-motion/api/python
./start_services.sh
```

### Option 3: Manual (3 Separate Terminals)
```bash
# Terminal 1 - Vision
cd /home/aft/dsr-motion/api/python
python -m connect4_robot.vision.service

# Terminal 2 - Orchestrator
cd /home/aft/dsr-motion/api/python
python -m connect4_robot.orchestrator_service

# Terminal 3 - Web UI
cd /home/aft/dsr-motion/api/python
python -m connect4_robot.web_service
```

## Accessing the UI

Open your web browser and go to:
```
http://localhost:8003
```

You should see:
- Connect4 game board (7x6 grid)
- Red (●) dots = Human pieces
- Yellow (●) dots = Robot pieces
- Reset Game button (homes robot & clears board)
- Pause button (pauses all services)
- Status indicators (Vision, Orchestrator, Robot)

## Testing Individual Services

### Health Check
```bash
curl http://localhost:8001/health  # Vision
curl http://localhost:8002/health  # Orchestrator
curl http://localhost:8003/health  # Web UI
```

### Get Status
```bash
curl http://localhost:8001/status  # Vision state
curl http://localhost:8002/status  # Game state
curl http://localhost:8003/api/status  # Full system
```

### Manual Commands
```bash
# Reset game
curl -X POST http://localhost:8003/api/reset

# Pause game
curl -X POST http://localhost:8003/api/pause

# Resume game
curl -X POST http://localhost:8003/api/resume
```

## Troubleshooting

### Web UI shows "loading..." or "checking..."

**Check 1: Are all services running?**
```bash
ps aux | grep "connect4_robot"
```

**Check 2: Can you reach each service?**
```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

If you see connection errors, check if ports are in use:
```bash
# Check if ports are already in use
lsof -i :8001  # Vision
lsof -i :8002  # Orchestrator
lsof -i :8003  # Web UI
```

If ports are in use, kill the old processes:
```bash
pkill -f "connect4_robot"
# Or manually kill by PID
kill -9 <PID>
```

### Vision Service won't start

**Problem: Camera not found**
```
[vision] Failed to open camera
```

**Solution:**
1. Check camera is connected
2. List available cameras:
```bash
ls -la /dev/video*
```

3. Update `CAMERA_INDEX` in `connect4_robot/config.py`
   - Usually 0 or 1
   - Try different numbers if unsure

4. Test with OpenCV:
```python
import cv2
cap = cv2.VideoCapture(0)  # or 1, 2, etc.
if cap.isOpened():
    print("Camera works!")
    cap.release()
```

**Problem: Grid not found**
```
[vision] Failed to find grid in camera feed
```

**Solution:**
1. Check board is visible in camera view
2. Verify ArUco tags are printed and visible
3. Ensure good lighting
4. Try more search frames (increase MAX_GRID_SEARCH_FRAMES)
5. Check tag quality - print ArUco markers again if faded

### Robot doesn't move

**Check 1: Is Arduino connected?**
```bash
ls -la /dev/ttyACM*  # or /dev/ttyUSB*
```

**Check 2: Check orchestrator logs**
Look for errors in orchestrator console output

**Check 3: Test motor manually**
The motor control is in `connect4_robot/motor_control/`
Review the motor stack code for connection issues

### Services lag or crash

**Memory leak?**
- Vision service should stabilize around 50-200MB
- Orchestrator around 30-50MB
- Web UI around 20-30MB

**Solution:** Restart services
```bash
pkill -f "connect4_robot"
python start_services.py
```

## Performance Tips

### Web UI loads slowly
- Polling happens every 500ms
- If still slow, reduce poll frequency in JavaScript or server
- Ensure localhost connection is working

### Vision detection is slow
- Default REFRESH_SECONDS = 5 (publishes updates every 5 seconds)
- Reduce for faster updates but uses more CPU
- Edit in `connect4_robot/config.py`

### Robot moves are jerky
- This is a motor driver issue, not software
- Check Arduino firmware timing
- Verify motor calibration parameters

## Configuration

All settings in: `connect4_robot/config.py`

```python
# Service URLs and ports
SERVICES.orchestrator_url = "http://localhost:8002"
SERVICES.vision_url = "http://localhost:8001"
SERVICES.web_port = 8003

# Vision settings
BOARD.rows = 6
BOARD.cols = 7

# Camera settings
VISION.camera_index = 1  # Which camera to use
VISION.refresh_seconds = 5  # How often to publish board updates
VISION.poll_period_s = 0.1  # How often to read camera frames
```

## Logs and Debugging

Each service prints logs with a prefix:
- `[vision]` - Vision service messages
- `[web]` - Web service messages
- Orchestrator prints directly (no prefix)

To capture logs:
```bash
# Run service and save to file
python -m connect4_robot.vision.service > vision.log 2>&1 &

# View logs in real-time
tail -f vision.log
```

## Architecture Files

- `MICROSERVICES.md` - Full architecture documentation
- `MIGRATION_GUIDE.md` - What changed from the old sim system
- `connect4_robot/vision/service.py` - Vision service code
- `connect4_robot/orchestrator_service.py` - Orchestrator service code
- `connect4_robot/web_service.py` - Web UI service code
- `start_services.py` - Simple startup script (Python)
- `start_services.sh` - Advanced startup script (Bash/Tmux)

## Next Steps

1. **Get it running** - Start services and verify board detection
2. **Test manually** - Place pieces and watch robot respond
3. **Monitor performance** - Check logs and status
4. **Tune settings** - Adjust vision/robot parameters as needed
5. **Extend functionality** - Add new AI strategies or features

## Still Having Issues?

1. Check `MICROSERVICES.md` for comprehensive guide
2. Review service console output for error messages
3. Test individual endpoints with curl
4. Check network connectivity between services
5. Verify Arduino and camera are properly connected

Good luck! 🎮🤖
