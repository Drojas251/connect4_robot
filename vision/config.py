from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MotorConfig:
    port: str = "/dev/ttyACM1"
    baudrate: int = 115200
    timeout_s: float = 0.1
    deg_per_mm: float = 5.54
    deg_offset: float = 0.0


@dataclass
class BoardConfig:
    rows: int = 6
    cols: int = 7
    column_centers_mm: List[float] = field(
        default_factory=lambda: [20.0, 58.0, 96.0, 134.0, 172.0, 210.0, 248.0]
    )
    home_mm: float = 0.0
    staging_mm: float = 10.0


@dataclass
class ServiceConfig:
    orchestrator_host: str = "0.0.0.0"
    orchestrator_port: int = 8000
    vision_host: str = "0.0.0.0"
    vision_port: int = 8001
    orchestrator_url: str = "http://127.0.0.1:8000"


@dataclass
class VisionSimConfig:
    poll_period_s: float = 0.4
    motion_wait_timeout_s: float = 30.0


MOTOR = MotorConfig()
BOARD = BoardConfig()
SERVICES = ServiceConfig()
VISION_SIM = VisionSimConfig()
