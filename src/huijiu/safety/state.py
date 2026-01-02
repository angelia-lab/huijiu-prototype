# src/huijiu/safety/state.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class VisionState:
    has_hand: bool
    hx: Optional[int]
    hy: Optional[int]


@dataclass
class SensorState:
    temp_c: Optional[float]
    distance_mm: Optional[float]


@dataclass
class SystemState:
    vision: VisionState
    sensors: SensorState


@dataclass
class SafetyDecision:
    ok: bool
    reason: str
    need_emergency_retreat: bool = False
