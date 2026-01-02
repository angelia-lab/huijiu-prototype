# src/huijiu/safety/rules.py
from __future__ import annotations

from .state import SystemState, SafetyDecision
from huijiu.core import config as cfg


def evaluate_safety(state: SystemState) -> SafetyDecision:
    """
    非常简单的一版安全规则：
    - 温度超过 TEMP_MAX_C => 不 OK，要求后退；
    - 距离小于 DIST_MIN_MM => 不 OK，要求后退；
    - 其它情况暂时都认为 OK。
    """
    t = state.sensors.temp_c
    d = state.sensors.distance_mm

    # 温度过高
    if t is not None and t > cfg.TEMP_MAX_C:
        return SafetyDecision(
            ok=False,
            reason=f"温度过高: {t:.1f}C > {cfg.TEMP_MAX_C}C",
            need_emergency_retreat=True,
        )

    # 距离过近
    if d is not None and d < cfg.DIST_MIN_MM:
        return SafetyDecision(
            ok=False,
            reason=f"距离过近: {d:.1f}mm < {cfg.DIST_MIN_MM}mm",
            need_emergency_retreat=True,
        )

    # 其它暂时 OK
    return SafetyDecision(
        ok=True,
        reason="OK",
        need_emergency_retreat=False,
    )
