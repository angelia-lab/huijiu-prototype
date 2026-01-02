# src/huijiu/sensors/manager.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .backends.base import I2CBackend
from .mlx90614 import MLX90614
from .vl53l0x import VL53L0X


@dataclass
class SensorReadings:
    timestamp: float
    temp_c: Optional[float]
    distance_mm: Optional[float]


class SensorManager:
    """
    统一管理 MLX90614（温度）和 VL53L0X（ToF 距离）。
    - 负责调用底层驱动、做异常捕获；
    - 暴露一个 read_all() 方法，供 orchestrator 调用。
    """

    def __init__(
        self,
        backend: I2CBackend,
        mlx_addr: int = 0x5A,
        vl53_addr: int = 0x53,
    ) -> None:
        self._backend = backend
        self._mlx = MLX90614(backend, address=mlx_addr)
        self._vl53 = VL53L0X(backend, address=vl53_addr)

        # 初始化 VL53L0X（上电只需 init 一次）
        try:
            self._vl53.init()
        except Exception as e:
            print(f"[WARN] VL53L0X init 失败: {e}")

    def read_all(self) -> SensorReadings:
        now = time.time()
        temp_c: Optional[float] = None
        dist_mm: Optional[float] = None

        # 读温度
        try:
            temp_c = float(self._mlx.read_object_c())
        except Exception as e:
            print(f"[WARN] 读取温度失败: {e}")

         # 读距离：用 measure_once()，只接受“可靠帧”
        try:
            sample = self._vl53.measure_once(delay_ms=50)  # 注意用你 vl53l0x.py 里已有的方法
            # 只有 reliable 才认，否则当作 None
            if sample is not None and getattr(sample, "reliable", False):
                dist = float(sample.distance_mm)
                if 0 < dist < 60000:
                    dist_mm = dist
                else:
                    dist_mm = None
            else:
                dist_mm = None
        except Exception as e:
            print(f"[WARN] 读取距离失败: {e}")
            dist_mm = None

        return SensorReadings(
            timestamp=now,
            temp_c=temp_c,
            distance_mm=dist_mm,
        )

    def close(self) -> None:
        try:
            self._backend.close()
        except Exception:
            pass
