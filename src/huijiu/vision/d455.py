"""
RealSense D455 封装

RealSenseD455 类，用于：
- 启动 / 停止管线
- 获取对齐到彩色图像坐标系的彩色 + 深度帧
- 返回的深度单位为米（float32）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少 pyrealsense2 依赖，请先安装 RealSense SDK 和 Python 包：\n"
        "  pip install pyrealsense2\n"
    ) from exc


@dataclass
class D455Frame:
    """一帧对齐后的 D455 数据."""
    color_bgr: Optional[np.ndarray]  # (H, W, 3), BGR 格式
    depth_m: Optional[np.ndarray]    # (H, W), float32，单位：米
    timestamp_ms: float              # 时间戳（毫秒）


class RealSenseD455:
    """Intel RealSense D455 的最小封装."""

    def __init__(
        self,
        color_resolution: Tuple[int, int] = (1280, 720),
        depth_resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        align_to_color: bool = True,
    ) -> None:
        """
        :param color_resolution: 彩色图分辨率 (width, height)
        :param depth_resolution: 深度图分辨率 (width, height)
        :param fps: 帧率
        :param align_to_color: 是否将深度对齐到彩色图坐标系
        """
        self._color_resolution = color_resolution
        self._depth_resolution = depth_resolution
        self._fps = fps
        self._align_to_color = align_to_color

        self._pipeline: Optional[rs.pipeline] = None
        self._config: Optional[rs.config] = None
        self._align: Optional[rs.align] = None
        self._depth_scale: Optional[float] = None
        self._started: bool = False

    # ------------------------------------------------------------------ #
    # 生命周期管理
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """启动 RealSense 管线。"""
        if self._started:
            return

        pipeline = rs.pipeline()
        config = rs.config()

        w_c, h_c = self._color_resolution
        w_d, h_d = self._depth_resolution

        # 启用彩色流 & 深度流
        config.enable_stream(rs.stream.color, w_c, h_c, rs.format.bgr8, self._fps)
        config.enable_stream(rs.stream.depth, w_d, h_d, rs.format.z16, self._fps)

        # 启动设备
        profile = pipeline.start(config)

        # 获取深度刻度（depth_scale）
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()  # 通常是 0.001（mm → m）

        self._pipeline = pipeline
        self._config = config
        self._depth_scale = float(depth_scale)
        self._started = True

        # 对齐器：将所有流对齐到彩色流
        if self._align_to_color:
            self._align = rs.align(rs.stream.color)
        else:
            self._align = None

    def stop(self) -> None:
        """停止管线，释放设备。"""
        if not self._started:
            return
        assert self._pipeline is not None
        self._pipeline.stop()
        self._pipeline = None
        self._config = None
        self._align = None
        self._started = False

    def __enter__(self) -> "RealSenseD455":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # 数据获取
    # ------------------------------------------------------------------ #

    def get_frame(self, timeout_ms: int = 1000) -> D455Frame:
        """
        阻塞等待一帧数据并返回。

        :param timeout_ms: 超时时间（毫秒）
        :return: D455Frame，包含 BGR 彩色图和深度（米）
        """
        if not self._started or self._pipeline is None:
            raise RuntimeError("RealSenseD455 未启动，请先调用 start()。")
        if self._depth_scale is None:
            raise RuntimeError("深度刻度未初始化。")

        # 等待帧
        frames: rs.composite_frame = self._pipeline.wait_for_frames(timeout_ms)

        # 可选：对齐到彩色流
        if self._align is not None:
            frames = self._align.process(frames)

        color_frame: Optional[rs.video_frame] = frames.get_color_frame()
        depth_frame: Optional[rs.depth_frame] = frames.get_depth_frame()

        color_bgr: Optional[np.ndarray]
        depth_m: Optional[np.ndarray]

        # 彩色
        if color_frame:
            color_bgr = np.asanyarray(color_frame.get_data())
        else:
            color_bgr = None

        # 深度：uint16 → float32 米
        if depth_frame:
            depth_raw = np.asanyarray(depth_frame.get_data())  # uint16
            depth_m = depth_raw.astype(np.float32) * self._depth_scale
        else:
            depth_m = None

        # 使用第一个可用帧的时间戳（毫秒）
        ts_ms: float
        if depth_frame:
            ts_ms = float(depth_frame.get_timestamp())
        elif color_frame:
            ts_ms = float(color_frame.get_timestamp())
        else:
            ts_ms = 0.0

        return D455Frame(
            color_bgr=color_bgr,
            depth_m=depth_m,
            timestamp_ms=ts_ms,
        )
