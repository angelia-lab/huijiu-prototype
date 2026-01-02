# src/huijiu/robot/mycobot280_m5.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from pymycobot.mycobot280 import MyCobot280

from huijiu.core import config as cfg


@dataclass
class RobotControlInputs:
    hx: Optional[int]        # 虎口像素 X（None 表示本帧没检测到手）
    hy: Optional[int]        # 虎口像素 Y
    temp_c: Optional[float]  # MLX 物体温度
    dist_mm: Optional[float] # ToF 距离
    dt: float                # 本帧时间间隔(s)


class MyCobot280Controller:
    """
    将“视觉跟随 + 温度闭环 + 距离闭环”统一在一个控制器里。
    - J1/J2：左右 / 上下小范围跟随虎口像素坐标；
    - J2/J3（这里先用 J2 简化）：根据温度 & 距离误差做前后微调；
    """

    def __init__(self, port: str, baud: int) -> None:
        self._forward_smooth = 0.0
        self._mc = MyCobot280(port, baud)
        time.sleep(2.0)

        try:
            self._mc.power_on()
        except Exception as e:
            print(f"[WARN] power_on 失败，可忽略: {e}")

        # 基准姿态 = 示教虎口姿态，但先夹紧到合法角度范围
        raw_base = cfg.HUKOU_REF_ANGLES
        self.base_angles: List[float] = [self._clamp_angle(a) for a in raw_base]
        self.current_angles: List[float] = list(self.base_angles)

        # 记录一下基准的 J1/J2，用来限制偏移
        self.j1_base = self.base_angles[0]
        self.j2_base = self.base_angles[1]

        print(f"[INFO] 回到虎口参考姿态: {self.base_angles}")
        self._mc.send_angles(self.base_angles, cfg.ROBOT_MOVE_SPEED)
        time.sleep(3.0)
        
        print(f"[INFO] 回到虎口参考姿态: {self.base_angles}")
        self._mc.send_angles(self.base_angles, cfg.ROBOT_MOVE_SPEED)
        time.sleep(3.0)
         # 记录上次下发命令的时间 & 目标，用于限频和避免微小抖动
        self._last_cmd_time: float = time.time()
        self._last_target: List[float] = list(self.base_angles)
        

    # ---------- 工具方法 ---------- #

    @staticmethod
    def _clamp_angle(a: float) -> float:
        return float(max(cfg.ANGLE_MIN, min(cfg.ANGLE_MAX, a)))

    def _clamp_j1j2_relative_to_base(self, j1: float, j2: float) -> tuple[float, float]:
        j1_min = self.j1_base - cfg.J1_OFFSET_LIMIT_DEG
        j1_max = self.j1_base + cfg.J1_OFFSET_LIMIT_DEG
        j2_min = self.j2_base - cfg.J2_OFFSET_LIMIT_DEG
        j2_max = self.j2_base + cfg.J2_OFFSET_LIMIT_DEG

        j1 = max(j1_min, min(j1_max, j1))
        j2 = max(j2_min, min(j2_max, j2))

        return self._clamp_angle(j1), self._clamp_angle(j2)

    # ---------- 高层 API ---------- #

    def decay_to_base(self, factor: float = 0.1) -> None:
        """没有手 / 传感器异常时，让 J1/J2 缓慢回基准姿态。"""
        j1_cur, j2_cur, j3_cur, j4_cur, j5_cur, j6_cur = self.current_angles

        j1_new = self.j1_base + (j1_cur - self.j1_base) * (1.0 - factor)
        j2_new = self.j2_base + (j2_cur - self.j2_base) * (1.0 - factor)
        j1_new, j2_new = self._clamp_j1j2_relative_to_base(j1_new, j2_new)

        # 其它关节保持基准
        j3_new, j4_new, j5_new, j6_new = self.base_angles[2:]

        target = [j1_new, j2_new, j3_new, j4_new, j5_new, j6_new]
        try:
            self._mc.send_angles(target, cfg.ROBOT_MOVE_SPEED)
            self.current_angles = target
            self._last_cmd_time = time.time()
            self._last_target = target
        except Exception as e:
            print(f"\n[ERROR] decay_to_base send_angles 失败: {e}")

    def apply_control(self, inputs: RobotControlInputs) -> None:
        """
        每一帧由 orchestrator 调用，传入当前虎口像素 + 温度 + 距离 + dt。
        内部做三件事：
        - 视觉跟随：J1/J2 跟踪 (hx, hy) 与 (HUKOU_REF_U, HUKOU_REF_V) 的偏差；
        - 温度闭环：温度高于 30℃ -> 后退；低于 30℃ -> 前进；
        - 距离闭环：ToF 距离偏离 140mm -> 前后微调；
        """
        # ------ 发送频率限制：最多 10Hz 下发控制指令 ------ #
        now = time.time()
        if now - getattr(self, "_last_cmd_time", 0.0) < 0.1:
            # 距离上次下发不足 0.1 秒，就先不动，等待下一帧
            return

        j1_cur, j2_cur, j3_cur, j4_cur, j5_cur, j6_cur = self.current_angles

        # ---- 1) 视觉：像素误差 -> J1/J2 小幅调整 ---- #
        if inputs.hx is not None and inputs.hy is not None:
             err_x = inputs.hx - cfg.HUKOU_REF_U
             err_y = inputs.hy - cfg.HUKOU_REF_V
             if abs(err_x) > cfg.PIXEL_DEADZONE:
                # 加负号做左右镜像：
                # 现在：err_x > 0（虎口在画面右边） => d_j1 < 0
                # 如果当前机械臂 J1 负方向是“朝右转”，就能对上你的直觉
                d_j1 = cfg.PIXEL_TO_J1_SIGN * cfg.K_J1_DEG_PER_PX * err_x
                d_j1 = float(np.clip(d_j1, -cfg.MAX_J_STEP_DEG, cfg.MAX_J_STEP_DEG))
                j1_cur += d_j1
            # 暂时不根据 err_y 调整 J2，避免和前后控制打架      
        else:
            # 没检测到手，视觉跟随不做增量（回收交给 decay_to_base）
            pass

        # ---- 2) 温度 & 距离：前后方向综合控制（这里先用 J2 做简化） ---- #
        forward_cmd_deg = 0.0

        # 温度闭环：目标 30℃，温度高 -> 后退（前后角度变小）
        if inputs.temp_c is not None:
            temp_err = inputs.temp_c - cfg.TEMP_TARGET_C
            if abs(temp_err) > cfg.TEMP_DEADBAND_C:
                # temp_err > 0 代表“太热”，我们希望后退 => 前向角度减少
                forward_cmd_deg += -cfg.TEMP_KP_DEG_PER_C * temp_err

        # 距离闭环：目标 140mm，距离大 -> 前进，距离小 -> 后退
        if inputs.dist_mm is not None:
            dist_err = inputs.dist_mm - cfg.DIST_TARGET_MM
            if abs(dist_err) > cfg.DIST_DEADBAND_MM:
                # dist_err > 0 (太远) => 前进 => 前向角度增加
                forward_cmd_deg += -cfg.DIST_KP_DEG_PER_MM * dist_err

        forward_cmd_deg = float(
            np.clip(
                forward_cmd_deg,
                -cfg.MAX_FORWARD_STEP_DEG,
                cfg.MAX_FORWARD_STEP_DEG,
            )
        )

        # 前后就先用 J2 来体现（以后可以在这里换成 J3 或组合）
        # 一阶低通平滑前后命令，避免抖动
        alpha = 0.3  # 平滑系数，0.3 比较柔和；越小越平滑但越“慢”
        self._forward_smooth = (1.0 - alpha) * self._forward_smooth + alpha * forward_cmd_deg

        j2_cur += self._forward_smooth

        # ---- 3) 统一做 J1/J2 限幅，保持其它关节在基准 ---- #
        j1_new, j2_new = self._clamp_j1j2_relative_to_base(j1_cur, j2_cur)
        j3_new, j4_new, j5_new, j6_new = self.base_angles[2:]

        target = [j1_new, j2_new, j3_new, j4_new, j5_new, j6_new]

        # ------ 如果与上一次目标几乎没差别，就不要浪费一次指令 ------ #
        if hasattr(self, "_last_target") and self._last_target:
            max_delta = max(abs(t - lt) for t, lt in zip(target, self._last_target))
            if max_delta < 0.3:  # 小于 0.3° 的微小变化直接忽略
                return

        try:
            self._mc.send_angles(target, cfg.ROBOT_MOVE_SPEED)
            self.current_angles = target
            self._last_cmd_time = now
            self._last_target = target

            print(
                f"\r[ROBOT] hx={inputs.hx} hy={inputs.hy} "
                f"T={inputs.temp_c}C D={inputs.dist_mm}mm "
                f"J1={j1_new:.2f} J2={j2_new:.2f}",
                end="",
                flush=True,
            )
        except Exception as e:
            print(f"\n[ERROR] apply_control send_angles 失败: {e}")

    def emergency_retreat(self) -> None:
        """简单的“后退”动作，用于安全规则触发时。这里先让 J2 增大一点。"""
        j1_cur, j2_cur, j3_cur, j4_cur, j5_cur, j6_cur = self.current_angles
        j2_new = j2_cur + 5.0  # 根据机械臂正方向实际表现再调
        j2_new = self._clamp_angle(j2_new)
        target = [j1_cur, j2_new, j3_cur, j4_cur, j5_cur, j6_cur]
        try:
            self._mc.send_angles(target, cfg.ROBOT_MOVE_SPEED)
            self.current_angles = target
            self._last_cmd_time = time.time()
            self._last_target = target
        except Exception as e:
            print(f"[ERROR] emergency_retreat 失败: {e}")
