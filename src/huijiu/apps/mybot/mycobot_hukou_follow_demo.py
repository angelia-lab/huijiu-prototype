"""
MyCobot 280 M5 + RealSense D455
虎口小范围跟随 Demo（约前后/左右 20cm 范围）

功能概述：
- 使用 D455 + MediaPipe Hands 获取【一只指定的手】的虎口像素坐标 (hx, hy)
- 机械臂先回到示教的虎口参考姿态（HUKOU_REF_ANGLES）
- 然后在该姿态附近，通过 J1/J2 小幅增量控制，使末端在约 ±20cm 的范围内
  跟随虎口的左右/上下移动

注意：
- 这里的 “20cm” 是近似量，通过限制 J1/J2 的偏转角度实现（例如 J1 ±40°，J2 ±30°）
- 只认一只“控制手”（TARGET_HAND_LABEL），避免左右手混乱
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands

from pymycobot.mycobot280 import MyCobot280

from huijiu.vision.d455 import RealSenseD455


# ======================= 机械臂相关配置 ======================= #

PORT = "COM4"         # TODO: 填你扫描脚本找到的“真正控制板端口”
BAUD = 115200

# 原始示教得到的虎口姿态（关节角）——来自 drag_teach_capture.py
RAW_HUKOU_REF_ANGLES: List[float] = [172.88, -92.54, -17.05, 111.79, -0.17, 119.00]

# SDK 的角度合法范围（来自异常：-168 ~ 168）
ANGLE_MIN = -168.0
ANGLE_MAX = 168.0


def clamp_angle(a: float, amin: float = ANGLE_MIN, amax: float = ANGLE_MAX) -> float:
    """把单个角度限制在 [amin, amax] 之间。"""
    return max(amin, min(amax, a))


# 夹紧之后的参考姿态（避免超出 SDK 限制）
HUKOU_REF_ANGLES: List[float] = [clamp_angle(a) for a in RAW_HUKOU_REF_ANGLES]


# ======================= 视觉侧：只认指定一只手 ======================= #

# 你现在发现“识别反了”，所以这里给你一个统一的控制开关：
# - 如果 MediaPipe 把你想用的手识别成 "Right"：TARGET_HAND_LABEL = "Right"
# - 如果把你想用的手识别成 "Left"：TARGET_HAND_LABEL = "Left"
TARGET_HAND_LABEL = "Right"   # 根据实际情况改成 "Left" 或 "Right"


@dataclass
class HukouPoint:
    x: int
    y: int


def compute_hukou_pixel(
    hand_landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList,
    image_width: int,
    image_height: int,
) -> HukouPoint:
    """
    根据 MediaPipe Hands 的关键点，估算“虎口”像素坐标：
    简化为：拇指 MCP(id=2) 与食指 MCP(id=5) 的中点。
    """
    lm = hand_landmarks.landmark
    thumb_mcp = lm[2]
    index_mcp = lm[5]

    x_norm = 0.5 * (thumb_mcp.x + index_mcp.x)
    y_norm = 0.5 * (thumb_mcp.y + index_mcp.y)

    x_px = int(x_norm * image_width)
    y_px = int(y_norm * image_height)

    x_px = max(0, min(image_width - 1, x_px))
    y_px = max(0, min(image_height - 1, y_px))

    return HukouPoint(x=x_px, y=y_px)


def pick_control_hand(
    result: mp_hands.Hands,
) -> Tuple[Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList], Optional[str]]:
    """
    从 MediaPipe 结果中，挑出一只“控制用的手”：
    - 只要 handedness.label == TARGET_HAND_LABEL 的手
    - 若没找到，返回 (None, None)
    """
    multi_landmarks = result.multi_hand_landmarks
    multi_handedness = result.multi_handedness

    if not multi_landmarks or not multi_handedness:
        return None, None

    for hand_lms, hand_info in zip(multi_landmarks, multi_handedness):
        label = hand_info.classification[0].label  # "Left" or "Right"
        if label == TARGET_HAND_LABEL:
            return hand_lms, label

    return None, None


# ======================= 机械臂跟随控制（基于 J1/J2 增量） ======================= #

class HukouFollowerByAngles:
    """
    用 J1/J2 在示教姿态附近做小幅关节偏移，实现“约 ±20cm 范围的虎口跟随”。

    设计要点：
    - 基准角度 = HUKOU_REF_ANGLES（示教姿态）
    - current_angles 从基准开始，随着每帧手势偏移做增量调整
    - 像素误差（hx - u0, hy - v0） -> J1/J2 的小角度增量
    - 限制 J1/J2 的总偏移量：例如 J1 最大偏移 ±40°，J2 ±30°
      这样在几百毫米的工作半径下，对应大约 ±20cm 的空间跟随范围
    """

    def __init__(self, port: str, baud: int) -> None:
        self.mc = MyCobot280(port, baud)
        time.sleep(2.0)

        try:
            self.mc.power_on()
        except Exception as e:
            print(f"[WARN] power_on 失败，可忽略: {e}")

        # 基准姿态（示教姿态）
        self.base_angles: List[float] = HUKOU_REF_ANGLES[:]
        self.current_angles: List[float] = HUKOU_REF_ANGLES[:]

        # 把机械臂先回到基准姿态
        print(f"[INFO] 回到虎口参考姿态（角度）：{self.base_angles}")
        self.mc.send_angles(self.base_angles, 40)
        time.sleep(3.0)

        # 每帧最大关节改变量（度），防止一下子跳太大
        self.max_step_deg = 2.0

        # 像素 -> 关节角映射系数（度 / 像素）
        # 注意：这里修正了方向：
        #   - err_x > 0（虎口在画面右侧），J1 朝右转；如果方向反了，再改符号即可
        self.k_j1 = 0.03  # 水平方向 -> J1
        self.k_j2 = 0.03  # 垂直方向 -> J2

        # 像素“死区”：在很小误差范围内不动，避免抖动
        self.dead_zone_px = 15

        # 约束 J1/J2 的总偏移范围：基于基准角度做“±偏移”
        self.j1_base = self.base_angles[0]
        self.j2_base = self.base_angles[1]
        self.j1_offset_limit = 40.0   # J1 相对基准最多偏转 ±40°（左右 ~20cm）
        self.j2_offset_limit = 30.0   # J2 相对基准最多偏转 ±30°（前后/上下 ~20cm）

    def _clamp_joint_relative_to_base(self, idx: int, val: float) -> float:
        """
        按“基准 + 偏移”的方式限制 J1/J2 总偏移范围，再叠加全局 [-168,168] 限制。
        """
        if idx == 0:  # J1
            min_j1 = self.j1_base - self.j1_offset_limit
            max_j1 = self.j1_base + self.j1_offset_limit
            return clamp_angle(max(min_j1, min(max_j1, val)))
        if idx == 1:  # J2
            min_j2 = self.j2_base - self.j2_offset_limit
            max_j2 = self.j2_base + self.j2_offset_limit
            return clamp_angle(max(min_j2, min(max_j2, val)))
        # 其它关节只做全局限制
        return clamp_angle(val)

    def decay_to_base(self, factor: float = 0.1) -> None:
        """
        当当前帧没有检测到控制手时，让 J1/J2 逐步回到基准姿态（平滑回收）。
        factor 越大，回归越快。
        """
        j1_cur, j2_cur, j3_cur, j4_cur, j5_cur, j6_cur = self.current_angles
        j1_new = self.j1_base + (j1_cur - self.j1_base) * (1.0 - factor)
        j2_new = self.j2_base + (j2_cur - self.j2_base) * (1.0 - factor)

        j1_new = self._clamp_joint_relative_to_base(0, j1_new)
        j2_new = self._clamp_joint_relative_to_base(1, j2_new)

        # 其它关节保持基准
        j3_new, j4_new, j5_new, j6_new = self.base_angles[2:]

        target = [j1_new, j2_new, j3_new, j4_new, j5_new, j6_new]
        try:
            self.mc.send_angles(target, 40)
            self.current_angles = target
        except Exception as e:
            print(f"\n[ERROR] decay_to_base send_angles 失败: {e}")

    def step_follow(self, hx: int, hy: int, ref_u: int, ref_v: int) -> None:
        """
        根据虎口像素位置 (hx, hy) 与参考像素 (ref_u, ref_v) 的偏差，
        对当前 J1/J2 做一次小角度增量。
        """
        err_x = hx - ref_u  # >0: 虎口在画面右侧
        err_y = hy - ref_v  # >0: 虎口在画面下方

        # 死区：误差很小就不动，避免机械臂抖动
        if abs(err_x) < self.dead_zone_px and abs(err_y) < self.dead_zone_px:
            print(
                f"\r[FOLLOW] within dead zone, no move. pixel=({hx:4d},{hy:4d})",
                end="",
                flush=True,
            )
            return

        # 像素误差 -> 关节角增量（单位：度）
        # 这里修正了符号，使得“手往左，机械臂也往左”：
        # - 当虎口在画面右侧（err_x>0），希望底座向右跟随，所以 d_j1 > 0。
        d_j1 = -self.k_j1 * err_x
        # - 当虎口在画面下方（err_y>0），让手臂稍微“往下/往前”一点，符号需要你根据实际机械方向微调
        d_j2 = -self.k_j2 * err_y

        # 限制单步最大变化
        d_j1 = float(np.clip(d_j1, -self.max_step_deg, self.max_step_deg))
        d_j2 = float(np.clip(d_j2, -self.max_step_deg, self.max_step_deg))

        # 在 current_angles 基础上累计偏移
        j1_cur, j2_cur, j3_cur, j4_cur, j5_cur, j6_cur = self.current_angles

        j1_new = self._clamp_joint_relative_to_base(0, j1_cur + d_j1)
        j2_new = self._clamp_joint_relative_to_base(1, j2_cur + d_j2)

        # 其它关节保持基准
        j3_new, j4_new, j5_new, j6_new = self.base_angles[2:]

        target = [j1_new, j2_new, j3_new, j4_new, j5_new, j6_new]

        try:
            self.mc.send_angles(target, 40)
            self.current_angles = target
            print(
                f"\r[FOLLOW] pixel=({hx:4d},{hy:4d}) "
                f"err=({err_x:4d},{err_y:4d}) "
                f"dJ=({d_j1:5.2f},{d_j2:5.2f}) "
                f"J=({j1_new:6.2f},{j2_new:6.2f})",
                end="",
                flush=True,
            )
        except Exception as e:
            print(f"\n[ERROR] send_angles 失败: {e}")


# ======================= 主程序：D455 + Mediapipe + MyCobot ======================= #

def main() -> int:
    # 1. 启动 D455
    cam = RealSenseD455(
        color_resolution=(640, 480),
        depth_resolution=(640, 480),
        fps=30,
        align_to_color=True,
    )
    cam.start()
    print(
        f"[INFO] D455 已启动。按 'q' 或 ESC 退出。\n"
        f"       当前控制手目标：{TARGET_HAND_LABEL}\n"
    )

    # 2. 初始化 MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # 3. 初始化机械臂跟随控制器
    follower = HukouFollowerByAngles(PORT, BAUD)

    # 4. 参考像素：初始先用图像中心，可以之后用手固定在穴位位置时打印 hx, hy 精准标定
    ref_u, ref_v = 402, 169  # TODO: 后面可以用“穴位理想姿态”下的虎口像素值替换

    print(
        "[INFO] 开始联动：\n"
        f"      - 只使用被 MediaPipe 判定为 {TARGET_HAND_LABEL} 的那只手；\n"
        "      - 手在 D455 视野内左右/上下移动，机械臂在基准姿态附近约 ±20cm 跟随；\n"
        "      - 当前用的是 J1/J2 控制，前后/高度只是近似；\n"
        "      - 按 'q' 或 ESC 退出。\n"
    )

    try:
        while True:
            frame = cam.get_frame()
            color = frame.color_bgr
            if color is None:
                continue

            h, w, _ = color.shape
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)
            vis = color.copy()

            if result.multi_hand_landmarks:
                hand_lms, label = pick_control_hand(result)

                if hand_lms is not None and label is not None:
                    # 画出控制手骨架
                    mp_drawing.draw_landmarks(
                        vis,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                    )

                    hukou = compute_hukou_pixel(hand_lms, w, h)

                    # 可视化虎口点
                    cv2.circle(vis, (hukou.x, hukou.y), 8, (0, 0, 255), -1)
                    cv2.putText(
                        vis,
                        f"{label} Hukou",
                        (hukou.x + 10, hukou.y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # 机械臂跟随一步
                    follower.step_follow(hukou.x, hukou.y, ref_u, ref_v)

                else:
                    cv2.putText(
                        vis,
                        f"No {TARGET_HAND_LABEL} hand",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    print(
                        f"\r[INFO] 本帧未检测到 {TARGET_HAND_LABEL} 手，机械臂缓慢回基准姿态。",
                        end="",
                        flush=True,
                    )
                    follower.decay_to_base()

            else:
                cv2.putText(
                    vis,
                    "No hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                print(
                    "\r[INFO] 本帧未检测到任何手，机械臂缓慢回基准姿态。",
                    end="",
                    flush=True,
                )
                follower.decay_to_base()

            # 画参考像素十字线，方便你以后精调 ref_u, ref_v
            cv2.drawMarker(
                vis,
                (ref_u, ref_v),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
            )

            cv2.imshow("MyCobot + D455 Hukou Follow (±20cm approx)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("\n[INFO] 收到退出指令，正在关闭...")
                break

    finally:
        hands.close()
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] 已关闭。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
