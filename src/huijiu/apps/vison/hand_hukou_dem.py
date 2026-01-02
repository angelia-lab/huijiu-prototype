"""
D455 + MediaPipe Hands
只使用【指定一只手】的虎口位置（可选 Left / Right）

功能：
- 使用 RealSenseD455 获取彩色帧
- 使用 MediaPipe Hands 检测手部关键点
- 只选定一只“控制手”（TARGET_HAND_LABEL），忽略另一只
- 计算这只手的虎口像素坐标（拇指 MCP 与食指 MCP 的中点）

用法：
1. 确保已安装依赖：
   - pyrealsense2
   - opencv-contrib-python
   - mediapipe
2. 确保 huijiu.vision.d455.RealSenseD455 已实现并可用。
3. 根据你实际情况设置 TARGET_HAND_LABEL：
   - 如果你想用“MediaPipe 识别为 Left 的那只手”：TARGET_HAND_LABEL = "Left"
   - 如果你现在看到识别反了，想用被识别为 Right 的那只手：TARGET_HAND_LABEL = "Right"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands

from huijiu.vision.d455 import RealSenseD455

# ============================================================
# 配置：控制手是哪一只？"Left" 或 "Right"
# ============================================================
# 你现在遇到“识别反了”的情况：
# - 如果在画面里你伸的是右手，但 demo 写的是 Left / 反了，
#   就把下面这个改成 "Right"，只用被识别为 Right 的那只手。
TARGET_HAND_LABEL = "Left"     # 可改成 "Left" 或 "Right"


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
    从 MediaPipe 结果中，挑出一只“控制用的手”。

    当前策略：
    - 优先寻找 handedness.label == TARGET_HAND_LABEL 的那只手，
      比如 TARGET_HAND_LABEL = "Right" 就只要 Right；
    - 如果没找到，返回 (None, None)。
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
        f"[INFO] D455 已启动，按 'q' 或 ESC 退出。当前控制手目标：{TARGET_HAND_LABEL}\n"
        f"       只会使用被 MediaPipe 判定为 {TARGET_HAND_LABEL} 的那只手的虎口。\n"
    )

    # 2. 初始化 MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,           # 支持双手，但我们稍后只选一只
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
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
                # 只选指定 label 的那只手
                hand_lms, label = pick_control_hand(result)

                if hand_lms is not None and label is not None:
                    # 画出控制手骨架
                    mp_drawing.draw_landmarks(
                        vis,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                    )

                    # 计算虎口位置
                    hukou = compute_hukou_pixel(hand_lms, w, h)

                    # 可视化虎口
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

                    print(
                        f"\r[INFO] {label} hand hukou 像素坐标: "
                        f"x={hukou.x:4d}, y={hukou.y:4d}",
                        end="",
                        flush=True,
                    )
                else:
                    cv2.putText(
                        vis,
                        f"No {TARGET_HAND_LABEL} hand detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    print(
                        f"\r[INFO] 当前帧未检测到 {TARGET_HAND_LABEL} 手。",
                        end="",
                        flush=True,
                    )
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
                print("\r[INFO] 当前帧未检测到任何手。        ", end="", flush=True)

            cv2.imshow("D455 Hand Hukou Demo (Control Hand Only)", vis)
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
