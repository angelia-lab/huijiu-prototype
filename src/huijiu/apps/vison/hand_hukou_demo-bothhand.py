"""
D455 + MediaPipe 识别手的“虎口”位置 Demo

功能：
- 从 D455 读取对齐的彩色图 + 深度图
- 使用 MediaPipe Hands 检测手部 21 个关键点
- 以拇指 MCP(2) 和食指 MCP(5) 的中点，作为“虎口”位置
- 在彩色图上画出虎口位置，并打印像素坐标 + 深度（米）

运行：
(.venv) python -m huijiu.apps.hand_hukou_demo
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from huijiu.vision.d455 import RealSenseD455


# ----------------- Tiger mouth（虎口）计算 ----------------- #

def compute_hukou_pixel(
    hand_landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList,
    image_width: int,
    image_height: int,
) -> Tuple[int, int]:
    """
    根据 MediaPipe Hands 的关键点，估算“虎口”在图像中的像素坐标。

    这里简单取：
    - 拇指 MCP: id = 2
    - 食指 MCP: id = 5
    的中点。

    返回: (x_px, y_px)
    """
    lm = hand_landmarks.landmark

    thumb_mcp = lm[2]   # 拇指 MCP
    index_mcp = lm[5]   # 食指 MCP

    # 归一化坐标 -> 像素坐标
    x_norm = 0.5 * (thumb_mcp.x + index_mcp.x)
    y_norm = 0.5 * (thumb_mcp.y + index_mcp.y)

    x_px = int(x_norm * image_width)
    y_px = int(y_norm * image_height)

    # 防止溢出
    x_px = max(0, min(image_width - 1, x_px))
    y_px = max(0, min(image_height - 1, y_px))

    return x_px, y_px


# ----------------- 主逻辑 ----------------- #

def main() -> int:
    # 启动 D455，相机内参可以用默认，也可以按你的标定结果设置
    cam = RealSenseD455(
        color_resolution=(640, 480),
        depth_resolution=(640, 480),
        fps=30,
        align_to_color=True,  # 深度对齐到彩色，这是拿中心深度的前提
    )
    cam.start()
    print("[INFO] RealSense D455 已启动，按 'q' 或 ESC 退出。")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        while True:
            frame = cam.get_frame()
            color = frame.color_bgr
            depth_m = frame.depth_m  # float32，单位米

            if color is None or depth_m is None:
                continue

            h, w, _ = color.shape

            # MediaPipe 使用 RGB
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            vis = color.copy()

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # 画出整只手的关键点（调试用，可注释掉）
                    mp.solutions.drawing_utils.draw_landmarks(
                        vis,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style(),
                    )

                    # 计算虎口像素位置
                    hx, hy = compute_hukou_pixel(hand_landmarks, w, h)

                    # 在彩色图上画一个圆标出虎口
                    cv2.circle(vis, (hx, hy), 8, (0, 0, 255), -1)
                    cv2.putText(
                        vis,
                        "Hukou",
                        (hx + 10, hy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    # 读取虎口点的深度（米）
                    depth_val = float(depth_m[hy, hx])
                    if depth_val <= 0 or np.isnan(depth_val):
                        depth_text = "Z = invalid"
                    else:
                        depth_text = f"Z = {depth_val:.3f} m"

                    print(
                        f"Hukou pixel = ({hx}, {hy}), {depth_text}       ",
                        end="\r",
                    )

            else:
                cv2.putText(
                    vis,
                    "No hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                print("No hand detected                      ", end="\r")

            cv2.imshow("D455 Hand Hukou Demo", vis)
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
