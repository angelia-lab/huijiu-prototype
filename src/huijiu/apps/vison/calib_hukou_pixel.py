"""
标定虎口“理想中心位置”的像素坐标 (hx0, hy0)

流程：
1. 先运行 mycobot_goto_hukou_pose.py，让机械臂停在 HUKOU_REF_ANGLES 姿态；
2. 保持机械臂不动，用本脚本打开 D455 + MediaPipe；
3. 用【控制手】（TARGET_HAND_LABEL）把虎口放到你认为“刚好对准艾灸点”的位置；
4. 按一次空格键，本脚本会把当前 (hx, hy) 打印为一行配置，可以复制到 mycobot_hukou_follow_demo 里；
5. 按 'q' 或 ESC 退出。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands

from huijiu.vision.d455 import RealSenseD455

# 和跟随 demo 保持一致：用同一只“控制手”
TARGET_HAND_LABEL = "Right"  # 根据你的 hand_hukou_demo 实际情况改成 "Left" 或 "Right"


@dataclass
class HukouPoint:
    x: int
    y: int


def compute_hukou_pixel(hand_landmarks, image_width: int, image_height: int) -> HukouPoint:
    """拇指 MCP(id=2) 与食指 MCP(id=5) 的中点 ≈ 虎口像素坐标。"""
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


def pick_control_hand(result) -> Tuple[Optional[mp.framework.formats.landmark_pb2.NormalizedLandmarkList], Optional[str]]:
    """只选 handedness.label == TARGET_HAND_LABEL 的那只手。"""
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
        f"[INFO] D455 已启动。当前控制手目标: {TARGET_HAND_LABEL}\n"
        "       操作步骤：\n"
        "       1）先让机械臂用 mycobot_goto_hukou_pose 停在 HUKOU_REF_ANGLES；\n"
        "       2）本脚本运行中，用控制手把虎口放到你认为“对准穴位”的位置；\n"
        "       3）按一次【空格键】，会打印当前 (hx, hy) 为配置行；\n"
        "       4）按 'q' 或 ESC 退出。\n"
    )

    # 2. 初始化 MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    last_hukou: Optional[HukouPoint] = None

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
                    mp_drawing.draw_landmarks(
                        vis,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                    )

                    hukou = compute_hukou_pixel(hand_lms, w, h)
                    last_hukou = hukou

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

                    cv2.putText(
                        vis,
                        f"hx={hukou.x}, hy={hukou.y}",
                        (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    print(
                        f"\r[INFO] {label} hand hukou 像素: hx={hukou.x:4d}, hy={hukou.y:4d}",
                        end="",
                        flush=True,
                    )
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

            cv2.imshow("Calib Hukou Pixel", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):  # 空格键：采样当前坐标
                if last_hukou is not None:
                    print("\n[CALIB] 捕获到一帧虎口中心：")
                    print(
                        f"        建议在 mycobot_hukou_follow_demo.py 中设置：\n"
                        f"        ref_u, ref_v = {last_hukou.x}, {last_hukou.y}"
                    )
                else:
                    print("\n[CALIB] 当前没有有效的虎口坐标，无法采样。")

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
