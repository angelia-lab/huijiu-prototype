# src/huijiu/vision/hands.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands

from huijiu.core import config as cfg


@dataclass
class HukouDetection:
    """虎口识别结果（当前先专注这一部位，后面可以扩展更多部位）"""
    has_hand: bool
    label: Optional[str]      # "Left" / "Right"（已经按 config 做过镜像修正）
    hx: Optional[int]
    hy: Optional[int]
    annotated: Optional[object]  # BGR 图像（带骨架/文字标注）


class HandHukouTracker:
    """
    封装 MediaPipe Hands，用于：
    - 识别指定“控制手”（TARGET_HAND_LABEL）
    - 计算该手的虎口位置 (hx, hy)
    - 返回带可视化标注的图像，以后你再想看 wrist / fingertip 都可以加进来
    """

    def __init__(self) -> None:
        # MediaPipe Hands 初始化
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # 想用哪只手控制：来自 config
        self._target_label = cfg.TARGET_HAND_LABEL
        self._mirrored = getattr(cfg, "HAND_VIEW_MIRRORED", False)

    # ---------- 内部辅助方法 ---------- #

    def _normalize_label(self, mp_label: str) -> str:
        """
        MediaPipe 的 handedness.label 是从相机视角定义的。
        如果相机对着你拍，相当于你看到的是镜像，这时左右反了。
        HAND_VIEW_MIRRORED=True 时，我们直接在这里把 Left<->Right 互换。
        """
        if not self._mirrored:
            return mp_label

        if mp_label == "Left":
            return "Right"
        if mp_label == "Right":
            return "Left"
        return mp_label

    @staticmethod
    def _compute_hukou_pixel(hand_landmarks, image_width: int, image_height: int) -> Tuple[int, int]:
        """
        目前虎口的定义：拇指 MCP(id=2) 与食指 MCP(id=5) 的中点。
        将来你要加“腕、肘”等，只要再写类似的 compute_xxx_pixel 即可。
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

        return x_px, y_px

    def _pick_control_hand(self, result):
        """
        从 MediaPipe 的结果中挑出“一只控制手”：
        - 先对 handedness.label 做镜像修正；
        - 再和 TARGET_HAND_LABEL 对比，只取匹配的那只。
        """
        multi_landmarks = result.multi_hand_landmarks
        multi_handedness = result.multi_handedness

        if not multi_landmarks or not multi_handedness:
            return None, None

        for hand_lms, hand_info in zip(multi_landmarks, multi_handedness):
            raw_label = hand_info.classification[0].label  # 原始的 "Left"/"Right"
            norm_label = self._normalize_label(raw_label)

            if norm_label == self._target_label:
                # 返回修正后的 label（符合你逻辑上的 Left/Right）
                return hand_lms, norm_label

        return None, None

    # ---------- 对外主要接口 ---------- #

    def process(self, bgr_image) -> HukouDetection:
        """
        输入一帧 BGR 图像：
        - 内部转换为 RGB 喂给 MediaPipe；
        - 只挑出一只“控制手”（config.TARGET_HAND_LABEL）；
        - 计算该手的虎口像素；
        - 在图像上画骨架 / 虎口点 / 文本标注。
        """
        h, w, _ = bgr_image.shape
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        vis = bgr_image.copy()

        if not result.multi_hand_landmarks:
            # 没手：只画一行提示
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
            return HukouDetection(
                has_hand=False,
                label=None,
                hx=None,
                hy=None,
                annotated=vis,
            )

        hand_lms, label = self._pick_control_hand(result)
        if hand_lms is None or label is None:
            # 有手，但不是你指定的那只（比如你设 Left，结果画面里只有 Right）
            cv2.putText(
                vis,
                f"No {self._target_label} hand",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return HukouDetection(
                has_hand=False,
                label=None,
                hx=None,
                hy=None,
                annotated=vis,
            )

        # 画骨架
        mp_drawing.draw_landmarks(
            vis,
            hand_lms,
            mp_hands.HAND_CONNECTIONS,
        )

        # 计算虎口像素
        hx, hy = self._compute_hukou_pixel(hand_lms, w, h)
        cv2.circle(vis, (hx, hy), 8, (0, 0, 255), -1)
        cv2.putText(
            vis,
            f"{label} Hukou",
            (hx + 10, hy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # 底部打印坐标方便校准
        cv2.putText(
            vis,
            f"hx={hx}, hy={hy}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return HukouDetection(
            has_hand=True,
            label=label,
            hx=hx,
            hy=hy,
            annotated=vis,
        )

    def close(self) -> None:
        self._hands.close()
