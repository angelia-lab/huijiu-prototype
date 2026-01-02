"""
D455 预览 Demo

功能：
- 打开 D455
- 实时显示彩色图 + 深度伪彩
- 在中心点画十字，并打印中心点深度（米）
"""
import sys
from typing import Optional

import cv2
import numpy as np

from huijiu.vision.d455 import RealSenseD455


def depth_to_colormap(
    depth_m: np.ndarray,
    max_distance_m: float = 1.5,
) -> np.ndarray:
    """
    将深度（米）转换为可视化的伪彩色图像。

    :param depth_m: (H, W) float32，单位米
    :param max_distance_m: 伪彩映射的最大距离（更远的都当作 max）
    :return: (H, W, 3) uint8，BGR 伪彩
    """
    # 处理无效深度（0 或 NaN）
    depth = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)

    # 限制在 [0, max_distance_m]
    depth_clipped = np.clip(depth, 0.0, max_distance_m)

    # 归一化到 [0, 255]
    depth_norm = (depth_clipped / max_distance_m * 255.0).astype(np.uint8)

    # 应用伪彩色
    depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    return depth_colormap


def main() -> int:
    # 初始化相机
    cam = RealSenseD455(
        color_resolution=(640, 480),
        depth_resolution=(640, 480),
        fps=30,
        align_to_color=True,
    )

    try:
        cam.start()
    except Exception as exc:
        print(f"[ERROR] 无法启动 RealSense D455：{exc}", file=sys.stderr)
        return 1

    print("[INFO] RealSense D455 已启动，按 'q' 或 ESC 退出。")

    try:
        while True:
            try:
                frame = cam.get_frame(timeout_ms=1000)
            except Exception as exc:
                print(f"[WARN] 获取帧失败：{exc}", file=sys.stderr)
                continue

            color = frame.color_bgr
            depth_m = frame.depth_m

            if color is None or depth_m is None:
                print("[WARN] 本帧缺少彩色或深度数据。")
                continue

            h, w = depth_m.shape
            cx, cy = w // 2, h // 2

            center_depth = float(depth_m[cy, cx])

            # 将过小的值视为无效
            if center_depth <= 0.0:
                center_text = "center: --.-- m"
            else:
                center_text = f"center: {center_depth:5.3f} m"

            # 构造深度伪彩图
            depth_vis = depth_to_colormap(depth_m, max_distance_m=1.5)

            # 在彩色图和深度图上画十字
            cv2.drawMarker(color, (cx, cy), (0, 0, 255),
                           markerType=cv2.MARKER_CROSS,
                           markerSize=15, thickness=2)
            cv2.drawMarker(depth_vis, (cx, cy), (255, 255, 255),
                           markerType=cv2.MARKER_CROSS,
                           markerSize=15, thickness=2)

            # 在彩色图左上角写中心点距离
            cv2.putText(
                color,
                center_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("D455 Color", color)
            cv2.imshow("D455 Depth (pseudo-color)", depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("\n[INFO] 收到退出指令，正在关闭...")
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] RealSense D455 已停止。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
