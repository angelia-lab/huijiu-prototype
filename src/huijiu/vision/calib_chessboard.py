"""
使用 D455 + A4 棋盘格相机标定脚本

- 使用已有的 RealSenseD455 封装，从相机实时采集图像
- 检测 9x6 内角点的棋盘格
- 按键 's'：保存一帧检测到的角点用于标定
- 按键 'c'：执行标定并将结果保存到 d455_intrinsics.json
- 按键 'q' 或 ESC：退出

操作说明：
1、确认 D455 已连接，pyrealsense2 正常
2、运行此程序python -m huijiu.vision.calib_chessboard
3、标定采集：
1）拿 A4 棋盘格放在不同位置/角度（左上、右下、近一点、远一点、有一点旋转）；
2）每出现绿色角点框时，按 s 采样；
3）收集 8–15 张不同姿态后，按 c 执行标定；
4）标定成功后，会在 src/hardware/calib/ 下生成 d455_intrinsics.json。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from huijiu.vision.d455 import RealSenseD455
from huijiu.core import config as cfg


# === 棋盘格配置，根据实际板子(A4棋盘格规格：单格边长：35 mm，内角数：9×6 内角) ===
CHESSBOARD_COLS = 9          # 内角：列数
CHESSBOARD_ROWS = 6          # 内角：行数
CHESSBOARD_SIZE = (CHESSBOARD_COLS, CHESSBOARD_ROWS)

SQUARE_SIZE_M = 0.035        # 单格边长，单位：米


@dataclass
class CameraIntrinsics:
    """用于保存和加载的相机内参结构."""
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: list[float]
    image_width: int
    image_height: int
    rms: float


def _create_object_points() -> np.ndarray:
    """
    为一个棋盘格视图生成 3D 参考点：
    Z=0 平面上的 (X, Y, 0)，单位米。
    """
    objp = np.zeros((CHESSBOARD_ROWS * CHESSBOARD_COLS, 3), np.float32)
    # 网格排列：X 方向 COLS，Y 方向 ROWS
    objp[:, :2] = (
        np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS]
        .T.reshape(-1, 2)
        * SQUARE_SIZE_M
    )
    return objp


def calibrate_from_samples(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    image_size: Tuple[int, int],
) -> CameraIntrinsics:
    """
    调用 OpenCV 的 calibrateCamera，返回内参结果。
    """
    # OpenCV 要求 image_size 为 (width, height)
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        dist_coeffs=dist_coeffs.ravel().tolist(),
        image_width=image_size[0],
        image_height=image_size[1],
        rms=float(ret),
    )


def save_intrinsics(
    intr: CameraIntrinsics,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(intr), f, indent=2, ensure_ascii=False)
    print(f"[INFO] 标定结果已保存到: {output_path}")


def main() -> int:
    # 输出文件路径
    output_path =cfg.D455_INSTRINSICS
    
    
    cam = RealSenseD455(
        color_resolution=(640, 480),
        depth_resolution=(640, 480),
        fps=30,
        align_to_color=True,
    )
    cam.start()
    print("[INFO] D455 已启动，开始采集棋盘格图像。")
    print("      按 's' 保存当前视图的角点用于标定")
    print("      按 'c' 执行标定并保存结果")
    print("      按 'q' 或 ESC 退出")

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []

    objp_template = _create_object_points()

    try:
        while True:
            frame = cam.get_frame()
            color = frame.color_bgr
            if color is None:
                continue

            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            image_size = (gray.shape[1], gray.shape[0])

            # 查找棋盘格角点
            found, corners = cv2.findChessboardCorners(
                gray,
                CHESSBOARD_SIZE,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_FAST_CHECK,
            )

            vis = color.copy()

            if found:
                # 角点细化
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                corners_refined = cv2.cornerSubPix(
                    gray,
                    corners,
                    winSize=(11, 11),
                    zeroZone=(-1, -1),
                    criteria=criteria,
                )

                cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners_refined, found)

                # 显示当前已采样数量
                cv2.putText(
                    vis,
                    f"samples: {len(objpoints)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            else:
                cv2.putText(
                    vis,
                    "Chessboard NOT found",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Calib - Color", vis)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                print("\n[INFO] 退出标定脚本。")
                break

            if key == ord("s") and found:
                # 保存当前视图的角点
                objpoints.append(objp_template.copy())
                imgpoints.append(corners_refined.reshape(-1, 2))
                print(f"[INFO] 已采集样本数: {len(objpoints)}")

            if key == ord("c"):
                if len(objpoints) < 5:
                    print("[WARN] 样本数量不足，建议至少 5–10 张不同姿态的棋盘格。")
                    continue

                print("[INFO] 正在执行标定，请稍候...")
                intr = calibrate_from_samples(objpoints, imgpoints, image_size)
                print(
                    f"[INFO] 标定完成: rms = {intr.rms:.4f}, "
                    f"fx={intr.fx:.1f}, fy={intr.fy:.1f}, "
                    f"cx={intr.cx:.1f}, cy={intr.cy:.1f}"
                )
                save_intrinsics(intr, output_path)

    finally:
        cam.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
