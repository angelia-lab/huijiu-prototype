# src/huijiu/vision/aruco_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from huijiu.core import config as cfg

CAMERA_INTRINSICS_NAME = "d455_intrinsics.json"

# 先统一用 4x4_100，如果后面你确实用别的，再一起改
ARUCO_DICT_NAME = "DICT_4X4_100"
ARUCO_MARKER_LENGTH_M = 0.03  # 3cm


def load_camera_intrinsics() -> Tuple[np.ndarray, np.ndarray, int, int]:
    intr_path = cfg.D455_INSTRINSICS
    if not intr_path.exists():
        raise FileNotFoundError(f"找不到相机内参文件: {intr_path}")

    with intr_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    fx = float(data["fx"])
    fy = float(data["fy"])
    cx = float(data["cx"])
    cy = float(data["cy"])
    width = int(data["image_width"])
    height = int(data["image_height"])

    dist = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return K, dist, width, height


def get_aruco_dict_and_params():
    # 字典：从 ARUCO_DICT_NAME 字符串映射到 cv2.aruco 常量
    dict_id = getattr(cv2.aruco, ARUCO_DICT_NAME)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    # DetectorParameters：不同 OpenCV 版本有 create() / 直接构造两种写法
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        params = cv2.aruco.DetectorParameters_create()
    else:
        params = cv2.aruco.DetectorParameters()

    # 调宽松一点的阈值（让检测更容易成功）
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    return aruco_dict, params
