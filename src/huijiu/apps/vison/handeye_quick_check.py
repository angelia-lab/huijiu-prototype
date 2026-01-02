"""
handeye_quick_check.py

用途：
- 读取 D455 内参 (d455_intrinsics.json)
- 读取 手眼标定结果 (handeye_result.json / handeye_extrinsics.json)
- 实时检测 ArUco 标记
- 在终端和画面上同时打印：
    - 相机坐标系下的 (x_c, y_c, z_c) [mm]
    - 机械臂基坐标系下的 (x_b, y_b, z_b) [mm]
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from huijiu.core import config as cfg

# 路径配置：保持与你工程里的配置一致
CAM_INTRINSICS_PATH = cfg.D455_INSTRINSICS      # e.g. PROJECT_ROOT / "hardware/calib/d455_intrinsics.json"
HANDEYE_PATH        = cfg.OUT_JSON              # e.g. PROJECT_ROOT / "hardware/calib/handeye_result.json"

# 你的 ArUco 实物边长（mm）
MARKER_LENGTH_MM = 40.0


# ========= 工具函数：加载相机内参 =========
def load_intrinsics():
    data = json.loads(CAM_INTRINSICS_PATH.read_text(encoding="utf-8"))

    # --- 1) 先拿相机矩阵 ---
    if "camera_matrix" in data:
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
    else:
        fx, fy = data["fx"], data["fy"]
        cx, cy = data["cx"], data["cy"]
        camera_matrix = np.array(
            [[fx, 0,  cx],
             [0,  fy, cy],
             [0,  0,  1 ]],
            dtype=np.float32,
        )

    # --- 2) 再拿畸变系数 ---
    if "dist_coeffs" in data:
        dist_list = data["dist_coeffs"]
        dist_coeffs = np.array(dist_list, dtype=np.float32).reshape(-1)
    elif "dist" in data:
        dist_list = data["dist"]
        dist_coeffs = np.array(dist_list, dtype=np.float32).reshape(-1)
    elif "k1" in data:
        k1, k2 = data["k1"], data["k2"]
        p1, p2 = data["p1"], data["p2"]
        k3 = data.get("k3", 0.0)
        dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    else:
        # 如果压根没存畸变，就当作 0 处理
        dist_coeffs = np.zeros(5, dtype=np.float32)

    print(f"[INFO] 相机内参加载成功: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print(f"[INFO] 畸变系数 dist_coeffs={dist_coeffs}")
    return camera_matrix, dist_coeffs


# ========= 工具函数：加载手眼标定外参 =========
def load_handeye():
    data = json.loads(HANDEYE_PATH.read_text(encoding="utf-8"))
    # 注意这里的键名要和 handeye_result.json 里对应
    R = np.array(data["R_base_cam"], dtype=np.float32)                # 3x3
    t = np.array(data["t_base_cam"], dtype=np.float32).reshape(3, 1)  # 3x1, 单位一般是 mm
    return R, t


# ========= 工具函数：兼容不同版本的 ArUco API =========
def create_aruco_detector():
    # 这里假设你用的是 4x4_50 字典（和之前生成的 marker 对应）
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    if hasattr(cv2.aruco, "DetectorParameters_create"):
        # 旧接口 (OpenCV <= 4.5.x 常见)
        params = cv2.aruco.DetectorParameters_create()
        detector = None  # 用全局函数 detectMarkers
        print("[INFO] 使用旧版 ArUco API: DetectorParameters_create + cv2.aruco.detectMarkers")
    else:
        # 新接口 (OpenCV 4.7+/4.10+ 一类)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        print("[INFO] 使用新版 ArUco API: DetectorParameters + ArucoDetector.detectMarkers")

    return aruco_dict, params, detector


# ========= 主流程 =========
def main():
    cam_mtx, dist = load_intrinsics()
    R_bc, t_bc    = load_handeye()
    print("[INFO] 内参 & 手眼外参已加载。")

    # --- RealSense 管线 ---
    pipeline = rs.pipeline()
    rs_cfg   = rs.config()
    rs_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(rs_cfg)
    print("[INFO] D455 已启动，按 q 退出。")

    aruco_dict, aruco_params, aruco_detector = create_aruco_detector()
    marker_length_m = MARKER_LENGTH_MM / 1000.0  # OpenCV 习惯用米

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            gray  = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            # --- 兼容不同版本 detectMarkers ---
            if aruco_detector is None:
                # 旧版：直接用函数
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, aruco_dict, parameters=aruco_params
                )
            else:
                # 新版：用 ArucoDetector 对象
                corners, ids, _ = aruco_detector.detectMarkers(gray)

            if ids is not None and len(ids) > 0:
                # 画出检测到的标记框
                cv2.aruco.drawDetectedMarkers(color, corners, ids)

                # 位姿估计：得到每个 marker 在“相机坐标系”下的位置 (米)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_length_m, cam_mtx, dist
                )

                for i, marker_id in enumerate(ids.flatten()):
                    rvec = rvecs[i].reshape(3, 1)
                    tvec = tvecs[i].reshape(3, 1) * 1000.0  # m → mm，和 handeye 的单位对齐

                    # 相机坐标系下：z 轴一般是“往前”，x 右、y 下（取决于约定）
                    x_c, y_c, z_c = tvec.flatten().tolist()

                    # 转到机械臂基坐标系：P_base = R_bc * P_cam + t_bc
                    p_base = R_bc @ tvec + t_bc
                    x_b, y_b, z_b = p_base.flatten().tolist()

                    text = (
                        f"id={marker_id} "
                        f"cam(mm)=({x_c:.1f},{y_c:.1f},{z_c:.1f}) "
                        f"base(mm)=({x_b:.1f},{y_b:.1f},{z_b:.1f})"
                    )
                    print(text)

                    cv2.putText(
                        color, text, (10, 30 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                    )

            cv2.imshow("handeye_quick_check", color)
            key = cv2.waitKey(1)
            if key in (ord("q"), 27):  # q 或 ESC
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] 结束。")


if __name__ == "__main__":
    main()
