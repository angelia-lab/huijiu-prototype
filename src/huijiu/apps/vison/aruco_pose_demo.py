#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D455 + ArUco + 手眼标定 演示脚本

- 从 JSON 读取相机内参
- 从 handeye_result.json 读取 R_base_cam / t_base_cam
- 实时检测 ArUco，显示：
    * Marker 在相机坐标系下坐标 (mm)
    * Marker 在机械臂基坐标系下坐标 (mm)
    * 中心点深度 (m)
- 按 'q' 或 ESC 退出
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from huijiu.core import config as cfg

# ========= 配置区域：请按你的实际情况修改 =========

# 相机内参文件路径（根据你的项目调整）
CAM_INTRINSICS_PATH = cfg.D455_INSTRINSICS

# 手眼标定结果文件路径
HAND_EYE_PATH = cfg.OUT_JSON

# ArUco 标记实物边长（毫米）
MARKER_LENGTH_MM = 40.0  # 例如 4cm × 4cm

# 使用的 ArUco 字典（根据你生成的实际字典修改）
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50


# ========= 工具函数 =========

def load_camera_intrinsics():
    """
    从 JSON 读取相机内参，兼容两种常见格式：
    1) {"camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]], "dist_coeffs": [k1, k2, p1, p2, k3]}
    2) {"fx":..., "fy":..., "cx":..., "cy":..., "dist_coeffs":[...]} 或无畸变
    """
    data = json.loads(CAM_INTRINSICS_PATH.read_text(encoding="utf-8"))

    if "camera_matrix" in data:
        K = np.array(data["camera_matrix"], dtype=np.float64)
    elif {"fx", "fy", "cx", "cy"} <= set(data.keys()):
        fx = float(data["fx"])
        fy = float(data["fy"])
        cx = float(data["cx"])
        cy = float(data["cy"])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    else:
        raise RuntimeError(
            f"不支持的相机内参 JSON 格式，请检查: {CAM_INTRINSICS_PATH}"
        )

    if "dist_coeffs" in data:
        dist = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    else:
        # 如果没有畸变参数，就用 0
        dist = np.zeros((5, 1), dtype=np.float64)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print("[INFO] 已加载相机内参。")
    print(f"       fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    return K, dist


def load_handeye():
    """
    从 handeye_result.json 读取 R_base_cam, t_base_cam
    预期格式:
    {
      "R_base_cam": [[...],[...],[...]],
      "t_base_cam": [x, y, z]  # 单位：mm
    }
    """
    data = json.loads(HAND_EYE_PATH.read_text(encoding="utf-8"))
    R = np.array(data["R_base_cam"], dtype=np.float64)
    t = np.array(data["t_base_cam"], dtype=np.float64).reshape(3, 1)
    print("[INFO] 已加载手眼标定结果。")
    return R, t


def create_aruco_detector():
    """
    创建 ArUco 检测器（OpenCV 4.7+ 推荐用法）
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    return detector


# ========= 主流程 =========

def main():
    # 1. 加载相机内参与手眼标定
    camera_matrix, dist_coeffs = load_camera_intrinsics()
    R_base_cam, t_base_cam = load_handeye()
    detector = create_aruco_detector()

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # 2. 启动 RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    # 对齐深度到彩色
    align_to = rs.stream.color
    align = rs.align(align_to)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] 深度 scale: {depth_scale} (m/单位)")
    print("[INFO] D455 已启动，正在检测 ArUco 标记。按 'q' 或 ESC 退出。")

    aruco_axis_len = MARKER_LENGTH_MM * 0.5  # 用于画坐标轴（mm）

    try:
        while True:
            # 3. 取一帧，并对齐
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            if color_img is None or color_img.size == 0:
                print("[WARN] color_img 为空，跳过本帧")
                continue

            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

            # 4. 检测 ArUco
            try:
                corners_list, ids, rejected = detector.detectMarkers(gray)
            except cv2.error as e:
                print("[ERROR] detectMarkers 失败:", e)
                print("    gray shape:", gray.shape, "dtype:", gray.dtype)
                continue

            if ids is not None and len(ids) > 0:
                # 绘制检测到的 marker 边框
                cv2.aruco.drawDetectedMarkers(color_img, corners_list, ids)

                # 使用 ArUco 求位姿（注意：marker_length 单位是什么，tvec 就是什么单位）
                marker_length = MARKER_LENGTH_MM  # 这里统一按 mm 来
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners_list, marker_length, camera_matrix, dist_coeffs
                )

                for idx, (rvec, tvec, mid) in enumerate(
                    zip(rvecs, tvecs, ids)
                ):
                    marker_id = int(mid[0])

                    # 在画面上画坐标轴（单位：与 marker_length 一致）
                    cv2.drawFrameAxes(
                        color_img,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        aruco_axis_len,
                    )

                    # ----- 相机坐标系下的位置（mm） -----
                    # tvec: shape (1, 3)，单位 = marker_length 的单位 (mm)
                    p_cam_mm = tvec.reshape(3, 1)  # (3,1)

                    # ----- 转机械臂基坐标系 -----
                    # p_base = R * p_cam + t
                    p_base_mm = R_base_cam @ p_cam_mm + t_base_cam
                    x_b, y_b, z_b = p_base_mm.reshape(3)

                    # ----- 中心点深度（来自深度图） -----
                    corners = corners_list[idx][0]  # (4, 2)
                    u_center = int(round(corners[:, 0].mean()))
                    v_center = int(round(corners[:, 1].mean()))
                    depth_m = depth_frame.get_distance(u_center, v_center)

                    # 打印调试信息
                    print(
                        f"[INFO] Marker id={marker_id} | "
                        f"cam(mm)=({p_cam_mm[0,0]:.1f},{p_cam_mm[1,0]:.1f},{p_cam_mm[2,0]:.1f}) | "
                        f"base(mm)=({x_b:.1f},{y_b:.1f},{z_b:.1f}) | "
                        f"depth={depth_m:.3f} m (u={u_center}, v={v_center})"
                    )

                    # 在画面上叠加文字
                    text_org = (u_center + 5, v_center - 10)
                    cv2.putText(
                        color_img,
                        f"id={marker_id}",
                        text_org,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    text_org2 = (u_center + 5, v_center + 10)
                    cv2.putText(
                        color_img,
                        f"Base({x_b:.0f},{y_b:.0f},{z_b:.0f})mm",
                        text_org2,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    text_org3 = (u_center + 5, v_center + 30)
                    cv2.putText(
                        color_img,
                        f"Z_cam={p_cam_mm[2,0]:.0f}mm depth={depth_m:.3f}m",
                        text_org3,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            # 5. 显示画面
            cv2.imshow("D455 ArUco Pose (Camera + Base)", color_img)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] 已关闭。")


if __name__ == "__main__":
    raise SystemExit(main())
