"""
aruco_go_demo.py

功能：
- 启动 D455，实时检测 ArUco 标记；
- 用手眼标定外参，把“相机下的 ArUco 坐标”转换到“机械臂基坐标系”；
- 按 s 键时，让 myCobot 运动到该 ArUco 上方（悬停高度 = HOVER_OFFSET_MM）。

安全提示：
- 第一次跑前务必清空机械臂周围空间；
- 先用比较大的 HOVER_OFFSET_MM（比如 100mm）做实验；
- 确保手眼标定是最近且有效的。
"""

import json
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import pyrealsense2 as rs
from pymycobot.mycobot280 import MyCobot280

from huijiu.core import config as cfg

# ----------------- 路径与参数配置 -----------------

CAM_INTRINSICS_PATH: Path = cfg.D455_INSTRINSICS
HANDEYE_PATH: Path        = cfg.OUT_JSON

MARKER_LENGTH_MM: float = 40.0

ROBOT_PORT: str = "COM4"
BAUD: int       = 115200

TARGET_MARKER_ID: int = 2

HOVER_OFFSET_MM: float = 150.0

HOME_ANGLES = [0, 0, 0, 0, 0, 0]

MOVE_SPEED: int = 20
MOVE_MODE: int  = 1   # <<< 关键修改：强制用 mode=1（大多数 280 固件更稳定）


# ----------------- 相机内参 & 手眼外参 -----------------

def load_intrinsics() -> Tuple[np.ndarray, np.ndarray]:
    data = json.loads(CAM_INTRINSICS_PATH.read_text(encoding="utf-8"))

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

    if "dist_coeffs" in data:
        dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float32).reshape(-1)
    elif "dist" in data:
        dist_coeffs = np.array(data["dist"], dtype=np.float32).reshape(-1)
    elif "k1" in data:
        k1, k2 = data["k1"], data["k2"]
        p1, p2 = data["p1"], data["p2"]
        k3 = data.get("k3", 0.0)
        dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    else:
        dist_coeffs = np.zeros(5, dtype=np.float32)

    print(f"[INFO] 相机内参加载成功: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print(f"[INFO] 畸变系数 dist_coeffs={dist_coeffs}")
    return camera_matrix, dist_coeffs


def load_handeye() -> Tuple[np.ndarray, np.ndarray]:
    data = json.loads(HANDEYE_PATH.read_text(encoding="utf-8"))
    R = np.array(data["R_base_cam"], dtype=np.float32)
    t = np.array(data["t_base_cam"], dtype=np.float32).reshape(3, 1)
    print(f"[INFO] 手眼外参加载成功。")
    print(f"       R_base_cam =\n{R}")
    print(f"       t_base_cam = {t.ravel()}")
    return R, t


# ----------------- ArUco 兼容封装 -----------------

def create_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    if hasattr(cv2.aruco, "DetectorParameters_create"):
        params = cv2.aruco.DetectorParameters_create()
        detector = None
        print("[INFO] 使用旧版 ArUco API: DetectorParameters_create + cv2.aruco.detectMarkers")
    else:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        print("[INFO] 使用新版 ArUco API: DetectorParameters + ArucoDetector.detectMarkers")

    return aruco_dict, params, detector


def detect_markers(gray, aruco_dict, params, detector):
    if detector is None:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=params
        )
    else:
        corners, ids, rejected = detector.detectMarkers(gray)
    return corners, ids, rejected


# ----------------- 机械臂控制 -----------------

def connect_robot() -> MyCobot280:
    print(f"[INFO] 正在连接机械臂: port={ROBOT_PORT}, baud={BAUD}")
    mc = MyCobot280(ROBOT_PORT, BAUD)

    # <<< 关键修改：无论当前 fresh_mode 是多少，都强制切到 1
    import time
    try:
        fm_before = mc.get_fresh_mode()
        print(f"[INFO] get_fresh_mode() before = {fm_before}")
    except Exception as e:
        print(f"[WARN] get_fresh_mode() 异常：{e}")
        fm_before = None

    try:
        print("[INFO] set_fresh_mode(1) 启用坐标模式 ...")
        mc.set_fresh_mode(1)
        time.sleep(0.5)
        try:
            fm_after = mc.get_fresh_mode()
            print(f"[INFO] get_fresh_mode() after  = {fm_after}")
        except Exception as e2:
            print(f"[WARN] get_fresh_mode() after 异常：{e2}")
    except Exception as e:
        print(f"[WARN] set_fresh_mode(1) 异常：{e}")

    return mc


def robot_go_home(mc: MyCobot280):
    import time
    print("[INFO] 上电 servos ...")
    mc.power_on()
    time.sleep(1.0)

    print(f"[INFO] 回到 HOME 姿态: {HOME_ANGLES}")
    mc.send_angles(HOME_ANGLES, 20)
    time.sleep(4.0)


def robot_move_to_base_point(mc: MyCobot280, x_b: float, y_b: float, z_b: float):
    import time

    current = mc.get_coords()
    print(f"[DEBUG] before move get_coords() = {current}")
    if not current or len(current) != 6:
        print(f"[WARN] get_coords() 异常，使用默认姿态。current={current}")
        current = [0.0, 0.0, 200.0, -90.0, 0.0, -90.0]

    x0, y0, z0, rx, ry, rz = current

    # 限幅
    x_clamped = float(np.clip(x_b, -270.0, 270.0))
    y_clamped = float(np.clip(y_b, -270.0, 270.0))
    z_clamped = float(np.clip(z_b,  50.0,  420.0))

    target = [x_clamped, y_clamped, z_clamped, rx, ry, rz]
    print(f"[DEBUG] raw base target (mm): x={x_b:.2f}, y={y_b:.2f}, z={z_b:.2f}")
    print(f"[DEBUG] clamped base target (mm): x={x_clamped:.2f}, y={y_clamped:.2f}, z={z_clamped:.2f}")
    print(f"[DEBUG] target coords (after clamp) = {target}")
    print(f"[DEBUG] calling send_coords(..., speed={MOVE_SPEED}, mode={MOVE_MODE}) ...")

    try:
        mc.send_coords(target, MOVE_SPEED, MOVE_MODE)
    except Exception as e:
        print(f"[ERROR] send_coords 异常：{e}")
        return

    # 简单观测 5 秒，看返回的 coords 有没有变化
    for i in range(10):
        time.sleep(0.5)
        c = mc.get_coords()
        print(f"[DEBUG] t={(i+1)*0.5:.1f}s, get_coords() = {c}")

    print("[INFO] 已发送运动指令。")


# ----------------- 主流程 -----------------

def main():
    cam_mtx, dist = load_intrinsics()
    R_bc, t_bc    = load_handeye()

    pipeline = rs.pipeline()
    rs_cfg   = rs.config()
    rs_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(rs_cfg)
    print("[INFO] D455 已启动。")
    print("[INFO] 操作说明：")
    print("       - 画面中看到 ArUco 码时，会显示 cam/base 坐标")
    print("       - 键盘按 's'：让机械臂运动到目标 ArUco 上方悬停")
    print("       - 键盘按 'q' 或 ESC：退出")

    mc = connect_robot()
    robot_go_home(mc)

    aruco_dict, aruco_params, aruco_detector = create_aruco_detector()
    marker_length_m = MARKER_LENGTH_MM / 1000.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            gray  = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = detect_markers(gray, aruco_dict, aruco_params, aruco_detector)

            target_base_point_mm: Optional[np.ndarray] = None

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(color, corners, ids)

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_length_m, cam_mtx, dist
                )

                for i, marker_id in enumerate(ids.flatten()):
                    rvec   = rvecs[i].reshape(3, 1)
                    tvec_m = tvecs[i].reshape(3, 1)
                    tvec_mm = tvec_m * 1000.0

                    x_c, y_c, z_c = tvec_mm.flatten().tolist()

                    p_base = R_bc @ tvec_mm + t_bc
                    x_b, y_b, z_b = p_base.flatten().tolist()

                    text = (
                        f"id={marker_id} "
                        f"cam(mm)=({x_c:.1f},{y_c:.1f},{z_c:.1f}) "
                        f"base(mm)=({x_b:.1f},{y_b:.1f},{z_b:.1f})"
                    )
                    cv2.putText(
                        color, text, (10, 30 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                    )

                    if int(marker_id) == TARGET_MARKER_ID:
                        target_base_point_mm = p_base

            cv2.putText(
                color,
                f"Target ID: {TARGET_MARKER_ID}  |  Press 's' = go above marker, 'q' = quit",
                (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255),
                1,
            )

            cv2.imshow("aruco_go_demo", color)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("s"):
                if target_base_point_mm is None:
                    print(f"[WARN] 当前画面中没有检测到目标 ID={TARGET_MARKER_ID}，忽略本次 's' 操作。")
                    continue

                x_b, y_b, z_b = target_base_point_mm.flatten().tolist()
                hover_z = HOVER_OFFSET_MM

                print(f"[INFO] 捕获到目标 ArUco (ID={TARGET_MARKER_ID}) 基坐标："
                      f"({x_b:.1f}, {y_b:.1f}, {z_b:.1f}) mm")
                print(f"[INFO] 计划运动到悬停点：({x_b:.1f}, {y_b:.1f}, {hover_z:.1f}) mm")

                robot_move_to_base_point(mc, x_b, y_b, hover_z)

    finally:
        print("[INFO] 收到退出请求，正在清理资源 ...")
        pipeline.stop()
        cv2.destroyAllWindows()

        try:
            mc.release_all_servos()
            print("[INFO] 已 release_all_servos。")
        except Exception as e:
            print(f"[WARN] 释放伺服时出现异常：{e}")


if __name__ == "__main__":
    main()
