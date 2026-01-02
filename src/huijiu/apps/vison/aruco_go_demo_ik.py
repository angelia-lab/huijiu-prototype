"""
aruco_go_demo_ik.py

功能：
- 启动 D455，实时检测 ArUco 标记；
- 用手眼标定外参，把“相机下的 ArUco 坐标”转换到“机械臂基坐标系”；
- 使用 URDF + ikpy 做逆解：基坐标系下的 (x,y,z) -> 关节角；
- 按 s 键时，让 myCobot 运动到该 ArUco 上方（悬停高度 = HOVER_OFFSET_MM）。

安全提示：
- 第一次跑前务必清空机械臂周围空间；
- 先用比较大的 HOVER_OFFSET_MM（比如 60~80mm）做实验；
- 确保手眼标定是最近且有效的。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np
import pyrealsense2 as rs
from ikpy.chain import Chain
from ikpy.link import URDFLink
from pymycobot.mycobot280 import MyCobot280

from huijiu.core import config as cfg


# ----------------- 路径与参数配置（根据你工程实际情况调整） -----------------

# 相机内参与手眼外参路径（来自你的 config）
CAM_INTRINSICS_PATH: Path = cfg.D455_INSTRINSICS      # e.g. PROJECT_ROOT / "hardware/calib/d455_intrinsics.json"
HANDEYE_PATH: Path        = cfg.OUT_JSON              # e.g. PROJECT_ROOT / "hardware/calib/handeye_result.json"

# 机器人 URDF（你之前已经配置好的）
ROBOT_URDF: Path = cfg.ROBOT_URDF                     # e.g. PROJECT_ROOT / "hardware/urdf/mycobot_280m5_with_gripper_up.urdf"

# 实物 ArUco 的边长（mm）
MARKER_LENGTH_MM: float = 40.0

# 机械臂连接参数
ROBOT_PORT: str = "COM4"      # 如果串口变了，改这里
BAUD: int       = 115200

# 指定我们要“跟踪”的 ArUco ID
TARGET_MARKER_ID: int = 1

# 悬停高度：让机械臂末端停在 ArUco 上方多少 mm（基于 base 坐标系的 Z）
HOVER_OFFSET_MM: float = 40.0

# 工具末端（真正想对齐的“抓夹嘴中心点”）相对于 IK 末端 link
# 在“机械臂基坐标系”下的偏移量（mm）。
# 先设为 [0,0,0]，相当于不补偿；后面我们再做 TCP 标定后，把这里改成实测值即可。
TOOL_OFFSET_BASE_MM = np.array([0.0, 0.0, 0.0], dtype=np.float32)

# 回到“安全姿态”的关节角（单位：度）
HOME_ANGLES = [0, 0, 0, 0, 0, 0]

# 关节运动速度
JOINT_SPEED: int = 20


# ----------------- 工具函数：加载相机内参 & 手眼外参 -----------------

def load_intrinsics() -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 d455_intrinsics.json，兼容几种常见结构：
    - { "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]], "dist_coeffs": [...] }
    - { "fx": ..., "fy": ..., "cx": ..., "cy": ..., "k1":..., "k2":..., "p1":..., "p2":..., "k3":... }
    - { "fx":..., "fy":..., "cx":..., "cy":..., "dist":[...] }
    """
    data = json.loads(CAM_INTRINSICS_PATH.read_text(encoding="utf-8"))

    # 相机矩阵
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

    # 畸变系数
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
    """
    读取手眼标定结果，约定键名为：
    - R_base_cam: 3x3 旋转矩阵
    - t_base_cam: 3 元向量，单位 mm

    给出的意义：
      相机坐标系中的一点 P_cam（单位 mm），
      对应到机械臂基坐标系为：
         P_base = R_base_cam @ P_cam + t_base_cam
    """
    data = json.loads(HANDEYE_PATH.read_text(encoding="utf-8"))
    R = np.array(data["R_base_cam"], dtype=np.float32)                # 3x3
    t = np.array(data["t_base_cam"], dtype=np.float32).reshape(3, 1)  # 3x1, 单位 mm

    print(f"[INFO] 手眼外参加载成功。")
    print(f"       R_base_cam =\n{R}")
    print(f"       t_base_cam = {t.ravel()}")
    return R, t


# ----------------- ArUco 兼容封装 -----------------

def create_aruco_detector():
    """兼容不同 OpenCV 版本的 ArUco 检测 API。"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # OpenCV 旧版: DetectorParameters_create + cv2.aruco.detectMarkers
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        params = cv2.aruco.DetectorParameters_create()
        detector = None  # 用 cv2.aruco.detectMarkers
        print("[INFO] 使用旧版 ArUco API: DetectorParameters_create + cv2.aruco.detectMarkers")
    else:
        # OpenCV 新版: DetectorParameters + ArucoDetector.detectMarkers
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        print("[INFO] 使用新版 ArUco API: DetectorParameters + ArucoDetector.detectMarkers")

    return aruco_dict, params, detector


def detect_markers(gray, aruco_dict, params, detector):
    """封装 detectMarkers 调用，兼容旧/新 API。"""
    if detector is None:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=params
        )
    else:
        corners, ids, rejected = detector.detectMarkers(gray)
    return corners, ids, rejected


# ----------------- IK 相关：URDF -> Chain -> 6 关节角 -----------------

def build_ik_chain(urdf_path: Path) -> Tuple[Chain, List[int]]:
    """
    用 ikpy 从 URDF 构建 Kinematic Chain，并找出“机械臂 6 个关节”的 DOF 索引。

    返回：
        chain: ikpy.chain.Chain
        arm_dof_indices: 长度为 6 的整数列表，对应 J1~J6 在 chain.inverse_kinematics
                         返回向量中的下标。
    """
    print(f"[INFO] 使用 URDF 构建 IK Chain: {urdf_path}")

    # 显式指定 base_elements=['g_base']，避免默认找 'base_link'
    chain = Chain.from_urdf_file(
        str(urdf_path),
        base_elements=["g_base"],
    )

    print(f"[INFO] Chain 链接数 = {len(chain.links)}")
    print("[INFO] 链上各 link / joint：")
    for idx, link in enumerate(chain.links):
        jt = getattr(link, "joint_type", None)
        if jt is None:
            jt = "fixed"
        mark = ""
        print(f"  [{idx:02d}] name={link.name:30s} joint_type={jt}{mark}")

    # 找出所有 joint_type != 'fixed' 的 link，它们就是 DOF
    dof_indices: List[int] = []
    for i, link in enumerate(chain.links):
        jt = getattr(link, "joint_type", None)
        if jt is not None and jt != "fixed":
            dof_indices.append(i)

    print(f"[INFO] 总 DOF（joint_type != 'fixed'）= {len(dof_indices)}")
    print(f"[INFO] 所有 DOF indices = {dof_indices}")

    # 约定：前 6 个 DOF 对应 myCobot J1~J6（抓夹的几个 DOF 不参与 IK 控制）
    if len(dof_indices) < 6:
        raise RuntimeError(f"可动关节不足 6 个，当前 DOF={len(dof_indices)}: {dof_indices}")

    arm_dof_indices = dof_indices[:6]
    print(f"[INFO] 机械臂 6 关节 DOF indices = {arm_dof_indices}")

    return chain, arm_dof_indices


def solve_ik_to_joint_angles(
    chain: Chain,
    arm_dof_indices: List[int],
    target_xyz_m: np.ndarray,
) -> List[float]:
    """
    使用 ikpy 做 IK，让“末端坐标系原点”到达 target_xyz_m（单位：m，基坐标系）。

    说明：
    - 这里只约束位置（x,y,z），不强行约束姿态；
    - chain.inverse_kinematics 返回的是“全链条各 DOF 的角度（弧度）”，
      我们只抽取前 6 个指定 DOF，并做弧度 -> 度的转换。
    """
    if target_xyz_m.shape != (3,):
        target_xyz_m = target_xyz_m.reshape(3,)

    print(f"[IK] 目标位置（m）= {target_xyz_m}")

    # 你的 ikpy 版本 API 是：inverse_kinematics(target_position, ...)
    full_angles_rad = chain.inverse_kinematics(target_xyz_m)
    print(f"[IK] full angles (rad) = {full_angles_rad}")

    arm_angles_rad = [full_angles_rad[i] for i in arm_dof_indices]
    arm_angles_deg = [math.degrees(a) for a in arm_angles_rad]

    print(f"[IK] 机械臂 6 关节角（deg, J1~J6）= {arm_angles_deg}")
    return arm_angles_deg


# ----------------- 机械臂控制相关 -----------------

def connect_robot() -> MyCobot280:
    print(f"[INFO] 正在连接机械臂: port={ROBOT_PORT}, baud={BAUD}")
    mc = MyCobot280(ROBOT_PORT, BAUD)
    return mc


def robot_go_home(mc: MyCobot280):
    """上电并回到一个安全的 home 姿态。"""
    print("[INFO] 上电 servos ...")
    mc.power_on()
    time.sleep(1.0)

    print(f"[INFO] 回到 HOME 姿态: {HOME_ANGLES}")
    mc.send_angles(HOME_ANGLES, JOINT_SPEED)
    time.sleep(4.0)  # 留足够时间让机械臂运动完成


def send_angles_to_robot(mc: MyCobot280, joint_angles_deg: List[float]):
    """
    把 6 个关节角（度）下发给 myCobot。
    """
    if len(joint_angles_deg) != 6:
        raise ValueError(f"关节角长度必须为 6，当前={len(joint_angles_deg)}")

    print(f"[CTRL] send_angles = {joint_angles_deg}, speed={JOINT_SPEED}")
    mc.send_angles(joint_angles_deg, JOINT_SPEED)

    # 简单观测 5s，看一下角度是否在变化
    t0 = time.time()
    while time.time() - t0 < 5.0:
        ang = mc.get_angles()
        print(f"[CTRL] t={time.time()-t0:.1f}s, angles={ang}")
        time.sleep(0.5)


# ----------------- 主流程 -----------------

def main():
    # 1. 加载相机内参 & 手眼外参
    cam_mtx, dist = load_intrinsics()
    R_bc, t_bc    = load_handeye()

    # 2. 构建 IK chain
    chain, arm_dof_indices = build_ik_chain(ROBOT_URDF)

    # 3. 启动 D455
    pipeline = rs.pipeline()
    rs_cfg   = rs.config()
    rs_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(rs_cfg)
    print("[INFO] D455 已启动。")
    print("[INFO] 操作说明：")
    print("       - 画面中看到 ArUco 码时，会显示 cam/base 坐标")
    print("       - 键盘按 's'：用 IK 让机械臂运动到 ArUco 上方（含悬停+工具偏移）")
    print("       - 键盘按 'q' 或 ESC：退出")

    # 4. 连接机械臂 & 回 home
    mc = connect_robot()
    robot_go_home(mc)

    # 5. 创建 ArUco 检测器
    aruco_dict, aruco_params, aruco_detector = create_aruco_detector()
    marker_length_m = MARKER_LENGTH_MM / 1000.0  # mm → m（OpenCV 习惯用米）

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            gray  = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            # ArUco 检测
            corners, ids, _ = detect_markers(gray, aruco_dict, aruco_params, aruco_detector)

            target_base_point_mm: Optional[np.ndarray] = None

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(color, corners, ids)

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_length_m, cam_mtx, dist
                )

                for i, marker_id in enumerate(ids.flatten()):
                    rvec = rvecs[i].reshape(3, 1)
                    tvec_m = tvecs[i].reshape(3, 1)          # 单位：m（相机坐标）
                    tvec_mm = tvec_m * 1000.0                # m → mm

                    x_c, y_c, z_c = tvec_mm.flatten().tolist()

                    # P_base = R_bc * P_cam + t_bc
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

                    # 把目标 ID 的点存下来，方便按键时用
                    if int(marker_id) == TARGET_MARKER_ID:
                        target_base_point_mm = p_base

            # 在画面上给用户一点指引
            cv2.putText(
                color,
                f"Target ID: {TARGET_MARKER_ID}  |  Press 's' = go above marker (IK), 'q' = quit",
                (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255),
                1,
            )

            cv2.imshow("aruco_go_demo_ik", color)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q 或 ESC
                break

            if key == ord("s"):
                if target_base_point_mm is None:
                    print(f"[WARN] 当前画面中没有检测到目标 ID={TARGET_MARKER_ID}，忽略本次 's' 操作。")
                    continue

                # target_base_point_mm: ArUco 中心在“机械臂基坐标系”下的位置（mm, 3x1）
                marker_base = target_base_point_mm.reshape(3, 1)
                x_m, y_m, z_m = marker_base.flatten().tolist()

                # 先在 Z 方向加一个悬停高度
                hover_base = marker_base.copy()
                hover_base[2, 0] += HOVER_OFFSET_MM

                # 再加上工具末端 TCP 的偏移（在“基坐标系”下定义）
                # 直观理解：我们想让「抓夹真正末端点」落在 ArUco 上方，
                # 所以要把 IK 的目标点往相反方向平移一个 TOOL_OFFSET_BASE_MM。
                ik_target_base = hover_base + TOOL_OFFSET_BASE_MM.reshape(3, 1)

                x_b, y_b, z_b = ik_target_base.flatten().tolist()

                print(f"[INFO] 捕获到目标 ArUco (ID={TARGET_MARKER_ID}) 基坐标："
                      f"({x_m:.1f}, {y_m:.1f}, {z_m:.1f}) mm")
                print(
                    f"[INFO] 计划让 TCP 悬停在：({x_b:.1f}, {y_b:.1f}, {z_b:.1f}) mm "
                    f"(HOVER_OFFSET={HOVER_OFFSET_MM:.1f}mm, TOOL_OFFSET={TOOL_OFFSET_BASE_MM})"
                )

                # mm -> m
                target_xyz_m = (ik_target_base / 1000.0).flatten()

                try:
                    joint_angles_deg = solve_ik_to_joint_angles(
                        chain,
                        arm_dof_indices,
                        target_xyz_m,
                    )
                except Exception as e:
                    print(f"[ERROR] IK 求解失败：{e}")
                    continue

                try:
                    send_angles_to_robot(mc, joint_angles_deg)
                except Exception as e:
                    print(f"[ERROR] 下发关节角失败：{e}")

    finally:
        print("[INFO] 收到退出请求，正在清理资源 ...")
        pipeline.stop()
        cv2.destroyAllWindows()

        # 出于安全考虑，这里不做 release_all_servos，避免突然断电掉落
        try:
            print("[INFO] 保持当前关节上电姿态，你可以手动回 home 或关闭电源。")
        except Exception:
            pass


if __name__ == "__main__":
    main()
