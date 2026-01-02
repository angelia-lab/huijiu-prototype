"""
handeye_solve.py

读取 handeye_samples.json，使用 OpenCV 进行手眼标定，
计算相机坐标系到机械臂基坐标系的变换 T_base_cam。

使用前请根据你的 JSON 结构修改 load_samples() 里的字段映射。
"""

import json
from pathlib import Path
import numpy as np
import cv2
from huijiu.core import config as cfg

CALIB_JSON = cfg.CALIB_JSON
OUT_JSON = cfg.OUT_JSON


# ====== 工具函数：角度/欧拉角 -> 旋转矩阵 ======

def euler_deg_to_Rxyz(rx_deg, ry_deg, rz_deg):
    """
    示例：假设 handeye_samples.json 里保存的是 XYZ 顺序的欧拉角（单位：度）。
    如果你在 handeye_collect_samples.py 里用的是其他约定（比如 ZYX），
    请保证这里和采样时一致。
    """
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0,           1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,          1]])
    # 这里用 R = Rz * Ry * Rx 只是一个常见约定，必须与你采样时保持一致
    R = Rz @ Ry @ Rx
    return R


# ====== 第 1 步：读取样本并构造 OpenCV 需要的输入 ======

def load_samples(path: Path):
    """
    读取 JSON，返回：
      - R_gripper2base_list: [N, 3, 3]
      - t_gripper2base_list: [N, 3, 1]
      - R_target2cam_list:   [N, 3, 3]
      - t_target2cam_list:   [N, 3, 1]
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []

    for i, s in enumerate(raw):
        # === 根据你自己的 JSON 结构修改这里 ===
        # 假设结构类似：
        # {
        #   "robot":  [x, y, z, rx_deg, ry_deg, rz_deg],   # 端执行器在 base 下
        #   "marker_rvec": [r1, r2, r3],                   # ArUco 在相机坐标系下（rvec）
        #   "marker_tvec": [t1, t2, t3]                    # 同上（tvec）
        # }

        # TODO: 把下面这 3 行里的字段名改成你 JSON 里的实际字段
        robot_pose = s["base_to_ee_coords"]              # 例如 [x, y, z, rx, ry, rz]
        marker_rvec = np.array(s["cam_rvec"], dtype=np.float64)
        marker_tvec = np.array(s["cam_tvec"], dtype=np.float64).reshape(3, 1)

        # 1) 机械臂端执行器在 base 下的位姿：T_base_gripper
        x, y, z, rx_deg, ry_deg, rz_deg = robot_pose
        t_base_gripper = np.array([[x], [y], [z]], dtype=np.float64)
        R_base_gripper = euler_deg_to_Rxyz(rx_deg, ry_deg, rz_deg)

        # OpenCV 需要的是 R_gripper2base / t_gripper2base，即 T_gripper_base = T_base_gripper 的逆
        R_gripper2base = R_base_gripper.T
        t_gripper2base = -R_gripper2base @ t_base_gripper

        R_gripper2base_list.append(R_gripper2base)
        t_gripper2base_list.append(t_gripper2base)

        # 2) 标定板（target）在相机坐标系下的位姿：T_cam_target
        R_cam_target, _ = cv2.Rodrigues(marker_rvec)      # rvec -> R_cam_target
        t_cam_target = marker_tvec                        # [3,1]

        # OpenCV 需要的是 R_target2cam / t_target2cam = T_target_cam = T_cam_target 的逆
        R_target2cam = R_cam_target.T
        t_target2cam = -R_target2cam @ t_cam_target

        R_target2cam_list.append(R_target2cam)
        t_target2cam_list.append(t_target2cam)

    print(f"[INFO] 从 {path} 读取样本 {len(R_gripper2base_list)} 组")
    return (
        R_gripper2base_list,
        t_gripper2base_list,
        R_target2cam_list,
        t_target2cam_list,
    )


def calibrate_handeye(Rg2b, tg2b, Rt2c, tt2c):
    """
    使用 OpenCV 的手眼标定算法，计算 T_base_cam:
      base_T_cam = [R, t]
    """
    # OpenCV 需要 numpy 数组列表
    R_gripper2base = [np.asarray(R, dtype=np.float64) for R in Rg2b]
    t_gripper2base = [np.asarray(t, dtype=np.float64) for t in tg2b]
    R_target2cam   = [np.asarray(R, dtype=np.float64) for R in Rt2c]
    t_target2cam   = [np.asarray(t, dtype=np.float64) for t in tt2c]

    # 可选算法：Tsai、Park、Horaud 等，这里用 Tsai（默认）
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    # 我们更关心的是 base_T_cam（相机在 base 坐标系下的位姿）
    # 手眼模型有多种形式：这里假设是「外眼」（相机固定在世界/基座附近），
    # 上面求的是 T_cam_gripper。结合一组样本的 T_base_gripper 可以求 T_base_cam，
    # 简单做法：取第一个样本：
    Rg2b0 = R_gripper2base[0]
    tg2b0 = t_gripper2base[0]

    # base_T_gripper = (R_bg, t_bg) = (R_g2b^T, -R_g2b^T * t_g2b)
    R_bg = Rg2b0.T
    t_bg = -R_bg @ tg2b0

    # cam_T_gripper = (R_cg, t_cg) = (R_cam2gripper, t_cam2gripper)
    R_cg = R_cam2gripper
    t_cg = t_cam2gripper

    # base_T_cam = base_T_gripper * gripper_T_cam
    # 其中 gripper_T_cam = (R_gc, t_gc) = (R_cg^T, -R_cg^T * t_cg)
    R_gc = R_cg.T
    t_gc = -R_gc @ t_cg

    R_bc = R_bg @ R_gc
    t_bc = R_bg @ t_gc + t_bg

    return R_bc, t_bc, R_cam2gripper, t_cam2gripper


def save_result(R_bc, t_bc):
    """
    保存 base_T_cam 结果到 JSON，方便后续加载。
    """
    result = {
        "R_base_cam": R_bc.tolist(),
        "t_base_cam": t_bc.reshape(3).tolist(),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[INFO] 手眼标定结果已写入: {OUT_JSON}")


def main():
    (
        Rg2b,
        tg2b,
        Rt2c,
        tt2c,
    ) = load_samples(CALIB_JSON)

    R_bc, t_bc, R_cg, t_cg = calibrate_handeye(Rg2b, tg2b, Rt2c, tt2c)

    print("[INFO] base_T_cam 旋转矩阵 R_base_cam =")
    print(R_bc)
    print("[INFO] base_T_cam 平移向量 t_base_cam =")
    print(t_bc.reshape(3))

    save_result(R_bc, t_bc)


if __name__ == "__main__":
    main()
