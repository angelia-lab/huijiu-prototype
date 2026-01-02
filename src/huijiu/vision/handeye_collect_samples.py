# src/huijiu/apps/handeye_collect_samples.py
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
from pymycobot.mycobot280 import MyCobot280

from huijiu.vision.d455 import RealSenseD455
from huijiu.vision.aruco_utils import (
    load_camera_intrinsics,
    get_aruco_dict_and_params,
    ARUCO_MARKER_LENGTH_M,
)
from huijiu.core import config as cfg


@dataclass
class HandEyeSample:
    base_to_ee_coords: List[float]   # [x,y,z,rx,ry,rz] // mm,deg
    cam_rvec: List[float]            # 3
    cam_tvec: List[float]            # 3


OUT_PATH = cfg.CALIB_JSON


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 连接 MyCobot: port={cfg.ROBOT_PORT}, baud={cfg.ROBOT_BAUD}")
    mc = MyCobot280(cfg.ROBOT_PORT, cfg.ROBOT_BAUD)
    time.sleep(2.0)

    # 默认不上电，让你先决定是拖动还是直接用当前姿态
    motor_on = False
    print(
        "[INFO] 机械臂当前电机默认未上电。\n"
        "       在采集窗口中按键：\n"
        "         P : power_on 上电/锁定当前位置\n"
        "         R : release_all_servos 断电/可用手拖动\n"
        "         S : 记录当前这一帧的手眼样本（坐标 + ArUco 位姿）\n"
        "         Q 或 ESC : 退出采集\n"
    )

    # 1) 相机内参 + ArUco
    K, dist_coeffs, width, height = load_camera_intrinsics()
    aruco_dict, aruco_params = get_aruco_dict_and_params()

    # 2) 启动 D455
    cam = RealSenseD455(
        color_resolution=(width, height),
        depth_resolution=(width, height),
        fps=getattr(cfg, "D455_FPS", 30),
        align_to_color=True,
    )
    cam.start()
    print("[INFO] D455 已启动。")

    samples: List[HandEyeSample] = []

    try:
        idx = 0
        while True:
            frame = cam.get_frame()
            color = frame.color_bgr
            if color is None:
                continue

            vis = color.copy()
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=aruco_params
            )
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)

            status_str = f"{'ON' if motor_on else 'OFF'}"
            cv2.putText(
                vis,
                f"samples={len(samples)}  [P 上电] [R 断电] [S 记录] [Q 退出]  motor={status_str}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            cv2.imshow("HandEye collect", vis)
            key = cv2.waitKey(1) & 0xFF

            # ---- 键盘控制 ----
            if key in (ord("q"), 27):
                print("[INFO] 收到退出指令。")
                break

            if key == ord("p"):
                try:
                    mc.power_on()
                    motor_on = True
                    print("[INFO] 已上电 (power_on)，机械臂会锁定当前位置。")
                except Exception as e:
                    print(f"[WARN] power_on 失败: {e}")

            if key == ord("r"):
                try:
                    mc.release_all_servos()
                    motor_on = False
                    print("[INFO] 已断电 (release_all_servos)，可以用手拖动机械臂。")
                except Exception as e:
                    print(f"[WARN] release_all_servos 失败: {e}")

            if key == ord("s"):
                # 记录样本
                if ids is None or len(ids) == 0:
                    print("[WARN] 当前没看到任何 ArUco，无法记录样本。")
                    continue

                # 只取第一个 marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners,
                    ARUCO_MARKER_LENGTH_M,
                    K,
                    dist_coeffs,
                )
                rvec = rvecs[0][0]
                tvec = tvecs[0][0]

                coords = mc.get_coords()
                if not coords or len(coords) != 6:
                    print("[WARN] get_coords() 失败，忽略本次。 coords =", coords)
                    continue

                sample = HandEyeSample(
                    base_to_ee_coords=list(coords),
                    cam_rvec=list(map(float, rvec)),
                    cam_tvec=list(map(float, tvec)),
                )
                samples.append(sample)
                idx += 1
                print(f"[INFO] 已记录样本 {idx}: coords={coords}")

        # 保存所有样本
        data = [asdict(s) for s in samples]
        OUT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[INFO] 共记录 {len(samples)} 组样本，已写入 {OUT_PATH}")

    finally:
        cam.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
