# src/huijiu/apps/aruco_debug_demo.py
from __future__ import annotations

import cv2

from huijiu.vision.d455 import RealSenseD455
from huijiu.vision.aruco_utils import (
    load_camera_intrinsics,
    get_aruco_dict_and_params,
    ARUCO_MARKER_LENGTH_M,
)


def main() -> int:
    K, dist_coeffs, width, height = load_camera_intrinsics()
    aruco_dict, aruco_params = get_aruco_dict_and_params()

    cam = RealSenseD455(
        color_resolution=(width, height),
        depth_resolution=(width, height),
        fps=30,
        align_to_color=True,
    )
    cam.start()
    print("[INFO] D455 已启动。按 Q / ESC 退出。")

    try:
        while True:
            frame = cam.get_frame()
            color = frame.color_bgr
            if color is None:
                continue

            vis = color.copy()
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = cv2.aruco.detectMarkers(
                gray,
                aruco_dict,
                parameters=aruco_params,
            )

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                # 估算姿态，只是为了验证
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners,
                    ARUCO_MARKER_LENGTH_M,
                    K,
                    dist_coeffs,
                )
                for i, marker_id in enumerate(ids.flatten()):
                    rvec = rvecs[i]
                    tvec = tvecs[i]
                    cv2.drawFrameAxes(
                        vis,
                        K,
                        dist_coeffs,
                        rvec,
                        tvec,
                        ARUCO_MARKER_LENGTH_M * 2.0,
                    )
                text = f"Detected ids: {ids.flatten().tolist()}"
            else:
                text = "No markers"

            cv2.putText(
                vis,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("ArUco debug", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] 已关闭。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
