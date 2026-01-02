# src/huijiu/core/orchestrator.py
import cv2
import time

from huijiu.core import config as cfg
from huijiu.vision.d455 import RealSenseD455
from huijiu.vision.hands import HandHukouTracker
from huijiu.sensors.backends.mcp2221_easy import MCP2221EasyBackend
from huijiu.sensors.manager import SensorManager
from huijiu.robot.mycobot280_m5 import MyCobot280Controller, RobotControlInputs
from huijiu.safety.state import VisionState, SensorState, SystemState
from huijiu.safety.rules import evaluate_safety


class IntegratedOrchestrator:
    """
    把 vision + sensors + robot 串起来的主控类。
    """

    def __init__(self) -> None:
        # 1. 视觉：D455
        self.cam = RealSenseD455(
            color_resolution=(640, 480),
            depth_resolution=(640, 480),
            fps=30,
            align_to_color=True,
        )
        self.cam.start()
        print(f"[INFO] D455 已启动，控制手目标: {cfg.TARGET_HAND_LABEL}, "
              f"镜像模式: {getattr(cfg, 'HAND_VIEW_MIRRORED', False)}")

        # 手部 + 虎口识别
        self._hukou_tracker = HandHukouTracker()

        # 2. 传感器：MCP2221 + SensorManager
        backend = MCP2221EasyBackend(bus_speed=cfg.I2C_BUS_SPEED)
        self.sensors = SensorManager(
            backend=backend,
            mlx_addr=cfg.MLX90614_ADDR,
            vl53_addr=cfg.VL53L0X_ADDR,
        )

        # 3. 机械臂
        self.robot = MyCobot280Controller(cfg.ROBOT_PORT, cfg.ROBOT_BAUD)
        self._last_time = time.time()

    def run_loop(self) -> None:
        try:
            while True:
                now = time.time()
                dt = now - self._last_time
                self._last_time = now

                frame = self.cam.get_frame()
                color = frame.color_bgr
                if color is None:
                    continue

                # ---- 1) 用 HandHukouTracker 做手 + 虎口识别 ---- #
                det = self._hukou_tracker.process(color)
                vis = det.annotated
                hx, hy = det.hx, det.hy
                has_hand = det.has_hand

                h, w, _ = vis.shape

                # 画参考中心点（校准用）
                cv2.drawMarker(
                    vis,
                    (cfg.HUKOU_REF_U, cfg.HUKOU_REF_V),
                    (0, 255, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=12,
                    thickness=2,
                )

                # ---- 2) 读温度 & 距离 ---- #
                readings = self.sensors.read_all()
                temp_c = readings.temp_c
                dist_mm = readings.distance_mm

                if temp_c is not None:
                    cv2.putText(
                        vis,
                        f"T={temp_c:.1f}C",
                        (10, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                if dist_mm is not None:
                    cv2.putText(
                        vis,
                        f"D={dist_mm:.0f}mm",
                        (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                # ---- 3) 安全规则 ---- #
                sys_state = SystemState(
                    vision=VisionState(has_hand=has_hand, hx=hx, hy=hy),
                    sensors=SensorState(temp_c=temp_c, distance_mm=dist_mm),
                )
                decision = evaluate_safety(sys_state)

                if not decision.ok and decision.need_emergency_retreat:
                    print(f"\n[SAFETY] {decision.reason} -> emergency_retreat()")
                    self.robot.emergency_retreat()
                else:
                    # ---- 4) 正常控制 ---- #
                    if has_hand or (temp_c is not None) or (dist_mm is not None):
                        inputs = RobotControlInputs(
                            hx=hx,
                            hy=hy,
                            temp_c=temp_c,
                            dist_mm=dist_mm,
                            dt=dt,
                        )
                        self.robot.apply_control(inputs)
                    else:
                        self.robot.decay_to_base()

                # ---- 5) 显示 ---- #
                cv2.imshow("Integrated Safe Loop", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("\n[INFO] 收到退出指令，正在关闭...")
                    break

        finally:
            self._hukou_tracker.close()
            self.cam.stop()
            self.sensors.close()
            cv2.destroyAllWindows()
            print("[INFO] Orchestrator 已关闭。")
