from __future__ import annotations

import time
from typing import List, Optional, Union

from pymycobot.mycobot280 import MyCobot280

PORT = "COM4"
BAUD = 115200

# 原始示教得到的虎口姿态
RAW_HUKOU_REF_ANGLES: List[float] =[-168.04, -34.80, -18.72, 5.18, 3.07, 139.48]

# MyCobot280 SDK 要求的默认范围是 -168 ~ 168（从异常信息里来的）
ANGLE_MIN = -168.0
ANGLE_MAX = 168.0


def clamp_angle(a: float, amin: float = ANGLE_MIN, amax: float = ANGLE_MAX) -> float:
    """把单个角度限制在 [amin, amax] 之间。"""
    if a < amin:
        return amin
    if a > amax:
        return amax
    return a


def main() -> int:
    # 先把示教角度“夹紧”到合法范围内
    hukou_angles = [clamp_angle(a) for a in RAW_HUKOU_REF_ANGLES]
    print(f"[INFO] 原始示教角度: {RAW_HUKOU_REF_ANGLES}")
    print(f"[INFO] 夹紧后的角度: {hukou_angles}")

    print(f"[INFO] 连接 MyCobot 280 M5: port={PORT}, baud={BAUD} ...")
    mc = MyCobot280(PORT, BAUD)
    time.sleep(2.0)

    print("[INFO] 上电使能电机（power_on）...")
    try:
        mc.power_on()
    except Exception as e:
        print(f"[WARN] power_on 失败，可忽略: {e}")
    time.sleep(1.0)

    # 打印当前关节角
    try:
        cur: Union[int, List[float]] = mc.get_angles()
    except Exception as e:
        cur = None
        print(f"[WARN] get_angles() 异常: {e}")
    else:
        print(f"[INFO] 当前关节角: {cur}")

    print(f"[INFO] 准备移动到虎口参考姿态（合法范围内）: {hukou_angles}")
    speed = 40
    mc.send_angles(hukou_angles, speed)

    time.sleep(4.0)

    try:
        cur2: Union[int, List[float]] = mc.get_angles()
    except Exception as e:
        cur2 = None
        print(f"[WARN] get_angles() 异常: {e}")
    else:
        print(f"[INFO] 到位后的关节角: {cur2}")

    print("[INFO] 完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
