"""
简单 MyCobot 280 M5 自测 Demo

功能：
- 连接机械臂
- 打印固件版本 & 当前关节角
- 运动到一个安全初始姿态
- 做一个小范围“点头 + 摇头”动作
"""

from __future__ import annotations

import time
from typing import List

from pymycobot.mycobot280 import MyCobot280


def wait_and_print_angles(mc: MyCobot280, desc: str, sleep_s: float = 2.0) -> None:
    """等待一会儿，然后读取并打印当前关节角。"""
    time.sleep(sleep_s)
    try:
        angles: List[float] = mc.get_angles()
        print(f"{desc} 关节角：{angles}")
    except Exception as e:
        print(f"[WARN] 读取关节角失败（{desc}）：{e}")


def main() -> int:
    # 1. 修改为你的实际串口号
    port = "COM4"      # TODO: 改成你的串口，比如 "COM4"
    baud = 115200

    print(f"[INFO] 正在连接 MyCobot 280 M5: port={port}, baud={baud} ...")
    mc = MyCobot280(port, baud)
    time.sleep(2.0)  # 给一点时间让串口稳定

    try:
        # 2. 打印固件版本
        try:
            version = mc.get_version()
        except Exception as e:
            version = f"获取失败: {e}"
        print(f"[INFO] 固件版本: {version}")

        # 3. 上电使能
        print("[INFO] 上电使能电机（power_on）...")
        mc.power_on()
        time.sleep(1.0)

        # 4. 打印当前关节角
        wait_and_print_angles(mc, desc="初始")

        # 5. 运动到一个较安全的初始姿态（需要根据你当前机械位置确认是否安全）
        #    格式：[J1, J2, J3, J4, J5, J6]，单位：度
        safe_angles = [0, 0, 90, 0, 90, 0]
        print(f"[INFO] 移动到安全初始姿态: {safe_angles}")
        mc.send_angles(safe_angles, 40)  # 速度 0~100，建议 20~50 之间
        wait_and_print_angles(mc, desc="到达安全姿态")

        # 6. 做一个“小点头 + 摇头”动作
        print("[INFO] 开始做测试动作（小范围点头 + 摇头）...")

        # 以当前角度为基准
        base_angles = mc.get_angles()
        if not base_angles or len(base_angles) != 6:
            base_angles = safe_angles
        print(f"[INFO] 测试动作基准角度: {base_angles}")

        # 6.1 关节 2（类似肩/肘）小幅度上下点头
        for i in range(2):
            a = base_angles.copy()
            a[1] = a[1] + 10   # J2 +10°
            print(f"[INFO] 点头 上: {a}")
            mc.send_angles(a, 40)
            time.sleep(2.0)

            a = base_angles.copy()
            a[1] = a[1] - 10   # J2 -10°
            print(f"[INFO] 点头 下: {a}")
            mc.send_angles(a, 40)
            time.sleep(2.0)

        # 回到基准
        mc.send_angles(base_angles, 40)
        time.sleep(2.0)

        # 6.2 关节 1 小幅度左右摇头
        for i in range(2):
            a = base_angles.copy()
            a[0] = a[0] + 15   # J1 +15°
            print(f"[INFO] 摇头 右: {a}")
            mc.send_angles(a, 40)
            time.sleep(2.0)

            a = base_angles.copy()
            a[0] = a[0] - 15   # J1 -15°
            print(f"[INFO] 摇头 左: {a}")
            mc.send_angles(a, 40)
            time.sleep(2.0)

        # 回到基准
        print("[INFO] 回到测试结束姿态（基准角度）...")
        mc.send_angles(base_angles, 40)
        wait_and_print_angles(mc, desc="结束姿态")

        print("[INFO] 测试动作完成。按 Ctrl+C 可以随时中断。")

    finally:
        # 你可以选择是否在结束时断电
        # 如果希望手动拖动机械臂，可以取消下面两行注释：
        # print("[INFO] 关闭电机扭矩（power_off），可手动拖动机械臂。")
        # mc.power_off()
        print("[INFO] Demo 结束。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
