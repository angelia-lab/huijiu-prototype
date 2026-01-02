"""
MyCobot 280 M5 拖动示教 + 姿态捕捉 Demo

用途：
- 让电机“松扭矩”（可以用手拖动机械臂）
- 你用手把艾灸头拖到穴位位置
- 回车后读取当前关节角（或末端坐标，如果固件支持），作为“理想对准姿态”
"""

from __future__ import annotations

import time
from typing import List, Optional, Union

from pymycobot.mycobot280 import MyCobot280


PORT = "COM4"      # 根据实际情况修改
BAUD = 115200


def release_servos(mc: MyCobot280) -> None:
    """
    将电机松扭矩，方便手动拖动。
    不同固件/库的接口可能不一样，这里做兼容：
    - 优先尝试 release_all_servos()
    - 否则退回 power_off()
    """
    if hasattr(mc, "release_all_servos"):
        try:
            mc.release_all_servos()
            print("[INFO] 已调用 release_all_servos()，可以用手拖动机械臂。")
            return
        except Exception as e:
            print(f"[WARN] release_all_servos() 失败，尝试 power_off(): {e}")

    try:
        mc.power_off()
        print("[INFO] 已调用 power_off()，电机无力，可手动拖动机械臂。")
    except Exception as e:
        print(f"[WARN] power_off() 也失败: {e}")


def main() -> int:
    print(f"[INFO] 连接 MyCobot 280 M5: port={PORT}, baud={BAUD} ...")
    mc = MyCobot280(PORT, BAUD)
    time.sleep(2.0)

    print("[INFO] 不读取固件版本（当前类无 get_version 接口）。")

    # 1. 松扭矩，让你可以用手拖动
    release_servos(mc)

    print(
        "\n[步骤说明]\n"
        "1）现在可以用手抓住机械臂各段，把艾灸头拖到你想要的穴位位置；\n"
        "2）拖动过程中，这个脚本会一直保持运行；\n"
        "3）拖好后，轻轻扶住机械臂别晃，回到电脑按一次回车；\n"
        "4）脚本会尝试读取当前关节角 / 坐标并打印出来。\n"
    )

    input("[INFO] 当你已经拖到理想穴位位置后，按回车键继续...")

    # 2. 尝试读取坐标 & 关节角
    last_coords: Optional[List[float]] = None
    last_angles: Optional[List[float]] = None

    # 2.1 末端坐标（如果固件支持）
    try:
        raw_coords: Union[int, List[float]] = mc.get_coords()
    except Exception as e:
        raw_coords = None
        print(f"[WARN] get_coords() 异常: {e}")
    else:
        if isinstance(raw_coords, (list, tuple)) and len(raw_coords) == 6:
            last_coords = list(raw_coords)
        elif isinstance(raw_coords, int):
            print(f"[WARN] get_coords() 返回错误码 {raw_coords}，当前固件可能不支持坐标读取。")
        else:
            print(f"[WARN] get_coords() 返回类型 {type(raw_coords)}，无法解析。")

    # 2.2 关节角（大部分固件一定支持）
    try:
        raw_angles = mc.get_angles()
    except Exception as e:
        raw_angles = None
        print(f"[WARN] get_angles() 异常: {e}")
    else:
        if isinstance(raw_angles, (list, tuple)) and len(raw_angles) == 6:
            last_angles = list(raw_angles)
        else:
            print(f"[WARN] get_angles() 返回类型 {type(raw_angles)}，无法解析。")

    print("\n[结果] ===============================")

    if last_coords is not None:
        x, y, z, rx, ry, rz = last_coords
        print(
            "[INFO] 捕捉到的末端坐标（如果固件支持）：\n"
            f"       coords = [{x:.2f}, {y:.2f}, {z:.2f}, {rx:.2f}, {ry:.2f}, {rz:.2f}]"
        )
    else:
        print("[INFO] 未成功获得有效的 coords（末端坐标），后续可以只用关节角。")

    if last_angles is not None:
        a1, a2, a3, a4, a5, a6 = last_angles
        print(
            "[INFO] 捕捉到的关节角（推荐用来做“理想对准姿态”）：\n"
            f"       angles = [{a1:.2f}, {a2:.2f}, {a3:.2f}, {a4:.2f}, {a5:.2f}, {a6:.2f}]"
        )
        
        
        print(
            "\n[下一步建议]\n"
            "  - 把上面这组 angles 复制到你的控制代码里，例如：\n"
            "      HUKOU_REF_ANGLES = "
            f"[{a1:.2f}, {a2:.2f}, {a3:.2f}, {a4:.2f}, {a5:.2f}, {a6:.2f}]\n"
            "  - 后面自动对准穴位时，直接调用 mc.send_angles(HUKOU_REF_ANGLES, 40) 即可回到这个姿态。\n"
        )
    else:
        print("[INFO] 未成功获得有效的 angles（关节角），请检查固件 / 通信状态。")

    print("[INFO] 结束。")
    
    
    
    
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
