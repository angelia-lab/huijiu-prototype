"""
ik_demo.py

目标：
- 用 ikpy 加载 mycobot_280m5_with_gripper_up.urdf；
- 选一个很“朴素”的目标点（基坐标系下 X 前、Z 向上）；
- 调用 IK 求解得到 6 轴关节角；
- 把前 6 个 DOF 映射为 myCobot J1~J6，通过 send_angles 让真实机械臂动起来。

安全提醒：
- 第一次跑务必要把周围障碍物清干净；
- 先只当「能不能动起来」的实验，不保证姿态一定好看；
- 如果姿态很扭曲，就立刻断电或按急停，以免机械臂撞自己。
"""

from pathlib import Path
import math
import time

import numpy as np
from ikpy.chain import Chain
from pymycobot.mycobot280 import MyCobot280

from huijiu.core import config as cfg

# 你的 URDF 路径：config 里已经设置过
URDF_PATH: Path = cfg.ROBOT_URDF

# 机械臂串口（按你现在 Windows 使用的串口改）
ROBOT_PORT: str = cfg.ROBOT_PORT
BAUD: int = cfg.ROBOT_BAUD

# 目标点（单位：米，URDF 基坐标系）
# 假设：g_base 在桌面上，X 朝前，Z 向上
TARGET_XYZ_M = np.array([0.18, 0.0, 0.20], dtype=float)  # 18cm 前方、20cm 高


# ----------------------------------------------------------------------
# 1. 构建 Chain，并打印结构 / DOF 映射
# ----------------------------------------------------------------------
def build_chain() -> Chain:
    print(f"[INFO] 使用 URDF 构建 Chain: {URDF_PATH}")
    chain = Chain.from_urdf_file(
        str(URDF_PATH),
        base_elements=["g_base"],
    )

    print(f"[INFO] Chain 链接数 = {len(chain.links)}")
    print("[INFO] 链上各 link / joint：")
    dof_index = 0
    for idx, link in enumerate(chain.links):
        jt = link.joint_type
        if jt != "fixed":
            print(f"  [{idx:02d}] name={link.name:<30} joint_type={jt:<9}  --> DOF[{dof_index}]")
            dof_index += 1
        else:
            print(f"  [{idx:02d}] name={link.name:<30} joint_type={jt:<9}")
    print(f"[INFO] 总 DOF（joint_type != 'fixed'）= {dof_index}")
    print("[HINT] 默认假设 DOF[0..5] ≈ 机械臂 J1~J6，DOF[6..] 为抓夹内部关节。")
    return chain


# ----------------------------------------------------------------------
# 2. 计算 IK，并提取前 6 个 DOF → 机械臂关节角
# ----------------------------------------------------------------------
def solve_ik(chain: Chain, target_xyz_m: np.ndarray):
    """
    输入：目标位置（米），不管姿态；
    输出：
      - full_angles_rad: 长度 = len(chain.links)，每个 link 对应一个角
      - arm_dofs_deg:    长度 = 6，映射为 myCobot 的 J1~J6（单位：度）
    """
    print(f"[INFO] 目标点（基坐标系，m）：{target_xyz_m}")

    # 关键修正：这里只给【位置】给 inverse_kinematics，不再传 4x4 矩阵
    full_angles_rad = chain.inverse_kinematics(target_xyz_m)
    print(f"[INFO] IK 解（rad，按 chain.links 顺序） =\n  {full_angles_rad}")

    # 为了验证 IK 是否靠谱，跑一下 FK 再算回去
    fk_matrix = chain.forward_kinematics(full_angles_rad)
    pos_fk = fk_matrix[:3, 3]
    print(f"[INFO] 用 IK 解做 FK 得到的末端位置（m） = {pos_fk}")
    print(f"[INFO] 位置误差（m） = {pos_fk - target_xyz_m}")

    # 把前 6 个 DOF 抽出来，对齐到 myCobot 的 J1~J6
    arm_dofs_deg = []
    dof_count = 0
    for link_index, link in enumerate(chain.links):
        if link.joint_type == "fixed":
            continue
        angle_rad = full_angles_rad[link_index]
        arm_dofs_deg.append(math.degrees(angle_rad))
        dof_count += 1
        if dof_count == 6:
            break

    if len(arm_dofs_deg) < 6:
        raise RuntimeError(f"[FATAL] 这个 URDF 里可动关节少于 6 个，得到的 DOF={len(arm_dofs_deg)}")

    print(f"[INFO] 映射到机械臂 J1~J6 的角度（deg） = {arm_dofs_deg}")
    return full_angles_rad, arm_dofs_deg


# ----------------------------------------------------------------------
# 3. 把关节角发送给真实 myCobot
# ----------------------------------------------------------------------
def move_real_robot(angles_deg):
    print(f"[INFO] 准备连接机械臂: port={ROBOT_PORT}, baud={BAUD}")
    mc = MyCobot280(ROBOT_PORT, BAUD)

    print("[INFO] power_on ...")
    mc.power_on()
    time.sleep(1.0)

    print("[INFO] 先回到一个简单 HOME 姿态: [0, 0, 0, 0, 0, 0]")
    mc.send_angles([0, 0, 0, 0, 0, 0], 20)
    time.sleep(4.0)

    print(f"[INFO] 发送 IK 计算得到的关节角（deg）: {angles_deg}")
    mc.send_angles(angles_deg, 10)  # 速度先用 10 慢一点
    for t in np.linspace(0.5, 5.0, 10):
        time.sleep(0.5)
        coords = mc.get_coords()
        print(f"[DEBUG] t={t:.1f}s, coords={coords}")

    print("[INFO] demo 结束，不主动 release，你自己确认安全后再断电/关机。")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    chain = build_chain()

    full_angles_rad, arm_dofs_deg = solve_ik(chain, TARGET_XYZ_M)

    # 先只看打印结果，如果感觉关节角很夸张，可以暂时注释掉真实机械臂这一步
    move_real_robot(arm_dofs_deg)


if __name__ == "__main__":
    main()
