"""
urdf_inspect.py

功能：
1. 解析 mycobot_280m5_with_gripper_up.urdf，打印所有 link / joint，并用“现实零件”语言解释。
2. 用 ikpy 简单检查一下 DOF 顺序（只做分析，失败也无所谓）。
3. 连接真实 myCobot 280：
   - 上电 + 回 HOME 姿态；
   - 依次轻微摆动 J1~J6（腰、大臂、小臂、腕1、腕2、末端自转）；
   - 最后做一组夹爪“开-合-开”，对应 URDF 中 8~13 这些抓夹连杆的整体运动。
4. 结束时不释放电机（不调用 release_all_servos），避免“突然掉下去”。

注意：
- 请确认 PORT / BAUD / HOME_ANGLES 正确。
- 第一次跑前务必清空机械臂周围空间。
"""

from __future__ import annotations

import time
from pathlib import Path
import xml.etree.ElementTree as ET

from ikpy.chain import Chain
from pymycobot.mycobot280 import MyCobot280

# 可按你项目情况改成从 cfg 里取，这里写死方便单文件测试
PROJECT_ROOT = Path(__file__).resolve().parents[3]
URDF_PATH = PROJECT_ROOT / "hardware" / "urdf" / "mycobot_280m5_with_gripper_up.urdf"

ROBOT_PORT = "COM4"      # 按你当前 Windows 上的串口改
ROBOT_BAUD = 115200

HOME_ANGLES = [0, 0, 0, 0, 0, 0]  # 机械臂回零姿态，可以按你习惯调整

# 摆动幅度 & 速度
WIGGLE_DELTA_DEG = 10.0
WIGGLE_SPEED = 20
WIGGLE_WAIT = 2.0


# ---------------------- 1. 解析 URDF：link & joint ----------------------


def explain_link(name: str) -> str:
    """用“现实零件”语言给每个 link 写注释。"""
    mapping = {
        "g_base": "机械臂总底座：固定在桌面/支架上的基座，是整机坐标系起点。",
        "joint1": "第一节立柱外壳：围绕竖直轴旋转，相当于“腰部”，决定整臂朝向。",
        "joint2": "第二节大臂：从腰部伸出的第一段“大臂”，主要做上下抬升。",
        "joint3": "第三节小臂：连接大臂和前臂的“肘部”部分，改变臂的伸出长度。",
        "joint4": "第四节腕部 1：靠近末端的腕关节之一，调整末端俯仰姿态。",
        "joint5": "第五节腕部 2：紧接着的腕关节，进一步调整末端姿态（翻转/摆动）。",
        "joint6": "第六节腕部 3：最靠近法兰的腕关节，通常控制工具绕自身轴旋转。",
        "joint6_flange": "法兰盘：安装工具/夹爪的圆盘，工业机器人常用的“工具安装面”。",
        "gripper_base": "抓夹基座：夹爪主体，固定在法兰盘上，支撑左右爪结构。",
        "gripper_left1": "左侧爪尖：真正接触并夹住物体的部分。",
        "gripper_left2": "左侧中间连杆：连接爪尖与基座的连杆。",
        "gripper_left3": "左侧传动连杆：靠近基座，把驱动关节的转动传递给爪尖。",
        "gripper_right1": "右侧爪尖：与左爪尖配合夹住物体。",
        "gripper_right2": "右侧中间连杆：连接爪尖与基座的连杆。",
        "gripper_right3": "右侧传动连杆：靠近基座，把驱动关节的转动传递给右爪尖。",
    }
    return mapping.get(name, "（未特别标注，可视作结构件或装饰件）")


def explain_joint(name: str, joint_type: str) -> str:
    """用“现实动作”语言解释每个 joint 代表的运动含义。"""
    if joint_type == "fixed":
        fixed_note = "固定关节：不提供运动自由度，只是把两个 link 刚性连接起来。"
    else:
        fixed_note = ""

    mapping = {
        "g_base_to_joint1": (
            "把抽象的 base_link g_base 和第一节外壳 joint1 连起来；"
            "URDF 里叫 joint，但类型为 fixed，不产生实际运动。"
        ),
        "joint2_to_joint1": "J1：腰部旋转，围绕竖直轴转动，决定手臂朝向。",
        "joint3_to_joint2": "J2：大臂抬升，相当于人从肩膀处抬手。",
        "joint4_to_joint3": "J3：肘部弯曲，决定末端远近。",
        "joint5_to_joint4": "J4：腕部自由度之一，配合 J5 调整末端俯仰/翻转。",
        "joint6_to_joint5": "J5：腕部自由度之一，与 J4 联合作用，控制末端姿态。",
        "joint6output_to_joint6": "J6：末端自转，控制工具/夹爪绕自身轴旋转。",
        "joint6output_to_gripper_base": "固定：把法兰盘和抓夹基座刚性连接在一起。",
        "gripper_controller": (
            "抓夹主控制关节：由一个电机驱动，通过连杆机构带动左右爪张开/闭合；"
            "现实里你能控制的是“夹爪电机”，URDF 把内部连杆也拆成多个 joint。"
        ),
        "gripper_base_to_gripper_left2": "左侧辅助连杆关节：连杆机构一部分，配合同步开合。",
        "gripper_left3_to_gripper_left1": "左爪尖连杆关节：直接驱动左侧爪尖运动。",
        "gripper_base_to_gripper_right3": "右侧辅助连杆关节：与左侧对称，参与同步开合。",
        "gripper_base_to_gripper_right2": "右侧辅助连杆关节：连杆机构一部分。",
        "gripper_right3_to_gripper_right1": "右爪尖连杆关节：直接驱动右侧爪尖，与左爪夹住物体。",
    }

    body = mapping.get(name, "")
    if fixed_note and body:
        return f"{body} {fixed_note}"
    elif fixed_note:
        return fixed_note
    elif body:
        return body
    else:
        return "（未特别标注的运动/结构关节）"


def parse_urdf_and_print(urdf_path: Path) -> None:
    print(f"[INFO] URDF 路径: {urdf_path}")

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # 1) link 列表
    links = root.findall("link")
    print("\n[INFO] 所有 link：")
    for idx, link in enumerate(links):
        name = link.attrib.get("name", "")
        print(f"  [{idx:02d}] {name}")
        print(f"       ↳ 说明: {explain_link(name)}")

    # 2) joint 列表
    joints = root.findall("joint")
    print("\n[INFO] 所有关节（joint）：")
    for idx, joint in enumerate(joints):
        name = joint.attrib.get("name", "")
        jt = joint.attrib.get("type", "")
        parent = joint.find("parent").attrib.get("link", "") if joint.find("parent") is not None else "?"
        child = joint.find("child").attrib.get("link", "") if joint.find("child") is not None else "?"
        origin = joint.find("origin")
        if origin is not None:
            xyz = origin.attrib.get("xyz", "0 0 0")
            rpy = origin.attrib.get("rpy", "0 0 0")
        else:
            xyz = "0 0 0"
            rpy = "0 0 0"

        print(
            f"  [{idx:02d}] name={name:<32} type={jt:<9} "
            f"parent={parent:<24} child={child:<24} origin.xyz={xyz:<12} rpy={rpy}"
        )
        print(f"       ↳ 说明: {explain_joint(name, jt)}")

    # 3) 基本的 base_link / end_link 猜测（不严格，只是帮助你理解）
    parent_links = {j.find('parent').attrib['link'] for j in joints if j.find('parent') is not None}
    child_links = {j.find('child').attrib['link'] for j in joints if j.find('child') is not None}

    base_candidates = parent_links - child_links
    end_candidates = child_links - parent_links

    print("\n[INFO] 可能的 base_link（仅做参考）：")
    for name in base_candidates:
        mark = " <= 推荐作为整机基座坐标系" if name == "g_base" else ""
        print(f"  - {name} {mark}")

    print("\n[INFO] 可能的 end_link（仅做参考）：")
    for name in end_candidates:
        print(f"  - {name}")


# ---------------------- 2. 用 ikpy 看 DOF 顺序 ----------------------


def inspect_chain_with_ikpy():
    print("\n[INFO] 使用 ikpy 从 URDF 构建 Chain（仅做 DOF 检查，不依赖 mesh）...")
    try:
        # 显式告诉 ikpy：根节点是 g_base
        chain = Chain.from_urdf_file(str(URDF_PATH), base_elements=["g_base"])
        print(f"[INFO] Chain 链接数 = {len(chain.links)}")

        dof_idx = 0
        print("[INFO] 链上各 link / joint：")
        for i, link in enumerate(chain.links):
            jt = link.joint_type  # 'fixed' / 'revolute' / 'prismatic'...
            line = f"  [{i:02d}] name={link.name:<30} joint_type={jt:<9}"
            if jt != "fixed":
                line += f"  --> DOF[{dof_idx}]"
                dof_idx += 1
            print(line)
        print(f"[INFO] 总 DOF（joint_type != 'fixed'）= {dof_idx}")

        print(
            "\n[HINT] 你可以把上面 DOF[0..] 和前面 joint 列表结合起来看：\n"
            "      - DOF[0..5] ≈ 机械臂 J1~J6 的关节角（真实有电机的关节）；\n"
            "      - DOF[6..] 是抓夹连杆相关的虚拟关节，现实中由一个夹爪电机带动。"
        )
    except Exception as e:
        print(f"[WARN] 使用 ikpy 构建 Chain 失败：{e}")
        print("[WARN] 这只是分析工具，不影响下面真机的关节摆动演示。")


# ---------------------- 3. 真实机械臂：依次摆 J1~J6 + 夹爪 ----------------------


def connect_robot() -> MyCobot280:
    print(f"\n[ROBOT] 正在连接机械臂: port={ROBOT_PORT}, baud={ROBOT_BAUD}")
    mc = MyCobot280(ROBOT_PORT, ROBOT_BAUD)
    time.sleep(0.5)
    return mc


def robot_go_home(mc: MyCobot280):
    print("[ROBOT] power_on ...")
    mc.power_on()
    time.sleep(1.0)
    print(f"[ROBOT] 回到 HOME 姿态: {HOME_ANGLES}")
    mc.send_angles(HOME_ANGLES, WIGGLE_SPEED)
    time.sleep(4.0)


def wiggle_joint(mc: MyCobot280, joint_id: int, delta_deg: float = WIGGLE_DELTA_DEG):
    """
    让某一个关节正负小角度摆动一下，其他关节保持当前角度。

    joint_id: 1~6（myCobot 的 J1~J6）
    """
    assert 1 <= joint_id <= 6, "joint_id 必须在 1~6 之间"

    angles = mc.get_angles()
    if not angles or len(angles) != 6:
        print(f"[ROBOT] get_angles() 返回异常: {angles}，跳过 J{joint_id}")
        return

    print(f"\n[ROBOT] >>> 摆动 J{joint_id}（其它关节尽量保持不动）")
    print(f"[ROBOT] 当前角度: {angles}")

    base_angles = list(angles)
    up_angles = list(angles)
    down_angles = list(angles)

    idx = joint_id - 1
    up_angles[idx] += delta_deg
    down_angles[idx] -= delta_deg

    # 限位简单保护：控制在 [-180, 180] 区间内
    up_angles[idx] = max(min(up_angles[idx], 180.0), -180.0)
    down_angles[idx] = max(min(down_angles[idx], 180.0), -180.0)

    print(f"[ROBOT]  -> 往正方向 (+{delta_deg}°): {up_angles}")
    mc.send_angles(up_angles, WIGGLE_SPEED)
    time.sleep(WIGGLE_WAIT)

    print(f"[ROBOT]  -> 往负方向 (-{delta_deg}°): {down_angles}")
    mc.send_angles(down_angles, WIGGLE_SPEED)
    time.sleep(WIGGLE_WAIT)

    print(f"[ROBOT]  -> 回到原来的姿态: {base_angles}")
    mc.send_angles(base_angles, WIGGLE_SPEED)
    time.sleep(WIGGLE_WAIT)


def wiggle_gripper(mc: MyCobot280):
    """
    把“夹爪开-合-开”作为 URDF 中 8~13 这些抓夹连杆关节的整体运动体验。

    注意：
    - myCobot 真机上，抓夹通常通过 set_gripper_state / set_gripper_value 控制；
    - URDF 里拆成了多个 joint（gripper_controller + 左右连杆），但这些在硬件上是被动的。
    """
    print(
        "\n[ROBOT] >>> 演示抓夹运动（对应 URDF 中 joint 8~13：gripper_controller + 左/右连杆）"
    )
    try:
        # 按你之前 blockly 的习惯，这里用 set_gripper_state 版本：
        # state: 0=open, 1=close, speed: 0~100
        print("[ROBOT] 夹爪 -> 打开")
        mc.set_gripper_state(0, 50)
        time.sleep(2.0)

        print("[ROBOT] 夹爪 -> 闭合")
        mc.set_gripper_state(1, 50)
        time.sleep(2.0)

        print("[ROBOT] 夹爪 -> 再次打开")
        mc.set_gripper_state(0, 50)
        time.sleep(2.0)

        print("[ROBOT] 抓夹演示完成。")
    except AttributeError:
        print(
            "[WARN] 你的 pymycobot 版本可能没有 set_gripper_state，"
            "可以改用 set_gripper_value(value, speed) 方式测试：\n"
            "      mc.set_gripper_value(0, 50)   # 一端极限\n"
            "      mc.set_gripper_value(100, 50) # 另一端极限"
        )
    except Exception as e:
        print(f"[WARN] 抓夹控制时出现异常：{e}")


# ---------------------- 4. main ----------------------


def main():
    # 1. 打印 URDF 结构 + 文本解释
    parse_urdf_and_print(URDF_PATH)

    # 2. 用 ikpy 看一下 DOF 顺序（只做分析，用 try 包起来）
    inspect_chain_with_ikpy()

    # 3. 真机演示
    print(
        "\n================================================================"
        "\n[ROBOT] 接下来要连接真实机械臂并依次摆动 J1~J6 + 夹爪。"
        "\n       请确认："
        "\n       - 机械臂周围空间已经清空；"
        "\n       - 电源和串口连接正常；"
        "\n       - 你在旁边可以随时按下急停 / 断电。"
        "\n================================================================"
    )
    input(">>> 如果一切准备就绪，按回车开始；如不想让真机运动，请 Ctrl+C 结束脚本：")

    mc = connect_robot()
    robot_go_home(mc)

    # 依次摆动 J1~J6
    for joint_id in range(1, 7):
        wiggle_joint(mc, joint_id, WIGGLE_DELTA_DEG)

    # 夹爪示意 URDF 8~13 的运动
    wiggle_gripper(mc)

    print(
        "\n[ROBOT] 演示结束。"
        "\n[ROBOT] 按要求：不调用 release_all_servos()，保持当前姿态，"
        "\n        可以手动在上位机 / 电源上安全断电或回 HOME。"
    )


if __name__ == "__main__":
    main()
