from pymycobot.mycobot280 import MyCobot280
import time

PORT = "COM4"     # 换成你现在真实用的
BAUD = 115200

mc = MyCobot280(PORT, BAUD)

print("[TEST] power_on ...")
mc.power_on()
time.sleep(1.0)

print("[TEST] go home ...")
mc.send_angles([0, 0, 0, 0, 0, 0], 20)
time.sleep(4.0)

print("[TEST] current coords =", mc.get_coords())

# 这个点你可以摆一块 ArUco，大致确认在工作空间中间
target = [-150, 150, 150, -90, 0, -90]
print("[TEST] send_coords to:", target)

mc.send_coords(target, 20, 0)

# 连续读位置，看是否真的在动
for i in range(20):
    time.sleep(0.5)
    cur = mc.get_coords()
    print(f"[TEST] t={0.5 * (i+1):.1f}s, coords={cur}")

print("[TEST] release_all_servos")
mc.release_all_servos()
