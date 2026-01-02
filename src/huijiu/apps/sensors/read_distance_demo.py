# read_distance_demo.py
import time
from huijiu.sensors.backends.base import I2CBackend
from huijiu.sensors.backends.mcp2221_easy import MCP2221EasyBackend
from huijiu.sensors.vl53l0x import VL53L0X
def main() -> None:
    # 1. 打开 MCP2221A，并初始化 I2C
    i2c = MCP2221EasyBackend(bus_speed=100_000)

    # 2. 创建传感器实例并做一次初始化
    sensor = VL53L0X(i2c)
    sensor.init()

    # 3. 循环读距离
    print("开始循环测距（Ctrl+C 退出）...")
    try:
        while True:
            try:
                distance_mm = sensor.read_range_single_mm()
                if distance_mm!=0:
                    print(f"距离：{distance_mm} mm")
            except Exception as e:
                print(f"[ERROR] 读取距离失败: {e}")

            time.sleep(0.2)  # 200 ms 一次
    except KeyboardInterrupt:
        print("退出测距。")


if __name__ == "__main__":
    main()
