# read_temp_demo.py
"""
使用 MCP2221A + MLX90614 连续读取温度并打印
"""

import time

from huijiu.sensors.backends.base import I2CBackend
from huijiu.sensors.backends.mcp2221_easy import MCP2221EasyBackend
from huijiu.sensors.mlx90614 import MLX90614


def main() -> None:
    # 1. 打开 MCP2221A，初始化 I2C
    i2c = MCP2221EasyBackend(bus_speed=100_000)

    # 2. 创建 MLX90614 传感器实例
    sensor = MLX90614(i2c)

    print("开始循环读取温度（Ctrl+C 退出）...")
    try:
        while True:
            try:
                ta = sensor.read_ambient_c()
                tobj = sensor.read_object_c()
                print(f"环境温度: {ta:6.2f} °C   物体温度: {tobj:6.2f} °C")
            except Exception as e:
                print(f"[ERROR] 读取温度失败: {e}")

            time.sleep(0.5)  # 每 0.5 秒读一次
    except KeyboardInterrupt:
        print("退出温度读取。")


if __name__ == "__main__":
    main()
