# mlx90614.py
"""
MLX90614 红外温度传感器最小驱动
- 依赖 i2c_backend.I2CBackend（write_reg / read_reg / close）
- 提供:
    - MLX90614.read_ambient_c()  : 环境温度 (°C)
    - MLX90614.read_object_c()   : 物体温度 (°C)
"""

from __future__ import annotations

from .backends.base import I2CBackend


class MLX90614:
    # 默认 7bit I2C 地址（很多文档写 0xB4/0xB5，那是 8bit 地址，这里统一用 0x5A）
    DEFAULT_I2C_ADDR = 0x5A

    # 寄存器地址
    REG_TA = 0x06   # Ambient temperature
    REG_TOBJ1 = 0x07  # Object temperature 1
    # REG_TOBJ2 = 0x08  # Object temperature 2 (一般没用)

    def __init__(self, backend: I2CBackend, address: int = DEFAULT_I2C_ADDR) -> None:
        """
        :param backend: 实现了 I2CBackend 的对象（例如 MCP2221EasyBackend 实例）
        :param address: 7bit I2C 地址，默认 0x5A
        """
        self._i2c = backend
        self.address = address

    # ---- 底层读 word 的工具函数 ---- #

    def _read_word(self, reg: int) -> int:
        """
        从 MLX90614 读一个 16bit 的原始数据（小端序）。
        MLX90614 返回的是 LSB, MSB, PEC，我们这里只用前两个字节。
        """
        raw = self._i2c.read_reg(self.address, reg & 0xFF, 3)
        if not raw or len(raw) < 2:
            raise RuntimeError("MLX90614 读寄存器失败")

        lsb = raw[0]
        msb = raw[1]
        value = (msb << 8) | lsb
        return value

    def _raw_to_celsius(self, raw: int) -> float:
        """
        数据手册公式：
        T(K) = raw * 0.02
        T(°C) = T(K) - 273.15
        """
        return raw * 0.02 - 273.15

    # ---- 对外接口 ---- #

    def read_ambient_c(self) -> float:
        """读取环境温度（°C）"""
        raw = self._read_word(self.REG_TA)
        return self._raw_to_celsius(raw)

    def read_object_c(self) -> float:
        """读取物体温度（°C）——对你来说就是皮表温度方向"""
        raw = self._read_word(self.REG_TOBJ1)
        return self._raw_to_celsius(raw)
