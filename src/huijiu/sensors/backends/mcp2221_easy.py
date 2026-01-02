# mcp2221_easymcp_backend.py
from typing import Dict

import EasyMCP2221  # pip install EasyMCP2221
from .base import I2CBackend

class MCP2221EasyBackend(I2CBackend):
    """
    基于 EasyMCP2221 的 MCP2221A I2C 适配器实现。

    - 负责跟 USB/HID 打交道
    - 对上暴露标准 I2CBackend 接口
    """

    def __init__(self, bus_speed: int = 100_000):
        """
        :param bus_speed: I2C 总线速度（Hz），VL53L0X 用 100k 足够。
        """
        # 连接到第一个 MCP2221 设备
        self._mcp = EasyMCP2221.Device()
        self._bus_speed = bus_speed
        self._slaves: Dict[int, object] = {}

    # 内部：按地址缓存一个 I2C_Slave 对象
    def _get_slave(self, dev_addr: int):
        if dev_addr not in self._slaves:
            self._slaves[dev_addr] = self._mcp.I2C_Slave(
                dev_addr,
                speed=self._bus_speed,
            )
        return self._slaves[dev_addr]

    def write_reg(self, dev_addr: int, reg_addr: int, data: bytes) -> None:
        """
        使用 I2C_Slave.write_register(reg, data) 写寄存器。
        """
        slave = self._get_slave(dev_addr)

        if isinstance(data, int):
            data = bytes([data])

        # reg_bytes=1：VL53L0X 的寄存器地址是 1 字节
        slave.write_register(reg_addr, data, reg_bytes=1)

    def read_reg(self, dev_addr: int, reg_addr: int, length: int) -> bytes:
        """
        使用 I2C_Slave.read_register(reg, length) 读寄存器。
        """
        slave = self._get_slave(dev_addr)

        raw = slave.read_register(reg_addr, length, reg_bytes=1)
        # EasyMCP2221 返回的是 bytes 或 bytearray，这里统一转成 bytes
        return bytes(raw)

    def close(self) -> None:
        """
        EasyMCP2221 会在对象销毁时自动关闭 USB，这里可以简单留空。
        真要手动清理，可以在后续版本里加上。
        """
        # 如果以后库提供了 close()，可以改成：self._mcp.close()
        pass
