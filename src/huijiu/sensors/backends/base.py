# i2c_backend.py
from abc import ABC, abstractmethod

class I2CBackend(ABC):
    """
    抽象 I2C 适配器接口。

    任何 I2C 控制器（MCP2221A、树莓派、USB 转 I2C 适配器……）
    只要实现这 3 个方法，就可以被上层传感器驱动复用。
    """

    @abstractmethod
    def write_reg(self, dev_addr: int, reg_addr: int, data: bytes) -> None:
        """
        向 dev_addr 设备的 reg_addr 寄存器写入若干字节 data。
        data 通常是 bytes 对象，比如 b'\\x01\\x02'。
        """
        raise NotImplementedError

    @abstractmethod
    def read_reg(self, dev_addr: int, reg_addr: int, length: int) -> bytes:
        """
        从 dev_addr 设备的 reg_addr 寄存器开始，连续读取 length 个字节。
        返回值必须是 bytes。
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        关闭底层连接（如有需要）。
        没有需要时可以实现成 pass。
        """
        raise NotImplementedError
