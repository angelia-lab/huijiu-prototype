# vl53l0x.py
"""
VL53L0X 激光 ToF 测距传感器最小驱动
- 依赖 i2c_backend.I2CBackend（write_reg / read_reg / close）
- 提供:
    - VL53L0X.init()           : 上电后的最小初始化
    - VL53L0X.measure_once()   : 返回一帧带可靠性标记的 RangeSample
    - VL53L0X.read_range_single_mm() : 只返回距离（mm）
"""

from __future__ import annotations
from dataclasses import dataclass
import time

from .backends.base import I2CBackend


@dataclass
class RangeSample:
    """一次测距结果 + 质量信息"""
    distance_mm: int
    ambient: int
    signal: int
    range_status: int   # 低 5 位 status
    raw_status: int     # 原始 status 寄存器值（0x14）
    reliable: bool      # 我们根据阈值判断出的“是否可靠”


class VL53L0X:
    """VL53L0X 激光 ToF 测距传感器（单次测距最小版）。"""

    # 7-bit I2C 地址：用 MCP2221 I2C/SMBus Terminal查看 Bus Scan 7bit 0X00-0X7F
    DEFAULT_I2C_ADDR = 0x29

    # 常用寄存器
    REG_IDENT_MODEL_ID = 0xC0
    REG_IDENT_REVISION_ID = 0xC2
    REG_SYSRANGE_START = 0x00
    REG_SYSTEM_INTERRUPT_CLEAR = 0x0B
    REG_RESULT_INTERRUPT_STATUS = 0x13
    REG_RESULT_RANGE_STATUS = 0x14  # 从这里起连续 12 字节包含状态+距离等信息
    REG_VHV_CONFIG_PAD_SCL_SDA__EXTSUP_HV = 0x89

    def __init__(self, backend: I2CBackend, address: int = DEFAULT_I2C_ADDR) -> None:
        """
        :param backend: 任意实现了 I2CBackend 接口的对象（MCP2221、以后换 HIDAPI 也一样）
        :param address: 7bit I2C 地址，默认 0x29
        """
        self._i2c = backend
        self.address = address

        # ---- 可靠性判断的参数 ----

        # 哪些 status 被认为是“测距有效”（低 5 位取值）
        # 0x00 = Range Valid，0x0B(11) = 很多实板会返回的“有效”状态
        self.valid_status_codes = {0x00, 0x0B}

        # 最小信号强度（太小就认为不可靠）
        self.min_signal = 1

        # ambient 的上限（环境太亮，容易噪声太大）
        self.max_ambient = 65535

        # 最小信噪比：signal 至少要大于 ambient 的多少倍
        self.min_signal_to_ambient_ratio = 0

        # 最近一次的测量结果（便于调试）
        self.last_sample: RangeSample | None = None

    # ----------- 基础读写封装（只用 write_reg / read_reg） ----------- #

    def _write_reg8(self, reg: int, value: int) -> None:
        """向 8bit 寄存器写 1 个字节。"""
        self._i2c.write_reg(self.address, reg & 0xFF, bytes([value & 0xFF]))

    def _read_reg8(self, reg: int) -> int:
        """从 8bit 寄存器读 1 个字节。"""
        data = self._i2c.read_reg(self.address, reg & 0xFF, 1)
        if not data or len(data) < 1:
            raise RuntimeError("I2C 读寄存器失败：没有返回数据")
        return data[0]

    def _read_multi(self, reg: int, length: int) -> bytes:
        """从 reg 开始连续读取 length 个字节。"""
        data = self._i2c.read_reg(self.address, reg & 0xFF, length)
        if not data or len(data) < length:
            raise RuntimeError(
                f"I2C 读多字节失败：期望 {length} 字节，实际 {len(data) if data else 0}"
            )
        return data

    # ----------- 初始化（上电后调用一次） ----------- #

    def init(self) -> None:
        """
        做一次最小初始化：
        1) 把 SCL/SDA 垫片配置到 2.8V 模式
        2) 写一串官方推荐的“魔法寄存器”，让内部状态机进入可测距状态
        """
        # 1) 2.8V 模式
        vhv = self._read_reg8(self.REG_VHV_CONFIG_PAD_SCL_SDA__EXTSUP_HV)
        self._write_reg8(
            self.REG_VHV_CONFIG_PAD_SCL_SDA__EXTSUP_HV,
            (vhv & 0xFE) | 0x01,
        )

        # 2) 官方推荐初始化序列（大部分 Arduino/STM32 库都这么写）
        self._write_reg8(0x88, 0x00)
        self._write_reg8(0x80, 0x01)
        self._write_reg8(0xFF, 0x01)
        self._write_reg8(0x00, 0x00)
        _ = self._read_reg8(0x91)  # 读一下但不用，保持和参考实现一致
        self._write_reg8(0x91, 0x3C)
        self._write_reg8(0x00, 0x01)
        self._write_reg8(0xFF, 0x00)
        self._write_reg8(0x80, 0x00)

        # 3) 读型号信息做 sanity check
        try:
            model = self._read_reg8(self.REG_IDENT_MODEL_ID)
            rev = self._read_reg8(self.REG_IDENT_REVISION_ID)
            print(f"VL53L0X detected: model 0x{model:02X}, rev 0x{rev:02X}")
        except Exception as e:
            print(f"[WARN] 读取设备 ID 失败（不影响继续尝试测距）: {e}")

    # ----------- 核心：判断一帧是否可靠（你以后主要改这块） ----------- #

    def _evaluate_sample(self, sample: RangeSample) -> bool:
        """
        按当前规则判断一帧数据是否“可靠”。
        当前版本：非常宽松，只在明显异常时标记为不可靠。
        """
        rs = sample.range_status
        ambient = sample.ambient
        signal = sample.signal

        # 1) status 必须在“允许的集合”里
        if rs not in self.valid_status_codes:
            #调试用：看下出现了哪些异常状态
            print(f"[DEBUG] range_status 不在有效集合: 0x{rs:02X}")
            return False

        # 2) signal 不能太小
        if signal < self.min_signal:
            print(f"[DEBUG] signal 太小: {signal}")
            return False

        # 3) ambient 不能太大
        if ambient > self.max_ambient:
            return False

        # 4) 信噪比：signal 至少要比 ambient 高一定倍数
        if ambient > 0:
            ratio = signal / float(ambient)
            if ratio < self.min_signal_to_ambient_ratio:
                return False

        # 如果都通过，就认为这帧“可用”
        return True

    # ----------- 单次测距，返回 RangeSample ----------- #

    def measure_once(self, delay_ms: int = 50) -> RangeSample:
        """
        触发一次测距，返回包含距离 + 质量信息的 RangeSample。
        delay_ms: 触发后等待多少毫秒再读取结果。
        """
        # 1) 启动单次测距
        self._write_reg8(self.REG_SYSRANGE_START, 0x01)

        # 2) 等待测量完成（简单粗暴睡一会儿）
        time.sleep(delay_ms /500.0)

        # 3) 读取结果数据块（12 字节）
        raw = self._read_multi(self.REG_RESULT_RANGE_STATUS, 12)

        # 解析：参考常见 Arduino/DFRobot 驱动
        raw_status = raw[0]                     # 原始状态寄存器值
        range_status = (raw_status & 0x78) >> 3 # 官方推荐的 RangeStatus 计算方式
        ambient = (raw[6] << 8) | raw[7]
        signal = (raw[8] << 8) | raw[9]
        distance = (raw[10] << 8) | raw[11]


        # 4) 清除中断，为下一次测距做准备
        self._write_reg8(self.REG_SYSTEM_INTERRUPT_CLEAR, 0x01)

        # 5) 判断可靠性
        sample = RangeSample(
            distance_mm=distance,
            ambient=ambient,
            signal=signal,
            range_status=range_status,
            raw_status=raw_status,
            reliable=False,  # 先占位，下面 evaluate
        )
        sample.reliable = self._evaluate_sample(sample)

        # 记录为最近一次测量，方便外部调试查看
        self.last_sample = sample

        # 有需要可以在这里打印调试信息：
        # print(sample)

        return sample

    # ----------- 兼容之前 demo 的简化接口 ----------- #

    def read_range_single_mm(self, delay_ms: int = 50) -> int:
        """
        兼容旧接口：只返回距离（mm）。
        如果你关心数据质量，请用 measure_once()。
        """
        sample = self.measure_once(delay_ms=delay_ms)

        if not sample.reliable:
            print(
                f"[WARN] 不可靠测距: "
                f"dist={sample.distance_mm}mm, "
                f"status=0x{sample.range_status:02X}, "
                f"ambient={sample.ambient}, signal={sample.signal}"
            )

        return sample.distance_mm
