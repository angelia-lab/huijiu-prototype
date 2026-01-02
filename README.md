代码逻辑结构:
huijiu-prototype/
├─ pyproject.toml / requirements.txt
├─ README.md
├─ docs/                # 设计文档、笔记
├─ hardware/            # 电路图、结构图等非代码文件（后面用）
├─ src/
│  └─ huijiu/
│     ├─ sensors/       # 各类传感器 & I2C backend
│     │  ├─ backends/
│     │  │  ├─ __init__.py
│     │  │  ├─ base.py              # I2CBackend 抽象
│     │  │  └─ mcp2221_easy.py      # MCP2221EasyBackend 实现
│     │  ├─ __init__.py
│     │  ├─ mlx90614.py
│     │  ├─ vl53l0x.py
│     │  └─ manager.py              # 可选：统一管理多个传感器
│     │
│     ├─ vision/
│     │  ├─ __init__.py
│     │  ├─ d455.py                 # RealSense D455 封装
│     │  ├─ calib_chessboard.py     # 相机标定
│     │  └─ calib_aruco.py          # ArUco / AprilTag 位姿
│     │
│     ├─ robot/
│     │  ├─ __init__.py
│     │  ├─ mycobot280_m5.py        # 机械臂控制封装
│     │  ├─ servo_head.py           # 上下/左右两个舵机的云台封装
│     │  └─ interfaces.py           # Actuator / Gimbal 抽象接口
│     │
│     ├─ devices/
│     │  ├─ __init__.py
│     │  ├─ moxibustion_head.py     # 艾灸头这一整套的抽象（含舵机+传感器位姿）
│     │  └─ exhaust_fan.py          # 排烟机控制
│     │
│     ├─ safety/
│     │  ├─ __init__.py
│     │  ├─ state.py                # SystemState 数据结构
│     │  └─ rules.py                # 安全规则/状态机
│     │
│     ├─ apps/                      # 各种 demo / 实验脚本
│     │  ├─ __init__.py (可选)
│     │  ├─ read_temp_demo.py
│     │  ├─ read_distance_demo.py
│     │  ├─ sensors_loop_demo.py
│     │  ├─ d455_preview_demo.py
│     │  ├─ mycobot_basic_demo.py
│     │  ├─ servo_head_demo.py
│     │  └─ integrated_safe_loop_demo.py
│     │
│     ├─ core/                      # 整体调度、配置、运行时
│     │  ├─ __init__.py
│     │  ├─ config.py               # 参数 & 阈值
│     │  ├─ orchestrator.py         # 把 sensors+vision+robot+devices 串起来
│     │  └─ logging.py              # 统一日志配置
│     │
│     └─ __init__.py
└─ tests/                           # 单元测试（以后再补）
2、写 import 的简易规则
同一个子包内部（sensors 里面互相 import）
用相对导入：
from .backends.base import I2CBackend
from .mlx90614 import MLX90614


跨子包（robot 里想用 sensors）
用绝对导入：
from huijiu.sensors.mlx90614 import MLX90614
from huijiu.sensors.vl53l0x import VL53L0X

3、在根目录下安装，即huijiu-prototype下
python.exe -m pip install -e .
4、“入口脚本”自检
python -m huijiu.apps.sandbox_check_imports

5、参考标记析：
A4棋盘格规格：单格边长：35 mm，内角数：9×6 内角
ArUco单牌尺寸：50–60 mm 边长，A4 排布：2×3 或 3×3 网格，间距≥20 mm空白边