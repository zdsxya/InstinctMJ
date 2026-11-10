# 多任务手柄遥操作

`instinct-teleop` 提供单机器人实时交互入口，与 `instinct-play` 分开。手柄发送机体坐标系下的
前后速度、侧向速度和转向速度，策略负责生成关节动作。当前版本复用各 task 已有的 mjlab 环境：
MILD 保留自定义接触求解，Parkour 保留深度图及观测历史。

## 启动

以下命令从项目根目录运行。已有环境可先执行 `uv sync`，使新增入口生效。

```bash
# 查看支持的任务
uv run instinct-teleop --list-tasks

# MILD-Walk：换成对应训练任务的 .pt checkpoint
uv run instinct-teleop mild_walk \
  --checkpoint-file logs/instinct_rl/<experiment>/<run>/model_<iteration>.pt \
  --viewer native

# Parkour PPO/AMP
uv run instinct-teleop parkour_amp \
  --checkpoint-file /absolute/path/to/parkour_ppo/model_<iteration>.pt \
  --viewer native

# Parkour SuperMimic
uv run instinct-teleop parkour_supermimic \
  --checkpoint-file /absolute/path/to/parkour_supermimic/model_<iteration>.pt \
  --viewer viser
```

路径中的 `<...>` 是占位符，需要替换。当前加载 `.pt` checkpoint，直接使用其中的策略参数及
适用的观测归一化状态；不需要初始化训练算法或加载 AMP 参考动作数据。请选择与 checkpoint
对应的任务，观测或模型结构不匹配时会报错。当前入口尚未提供 ONNX 推理选项。

| 简称 | 对应完整 task ID | 默认控制范围 |
| --- | --- | --- |
| `mild_walk` | `Instinct-Locomotion-Mild-Walk-G1-v0` | 前后 `[-0.5, 1.0] m/s`，转向 `[-1, 1] rad/s` |
| `parkour_amp` | `Instinct-Parkour-Target-Amp-G1-v0` | 前进 `[0, 0.8] m/s`，转向 `[-1, 1] rad/s` |
| `parkour_supermimic` | `Instinct-Parkour-Target-Amp-SuperMimic-G1-v0` | 前进 `[0, 0.8] m/s`，转向 `[-1, 1] rad/s` |

首个位置参数也接受表中的完整 task ID，以及其 `-Play-v0` 形式。它们都映射到对应遥操作配置，
不会使用 Play 的固定速度覆盖。任务通过首个位置参数选择，不要再传 `--task`。

MILD 默认允许倒退；当前 Parkour 策略的训练配置以向前运动为主，因此默认不允许倒退。
三个配置的横向速度范围都是 `(0.0, 0.0)`。增加命令范围不会自动赋予模型新的运动能力；
使用支持侧移或倒退的新 checkpoint 后，再在对应 task 配置中调整范围。

## 手柄和交互

手柄后端读取 Linux 的 `/dev/input/js0`，不依赖窗口焦点。手柄必须连接到运行 Python 进程的
主机，且当前用户具有该设备的读取权限。通过 Viser 在另一台电脑上看画面时，浏览器不会传输
那台电脑的手柄输入。

默认按常见 Xbox 手柄映射设置，设备驱动不同可能需要调整索引：

| 输入 | 默认 Linux 索引 | 作用 |
| --- | --- | --- |
| 左摇杆上下 | axis `1` | 前进、后退 |
| 左摇杆左右 | axis `0` | 左右侧移，受当前任务的范围限制 |
| 右摇杆左右 | axis `3` | 左右转向 |
| Start 类按键 | button `7` | 暂停、继续 |
| A 类按键 | button `0` | 机器人复位，并保持暂停 |
| Back 类按键 | button `6` | 退出 |

正方向为前、左、逆时针转向；三个轴的默认符号都是 `-1`。默认死区是 `0.1`。
以下示例更换设备、转向轴、方向和按钮：

```bash
uv run instinct-teleop mild_walk \
  --checkpoint-file /absolute/path/to/model.pt \
  --gamepad.device /dev/input/js1 \
  --gamepad.yaw-axis 2 \
  --gamepad.yaw-sign 1.0 \
  --gamepad.deadzone 0.15 \
  --gamepad.pause-button 9
```

默认启动后暂停。先让摇杆回中，再按暂停/继续键开始；暂停、复位及断线后都需要摇杆回中。
断线会清空命令并暂停，后端会尝试重连；重连后需要手动继续。机器人跌倒或触发任务终止时暂停，
按复位键后再继续。终止不会自动把机器人送回出生点。

Native 窗口也支持空格暂停/继续、Enter 复位；Viser 使用界面中的对应按钮。窗口显示连接状态、
目标速度、经过限速和平滑后的命令，以及机器人实际速度。可以用 `Ctrl+C` 退出进程。

手柄目标在环境的命令管理器更新时生效，随后生成观测，因此具有一个策略步的命令延迟。
这样可以让速度观测、Parkour 历史缓存和 MILD 步态时钟使用同一份命令；渲染和读取手柄不会推进
观测历史。松开摇杆后目标归零，实际命令按配置的加速度限制归零；暂停或断线会立即清零命令。

## 修改配置的位置

配置使用 Python dataclass。日常参数修改集中在以下文件，不需要修改公共运行循环：

| 修改内容 | 文件 |
| --- | --- |
| MILD 默认速度范围、加速度、场景、相机 | [tasks/mild_walk/config.py](tasks/mild_walk/config.py) |
| Parkour 默认速度范围、地形、出生偏移、相机；PPO 与 SuperMimic 共用 | [tasks/parkour/config.py](tasks/parkour/config.py) |
| 公共配置字段及默认值 | [config.py](config.py) |
| 手柄轴、按钮、死区、设备路径的公共默认值 | [inputs/gamepad.py](inputs/gamepad.py) |
| MILD 任务接入、复位与接触配置 | [tasks/mild_walk/adapter.py](tasks/mild_walk/adapter.py) |
| Parkour 任务接入、地形筛选与传感器依赖 | [tasks/parkour/adapter.py](tasks/parkour/adapter.py) |
| 支持的任务及对应适配器 | [registry.py](registry.py) |
| 策略加载 | [policies/loader.py](policies/loader.py) |
| 公共运行及交互状态 | [runtime.py](runtime.py)、[viewers.py](viewers.py) |

基础机器人、物理步长、动作缩放和观测定义仍引用原 task 配置。覆盖顺序为：

```text
原 task 配置 → 公共遥操作默认值 → task 遥操作默认值 → 命令行参数
```

任务适配器负责落实交互设置：单环境、手动命令、确定性出生、取消定时结束、去掉训练随机化、
奖励、课程和 AMP 参考依赖。原 task 的物理积分和策略所需的观测历史保持原有定义。

临时参数用命令行覆盖。布尔值显式写 `True` / `False`；元组使用带引号的 Python tuple：

```bash
uv run instinct-teleop mild_walk \
  --checkpoint-file /absolute/path/to/model.pt \
  --control.lin-vel-x '(-0.3, 0.8)' \
  --control.linear-acceleration 0.6 \
  --camera.distance 4.0 \
  --camera.follow True \
  --scene.spawn-offset '(0.1, 0.0, 0.0)'

uv run instinct-teleop parkour_amp \
  --checkpoint-file /absolute/path/to/model.pt \
  --scene.terrain pyramid_stairs \
  --scene.terrain-difficulty 0.3
```

Parkour 默认只生成一块 `perlin_rough` 地形。`scene.terrain` 可以选原 Parkour `sub_terrains`
中的名称，例如 `pyramid_stairs`、`square_gaps`、`boxes`；`task` 表示保留原任务的所有地形类型，
再按 `terrain_rows`、`terrain_cols` 和 seed 采样生成。MILD 当前使用平面接触，支持 `plane` 或
`task`。`spawn_offset` 是相对原任务出生位置和地形原点的米制偏移，`spawn_yaw` 单位为弧度。

查看最终配置，不启动仿真或读取手柄：

```bash
uv run instinct-teleop mild_walk --print-config True

uv run instinct-teleop parkour_amp --print-config True \
  --scene.terrain pyramid_stairs \
  --dump-config outputs/teleop/parkour_config.json
```

JSON 包括最终参数、实际物理步长/策略步长和传感器等摘要，以及 `sources` 文件路径索引。
`sources` 指向公共配置、任务配置、适配器和原 task 配置，便于定位修改；它不提供逐字段的赋值追踪。
正常运行时也可传 `--dump-config` 保存配置快照，其中还会记录实际选择的设备和 viewer。

## 显示方式与无手柄检查

| `--viewer` | 行为 |
| --- | --- |
| `auto` | 检测到 `DISPLAY` / `WAYLAND_DISPLAY` 时使用 Native，否则使用 Viser |
| `native` | 本机 MuJoCo 窗口，需要可用桌面显示 |
| `viser` | 浏览器显示，打开终端输出的地址；手柄仍由 Python 主机读取 |
| `none` | 不创建显示窗口 |

没有手柄时，`--input zero` 提供零速度目标。以下命令使用零关节动作检查环境能否运行，不加载策略：

```bash
uv run instinct-teleop mild_walk \
  --agent zero --input zero --viewer none \
  --start-paused False --max-steps 20

uv run instinct-teleop parkour_amp \
  --agent zero --input zero --viewer none \
  --start-paused False --max-steps 20
```

零动作检查不代表策略会走路或稳定站立。`max_steps` 是最多执行的策略控制步数；headless 模式
触发终止时会提前退出。`none` 与 `input zero` 同时使用时，必须指定 `--start-paused False`
和有限的 `--max-steps`。使用训练模型做零速检查时，换为 `--agent trained` 并提供 checkpoint。

## 新增 task

1. 在 `tasks/<name>/config.py` 提供 `default_config()`，集中该任务的交互默认值。
2. 为它选择或编写 adapter。物理环境由原 task registry 选择；适配器主要准备配置和绑定手动命令。
   若观测、复位和命令语义相同，可以复用现有适配器。
3. 在 [registry.py](registry.py) 中添加一个 `TaskSpec`，明确简称、原 task ID、配置模块、
   适配器入口和原配置模块。共享 `runtime.py` 不需要按 task 名增加条件分支。

接入时需要核对策略观测顺序、历史长度、动作维度、归一化及任务特有依赖。新的命令语义若不是
三维速度，应在对应适配器和命令模块中实现，保留公共手柄读取与运行循环。

## 验证

```bash
# CPU 单测：输入事件、命令、配置、策略加载及交互状态机；GPU 测试默认跳过
.venv/bin/python -m unittest discover -s tests -p 'test_teleop*.py'

# 实际 MILD / Parkour 环境：非零命令、观测历史、步态时钟及复位
INSTINCT_TELEOP_GPU_TESTS=1 .venv/bin/python -m unittest discover -s tests -p test_teleop_integration.py
```

实际手柄轴和按钮映射仍需按所连接设备确认；无手柄单测通过模拟 Linux 事件和输入状态验证交互。
