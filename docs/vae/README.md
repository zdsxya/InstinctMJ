# G1 29DoF VAE 地面适应实验

按 [init.md](init.md) 实现，2026-11-01。借鉴 [MILD 的状态估计与调制思路](https://arxiv.org/html/2608.19955)，机器人、动作、执行器及训练环境以当前项目的 G1 29DoF 配置为准。下面的网络尺寸、损失权重和状态缩放是工程默认值，不作为论文原始超参数的复现声明。

## 机器人与实验基线

三个新实验共同继承 `G1MildGaitEnvCfg`：保留全身 29 个关节的位置动作、关节顺序、默认姿态、按关节设置的动作缩放、原生 G1 PD 执行器、1 ms 物理步长、50 Hz 策略频率、现有 16 项基础奖励、速度指令、环境随机化和 MILD 接触模型，并统一加入下面的交替步行观测和奖励。上肢和腰部均由策略控制。VAE 不再使用旧实现的 12 关节动作、100 Hz 策略、专用腿部 PD 或论文式奖励替换。

这三组是共享基础 PPO 环境的消融实验。已有 `Instinct-Locomotion-Mild-Walk-G1-v0` 仍是原来的 AMP 步行任务，不会因本次修改自动启用 VAE；新实验也不加载 AMP 动作库或判别器。

三组与 `Mild-Walk` 共用 `gait_clock`（相位的 sin/cos）和三项奖励：`gait_contact`（+2.0，按相位要求左右脚支撑/摆动）、`swing_clearance`（+0.6，摆动脚跟随抬脚高度曲线）、`both_feet_airborne`（−2.0，惩罚双脚同时离地）。默认周期 0.8 s，左右脚相差半周期，支撑比例 0.6，最高抬脚目标 0.065 m，接触判定阈值 5 N。站立指令下时钟为零、目标为双脚支撑；原地转向仍启用步态。`gait_period`、`gait_stance_fraction`、`gait_clearance`、`gait_contact_threshold` 和 `randomize_gait_phase_by_env` 可通过 `--env.*` 配置。训练日志同时包含 `Step/Walk/phase_agreement` 和单/双支撑、腾空比例。

| 实验 | 训练任务 ID | Actor | Critic 网络输入 |
|---|---|---|---|
| Baseline | `Instinct-Locomotion-Mild-VAE-Baseline-G1-v0` | 当前观测（含时钟）→ PPO MLP | 基础观测 + 时钟，101 维 |
| VAE | `Instinct-Locomotion-Mild-VAE-G1-v0` | PPO + 多头 VAE，直接使用 μ | 基础观测 + 时钟，101 维 |
| Full | `Instinct-Locomotion-Mild-VAE-Full-G1-v0` | PPO + 多头 VAE + 足速调制 μ | 基础观测 + 时钟，101 维 |

三组均默认 seed 42、5000 次 PPO 更新、每个环境每次采样 24 步、初始动作噪声 1.0，PPO 优化参数相同。三组统一使用配置中的固定观测尺度，关闭运行均值/方差归一化。Baseline 的输入为单帧，VAE/Full 使用历史，这也是实验差异之一。

## 数据流和单位

当前 G1 每帧策略观测为 98 维：角速度 3、重力投影 3、速度指令 3、29 个关节相对默认位置的偏移、29 个关节速度、29 个上一时刻动作，以及步态时钟 2。默认 H=10，总输入 980 维；代码根据观测描述计算维度，不写死 98/980。时钟随其余观测一起进入历史编码和当前帧重建，当前时钟也直接输入 Actor。

```text
10 帧策略观测（980）→ Encoder [256,128]
                         ├→ 机体线速度（3）
                         ├→ 左右脚底速度（6）
                         ├→ 左右脚底高度（2）
                         └→ μ（16）、logvar（16）
                              ├→ 重参数化 z → Decoder [128,256] → 当前帧重建（98）
                              └→ 确定性 μ
估计脚底速度（6）→ Modulation [64,64] → scale、shift（各16）
调制结果 = (1 + scale) * μ + shift
当前观测（98）+ 缩放后的估计状态（11）+ latent（16）→ Actor [256,128,128] → 29 个动作
基础特权观测（99）+ 时钟（2）→ Critic [256,128,128] → V
```

VAE 组跳过调制，直接把 μ 输入 Actor。Full 的调制末层接近零初始化，使初始变换接近恒等。Actor 在 rollout、PPO 概率重算和推理时都使用确定性 μ；只有重建分支采样 z。

速度监督是**相对世界运动、在机器人根链节（torso）坐标系表达**的速度。脚底速度和高度按接触离散单元面积加权，高度是相对未变形地面的有符号高度，可以为负。监督标签和预测头均使用 SI 单位。默认 `state_scales=(1,1,10)` 分别缩放机体速度、足速、足高：只用于 Actor 的估计状态输入和监督误差，足高以 0.1 m 为参考尺度；调制输入保留 m/s。

真实机体速度、真实脚底状态及材料参数均不进入 Encoder。额外 11 维标签与 Critic 观测一起存入 rollout（VAE/Full 共 112 维），但标签不参与 Critic 前向；Critic 也不使用 latent。当前配置不额外向 Critic 加入材料参数。

历史复用 MJLab `ObservationManager` 的 GPU 并行缓冲区：按观测项依次排列，每项内部从旧帧到新帧；每个控制步更新一次。reset 仅将对应环境的历史填满为重置后首帧，其余环境保持不变。训练、Play 和导出模型遵循相同排列与重置规则。10 帧在 50 Hz 下首尾相隔 0.18 s。

## 损失与日志

默认联合目标：`0.7 * 原 PPO loss + 0.3 * (重建 MSE + 0.001 * KL + 三项缩放状态 MSE)`。KL 对 latent 维求和、batch 求平均。PPO 原有 GAE、clip、value loss、entropy、动作和旧 log probability 的存储与更新逻辑保留。

TensorBoard 与每个 run 的 `progress.jsonl` 包含：

- `Train/PPO_Actor_Loss`、`Train/PPO_Critic_Loss`、`Train/Policy_Entropy`。
- `VAE/Reconstruction_Loss`、`VAE/KL_Loss`、三项 `VAE/*_MSE`（物理单位），以及 `VAE/*_Scaled_MSE`（实际训练使用的缩放误差）。
- `VAE/Latent_Mean`、`VAE/Latent_Std`（后验标准差的均值）、`VAE/Modulation_Scale`（残差 scale 均值）、`VAE/Modulation_Shift`。
- `Evaluation/Velocity_Tracking_Error`（采样状态的平面速度误差范数）、`Evaluation/Episode_Reward`、`Evaluation/Episode_Length`（步数）。后两项在有 episode 结束后记录。
- `Evaluation/Fall_Rate`：累计因姿态倒地或机身接触结束的 episode 数 / 累计结束 episode 数；尚无结束 episode 时为 0，必须结合 `Counters/completed_episodes` 读取。单独的过深穿透终止不算跌倒。
- 各模块梯度范数、更新次数、采样数和耗时。

`Evaluation/` 是训练期间的诊断命名，不代表已做独立性能评估。不会因为奖励短时平台自动宣布收敛或停止训练。

## 启动与切换

从仓库根目录运行 Full：

```bash
.venv/bin/instinct-train Instinct-Locomotion-Mild-VAE-Full-G1-v0 \
  --num-envs 512 --device cuda:0 --agent.seed 42 --agent.max-iterations 5000
```

将任务 ID 换成上表的 Baseline 或 VAE 即切换实验。三组比较时使用相同环境数量、seed 和更新预算。小规模链路检查可以用 `--num-envs 8 --agent.max-iterations 5`。

主要参数由 CLI 暴露：

```bash
# 同步修改环境与策略历史长度，示例为 8 帧 / 32 维 latent
.venv/bin/instinct-train Instinct-Locomotion-Mild-VAE-Full-G1-v0 \
  --num-envs 512 --env.history-length 8 --agent.policy.history-length 8 \
  --agent.policy.latent-dim 32 --agent.policy.state-scales '(1, 1, 10)' \
  --agent.algorithm.regression-weight 0.3 --agent.algorithm.vae-kl-weight 0.001
```

Encoder/Decoder/Modulation/Actor/Critic 的隐藏层宽度、`use_modulation` 和 reconstruction/base_velocity/feet_velocity/feet_height 四个损失权重也都可配置。禁用整个 VAE 使用 Baseline 任务；仅设置 `env.use_vae=False` 而保留 VAE Runner 会因配置不一致报错。

此项目的 CLI 将列表和元组作为单个 Python 字面量解析，例如 `--agent.policy.encoder-hidden-dims '[256, 128]'`。

回放：

```bash
.venv/bin/instinct-play Instinct-Locomotion-Mild-VAE-Full-G1-Play-v0 \
  --checkpoint-file /absolute/path/to/model_5000.pt
```

每个实验都有对应的 `-Play-v0`。Play 从 checkpoint 恢复网络结构、状态尺度、历史长度及步态参数，保留 Play 的无观测噪声和确定性环境设置。恢复训练使用原实验 ID 和相同网络、步态参数，加 `--agent.load-run /absolute/path/to/run --agent.load-checkpoint model_5000.pt`；`--agent.max-iterations` 是本次追加更新次数。

checkpoint 保存策略、优化器、学习率、计数、随机数状态以及观测/控制约定；加载时核对关节顺序、动作缩放/偏移、默认姿态、PD 参数、控制周期、历史、步态和网络设置。加入时钟后 schema 为 v6，旧的无时钟 29DoF checkpoint（v5）及 12 动作 checkpoint 均不兼容，需要重新训练。恢复训练会重建环境，不宣称精确恢复仿真器状态或轨迹。

Play 的 `--export-onnx True` 可导出 Actor，旁边生成 `observation_contract.json`，记录历史布局、观测尺度、关节顺序、控制参数及步态时钟约定。ONNX 输入已经是处理好的历史观测，使用者须按该约定在外部生成时钟并维护缓存；项目现有 `--use-onnx` 回放入口仍只支持 Parkour。

## 文件职责

| 文件（相对仓库） | 作用 |
|---|---|
| `src/instinct_mj/tasks/locomotion/config/g1/mild_vae_env_cfg.py` | 29DoF 共享环境、历史和训练标签 |
| `src/instinct_mj/tasks/locomotion/config/g1/mild_gait_env_cfg.py` | Walk 与三组 VAE 共用的步态参数、时钟和三项交替奖励 |
| `src/instinct_mj/tasks/locomotion/config/g1/mild_vae_rl_cfg.py` | 三组配置及可调网络/损失参数 |
| `src/instinct_mj/tasks/locomotion/config/g1/__init__.py` | 三组 train/play 注册 |
| `src/instinct_mj/tasks/locomotion/mild/estimator.py` | 独立多头 VAE 和 Modulation 模块 |
| `src/instinct_mj/tasks/locomotion/mild/actor_critic.py` | Actor 数据流、原始 Critic、辅助损失、ONNX |
| `src/instinct_mj/tasks/locomotion/mild/ppo.py` | 原 PPO 上增加联合损失与诊断 |
| `src/instinct_mj/tasks/locomotion/mild/runner.py` | 本地模块构建、统一日志、保存/加载及控制约定 |
| `src/instinct_mj/tasks/locomotion/mild/vae_env.py` | 复用全身原生控制并记录环境指标 |
| `src/instinct_mj/tasks/locomotion/mild/vae_mdp.py` | 同时刻 SI 监督状态 |
| `src/instinct_mj/tasks/locomotion/mild/vae_wrapper.py` | 复用标准观测打包，避免重复历史 |
| `src/instinct_mj/tasks/locomotion/mild/deployment.py` | 从 checkpoint 恢复推理网络及历史配置 |
| `src/instinct_mj/scripts/instinct_rl/play.py` | 环境创建前调用可选的推理配置恢复钩子 |
| `tests/test_mild_vae.py`、`tests/test_mild_vae_deployment.py` | 网络、控制契约、历史、PPO、部署和 GPU 集成验证 |
| `.gitignore` | 允许跟踪 VAE 文档与新增部署测试，运行日志继续忽略 |

## 验证记录与边界

加入步态时钟后的 v6 已通过 50 项 CPU 测试和 1 项 RTX 5080 GPU 集成测试：覆盖三组共享配置、左右脚半周期切换与奖励、站立/转向、历史及局部 reset、PPO 更新、ONNX 对照、步态参数保存恢复，以及原接触和 AMP 手臂奖励回归。GPU 集成验证了时钟历史、三项奖励、步态诊断和 checkpoint；这些检查不代表训练后已经获得稳定交替步态。

以下是加入步态时钟之前（schema v5）的历史验证，不能作为当前 v6 的验证结果。原 `logs/vae29_validation_20261101/` 和短测训练目录已于 2026-11-04 按要求清理。短训练只检验链路和数值有效性，不代表行走性能或地面适应能力。

| 验证 | 结果 |
|---|---|
| 全部 MILD 相关单元及 GPU 集成测试 | 48 项全部通过，无跳过；含原接触、AMP 手臂奖励回归 |
| Baseline / VAE / Full 实际训练入口 | 每组 8 环境 × 24 rollout 步 × 5 更新 = 960 transitions；全部正常结束 |
| Full 保存后续训 | 从 5 更新恢复至 7 更新，累计 1344 transitions，计数与学习率恢复正常 |
| Full 标准 Play 和 ONNX 导出 | 2 环境运行 24 步成功；3 样本 ONNX 与 PyTorch 最大绝对误差 `1.50e-7` 以内 |
| 自定义结构贯通检查 | H=4、latent=7、不同隐藏层宽度、modulation=False、state_scales=(2,3,10)、初始噪声0.3；完成1更新并通过同一 Play 入口自动恢复配置 |
| 自定义 ONNX | 输入384、Actor输入114、输出29；3样本与PyTorch最大绝对误差 `1.05e-7` 以内 |
| 模型与优化器数值 | 三组、续训及自定义 checkpoint 的全部模型/优化器张量均有限 |

上述测试原始日志、短测 checkpoint 和导出产物已删除；正式训练、分段续训及其评估记录保留。清理明细见 `outputs/log_cleanup/20261104_125234/manifest.json`，运行产物按项目惯例不进入 Git。

```bash
MPLCONFIGDIR=/tmp/instinct-mpl INSTINCT_MILD_GPU_TESTS=1 \
  .venv/bin/python -m unittest discover -s tests -p 'test_mild*.py' -v
```

目前复用现有 MILD 接触参数随机化，未新增硬软空间交界地形，也未修改接触求解器。材料随机化不能代替硬地、软地、交界处的独立行走验收。长训练收敛、跨材料泛化及 G1 真机部署仍需验证；短时 smoke checkpoint 不用于性能结论。当前消融 Runner 只支持单 GPU。
