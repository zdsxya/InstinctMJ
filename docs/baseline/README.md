# MILD baseline 复现

> 此页记录首次 VAE+AMP baseline。当前调整后的 12 维 PPO+VAE 配置、UMR 核查和训练状态见 [baseline 调整 1](../change1.md)。首次训练旧日志按新 guide 的清理要求处理。

本目录对应 `docs/mytask.,md` 的「复现 baseline」。依据为仓库中的
[`MILD.pdf`](../MILD.pdf)，主要核对第 II 节、Fig. 4、Table I/II 和式 (1)–(17)。
论文页面：[arXiv:2608.19955](https://arxiv.org/abs/2608.19955)。

实现范围是 **MuJoCo Warp 上的 G1 方法复现，保留本仓库的 UMR AMP**：柔性接触、
多头 VAE 状态估计、隐式地形 latent、足速 modulation、非对称 actor/critic 和联合优化均已接通。
这不是作者 SA01/Isaac Gym 实验的逐数值复刻；论文未公开的超参数、公式歧义、G1 适配和
接触求解器差异均列在下文。训练启动不等于证明收敛，更不代表复现了论文的实机性能。

## 1. Task 整理

| Task | 处理 | 算法 |
| --- | --- | --- |
| `Instinct-Locomotion-Mild-G1-v0` / `-Play-v0` | 删除注册和独立工厂 | 原普通 PPO 接触消融 |
| `Instinct-Locomotion-Mild-Walk-G1-v0` / `-Play-v0` | 保留 | MLP + UMR AMP |
| `Instinct-Locomotion-vae-G1-v0` / `-Play-v0` | 新增，大小写按任务要求 | VAE + modulation + PPO + UMR AMP |

`G1MildEnvCfg` / `MildRlEnv` 仍作为共享接触配置和环境基类，不再作为独立训练任务。
`G1MildPPORunnerCfg` 是保留的内部类型化基类。旧训练数据和 checkpoint 没有删除。
既有用户修改的腰部、手臂参考约束继续生效。

## 2. 接触模型逐项核对

| 论文部分 | 实现与核对结果 |
| --- | --- |
| 足底离散，0.0195 m，154 单元 | 保留 0.0195 m 网格，对 G1 足底胶囊碰撞几何的**并集**做面积积分；G1 每脚 44 单元，共 88，不强制套用 SA01 的 154 |
| 式 (1)–(4)，锥体与附加质量 | `solver.granular_law` 用正侵入深度 `d=h0-z`，解析积分得到法向静力、导数、附加质量和侵入阻力；参考面积固定为 `0.0195²`，按实际单元面积加权，避免加密网格改变材料刚度 |
| 式 (4) 的符号 | 原式的积分方向与正的支撑力存在解释空间。本实现使用正的静力和锥侧面积幅值；不是直接照抄带符号积分 |
| 式 (5) 动态项 | 侵入时计算质量招募阻力；离开地面时无吸力、无招募阻力。附加质量通过广义质量矩阵隐式求解；额外使用 `dt²*dF/dd` 的一阶隐式刚度稳定项。这些是明确的工程选择 |
| 式 (6)–(10)，全身耦合 | 每个 1 ms 子步清除旧自定义外力，刷新 M、Jacobian、自由动力学，然后求法向与切向力，作用到足部 COM；左右脚共享同一个广义速度解 |
| 式 (11)–(13)，连续性与摩擦 | 同一刚体的点速度直接来自同一 `J*qvel`，连续性由运动学满足；摩擦使用耦合的投影梯度法和 `mu*Iz` 圆盘。没有按印刷式把标量连续性误差直接减到向量上，也没有用有歧义的 `mu*Ixy` 作半径 |
| 收敛条件 | 为 GPU graph 使用固定 24 次摩擦迭代、3 次外层耦合迭代；`1e-4 N*s` 用于诊断，不是自适应停止准则。记录实际法向本构残差和未达容差比例，不能假称每一步完全收敛 |
| 式 (14)，漂移 | 对每个持续侵入单元累计水平路径长度，卸载/reset 后清零；阻力反向于水平速度，零速度无除零 |
| 式 (15)，ZMP | 增加水平名义表面上的合力矩交点计算与 `Mild/max_zmp_offset`；采用 MuJoCo 的 `r×F` 力矩约定。法向支撑不足时返回无效 mask |
| 碰撞 | 只屏蔽足底与名义地面的原生碰撞，保留其他机器人碰撞。无隐藏硬底板；超过 0.20 m 侵入深度终止。原生非足部接触与自定义接触并非联合约束求解 |
| 时序 | 本次补充 policy 边界的 `sim.forward()`，让 reward/termination 的派生状态与最终积分位置一致。估计标签另用当前 Jacobian/足底位置，不从上个子步缓存的接触位置生成 |

Table II 的材料范围在新 VAE task 中如下；`Mild-Walk` 保留已有材料配置，便于比较旧模型。

| 参数 | 论文 | VAE task |
| --- | --- | --- |
| alpha | [0.698, 1.310] rad | 相同 |
| nu / recruitment | [0.8, 2.4] | 相同 |
| cg / mass_scale | [-1.4, 4.6] | **[0, 4.6]** |
| cd / drag_scale | [-6, 28] | **[0, 28]** |
| phi / volume_fraction | [0.50, 0.64] | 相同 |
| mu / friction | [0.2, 1.3] | 相同 |
| k_pen | [8, 1400] N/m | 相同，按参考单元面积归一化 |
| sigma_cone | [1e4, 2.4e5] N/m³ | 相同 |
| rho | 未公开 | 1600 kg/m³ |
| k_D / b_D | 未公开 | [1000,4000] / [0.5,2]，继承已有工程值 |

不对负 cg/cd 悄悄取绝对值：在当前正深度模型中，它们会产生负附加质量/主动注能。
训练明确采用表中非负子集，配置校验拒绝负数。尚未得到作者对负系数有效域的说明，
所以**完整有符号随机化范围未被复现**。材料独立均匀采样，每次 episode reset 更新，
同一环境双脚同材质；不支持地面永久形变或空间材料地图。这些均不冒充实测泡沫标定。

## 3. VAE、modulation 与数据流

实现位于 `src/instinct_mj/tasks/locomotion/mild/`。

```text
policy history (960 = 10×96，带噪，含当前帧)
        │
   Encoder [256,128]
        ├── state head：base v(3), feet v(6), feet h(2)
        └── latent mean / logvar (16)
                    │
估计 feet v(6) → Modulation [64,64] → scale / shift
                    │                 │
                    └──── z_tilde = (1+scale_delta)*mean + shift
                                      │
current o(96) + 估计 state(11) + z_tilde(16) → Actor [256,128,128] → 29 actions
clean o(96) + 真值 state(11) + 材料(22) + z_tilde(16) → Critic [256,128,128]

mean + exp(0.5*logvar)*noise → Decoder [128,256] → reconstructed current o(96)
```

- `actor_critic.py`：五个主要 MLP 和三类状态回归。actor 从 history 的每个具名 segment
  取最后一帧，不额外拼接完整原始 history；遵循 Fig. 4。history 是 oldest → newest。
- `vae_wrapper.py`：将**同一次采样的 policy history**附加到 critic 存储，供共享 encoder
  重新计算 latent。critic 存储宽度为 1089，但其 MLP 输入只有 145；history 不重复直连 critic。
- actor 不读取真实速度、足高、材料参数或参考动作。AMP reference 仅供 discriminator。
  modulation 在训练和推理时都使用**估计足速**，不存在 teacher forcing。
- latent 的策略/价值分支取均值，VAE 重构分支重参数采样；这样固定权重时 rollout/update
  的 action log-prob 一致，避免重新采样 latent 改变 PPO 概率比。这是论文未明确的实现选择。
- base/feet 速度均为相对世界的速度、表达在 root-link 坐标系；feet 取足底面积加权中心。
  足高相对未变形表面，支撑时可为负。足高标签乘 10，其他状态标签保持 SI 数值。
- 固定输入缩放：角速度 ×0.25、关节速度 ×0.05，其余本体输入不缩放；不使用运行统计
  normalizer。材料参数的固定尺度见 `vae_mdp.MATERIAL_SCALES`，防止 1e5 量级压力主导 critic。
- 监督标签与存储的 `o_t` 同时取得；step/reset 后的 `o_(t+1)` 不回填上一帧标签。
  部分 reset 只填充对应环境的历史，其他环境的时间轴不变。
- H=10、latent=16、网络宽度和 KL 权重均为工程超参数，论文未给数值。

联合 loss 对应式 (16)–(17)：

```text
L_reg = MSE(reconstructed_o, current_o) + 0.001 * KL(q(z|history) || N(0,I))
        + MSE(base_velocity) + MSE(feet_velocity) + MSE(feet_height * 10)
L_joint = 0.3 * L_reg + 0.7 * (L_surrogate + L_value - 0.008 * entropy)
```

三个监督 MSE 分别按本组维度取均值，KL 按 latent 维度求和、按 batch 取均值。
`ppo.py` 把联合权重只施加一次；编码器、回归头、调制网络、actor/critic 同步更新。
AMP discriminator 用独立 optimizer，保留 MSE / quadratic style reward；新任务每步权重
0.0125，对应旧任务 50 Hz 下 0.025 的每秒强度。AMP **不在原论文中**。
同时修正本地扩展中 upstream AMP `logit_weight_decay` loss key 与配置系数的名称对应。

## 4. 奖励、频率与随机化

接触和物理积分 1000 Hz，VAE task 动作更新 100 Hz，H=10 为约 0.1 s 历史。
PD 由 MuJoCo actuator 在物理子步计算；与论文所说的 100 Hz PD 实现仍有区别。
G1 29 自由度和现有关节 PD 参数保留，不套用论文 SA01 的 12 自由度 PD 参数。
AMP 原始 50 Hz 数据插值到 100 Hz，20 帧历史保持约 0.2 s 时间窗。

| Table I | 新任务实现 |
| --- | --- |
| 线速度 | `1.4*exp(-5*error_xy²)` |
| 角速度 | `1.1*exp(-5*error_yaw²)` |
| gait alignment | `1.4*(1.3*sum(contact==desired_support)-0.3)` |
| clearance | `-2.4*sum(abs(h-target_h)*swing)`；纠正印刷式中正误差被奖励的方向 |
| 足滑移 | `-0.1*sum(norm(foot_velocity)*contact)` |
| base acceleration | `0.2*exp(-3*norm(v_base[t]-v_base[t-1]))`；解释印刷式混用 base/feet 下标 |
| action smoothness | `-0.003*sum((a[t]-a[t-1])²+(a[t]-2a[t-1]+a[t-2])²)` |

gait 使用 0.8 s 周期、0.6 支撑比例、0.065 m 正弦摆动高度、5 N 接触阈值。
论文没有公开这些数值。新任务不向 actor 输入 gait phase；零速度命令改为双支撑；
原地转向也视为行走。保留 G1 的站姿/终止/关节限制/腰部/手臂参考约束和腾空惩罚，
这些是适配项。所有环境奖励沿用 manager 的 dt 缩放，只缩放一次。

前进命令 [0,1.2] m/s、转向 [-1.5,1.5] rad/s、横向 0，10% 站立环境。
Table II 的躯干附加质量 [-4,4] kg、COM 三轴各 [-6,6] mm 在 startup 采样；质量与惯性
沿用 pseudo-inertia 随机化。每次 reset 采样 motor offset ±0.035 rad 和 motor/IMU/command
各自独立的 1–10 ms 延迟，按 1 ms 向上量化。motor target、IMU 和命令缓存按物理子步推进，
不是用 10 ms policy lag 近似；reset 后填充新环境状态。速度 reward 跟踪外部原命令，
actor 看到延迟命令，此语义在训练/回放接口中固定。推扰关闭。原生碰撞材质随机化仍保留。

## 5. 运行

依赖现有 `.venv`、相邻 `mjlab` 和 UMR reference 数据：
`outputs/umr/motions/motions.yaml`。可通过 `UMR_MOTION_DIR` 改变数据目录。

```bash
# 带初始化计时和独立进程看护的正式训练
.venv/bin/python scripts/baseline/train_bounded.py \
  --num-envs 1024 --iterations 5000 --max-hours 9

# 直接 CLI 短训
.venv/bin/python -m instinct_mj.scripts.instinct_rl.train \
  Instinct-Locomotion-vae-G1-v0 --num-envs 8 \
  --agent.max-iterations 2 --agent.save-interval 1

# 原生窗口回放
.venv/bin/python -m instinct_mj.scripts.instinct_rl.play \
  Instinct-Locomotion-vae-G1-Play-v0 \
  --load-run /absolute/path/to/run --checkpoint-file /absolute/path/to/run/model_5000.pt --viewer native

# 无窗口回放、导出完整 encoder/modulation/actor
.venv/bin/python -m instinct_mj.scripts.instinct_rl.play \
  Instinct-Locomotion-vae-G1-Play-v0 \
  --load-run /absolute/path/to/run --checkpoint-file /absolute/path/to/run/model_5000.pt \
  --viewer none --max-steps 100 --export-onnx True

# 恢复：max-iterations 是追加的更新数，环境动态状态重新 reset
.venv/bin/python -m instinct_mj.scripts.instinct_rl.train \
  Instinct-Locomotion-vae-G1-v0 --num-envs 1024 \
  --agent.load-run /absolute/path/to/run --agent.load-checkpoint model_1000.pt \
  --agent.max-iterations 1000
```

保存 actor/critic/VAE/modulation、两个 optimizer、AMP discriminator、Torch RNG、实际学习率、
episode/transition 计数、训练配置、接触配置和观测/关节顺序 schema。加载时校验 schema，
拒绝旧 flat/walk 权重或不匹配的 history/关节顺序。checkpoint 不恢复 MuJoCo 环境的中间物理状态。
支持单 GPU；多 GPU 入口明确报错，避免错误的梯度同步和共享日志写入。

## 6. 训练预算与日志

2026-10-13，RTX 5080 16 GB，24 steps/env、5 PPO epochs、4 minibatches，短训实测
（排除第一轮预热，不含环境初始化）：

| 并行环境 | 每次更新 | policy transitions/s |
| --- | --- | --- |
| 256 | 约 2.5 s | 约 2440 |
| 512 | 约 3.3 s | 约 3770 |
| 1024 | 约 4.8 s | 约 5120 |

选择 1024 环境、5000 次更新，最多 122,880,000 policy transitions，预计约 6.7 小时。
这是测量外推，会随接触场景、系统负载和 GPU 频率变化。
runner 在 8.5 小时预算内提前留出至少 60 s 保存最终 checkpoint；外部 supervisor
从启动开始计时，在 9 小时前发出 SIGTERM，最后必要时终止其自己启动的进程组。
时限保证作业结束，不能保证任意机器上完成 5000 次更新或策略达到论文指标。

训练在 `logs/instinct_rl/g1_locomotion_mild_vae_amp/<run>/` 写入：

- `params/env.yaml`、`params/agent.yaml`：实际配置。
- `progress.jsonl`：每轮损失、完成更新数、completed episodes、policy/physics transitions 和耗时。
- TensorBoard：五项 estimator loss、PPO KL、各模块梯度范数、AMP、接触残差、步态指标。
- `model_*.pt`：约每 100 轮及退出时保存；恢复迭代数以文件内 `iter` 为准。
- `training_status.json`：结束原因，区分迭代完成、预算完成、中断和失败。

supervisor 额外写入 `outputs/baseline/launch_<timestamp>.json` 和 `train_<timestamp>.log`，
记录 PID、启动命令、包含初始化的 deadline、最终退出码。实际启动记录见 [training.md](training.md)。

## 7. 验证与未完成的实验

```bash
INSTINCT_MILD_GPU_TESTS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/umr/test_pipeline.py
```

覆盖接触静力/摩擦/卸载/网格面积/ZMP、MuJoCo Jacobian/质量矩阵/合力矩对照、
无双重地面支撑、GPU graph 与 eager 轨迹、部分 reset，以及：

- 改变 critic 特权输入不改变 actor，PPO 未更新时 ratio=1。
- actor-loss 梯度到达 encoder/state head/latent mean/modulation。
- 五项辅助 loss 和 0.3/0.7 联合权重；decoder/logvar 梯度非零。
- 同一历史的最新帧切片、真实 GPU 标签对齐、异步历史/延迟 reset。
- 保存/加载动作一致、schema 拒绝、完整 ONNX 对照 PyTorch。
- 真实 CLI 短训、回放及预算到期后保存退出。

本次任务启动一次有时限的训练。尚未做三随机种子统计、原论文各接触 baseline 的公平消融、
软硬地面连续切换评估或实机验证；不把这些实验写成已完成，也不以短训 loss 有限推断已经学会走路。
