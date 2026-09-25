# MILD contact + 普通 PPO：当前实现与运行

本阶段已注册 `Instinct-Locomotion-Mild-G1-v0` 和 `Instinct-Locomotion-Mild-G1-Play-v0`。范围按最新决定收缩为足地接触模型的消融：**没有 VAE、历史状态估计、latent modulation 或联合损失**。完整论文路线仍见 [mild.md](mild.md)，回放扩展路线见 [mild-play.md](mild-play.md)。

用户随后确认优先实现交替行走：新增独立的 [MILD Walk 实验](mild-walk.md)，复用这里的足底求解，增加步态观测和奖励。本文描述的原 contact-only 消融配置保持不变。

## 运行命令

在 `/home/aaa/InstinctMJ` 下，使用已经安装依赖的项目 `.venv`：

```bash
# 正式训练：先用 512 环境；PPO 仍为 flat 的 5000 iterations × 24 steps/env。
uv run --no-sync instinct-train Instinct-Locomotion-Mild-G1-v0 --num-envs 512

# 快速检查训练、梯度更新、checkpoint 保存。
uv run --no-sync instinct-train Instinct-Locomotion-Mild-G1-v0 \
  --num-envs 64 --agent.max-iterations 2 --agent.save-interval 1

# 无需 checkpoint 的接触调试：零动作保持默认关节目标，不代表已学会走路。
uv run --no-sync instinct-play Instinct-Locomotion-Mild-G1-Play-v0 --agent zero

# 加载同一实验目录最新的 checkpoint 回放。
uv run --no-sync instinct-play Instinct-Locomotion-Mild-G1-Play-v0

# 定量验证应明确指定 checkpoint。
uv run --no-sync instinct-play Instinct-Locomotion-Mild-G1-Play-v0 \
  --checkpoint-file /absolute/path/model.pt --viewer none --max-steps 200

# 从指定 run 的 checkpoint 续训；max-iterations 表示本次追加的更新轮数。
uv run --no-sync instinct-train Instinct-Locomotion-Mild-G1-v0 --num-envs 512 \
  --agent.load-run /absolute/path/to/run --agent.load-checkpoint model_100.pt \
  --agent.max-iterations 4900
```

`--no-sync` 使用现有虚拟环境，不更新依赖。首次安装依赖的机器先按 README 执行 `uv sync`。也可用 `.venv/bin/instinct-train` / `.venv/bin/instinct-play` 直接运行相同参数。

日志和权重保存在 `logs/instinct_rl/g1_locomotion_mild_contact/<run>/`；每次训练保存展开后的 `params/env.yaml` 与 `params/agent.yaml`。自动回放可能选中短训权重，因此看正式训练效果时优先使用明确的 checkpoint 路径。短训权重只验证代码链路，不具备已收敛的步行能力。

默认配置仍是 4096 环境；上面的 512 命令降低首次运行的显存和计算负担。与 flat 比较时应对两边设置同样的 `--num-envs`、iterations 和 rollout 长度，保证 policy transition 预算一致。接触求解消耗额外算力，不能把相同交互预算等同于相同 wall time。

## envcfg 和 agentcfg 的实际差异

| 项目 | 当前 MILD contact 实现 |
| --- | --- |
| 基础配置 | 继承 `G1LocomotionFlatEnvCfg`，原 flat 任务不变 |
| actor / critic | 原始 MLP [256,128,128]，ELU；普通 `ActorCritic` + `PPO` |
| 观测 | 原 flat 六项 policy = 96 维；critic 多 base 线速度 = 99 维；没有新增特权地面参数 |
| 动作 | 29 个关节位置偏移，原 default pose 与 `beyondmimic_action_scale` |
| PPO | 原 flat 的全部超参数、24 steps/env、5000 iterations、保存间隔；experiment name 改为 `g1_locomotion_mild_contact`，`load_run` 改用可从 CLI 传入的 `str \| None` 类型 |
| 时间 | 物理步长 1 ms，decimation 20；策略依旧 50 Hz |
| 足底 | 根据原 G1 capsule 并集进行面积采样，默认每脚 44 个积分单元，约 0.01146 m²/脚 |
| 接触来源 | 禁止足底和原生 plane 接触；身体碰撞/自碰撞继续由 MuJoCo 处理 |
| 材料 | 每次 episode reset 采样一次，双足共享同一参数；正值工程 profile，见下表 |
| reward | 保留原 16 项及全部权重；只替换 feet_air_time / feet_slide 所读取的接触数据 |
| commands / events | 训练继承 flat，包括命令分布、standing、heading、质量、初态和推扰；额外采样地面参数 |
| termination | 保留 timeout/body contact，新增过深下陷 >0.20 m、严重倾倒；非有限状态直接报错 |
| play | 1 环境、固定材料中值、关闭噪声与物理/初态随机化，固定 vx=0.6、vy=wz=0 |

原生足底传感器已从新任务移除，以免将空的引擎足地碰撞当作接触。自定义接触保存足部力历史、当前接触/腾空时长；奖励的数学定义及权重沿用 flat。推扰仍可施加到 torso，自定义施力只拥有双足的 `xfrc_applied` 槽位，不能另用事件覆盖同一双足槽位。

## 接触模型与数值实现

### 面积和几何

从 G1 原 MJCF 中读取脚底 capsule 的位置、方向、半径和长度，在投影并集上做 0.0195 m 尺度网格及 16×16 子采样；重叠区域只积分一次。单元位置采用占据子采样的面积中心与下包络平均高度，不把 ankle 原点或平面矩形替代真实足底。

各单元面积之和约 0.01146 m²/脚。加密网格改变求积精度，不增加整个足底的材料刚度。原资产和 flat 的碰撞参数没有修改，新任务通过独立 spec factory 设置碰撞 mask：地面 bit 1、足底 bit 2，其他碰撞体接受两类。

### 正深度工程约定

来源是用户提供的 MILD 论文 §II-A/B、式 (1)–(14)。论文中 h 的符号、附加质量、侧面积与式 (13) 的表达存在歧义，本实现采用以下**明确的正深度工程形式**，不宣称是未经修改的作者代码：

```text
d = max(surface_z - point_z, 0), a = nu / tan(alpha)
A_ref = (0.0195 m)^2, R = sqrt(A_ref / pi), w_i = area_i / A_ref
r(d) = R + a*d
V(d) = pi * (R^2*d + R*a*d^2 + a^2*d^3/3)
C(d) = (V(d) - A_ref*d) / cos(alpha)
F_qs = w_i * (k_pen/A_ref * V(d) + sigma_cone*C(d))
m_a = w_i * cg * phi * rho * nu * V(d)
F_drag = w_i * cd * cg * phi * rho * nu * pi*r(d)^2 * min(v_z,0)^2
```

使用增长锥和正的侧面积积分，侵入惯性拖曳仅在向下运动时生效。当前实现卸载时随深度减小释放有效附加质量，离地清零；没有永久压实记忆、网格变形或加载/卸载材料滞回标定。

法向反力包含隐式附加质量项 `-m_a*a_z`。为稳定 1 ms 积分，准静态力按下一子步深度线性化：`F_base=F_qs-dt*K*v_z+F_drag`、`m_eff=m_a+dt²*K`，其中 `K=dF_qs/dd`。全身增广质量矩阵为 `M + Jz^T diag(m_eff) Jz`，不使用上一帧法向加速度代替当前耦合加速度。

每子步从当前 MuJoCo 动力学读取全身质量矩阵、广义速度、自由力和点 Jacobian；用相邻 Jacobian 估算 `Jdot*qdot`。解小型全身矩阵而不创建巨大的稠密单元对矩阵。切向冲量采用矩阵隐式作用的投影梯度法，投影半径为 `mu*I_z`，默认 3 次法/切向耦合迭代、每次 24 次切向迭代。用有限迭代和 warm start 近似求解，同时输出法向本构残差与冲量迭代残差。

所有单元的速度都来自同一个全身广义速度，刚性足板的运动一致性直接成立，也保留双足之间的动力学耦合；没有实现论文有歧义的“标量一致性误差减二维冲量”修正式。

漂移阻力使用本次接触中的**累计水平路径长度**：

```text
D += norm(xy_t - xy_previous)       # 仅连续侵入时累加
F_D = -w_i*(k_D*D*d + b_D*norm(v_xy))*unit(v_xy)
```

零速度时该力为零。`F_D` 是摩擦冲量之外的阻力，不把总切向力再次裁进摩擦锥。每只脚汇总 `sum(F_i)` 和 `sum((p_i-p_COM)×F_i)`，按世界系写入 MuJoCo；离地清零，reset 只清对应环境。

超过 0.20 m 的真实下陷仍完整记录并触发失败；仅在该 policy step 剩余子步内把本构求值深度限制到 0.20 m，避免模型域外的多项式爆炸。没有隐藏硬底板帮助站立。

### 实际训练材料范围

| 字段 | 当前范围 |
| --- | --- |
| alpha | [0.8,1.2] rad |
| recruitment nu | [0.8,1.8] |
| mass_scale cg | [0.5,2.0] |
| drag_scale cd | [1.0,6.0] |
| volume_fraction phi | [0.50,0.64] |
| density rho | 1600 kg/m³ |
| friction mu | [0.4,1.0] |
| penetration_stiffness | [20,200] N/m，按参考单元定义 |
| conical_stress | [1e4,1.2e5] N/m³ |
| drift_stiffness | [1000,4000] N/m²，按参考单元定义 |
| drift_damping | [0.5,2.0] N·s/m，按参考单元定义 |

这些是用于 G1 接入与稳定性验证的工程范围，不是完整论文 Table II，也不是实测 foam/sand 参数。配置会拒绝负 cg/cd 等与本实现正深度物理约定冲突的数值，不静默取绝对值。后续扩大范围前要重新检查能量、接触残差与训练稳定性。

## 文件入口

- [mild_env_cfg.py](../src/instinct_mj/tasks/locomotion/config/g1/mild_env_cfg.py)：训练/play 配置。
- [mild_rl_cfg.py](../src/instinct_mj/tasks/locomotion/config/g1/mild_rl_cfg.py)：原 PPO 配置，仅切换实验目录。
- [config.py](../src/instinct_mj/tasks/locomotion/mild/config.py)：材料、采样、求解参数。
- [geometry.py](../src/instinct_mj/tasks/locomotion/mild/geometry.py)：足底包络和碰撞 mask。
- [solver.py](../src/instinct_mj/tasks/locomotion/mild/solver.py)：颗粒反力与耦合冲量求解。
- [contact.py](../src/instinct_mj/tasks/locomotion/mild/contact.py)：动力学接口、状态、CUDA Graph 和施力。
- [env.py](../src/instinct_mj/tasks/locomotion/mild/env.py)：物理子步时序与 manager/reset 契约。
- [mdp.py](../src/instinct_mj/tasks/locomotion/mild/mdp.py)：足部 reward 适配和终止项。

## 验证与当前限制

```bash
# 9 项 CPU 机制/配置测试；GPU 测试默认跳过。
uv run --no-sync python -m unittest discover -s tests -v

# 额外运行完整 GPU 集成测试。
INSTINCT_MILD_GPU_TESTS=1 uv run --no-sync python -m unittest discover -s tests -v
```

测试覆盖面积加密不改变总响应、承重、附加质量、摩擦锥/耗散、离地零力、配置隔离与非法范围；GPU 验证覆盖 CPU MuJoCo Jacobian/质量矩阵参考、逐单元力与质心合力矩等价、无原生重复支撑、部分 reset、CUDA Graph 与普通执行的轨迹一致。

2026-09-22 在本机 RTX 5080 / 项目 `.venv` 上完成初次验证：9 项测试通过；64 与 512 环境各两轮普通 PPO 更新、checkpoint 保存；加载短训权重完成 50 步 headless 回放。512 环境短训共 24,576 条 policy transitions，第二轮约 5.1 s（仅短程测量，不作为正式性能基准）。该次随机策略测试的 rollout 末步未达切向容差比例约 0.8%，尚未据此宣称所有工况均已收敛。

后续持续训练检查发现并修复：每个 control step 使用独立诊断字典，避免 rollout 内历史日志被最后一步覆盖；MILD runner 的 `load_run` 使用具体字符串类型，修复 tyro 将其锁为 `{fixed}` 导致无法 CLI 续训的问题。早期记录的接触指标只代表该 rollout 的最后一步，不能当作整段平均值；修复后的记录才是 runner 对逐步样本的聚合。PPO 算法及物理参数均未改变。

修复后全部 10 项测试通过。512 环境持续训练已产生 `model_100.pt`，其中 76 个模型/优化器/归一化张量均为有限值；已验证从该 checkpoint 恢复 PPO 更新，并完成 100 步 headless 权重回放。训练过程未见 NaN/Inf、CUDA 异常或内存耗尽；策略尚处早期训练阶段。相同 PPO 配置下的 flat 短程对照也存在初期短 episode，本次未因此改变 PPO 超参数。

TensorBoard 的 `Mild/*` 指标包含最大/平均下陷、足力、累积路径、切向冲量残差、法向本构残差和未达到切向容差的环境比例。`impulse_tolerance` 用于报告，不提前结束固定次数的图内迭代。未收敛比例持续升高时应检查材料和提高迭代数，不能只忽略该指标。

512 环境正式训练已完成 5000 轮，最终权重位于 `logs/instinct_rl/g1_locomotion_mild_contact/2026-09-22_10-37-06_G1MildContactPPO/model_5000.pt`，模型参数均为有限值。窗口回放发现并修复了两个问题：普通平面 `TerrainEntity` 没有可选的 `debug_vis` 接口，现仅在地形提供该接口时调用；MuJoCo 原生窗口关闭只请求渲染线程退出，现等待该线程结束后再允许进程清理 GLFW，避免退出时段错误。加载最终权重完成 300 步 native 窗口回放，进程正常退出（退出码 0）。这些修复不改变接触动力学或 PPO，不需要重新训练。

当前回放是均匀软介质；空间软硬材料带、系统化 COT/转换测试、真实泡沫标定仍未实现。短训和物理测试证明实现能运行，不代表普通 PPO 已收敛或论文性能已复现。

### 第 5000 轮步态验收：未通过交替行走目标

用户回放反馈双脚跳后，对最终权重进行了 12 s 定量回放：固定材料中值、vx=0.6 m/s、seed=42、4 个确定性初态副本，按 50 Hz 采样并剔除首秒。没有 episode reset；平均世界系 vx 约 0.358 m/s，竖直速度 RMS 约 0.438 m/s，左右足法向力相关系数约 0.723。使用任务原接触阈值 0.1 N，双脚同时无支撑占 19.6%，单脚支撑占 25.1%，双脚支撑占 55.3%；改用 5 N 阈值，双脚无支撑仍占 24.4%。这些是 policy step 末子步采样的比例，不是连续时间精确积分，也不是跨材料/随机种子的泛化评估。

结果与同步跳跃反馈一致；5000 轮完成不能作为步态质量验收。当前 `feet_air_time` 只在单脚支撑时给正奖励，不直接惩罚双脚无支撑，也不指定左右交替相位。接触模型的卸载响应及训练收敛问题仍需进一步排查，不能仅凭这个回放将根因全部归于奖励。若增加步态奖励，应另列实验并对 flat 对照使用相同改动，避免与原“仅替换接触模型”的消融混淆。

原始轨迹、诊断脚本和汇总保存在 `logs/mild_gait_diagnostics/model_5000/`。本次诊断没有修改动力学、奖励或权重，也没有启动新训练。
