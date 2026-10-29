# baseline 调整 1

修改对象由用户确认：`Instinct-Locomotion-vae-G1-v0` 及其 Play task。
依据为本地 `docs/MILD.pdf` 的 Fig. 4、Table I/II、式 (16)–(17)。
`Instinct-Locomotion-Mild-Walk-G1-v0` 保留原 AMP 实验。

## 1. 旧 baseline 的实际问题

上次训练完成 5000 次 PPO 更新，1024 个环境，共 122,880,000 次策略交互，耗时约 6.47 小时。
末 100 轮平均 episode reward 为 45.45，但不能据此判断步态合格。

使用旧 `model_5000.pt` 在固定材料上评估：8 个环境、前进命令 0.6 m/s、每个环境 15 秒，剔除重置后首秒。
结果：无跌倒，双脚支撑占 **80.06%**，单脚支撑仅 **19.94%**，相位匹配 **58.05%**，
前进速度 RMSE **0.2125 m/s**；两脚在目标摆动阶段的有效离地率均为 **0%**（最低足底点 > 2 cm）。
原始指标保存在 `outputs/change1/old_baseline/summary.json`，轨迹在同目录 `trajectory.npz`；
旧训练末段统计已提取到 `outputs/change1/old_training_summary.json`。

查到的配置问题：

- VAE task 继承 AMP walking task，仍有判别器、专家数据、手臂参考约束及多项非 Table I 奖励。
- 29 维全身动作允许策略通过上肢动作补偿运动；论文 SA01 仅有 12 个腿部自由度。
- 已移除相位观测，却保留按环境编号分配的不同奖励相位；相同历史状态可能对应不同目标。
- 100 Hz 仅是旧策略更新频率，旧 PD 实际随 1 kHz 物理步更新。

## 2. 配置与实现修改

| 项目 | 调整后 |
| --- | --- |
| 训练算法 | 普通 PPO + VAE 联合优化，一个 optimizer；移除 WasabiPPO/判别器 |
| AMP | VAE task 不创建 motion reference sensor、AMP observation、参考动作匹配事件或 dataset termination |
| 策略动作 | 12 个腿关节，动作直接表示相对默认关节角的弧度偏差，scale=1 |
| 策略观测 | 每帧 45 维：角速度 3、重力 3、命令 3、腿部关节角 12、速度 12、上一动作 12；10 帧共 450 维 |
| 状态估计 | 基座速度 3、双脚速度 6、双脚高度 2；足高为面积加权足底点高度，以米计算，取消旧版 ×10 的回归尺度 |
| Actor | 当前观测 45 + 估计状态 11 + 调制 latent 16 → 12 维动作 |
| Critic | 当前干净观测 45 + 真实状态 11 + 材料 22 + 调制 latent 16；另保存相同的 450 维 actor history 供重算 latent |
| 网络与联合损失 | 保留 VAE、足速调制和非对称 critic；0.3 回归/VAE + 0.7 PPO；KL 系数 0.001 |
| 腿部控制 | 位置目标 100 Hz；候选 G1 内环为 1 kHz，待用户选择后正式训练。支持 pd_decimation=10 的字面 100 Hz 消融；延迟作用于目标指令 |
| 腿部 PD | 每腿 Kp=[50,50,70,70,20,20]，Kd=[5,5,7,7,0.2,0.2]；暂按 G1 的髋俯仰、髋横滚、髋偏航、膝、踝俯仰、踝横滚解释 |
| 命令 | 前进 0–1.2 m/s，侧向 0，偏航 ±1.5 rad/s；取消 heading controller |
| 随机化 | 基座质量 ±4 kg、COM ±6 mm、腿部电机偏置 ±0.035 rad，电机/IMU/命令独立 1–10 ms 延迟 |
| 步态相位 | 各环境从 reset 的同一相位开始；关闭按环境编号的隐藏相位偏移及随机 episode age；仍只使用论文列出的六类策略观测 |
| Checkpoint | schema v4；旧 29 动作 checkpoint 会被拒绝，防止误加载 |

七项奖励通过显式白名单配置，不再继承额外姿态惩罚：

| 奖励 | 权重 |
| --- | ---: |
| 线速度跟踪 `exp(-5 * error²)` | 1.4 |
| 角速度跟踪 `exp(-5 * error²)` | 1.1 |
| 步态 `1.3 * sum(contact == target) - 0.3` | 1.4 |
| 摆动期足高误差 | -2.4 |
| 接触期足滑速度 | -0.1 |
| 基座速度变化的指数项 | 0.2 |
| 动作一阶和二阶差分平方 | -0.003 |

论文表 I 将足高**误差**写为正奖励；这里按最小化误差使用负号。
加速度行混用了 base/foot 下标，这里使用连续两个策略时刻的基座速度差。
这些解释与原有接触求解器的工程差异均不冒充作者原代码。

### G1 上身定姿与试跑排错

G1 的额外 17 个上身关节不再参与策略动作，使用原生伺服保持默认姿态。
肘角设为 1.4 rad，肩俯仰 0，肩横滚 ±0.3 rad。**该 MJCF 肘角 0 并非手臂伸直**：
用肩、肘、腕的实际空间位置计算，肘角 0.15 rad 对应约 73° 弯曲，1.4 rad 才接近伸直。

排错时发现：较小的肩横滚 ±0.15 rad 会产生手腕/大腿自碰撞；扩大到 ±0.3 rad 后，
1024 个随机 reset 姿态均无穿透。继续动态检查发现原低刚度手臂伺服仍会下垂碰腿，
把上身也改成延迟的 100 Hz 转矩保持还会引入额外腰部振荡。因此上身采用 1 kHz 原生定姿伺服，
腰部 Kp/Kd=100/6，手臂 50/2，独立腕组 20/0.5，保留 G1 原力矩限制。
最终动态探针未再观察到手腕/髋部碰撞；零策略的跌倒不作为已学会站立的证据。
上身控制属于 SA01 没有的 G1 适配。腿部内环频率另见下文悬空实验与待确认事项。

### 电机延迟排错

第一版按“延迟已计算的转矩”实现 motor latency。4096 环境试训 51 轮后，平均回合仍约 1 秒。
独立探针在 10 ms 延迟下仅施加一次 0.02 rad 髋偏航目标扰动，0.21 秒时最大关节速度达到
16.2 rad/s，随后出现放大振荡；这表明延迟转矩同时延迟了 PD 反馈校正。

修正为目标指令延迟：使用 ring buffer 保存目标，各环境按自己的指令到达相位每 10 ms 执行一次
PD，计算时读取**当前**关节位置和速度，转矩保持到下一次更新。相同探针在 0.21 秒时最大关节速度
降至 0.684 rad/s，0.41 秒时为 0.278 rad/s，髋偏航回到约 0.00054 rad。
无平衡策略的机器人之后仍会倒下；该探针只验证延迟环节不再放大微小扰动。

证据为 `outputs/change1/delayed_pd_probe.log` 与 `command_delay_pd_probe.log`；
失败试训摘要为 `torque_delay_trial.json`。新增回归检查同时覆盖 0 ms / 3 ms 指令延迟、
到达时读取新状态以及更新之间的转矩保持。论文没有公开延迟插入位置，这里采用电机目标指令
的延迟语义；不把延迟旧转矩造成的振荡当作地面随机化。

### 悬空腿 PD 稳定性与待确认事项

进一步把 G1 放在离地 3 m 处，保证测试期间足底完全不接触地面，只给左髋偏航一次 0.02 rad、
10 ms 的目标脉冲。相同目标延迟和相同论文 PD 增益下，字面 100 Hz 采样保持出现振荡：
0.11 s 时最大关节速度为 22.48 rad/s；1 kHz 内环在相同时刻约为 0.00392 rad/s。
整个 0.3 s 试验中，两者峰值分别为 **22.48 rad/s / 0.138 rad/s**，两组足底接触力均为零。
这个实验隔离了地面和训练策略，说明不能直接认定 SA01 的低频 PD 配置能稳定当前 G1 摆动腿。

![相同增益下 G1 的悬空腿响应](../outputs/change1/pd_stability.png)

候选适配保持 **100 Hz 策略/位置目标、论文 Kp/Kd、1–10 ms 目标延迟**，只把 G1 内环改为
1 kHz。`pd_decimation=1` 为候选适配，`pd_decimation=10` 为字面 100 Hz 对照；checkpoint schema
记录实际 PD 频率，禁止混用。证据可重复生成：

```bash
.venv/bin/python scripts/baseline/check_pd_stability.py
```

结果在 `outputs/change1/pd_stability.json`。已向用户请求选择这一与论文文字的差异；
**正式重训等待选择，不能把当前工作状态写成已收敛。**

### 尚不能声称逐数值复刻的部分

机器人仍为 G1，接触网格仍为每脚 44 个面积单元，论文为 SA01。
Table II 的负 cg/cd 与当前正深度、被动正定求解器不兼容，仍采用 cg∈[0,4.6]、cd∈[0,28]；
不能声称已经覆盖论文完整的有符号范围。其他已明确的材料范围按表 II 配置。
H=10、latent=16、网络宽度、KL 系数、0.8 s 周期、50% 支撑比、6.5 cm 摆动峰值及 PPO 超参数
属于论文未给全的实现选择。论文未说明相位是否直接提供给策略，当前按其观测清单保留历史输入；
去掉环境编号相位差后，周期学习效果仍须由训练和独立步态评估检验。

## 3. UMR 专家数据核查

新增可重复运行的检查：

```bash
/home/aaa/miniconda3/envs/umr/bin/python scripts/umr/audit_semantics.py
```

检查所有 18 个候选，并用 `motions.yaml` 标识实际进入 AMP 的 13 个：7 步行、4 转向、2 站立。
结果保存为 `outputs/change1/umr_semantics.json`。检查没有修改原始 CMU、SMPL-X 模型、UMR 输出或已选数据。

- 训练 MJCF 的 SHA256 与数据 manifest 一致。UMR 直接对 torso-root 训练 MJCF 求解，不需要后处理 pelvis→torso 转换。
- 原始 UMR、转换后动作和模型的 29 关节顺序一致；30→50 Hz 重采样复算最大位置误差约 4.8e-8，四元数符号/顺序正确。
- T-pose 肘角 π/2 是该 MJCF 的坐标约定，不能据此诊断成错误的 90° 弯肘映射。
- 选中步行动作平均根部平移速度约 **0.237–0.509 m/s**，与训练命令最高 1.2 m/s 存在覆盖差距。
- 步行动作两脚最高足底离地范围约 **2.31–4.25 cm**；转向约 **0.58–3.12 cm**，有些转向片段很接近拖脚。
- 用人体和机器人各自腿长归一化后，机器人踝关节竖直活动幅度平均约为人体的 **85%**。
  这与原始源动作的抬脚幅度损失相符；踝关节活动幅度与最低足底离地高度是两种不同指标。
- 肩、肘、腕空间夹角的平均绝对差约左 **27.2°**、右 **16.2°**，说明上肢重定向存在形态失真；
  机器人参考肘部整体比人体更直，不能把原策略的双手前伸直接归咎于参考肘关节角过大。

结论：未发现转换层的关节排列、左右交换或四元数错误；存在重定向形态误差、较低的参考抬脚幅度
和速度覆盖不足。旧审计只检查限位、穿透、滑移等合法性，不能保证这些动作满足本次步态目标。
本次 VAE 训练关闭 AMP，保留旧 AMP task 和源数据以便独立比较。

对比图：

![旧策略足底轨迹与 UMR 参考动作核查](../outputs/change1/diagnostics.png)

## 4. 训练、收敛与验证

训练使用 4096 环境、seed=42、24 steps/env，沿用 guide 原先的单次 9 小时范围，runner 在 8.5 小时内保存退出。
初始短测约 17–18 秒/更新。训练从头开始，不加载旧 AMP 模型。

收敛判据：至少 1000 次更新；比较相邻两个 250 轮窗口的平均每步 reward，相对变化 <2%；
最近窗口相对标准差 <5%；当前平均 episode 长度 >20 秒上限的 90%，
相位匹配 >80%、单脚支撑 >50%。达到迭代/时间上限本身不算收敛。
独立评估还检查无跌倒、速度 RMSE、交替触地以及两脚实际 >2 cm 的离地比例。

<!-- CHANGE1_TRAINING_STATUS_BEGIN -->
- **训练已暂停，等待用户选择 G1 内环采用 1 kHz 适配或保留字面 100 Hz PD。**
- 最后一次字面 100 Hz 试训在 35 轮后停止，尚未收敛；记录为 `outputs/change1/sampled_pd_trial.json`。
- 当前唯一保留的 VAE 训练目录：`logs/instinct_rl/g1_locomotion_mild_vae_change1/2026-10-14_20-26-48_G1MildVAEChange1/`。
- 1 kHz 实现与独立稳定性检查已准备好；本任务的正式训练及最终步态验收尚未完成。
<!-- CHANGE1_TRAINING_STATUS_END -->

已完成的验证：PPO/VAE 梯度、ONNX 对照、观测历史与局部 reset、checkpoint/schema、
物理步延迟、100 Hz 转矩保持、相位隔离、共享接触与旧 AMP 隔离；增加随机初始姿态自碰撞回归检查。
完整 GPU/单元测试 **33 项全部通过，无 skip**，结果在 `outputs/change1/tests_all.log`；UMR pipeline **3 项通过**，见 `outputs/change1/tests_umr.log`。

`scripts/baseline/report_change1.py` 持续更新本文训练状态；训练退出后自动评估固定材料和随机材料，保存 `outputs/change1/result.json` 及训练曲线。只有 reward 收敛且两组独立评估均通过才标记 `accepted=true`。

## 5. 文件与运行方法

核心文件：

- `src/instinct_mj/tasks/locomotion/config/g1/mild_vae_env_cfg.py` / `mild_vae_rl_cfg.py`
- `src/instinct_mj/tasks/locomotion/mild/actor_critic.py`、`ppo.py`、`runner.py`、`vae_env.py`、`vae_mdp.py`
- `scripts/umr/audit_semantics.py`
- `tests/test_mild_vae.py`

```bash
.venv/bin/python -u -m instinct_mj.scripts.instinct_rl.train \
  Instinct-Locomotion-vae-G1-v0 --num-envs 4096 \
  --agent.max-iterations 20000 --agent.save-interval 100 \
  --agent.wall-time-limit-s 30600

.venv/bin/python -m instinct_mj.scripts.instinct_rl.evaluate_mild_walk \
  --task Instinct-Locomotion-vae-G1-Play-v0 \
  --checkpoint /absolute/path/to/model_N.pt \
  --output outputs/change1/final_eval --num-envs 8 --steps 1500
```

训练日志清理范围为此次 VAE baseline 及其测试 run；保留最后一次正式训练目录。
其他 task 的训练目录和 UMR 数据不属于此次清理范围。删除前保留旧模型评估指标及试跑失败原因；
具体路径记录在 `outputs/change1/log_cleanup.json`。
