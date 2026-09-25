# MILD 足底求解 + 交替行走 PPO

> 2026-09-23：Walk 任务已增加 UMR AMP style reward 和判别器，策略仍为普通 MLP。当前数据准备、短训命令及验证结果见 [UMR 初始化记录](UMR/UMR_init.md)。下文保留原 PPO 行走阶段的设计与运行记录。

用户确认优先实现左右交替行走后，新增独立训练实验。原 `Mild-G1` 消融任务、奖励和权重仍保留；新任务不再属于“只替换接触模型、奖励完全不变”的严格消融。仍然没有 VAE、状态估计器或 latent modulation。

## 保持的物理实现

完全复用 `MildRlEnv` 的物理子步、足底离散单元、全身耦合求解、法向附加质量、切向摩擦、漂移阻力和质心合力矩施加。`solver.py`、`contact.py`、`geometry.py`、`env.py`、`config.py` 未修改。没有硬底板、原生足底接触、姿态强制覆盖或人为交替施力。

物理步长仍是 1 ms，策略周期 20 ms；每脚仍是 44 单元；材料参数范围及每次 reset 随机化不变。GPU 集成测试对两个任务输入相同动作，验证轨迹和逐单元足力在 GPU 浮点误差范围内一致。

## envcfg 改动

入口：[mild_walk_env_cfg.py](../src/instinct_mj/tasks/locomotion/config/g1/mild_walk_env_cfg.py)，继承 `G1MildEnvCfg`。

| 项目 | 配置与目的 |
| --- | --- |
| 步态周期 | 0.8 s，左右相差半周期；每脚支撑占 60%，目标包含 20% 双支撑，没有计划腾空阶段 |
| 时钟观测 | actor/critic 都追加 `sin(phase), cos(phase)`，观测由 96/99 维变成 98/101 维；每个环境有固定相位偏移，episode reset 重置时钟；站立指令时两维清零 |
| `gait_contact` | 新增，权重 +2.0；支撑脚奖励 `1-exp(-Fz/40N)`，摆动脚奖励 `exp(-Fz/10N)`，双脚取平均；Fz 来自真实 MILD 合力 |
| `swing_clearance` | 新增，权重 +0.6；仅摆动脚跟踪正弦净空，峰值 6.5 cm，误差尺度 2.5 cm；使用最低足底单元相对介质表面的高度，不限制支撑脚的下陷 |
| `both_feet_airborne` | 新增，权重 -2.0；双脚法向支撑力都不超过 5 N 时惩罚；不以踝关节高度推断接触 |
| `lin_vel_z_l2` | 权重由 -0.1 改为 -0.5，抑制竖直弹跳 |
| 其余奖励 | 原 16 项其余定义及权重保持不变，包括单脚支撑时间奖励 |
| 指令 | 第一阶段只训练 vx∈[0.2,0.6] m/s、vy=wz=0；10% 站立，关闭 heading；站立时目标为双支撑 |
| 扰动 | 第一阶段关闭周期 push；其他训练初态/质量随机化及地面材料随机化保留 |
| 回放 | 沿用 MILD 固定材料中值和确定性初态，vx=0.6 m/s |
| 日志 | 添加 `Walk/*`：腾空/单支撑/双支撑比例、相位一致率、速度误差、竖直速度平方及有效样本数 |

`Walk/*` 排除 reset 和每个 episode 首秒的过渡状态；没有有效样本时显示 0，必须结合 `Walk/sample_count` 阅读。TensorBoard 是逐 control step 聚合，最终验收使用下面的独立轨迹统计。

## agentcfg 改动

普通 MLP + PPO，隐藏层 [256,128,128]、优化超参数及每环境 24 步 rollout 不变。新增两维时钟自动改变首层输入维度；实验目录改为 `g1_locomotion_mild_walk`，run 名为 `G1MildWalkPPO`。从头训练，不直接加载之前双脚跳的 96/99 维权重；原权重与新网络维度不兼容。

## 运行

```bash
# 新训练：512 环境，5000 轮，每 100 轮保存。
uv run --no-sync instinct-train Instinct-Locomotion-Mild-Walk-G1-v0 \
  --num-envs 512 --agent.max-iterations 5000 --agent.save-interval 100

# 指定新任务自己的权重回放。
uv run --no-sync instinct-play Instinct-Locomotion-Mild-Walk-G1-Play-v0 \
  --checkpoint-file /absolute/path/to/mild_walk/model_5000.pt

# 8 环境、每环境 30 s 的固定材料验收，保存 summary.json 和 trajectory.npz。
uv run --no-sync python -m instinct_mj.scripts.instinct_rl.evaluate_mild_walk \
  --checkpoint /absolute/path/to/mild_walk/model_5000.pt \
  --output logs/mild_gait_diagnostics/walk_final

# 另测随机材料；应更换多个 seed，不能把单次中值回放当作鲁棒性结论。
uv run --no-sync python -m instinct_mj.scripts.instinct_rl.evaluate_mild_walk \
  --checkpoint /absolute/path/to/mild_walk/model_5000.pt \
  --randomize-material --seed 43 --output logs/mild_gait_diagnostics/walk_random43
```

不能给旧 `Mild-G1-Play-v0` 任务加载新 Walk 权重，也不能反过来。没有显式 checkpoint 时，自动选择最新 run 可能选到短训结果，验收应始终指定文件。

## 验收与范围

验收脚本按 50 Hz 采集真实足力、最低足底高度、速度、指令和 episode 状态。剔除每次 reset 后首秒，统计失败和超时，要求：无跌倒；无支撑比例 <5%；单支撑比例 >50%；前向速度 RMSE <0.2 m/s；每环境左右脚各至少 6 次落地且独立落地事件的交替比例 >80%；足力接触状态与目标相位一致率 >80%；左右摆动脚各至少一半的摆动采样具有 >2 cm 净空，排除仅卸载而拖脚。净空比例合并各环境统计。这些是本项目初步步行验收标准，不是论文原始指标。

短训只检查运行链路。正式训练完成后必须检查 `summary.json` 的 `passed` 及逐项 `checks`，并回放看步态；未通过就不能称为“已实现稳定交替行走”。这一阶段只覆盖前向步行，暂不宣称已经解决转弯、侧移、推扰恢复或所有材料的泛化。

## 本次运行记录

2026-09-22 已通过原接触力学测试、新增步态奖励测试和 GPU 动力学一致性/reset 测试；64 环境完成两轮 PPO 更新并保存 checkpoint，评估脚本成功加载短训权重，正确报告步态未通过。

正式训练于 21:38 启动：512 环境、从头训练 5000 轮、每 100 轮保存。启动记录在 `logs/mild_walk_launches/20260922_213815/`，`train.log` 查看进度，`status.json` 查看训练/评估阶段。后台管理脚本将在 `model_5000.pt` 产生后自动运行固定材料验收，报告保存到新 run 的 `evaluation/summary.json`，状态中的 `gait_passed` 表示结果，不能仅凭 `state=finished` 判断步态通过。
