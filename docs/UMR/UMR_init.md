# UMR 初始化与 MILD-Walk AMP

> 当前配置与结果（2026-09-28）：AMP 系数 `0.025`、胸部直立 `-3`、腰部回零 `-0.5`，仍只有三项 reward/AMP 权重改动，参考数据保持不变。512 环境实际续训 3500 轮后，选用 8300/8400 按 35%/65% 参数平均的策略；固定材料 seed 42、随机材料 seed 43/44 及未参与调参的 seed 45，站立与 0.2/0.4/0.6 m/s 全部通过既定工程验收，判定为任务基本收敛。0.4 m/s 胸部 pitch 从 13.92° 降至 -1.44°，仍有方向漂移与侧向摆动。模型、验证范围、优化器限制及从头训练命令见 [MILD-Walk 文档](../mild-walk.md#最终结论与推荐模型2026-09-28)。修改保持未提交；下方初始化与短训记录描述的是 2026-09-23 原始版本。

## 初始化实现范围（2026-09-23）

在 `Instinct-Locomotion-Mild-Walk-G1-v0` 及对应 Play 任务中增加 AMP style reward、判别器及专家观测。普通 ActorCritic MLP `[256,128,128]`、原有 policy/critic 观测、步态时钟、任务奖励、速度指令和 MILD 接触求解保持原有定义。没有引入 Parkour 的深度编码器、MoE、地形或机器人参考状态初始化。

判别器使用 Parkour 的现有参数：隐藏层 `[1024,512]`、ReLU、MSELoss、quad style reward、reward coefficient `0.25`、gradient penalty `5.0`、AdamW lr `1e-4`、betas `[0.9,0.999]`、weight decay loss coefficient `3e-4`、logit weight decay coefficient `0.04`。普通 PPO 参数沿用 Mild-Walk，包括 entropy coefficient `0.008`。

算法由 PPO 改为 `WasabiPPO`，style reward 由算法的 auxiliary reward 通道加入，不另设一个重复计算的环境奖励项。增加 `amp_policy` / `amp_reference` 两组观测：重力投影、相对默认位置的 29 个关节位置、29 个关节速度、机体线速度、机体角速度；每项使用 10 帧历史，关节速度缩放 `0.05`。参考片段耗尽时沿用 Parkour 的静默重采样机制，不结束机器人 episode。

实验目录改为 `logs/instinct_rl/g1_locomotion_mild_walk_amp`，避免和原有普通 PPO 训练混淆。原有 Mild-G1 接触消融任务不增加 AMP。转向参考加入专家库，但原有任务指令仍为向前行走和站立；本次不据此宣称学会了指令转向。

## 环境与模型

- 官方仓库：`/home/aaa/UMR`，来源 https://github.com/hanyang9/UMR 。任务开始时已有一个正在运行的克隆进程，等待其完成后复用。
- 固定提交：`b45d45c20967ac8db586587ba58cf17257a9a467`。
- 独立 conda 环境：`/home/aaa/miniconda3/envs/umr`，Python 3.12。
- 使用支持当前 RTX 5080 的 PyTorch 2.10.0 / CUDA 12.8 wheel，代替官方 README 的旧 Torch 2.4.1 / cu121；其余依赖来自该提交的 `requirements-umr.txt`。
- InstinctMJ 的 `.venv` 单独用于格式转换、训练与测试；UMR 的 MuJoCo 3.3.x 与训练的 MuJoCo 3.11.x 不混装。
- 人体模型复用 `/home/aaa/GMR/assets/body_models/smplx` 下已有的 SMPL-X 文件，不修改或复制进仓库。
- 目标模型直接使用 `src/instinct_mj/assets/resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml`。该模型以 `torso_link` 为自由根；没有先用骨盆根模型再直接冒用根位姿。UMR 输出的关节名称、顺序、模型路径和配置时 MJCF 的 SHA256 在转换时检查。

安装命令记录（当前已完成，无需重复执行）：

```bash
conda create -n umr python=3.12 pip -y
uv pip install --python /home/aaa/miniconda3/envs/umr/bin/python \
  torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /home/aaa/miniconda3/envs/umr/bin/python \
  -r /home/aaa/UMR/requirements-umr.txt
```

## 数据与可复现流程

数据来源为 CMU 动作捕捉库的 AMASS SMPL-X 拟合文件，本地已存在于 `/home/aaa/CMU`。另外从 CMU 官方站点下载对应原始 AMC 文件到 `outputs/umr/provenance`；UMR 实际读取同源 SMPL-X 拟合文件，不读取 AMC。

共处理 18 个不重叠时间片段：12 条直行、4 条转向、2 条站立。最初 14 条中部分直行未通过滑脚检查，追加了 4 条慢走候选。选择清单及秒级裁剪范围在 `scripts/umr/clips.json`：直行和转向来自 `120_19`（Walk slow）、`120_20`（Walk），站立来自 `82_08` 的静止开头和 `111_28`（Standing still）。这 18 个片段来自 4 个原始采集序列。原始人体形状参数、性别、时间尺度和单位保留（UMR 本身使用前 10 个 beta）；120 Hz 原始数据按每 4 帧抽取到 30 Hz。

官方来源：

- http://mocap.cs.cmu.edu/search.php?subjectnumber=120
- http://mocap.cs.cmu.edu/search.php?subjectnumber=82
- http://mocap.cs.cmu.edu/search.php?subjectnumber=111
- https://amass.is.tue.mpg.de/

所有生成数据位于已被 gitignore 排除的 `outputs/umr`。人体模型及原始动作没有加入代码提交。`manifest.json` 记录输入 SHA256、来源、裁剪范围、UMR 提交和机器人模型 SHA256。

```bash
# 准备片段和独立配置；在 InstinctMJ 根目录运行。
/home/aaa/miniconda3/envs/umr/bin/python scripts/umr/prepare.py

# 顺序运行官方 UMR：外表面采样、对应关系学习、重定向。
# 同一人体形状和机器人复用官方对应关系缓存。
/home/aaa/miniconda3/envs/umr/bin/python scripts/umr/retarget.py

# 在训练环境中转换和审核。
.venv/bin/python scripts/umr/convert_and_audit.py
MUJOCO_GL=egl .venv/bin/python scripts/umr/render.py
```

重定向使用官方 500 epoch 对应关系学习设置、双向 warm start，以及原有 LQR 轨迹滤波。源动作按 UMR 相同的 SMPL-X 模型计算全片最低身体表面，统一平移世界 z，使地面坐标一致，偏移量记入 manifest；这不是逐帧抬脚。每帧求解迭代提高到 4 次以减少跟随滞后。SMPL-X 批量限制为 64 帧、求解批量为 8；按片段顺序处理。保留各片段配置及完整日志以便复现。可用 `--only <clip_id>` 单独运行、审核或回放一个片段。

转换输出包含 `framerate`、`joint_names`、`joint_pos`、`base_pos_w`、`base_quat_w`；姿态为 wxyz，位置单位为米，关节单位为弧度。根旋转使用 SLERP、位置和关节使用线性插值重采样到 50 Hz，并保持四元数符号连续。

训练默认读取 `outputs/umr/motions/motions.yaml`。可通过 `UMR_MOTION_DIR=/absolute/path/to/motions` 覆盖数据目录。清单仅包含通过审核的文件。

## 动作质量检查

审核在训练版本的 MuJoCo 中逐帧执行实际训练模型正向运动学。脚本不以逐帧抬高根节点、关节截断等方式掩盖无效动作。

| 指标 | 初始验收上限 |
|---|---:|
| 关节硬限位越界 | 0.0001 rad |
| 相邻帧关节跳变（50 Hz） | 0.4 rad |
| 关节速度 | 20 rad/s |
| 根姿态相邻帧旋转 | 0.25 rad |
| 根平移速度 | 2 m/s |
| 根角速度 | 5 rad/s |
| 碰撞几何穿地深度 | 2 cm |
| MuJoCo 自碰撞穿透深度 | 2 cm |
| 支撑期脚部水平速度 p95 | 0.35 m/s |
| 双脚最低点均高于 4 cm 的帧比例 | 10% |

脚部最低点根据胶囊碰撞几何精确计算。滑脚以足底低于 5 mm、足底支撑点竖直速度绝对值低于 0.15 m/s 作为支撑代理条件，再通过 MuJoCo 点雅可比与 qvel 计算支撑点的材料水平速度，避免把脚跟/脚尖滚动时的踝关节移动误判为滑脚；最初 2.5 cm 的接触窗口会把低净空摆动误判成支撑，因此改用 5 mm 的几何接触容差，并单独记录每脚最大净空。这是保守筛查，不是力学接触或真实足底粘着验证。自碰撞遵循训练模型的碰撞几何与排除规则，不能代表所有视觉网格之间均无交叠。

脚本输出 `audit.json`、逐片段检查结果和通过清单；不足 11 条或缺少直行/转向/站立任一类别时以失败退出。另生成每条动作视频与汇总图进行视觉检查。这些是 AMP 输入质量检查，不等于训练收敛或真实机器人动态可执行性证明。

## 验证记录

2026-09-23 已完成重定向、数据审核、回放抽帧检查和短训。最终训练清单包含 **13 条：7 条直行、4 条转向、2 条站立，共 42.48 秒**。这 13 条均被实际训练加载，日志中确认 `All 13 motion files loaded`。

通过片段的直行速度中位数为 **0.243–0.503 m/s**；转向净航向变化绝对值为 **1.94–3.09 rad**；站立速度中位数为 **0.006–0.010 m/s**。所有通过片段中，关节限位越界为 0，关节速度最大 4.39 rad/s，地面穿透最大 1.42 mm，自碰撞穿透最大 5.83 mm，支撑点滑速 p95 的最大值为 0.341 m/s，双脚同时高于 4 cm 的帧比例为 0。

排除 5 条：`walk_slow_01`、`walk_slow_03`、`walk_normal_02`、`walk_normal_04`、`walk_slow_08`，均因滑脚指标未通过。没有放宽 0.35 m/s 的滑速阈值；文件留作诊断，不进入 `motions.yaml`。对 `walk_slow_01` 额外尝试过加强接触锚定，仍未通过，最终排除。

已生成并检查直行、转向、站立汇总抽帧；每个候选片段的视频保存在 `outputs/umr/renders`。原始审核数据为 `outputs/umr/audit.json`，通过名单为 `outputs/umr/motions/motions.yaml`。

### 短训命令与证据

在已有其他训练占用约 8.4 GiB 显存的情况下，执行：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MUJOCO_GL=egl \
.venv/bin/instinct-train Instinct-Locomotion-Mild-Walk-G1-v0 \
  --num-envs 16 --agent.max-iterations 3 \
  --agent.save-interval 1 --agent.log-interval 1
```

- 16 环境 × 每轮 24 步 × 3 轮 = **1,152 个环境步**，正常退出，无 OOM。
- rollout 与优化累计约 14.33 秒；未执行长时间训练。
- PPO 和判别器优化器都完成 **60 次更新**；判别器权重在两个保存点之间确实发生变化。
- policy/critic 输入仍为 98/101 维，输出为 29 个动作；AMP policy/reference 各 670 维；策略是 ActorCritic，非 MoE。
- 判别器损失三轮为 `0.5999 / 0.1230 / 0.1388`；AMP 总损失为 `1.1351 / 0.7548 / 0.7000`。
- 原始 style reward 均值为 `0.7471 / 0.3873 / 0.2878`；算法系数仍为 `0.25`。
- 所有 checkpoint 张量及上述训练指标均有限；训练过程中全局剩余显存最少 **5.786 GiB**。该数值包含正在运行的其他进程影响，不能当作本训练独占显存峰值。
- 运行目录：`logs/instinct_rl/g1_locomotion_mild_walk_amp/2026-09-23_10-37-54_G1MildWalkAMP/`。
- 最终 checkpoint：上述目录下 `model_3.pt`。
- 主日志：`logs/umr_init/train_smoke.log`。
- 校验报告：运行目录下 `smoke_verification.json`，副本在 `logs/umr_init/smoke_verification.json`。

可复核 checkpoint：

```bash
.venv/bin/python scripts/umr/verify_smoke.py \
  logs/instinct_rl/g1_locomotion_mild_walk_amp/2026-09-23_10-37-54_G1MildWalkAMP
```

短训初始策略仍会跌倒；本次完成标准是有限显存下训练链路和判别器更新可运行，不是收敛或稳定步行。没有继续占用 GPU 进行后台长训。

### 测试

- 原有接触与步态单元测试：13 项通过。
- 新增 UMR/AMP 检查：3 项通过，涵盖四元数符号连续性、穿地拒绝和配置隔离。
- GPU 集成测试：2 项通过，验证接触求解、CUDA graph 一致性，以及加入 AMP 后的动力学和 reset 时钟行为。
- 实际 AMP 训练及 checkpoint/优化器状态校验通过。

```bash
.venv/bin/python -m unittest discover -s tests -p test_mild_contact.py
.venv/bin/python scripts/umr/test_pipeline.py
INSTINCT_MILD_GPU_TESTS=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MUJOCO_GL=egl \
  .venv/bin/python -m unittest discover -s tests -p test_mild_integration.py
```

## 代码与产物

| 路径 | 用途 |
|---|---|
| `src/instinct_mj/tasks/locomotion/config/g1/mild_walk_amp_cfg.py` | 专家 sensor、AMP 观测、参考刷新配置 |
| `src/instinct_mj/tasks/locomotion/config/g1/mild_walk_env_cfg.py` | 在现有 Walk 任务中启用 AMP |
| `src/instinct_mj/tasks/locomotion/config/g1/mild_rl_cfg.py` | 保留普通 MLP/PPO 参数，加入判别器参数 |
| `scripts/umr/clips.json` | 原始动作与不重叠裁剪区间 |
| `scripts/umr/prepare.py` | 校准源地面，生成片段、配置和来源清单 |
| `scripts/umr/retarget.py` | 顺序调用官方 UMR，复用对应关系缓存 |
| `scripts/umr/convert_and_audit.py` | 转换为加载器格式、检查、生成通过清单 |
| `scripts/umr/render.py` | 每条片段视频与汇总抽帧 |
| `scripts/umr/test_pipeline.py` | UMR/AMP 回归检查 |
| `scripts/umr/verify_smoke.py` | 校验判别器、优化器与 TensorBoard 指标 |
| `outputs/umr/environment.txt` | UMR 实际安装版本快照 |

本次未修改 UMR 上游源码，所有任务特定参数位于 InstinctMJ 输出目录的配置中。已有工作区改动、已有训练进程和原有 conda 环境均保留。
