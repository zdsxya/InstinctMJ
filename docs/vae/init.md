# 任务：在 MJLab 中为 Unitree G1 实现 MILD 风格的 VAE 地面适应模块

## 一、任务背景与目标

我正在使用 MJLab 框架训练 Unitree G1 29DoF 人形机器人，研究其在刚性地面、柔性地面以及硬软交界处的行走适应问题。

现在希望参考论文 **MILD: Tractable Terrain Modeling for Learning Improved Bipedal Locomotion on Deformable Surfaces** 中的 VAE 与 Modulation Network，实现一套基于本体感知历史的隐式地面状态估计机制，并集成到现有 PPO 运动控制框架。

论文地址：https://arxiv.org/html/2608.19955

本次任务重点实现：

1. 历史本体观测缓冲区
2. 多头 VAE 状态估计网络
3. VAE 隐变量学习和重建
4. 基于双脚速度的 Modulation Network
5. VAE 与 PPO 的联合训练
6. 训练日志及基本测试

**注意：本次不要求复现 MILD 完整的柔性地面物理建模，只实现其中的 VAE、Modulation Network 及与 PPO 结合的部分。**

请先阅读当前 MJLab 项目结构、训练入口、Actor-Critic、PPO、观测管理及环境配置，再进行代码修改。优先复用现有架构，不要重新实现整套 PPO。

## 二、整体网络结构

目标网络的数据流如下：

```text
历史本体观测 O_t
        │
        ▼1
  Shared Encoder
        │
        ├── Base Velocity Head → 机身线速度 (3)
        │
        ├── Foot Velocity Head → 双脚速度 (6)
        │
        ├── Foot Height Head   → 双脚高度 (2)
        │
        └── Latent Head
                │
            μ, logvar
                │
                ├── 重参数化采样 z
                │        │
                │        ▼
                │      Decoder
                │        │
                │        ▼
                │     当前观测重建
                │
                └── 策略隐变量 μ
                         │
预测双脚速度 ──→ Modulation Network
                         │
                         ▼
                    调制 latent
                         │
                         ▼
当前观测 + 估计状态 + 调制 latent
                         │
                         ▼
                      PPO Actor
                         │
                         ▼
                    关节控制指令
```

这里使用 VAE 学习历史运动中隐含的地面接触特征，不需要显式预测地面刚度、摩擦系数等物理参数。

## 三、VAE 输入设计

使用机器人最近 H 帧的本体观测作为 Encoder 输入。

初始设置：

- 历史长度 H = 10
- 历史观测在时间维度展开后输入 MLP
- 控制频率沿用当前项目配置
- 不要硬编码观测维度

本体观测优先包含：

- IMU 角速度
- 投影重力方向
- 速度指令
- 关节位置
- 关节速度
- 上一步动作

不要将仿真中的真实机身线速度、真实足端状态、地面刚度等特权信息直接输入 VAE Encoder。

上述真实状态仅用于训练阶段的监督损失计算。

需要实现历史缓冲区的更新和环境 reset 逻辑，确保并行环境之间的数据独立。

## 四、VAE 网络设计

### 1. Shared Encoder

初始网络结构：

```python
Encoder:
    Input(H * obs_dim)
    Linear(256) + ELU
    Linear(128) + ELU
```

基于 Encoder 的共享特征分别构建：

**状态估计分支：**

- Base Velocity Head：输出 3 维机身线速度
- Foot Velocity Head：输出 6 维双脚线速度
- Foot Height Head：输出 2 维双脚高度

**隐变量分支：**

- Mu Head：输出 16 维
- Logvar Head：输出 16 维

Latent 维度初始设置为 16，要求可配置。

训练时使用 VAE 重参数化：

```python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + eps * std
```

为了降低 PPO 中策略输入的随机性，第一版让 Actor 使用确定性的 `mu`，Decoder 使用重参数化采样的 `z`。

### 2. Decoder

网络结构：

```python
Decoder:
    Input(latent_dim)
    Linear(128) + ELU
    Linear(256) + ELU
    Linear(obs_dim)
```

Decoder 重建当前时刻本体观测。

### 3. Modulation Network

使用估计得到的双脚速度作为调制输入。

```python
Modulation:
    Input(6)
    Linear(64) + ELU
    Linear(64) + ELU
    Linear(2 * latent_dim)
```

网络生成 scale 和 shift：

\[
\tilde z=(1+\text{scale})\odot\mu+\text{shift}
\]

这里使用残差缩放形式，初始化时尽可能使调制结果接近原始 latent。

Modulation Network 的参数与 Actor、VAE 联合训练。

## 五、PPO Actor-Critic 修改

### Actor

将 Actor 输入修改为：

```python
actor_input = concat(
    current_proprioception,
    estimated_base_velocity,
    estimated_foot_velocity,
    estimated_foot_height,
    modulated_latent
)
```

Actor 输出仍使用项目现有的关节动作定义，不修改动作空间及底层控制方式。

### Critic

优先保留现有 Critic 结构和特权观测，不要求 Critic 使用 VAE 输出。

### 训练一致性

必须保证：

- Rollout 和 PPO Update 使用一致的策略前向逻辑。
- PPO 更新时能够正确重算当前策略动作概率。
- 旧策略 log probability 和采样动作正确保存。
- VAE 使用的历史观测必须与对应 rollout 时刻对齐。
- 不允许未来时刻数据泄漏。
- 正确处理梯度反向传播和优化器参数注册。
- 不破坏现有 PPO 的 GAE、Clip、Value Loss 和 Entropy 计算。

## 六、联合训练损失

第一版采用以下损失：

### 1. VAE 重建损失

\[
L_{\mathrm{recon}}=\mathrm{MSE}(\hat o_t,o_t)
\]

### 2. KL 散度损失

\[
L_{\mathrm{KL}}=-\frac12\sum(1+\log\sigma^2-\mu^2-\sigma^2)
\]

实际实现时请正确处理 batch 和 latent 维度上的求和或平均。

VAE 损失：

\[
L_{\mathrm{VAE}}=L_{\mathrm{recon}}+\beta L_{\mathrm{KL}}
\]

初始建议：

```python
kl_weight = 0.001
```

该参数可配置。

### 3. 状态估计损失

从仿真器提取真实机身速度、足端速度、足端高度作为监督标签：

\[
L_{\mathrm{state}}=
L_{\mathrm{base}}+
L_{\mathrm{footvel}}+
L_{\mathrm{footheight}}
\]

每项使用 MSE。

注意对不同量纲的数据进行合理归一化，保持坐标系和定义一致。

### 4. PPO 联合损失

参考 MILD 的联合训练思想，初始设置：

\[
L_{\mathrm{total}}=
0.7L_{\mathrm{PPO}}+
0.3(L_{\mathrm{VAE}}+L_{\mathrm{state}})
\]

PPO Loss 应保留原项目中的 Actor、Critic 和 Entropy 组成。

上述参数作为第一版配置，不应假设其一定最优。

VAE 和 PPO 优先采用联合训练方式，不需要额外增加强制预训练阶段。

## 七、代码实现要求

请遵循以下原则：

1. 首先检查 MJLab 当前使用的 PPO 和 Actor-Critic 实现，确认可扩展接口。
2. 新建独立的 VAE 模块，尽量不要将全部网络实现混入环境文件。
3. Modulation Network 单独封装，方便消融实验。
4. 历史观测缓冲区应支持 GPU 并行环境。
5. 保留原始 PPO 训练模式，可通过配置开关控制是否启用 VAE。
6. 网络尺寸、历史长度、latent 维度、损失权重全部可配置。
7. 尽量使用 PyTorch 批量张量运算，不要使用逐环境 Python 循环。
8. 不依赖真实地面物理参数进行策略推理，确保未来可以移植到 G1 真机。
9. 保持现有训练入口、模型保存和模型加载机制正常工作。
10. 训练和部署阶段使用一致的观测归一化及历史缓存规则。

## 八、训练场景和实验设计

本次重点是实现算法模块，不要求重新构建复杂地形。

如果当前环境已有柔性地面或接触参数随机化机制，则直接复用。

如果没有，请检查 MuJoCo/MJLab 当前可用的接触参数配置，并说明如何构建硬地、软地和交界测试场景。

不要把简单修改摩擦系数等同于完整的柔性地面建模。

需要支持以下三个实验配置：

| 实验 | 结构 |
|---|---|
| Baseline | 原始 PPO |
| VAE | PPO + 多头 VAE |
| Full | PPO + VAE + Modulation Network |

为了保证可比较性，三个实验应尽量保持相同的奖励函数、训练预算和随机种子设置。

## 九、日志记录

需要记录：

```text
Train/
    PPO_Actor_Loss
    PPO_Critic_Loss
    Policy_Entropy

VAE/
    Reconstruction_Loss
    KL_Loss
    Base_Velocity_MSE
    Foot_Velocity_MSE
    Foot_Height_MSE
    Latent_Mean
    Latent_Std
    Modulation_Scale
    Modulation_Shift

Evaluation/
    Episode_Reward
    Velocity_Tracking_Error
    Episode_Length
    Fall_Rate
```

优先接入项目现有日志框架，不必重新开发可视化系统。

## 十、验证与验收

代码实现完成后，依次进行：

**第一阶段：网络前向验证**

- 测试历史观测输入维度。
- 测试每个预测 Head 的输出维度。
- 测试 VAE 重参数化和 Decoder。
- 测试 Modulation Network 输出。
- 确认 Actor 输入维度正确。

**第二阶段：反向传播验证**

- 对一批模拟数据执行前向与反向传播。
- 确认 VAE、Modulation、Actor 有正确梯度。
- 确认 KL、重建和监督损失均为有限数值。
- 检查是否存在 NaN 或梯度断裂。

**第三阶段：PPO 集成验证**

- 启动少量并行环境进行短时间训练。
- 确认 PPO rollout 和 update 能够正常运行。
- 确认历史缓冲区 reset 正常。
- 确认模型能够保存、加载和执行推理。
- 确认禁用 VAE 后仍可运行原始 PPO。

**第四阶段：训练验证**

如果有可用训练环境，启动短时 smoke test，观察各项损失变化。不要把短时间训练成功等同于算法性能验证。

## 十一、最终交付要求

请实际修改当前项目代码，而不是仅提供设计说明。

完成后给出：

1. 修改和新增的文件路径。
2. 各个文件的作用。
3. VAE 与 PPO 的最终数据流。
4. 训练启动命令。
5. Baseline、VAE、Full 三种实验的切换方法。
6. 已执行的测试及结果。
7. 尚未验证的问题和可能存在的风险。

如果论文未公开某个具体超参数，请明确区分“论文原始设计”和“工程默认设置”，不要虚构原文细节。

**执行顺序：先检查仓库，再实现网络，然后集成 PPO，最后测试。尽量采取最小侵入式修改，不要重构无关模块。**
