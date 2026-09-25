# Task: mild-play — G1 MILD 回放与评估计划

状态更新：均匀介质下的 MILD contact + 普通 PPO 回放入口已实现，见 [当前实现与命令](mild-contact.md)。下文保留完整论文的回放/评估计划；VAE 网络、空间材料带及系统化批量评估属于后续工作。

## 1. 任务定位

拟注册 `Instinct-Locomotion-Mild-G1-Play-v0`，基于 mild 的同一接触模型和训练完成的同一策略，执行固定条件回放、材料扫描、软硬切换和批量评估。不能在 play 时把地面退回普通 plane 接触，再用视频声称策略已经适应软地。

本任务与训练任务共享 `MildRlEnv`、`G1MildPPORunnerCfg`、`MildActorCritic`、normalization/schema、动作尺度与控制周期。只调整场景、随机化开关、命令生成和日志；不训练第二套 play 策略。

## 2. envcfg：相对现有 flat-play 和 mild 的改动

现有 `instinct_g1_locomotion_flat_env_cfg(play=True)` 只将环境数设为 1、CCD iterations 改为 50、命令重采样改为 2 s，并关闭 `base_external_force_torque` / `push_robot`。它**没有自动关闭** startup 材质/质量随机化、reset 随机化和 policy 观测噪声。MILD 的确定性评估必须逐项设置，不能直接继承后认为随机项已经全部移除。

| 配置区域 | mild-play 计划值 / 行为 |
| --- | --- |
| `scene.num_envs` | 交互回放 1；headless 评估可设 64 或 128 |
| 机器人、动作 | 与 mild checkpoint 一致，29 DOF、关节順序、default pose、逐关节 action scale 均校验 |
| `sim.mujoco.timestep` / `decimation` | 与训练一致，默认 0.001 / 20；不能为画面速度改物理周期 |
| 求解器 | 与训练相同容差、迭代上限、面积采样与附加质量处理；允许减少绘制密度，不减少物理单元 |
| `scene.terrain` / `deformable_contact` | 固定参数或固定空间材料图，正常计算 MILD 足地反力 |
| `observations` | 完全相同的 history 长度、排列、尺度和 frame；干净回放可关闭噪声 |
| `events` | 干净回放关闭随机质量/COM/偏置/延迟/推扰，固定初始状态；鲁棒性评估显式恢复对应 profile |
| `commands` | 固定命令、显式时间序列、交互命令三种来源；默认固定 vx=0.6、vy=0、wz=0 |
| `curriculum` | 关闭；评估过程中不因成功/失败改变材料难度 |
| `rewards` | 保留训练定义用于诊断；结果主要使用物理指标，不能只比较总 reward |
| `terminations` | 保留真实跌倒、非法接触、超深下陷、数值异常；区分 timeout 与 failure |
| `episode_length_s` | 基础定量评估 20 s；连续转换场景明确加长并固定统一时长 |
| `viewer` | 跟随机器人，显示名义表面、脚底单元、下陷、受力、CoP 和区域边界 |
| `monitors` | 输出时间序列、episode 汇总、参数/profile 与 checkpoint 标识 |

新增三个显式回放 profile：

- `clean`：固定物理与初始化，关闭观测噪声和各项随机化，验证模型/策略链路。
- `robustness`：固定测试 seeds，启用与训练相同的噪声、延迟、质量/COM/偏置；可另开推扰，单独报告。
- `transition`：固定空间材料带，其他随机项默认关闭，隔离在线地形适应。

关闭随机化意味着禁用事件或设固定值，并保存展开后的配置；不能仅靠 seed 把评估称为“无随机化”。异步 episode reset 继续清理对应环境的足底状态和 history，但**地形跨区时不清空本体 history**，让策略依靠新观测自行调整。

## 3. agentcfg：回放具体改动

| 项目 | 从 locomotion-flat 到 mild-play 的要求 |
| --- | --- |
| policy / algorithm / runner | 加载 mild 的网络和 runner；不再实例化普通 flat `ActorCritic` |
| encoder / modulation | 推理时必须保留；modulation 接收估计足速 |
| decoder / critic | checkpoint 可保留，但 actor 推理不依赖；仅诊断/训练时运行 |
| latent / 动作 | latent 使用训练定义的确定性 mean；输出动作分布均值，关闭动作采样 |
| normalization | 加载训练统计并冻结，禁止 play 过程中更新均值/方差 |
| history | H=10 等参数从 checkpoint schema 读取并核对，不能临时改成单帧输入 |
| privileged truth | 可以记录真值和估计误差，但不得送入 actor 或替换估计输出 |
| `experiment_name` | 同训练的 `g1_locomotion_mild`，避免自动查到 flat 模型 |
| checkpoint | 正式评估显式指定文件，记录 hash/训练配置/seed；不使用随运行变化的“最新模型”作为对照 |
| optimizer | 不更新；回放不执行 PPO 或 estimator fine-tuning |

`mild` 和 `mild-play` 的 `load_instinct_rl_cfg()` 返回兼容的同一 agent 配置。play 的 batch size 可以改变，但观测/动作 schema 和网络参数形状必须完全一致；不接受通过 `strict=False` 静默跳过 MILD 权重的加载。

若后续导出 ONNX，推理图包含：输入 normalization、history 到 estimator、modulation、actor。history 可在外部缓存，但必须明确帧次序、reset 填充、关节顺序和动作还原；不只导出最终 actor MLP。用固定输入和连续 rollout 验证 PyTorch/ONNX 数值一致。

## 4. 地面测试集与材料切换

### 4.1 固定材料扫描

测试包含原生刚地对照、参数化硬/中/软介质、训练范围边界、未见参数组合。每个 preset 保存完整材料字段，不只保存一个 `stiffness` 标量。材料参数表复用 [mild 的随机化定义](mild.md#46-材料与训练域随机化)。

首版先从通过接触单元测试的 `validated_ranges` 构建固定 preset；具体 rho、k_D、b_D 等标定完成前不伪造真实泡沫参数表。`soft/medium/hard` 仅表示仿真设定，不把它们自动映射成论文真实 28d/45d/60d PU foam、sand 或 grass。

主速度点采用论文 Table III 的 vx={0.3,0.6,0.9,1.2} m/s，vy=0、wz=0；另测站立、转向和倒退。倒退和横向若超出主训练范围，应明确标注域外命令结果。

### 4.2 空间材料带与在线适应

优先实现 `较硬介质 → 软介质 → 较硬介质`，两侧仍使用同一个 MILD 接触模型，仅材料不同。这能先检验在线适应，又避免原生/自定义接触边界机制未完成时做错误解释。

真正的 `原生刚地 → MILD 软地 → 原生刚地` 作为后续场景：将原生刚地几何限制在刚性区域，软区域不放与足底碰撞的名义平面；MILD 单元按世界 xy 判断当前所属区域。刚地碰撞几何需检查编译后的 convex/capsule 包络、接触点和边界容差，保证没有横跨软区的隐藏支撑。

每个单元在每个物理子步只由一个接触模型拥有。脚板跨边界时允许不同单元分别承受刚/软地反力；不能简单以 base 或整只脚中心的位置全局切换模型。区域坐标使用相对各环境 origin 的一致坐标系，避免并行环境读取错材料带。

跨介质边界时保留机器人本体 history；接触模型的 `h0`、累计漂移/压实状态按区域与当前侵入事件定义重新建立或连续迁移，并记录规则。不能在跨边界时触发环境 reset，也不能在脚正承重时每帧无条件清零接触历史。

初版不把这些突变带加入主训练域，从而检验论文强调的未见材料转换；若后续把它们加入训练，要另命名模型并分开报告。

## 5. 日志与定量指标

每个评估 case 保存 checkpoint hash、schema、seed、材料完整参数、初始化、命令、profile、环境/求解版本和实际运行步数。每项指标先按 episode 汇总，再按 case 与训练 seed 聚合，报告均值、离散程度和样本数。

| 指标 | 计算 / 注意事项 |
| --- | --- |
| 成功率 / 生存时间 | 完成指定时长/路程且未跌倒、未超深下陷、未数值失败；timeout 不记跌倒 |
| 速度跟踪 | vx/vy/wz MAE、RMSE；命令延迟时区分外部命令和策略收到的命令 |
| 下陷 | 每只脚、每个接触单元最大/均值/分位数，参照未变形表面 |
| 滑移 | 接触期足速、累计水平路径；区分脚中心与单元路径 |
| 足部净空 | swing 期相对未变形表面的高度，记录最小/峰值与拖脚比例 |
| 力与力矩 | 足底反力峰值、分布、CoP/力矩轨迹、关节力矩峰值、饱和比例 |
| COT | 行走 episode 用 `integral(sum(abs(tau*qdot))) / (m*g*实际行进距离)`；站立/近零距离单列能耗，不用极小分母制造分数 |
| 估计器精度 | base velocity、双足 velocity、双足 height 的分项误差；真值仅作诊断 |
| 转换响应 | 接触跨区时刻、最大跟踪偏差、跌倒率、恢复时间与轨迹 |
| 数值质量 | 求解残差、迭代数、未收敛比例、力裁剪率、异常参数计数 |
| 运行成本 | policy steps/s、physics steps/s、峰值显存、实际 wall time |

论文 COT 用 Power/(Weight×Velocity)，这里将恒速定义推广为整段能量/行进距离，明确报告该口径。失败 episode 的能耗和完成路程必须同时保留，不能只选成功样本让 COT 看起来更好。

转换恢复时间的工程初值定义为：跨区后速度误差回到预设阈值（例如 vx 误差 <0.1 m/s）并持续 1 s 的首次时刻；阈值在评估前固定，超时记未恢复。不能看曲线后再选有利阈值。

记录 z 和 z_tilde，必要时用同一组采样和固定降维 seed 做 t-SNE 可视化。但 latent 分群只是描述，不单独证明物理辨识正确；应联合看估计误差、转换响应和任务成功率。

## 6. 公平评估与验收

至少三个训练 seeds；工程建议每个模型×材料×命令使用相同的 20 个评估 seeds。相同评估 seed 生成相同材料、初始化、命令与扰动，模型之外条件相同。

所有模型在**同一评估接触场景**测试，包括原始 flat、Rigid+MILD 网络、MILD+普通 PPO、无 modulation 和完整模型。将“改善地面模型”和“改善策略结构”的贡献分别比较。单一模拟模型的优越结果不能直接等同于真实材料性能，实机材料标定和部署作为后续工作。

验收顺序：

1. 一环境固定条件：可加载完整 checkpoint，运行稳定，观测 history/normalization 和训练一致。
2. 接触可视化：下陷、接触区域、力与力矩方向正确，刚软区域没有重复支撑；验证可视化不会改物理状态。
3. 固定 seed 可重复回放，允许声明范围内的 GPU 数值差异；记录并保持配置一致。
4. 批量材料/速度评估：所有 case 均生成结果，失败/超时/数值异常有独立计数。
5. 空间转换：无 reset、无材料真值提示、无 estimator 真值替换，记录完整转换过程。
6. 如需导出，完成 PyTorch/ONNX 连续推理一致性验证。

入口如下（当前已支持接触消融 + 普通 PPO，完整自适应网络仍待实现）：

```bash
uv run instinct-train Instinct-Locomotion-Mild-G1-v0
uv run instinct-play Instinct-Locomotion-Mild-G1-Play-v0 --checkpoint-file /absolute/path/model.pt
```

批量评估 CLI 和 profile 参数在实现 `evaluation.py` 时再定义；本文不虚构现有脚本尚不支持的命令行参数。首版交付应包括可读的评估配置、CSV/JSON 汇总和选定 case 的受力/轨迹图。
