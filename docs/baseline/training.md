# 本次 baseline 训练记录

> 此页记录首次 VAE+AMP baseline。当前调整后的 12 维 PPO+VAE 配置、UMR 核查和训练状态见 [baseline 调整 1](../change1.md)。首次训练旧日志按新 guide 的清理要求处理。

## 正式运行

- 任务：`Instinct-Locomotion-vae-G1-v0`。
- 启动：2026-10-13 11:21:42，Asia/Shanghai。
- 外部硬时限：2026-10-13 20:21:42；内部正常保存退出预算为 8.5 小时。
- GPU：NVIDIA GeForce RTX 5080，16 GB；训练开始阶段约占用 3 GB。
- 随机种子：42；1024 环境；24 steps/env；目标 5000 PPO updates；5 epochs、4 minibatches。
- 总交互上限：122,880,000 policy transitions，1,228,800,000 physics transitions。
- 按稳定约 4.8 s/update 外推，预计约 6.7 小时，约 18:02 结束；实际时间以日志为准。
- supervisor PID：`1037581`；训练 PID：`1037582`。这些 PID 仅指本次启动。
- run：`logs/instinct_rl/g1_locomotion_mild_vae_amp/2026-10-13_11-21-47_G1MildVAEAMP/`。
- stdout：`outputs/baseline/train_2026-10-13_11-21-42.log`。
- supervisor：`outputs/baseline/formal_supervisor.log`。
- 启动元数据：`outputs/baseline/launch_2026-10-13_11-21-42.json`。

启动命令：

```bash
.venv/bin/python -u scripts/baseline/train_bounded.py \
  --num-envs 1024 --iterations 5000 --max-hours 9
```

已将 supervisor 作为独立后台进程启动，终端/本次对话结束后继续运行。
内部 trainer 同样有独立进程组，watchdog 只向该进程组发停止信号。

交付前观察到训练已完成至少 12 次更新，累计 294,912 policy transitions、4,656 个
completed episodes；该轮耗时 4.79 s，所有记录的训练损失均为有限值。
当时 `regression_loss=0.4567`、`value_loss=0.0294`。这些只用于确认训练链路工作，
**不能用于判断已经学会行走或复现论文指标**。此文是启动记录，实时状态读取下列日志。

```bash
tail -n 1 logs/instinct_rl/g1_locomotion_mild_vae_amp/2026-10-13_11-21-47_G1MildVAEAMP/progress.jsonl
tail -f outputs/baseline/train_2026-10-13_11-21-42.log

# 完成后查看结束原因、完成轮数和实际退出码
cat outputs/baseline/launch_2026-10-13_11-21-42.json
cat logs/instinct_rl/g1_locomotion_mild_vae_amp/2026-10-13_11-21-47_G1MildVAEAMP/training_status.json

# 如需主动停止，trainer 会处理 SIGTERM 并保存
kill -TERM 1037582
```

## 验证记录

| 检查 | 结果 | 日志 |
| --- | --- | --- |
| 全套单元与真实 GPU 集成测试 | 29 项通过，无 skip | `outputs/baseline/gpu_tests.log` |
| UMR pipeline | 3 项通过 | `outputs/baseline/umr_tests.log` |
| 8 环境短训 | 完成 2 轮；checkpoint 可加载 | `outputs/baseline/smoke.log` |
| 256 / 512 / 1024 环境吞吐 | 每轮约 2.5 / 3.3 / 4.8 秒 | `outputs/baseline/benchmark_*.log` |
| supervisor 正常结束 | 完成 2 轮并保存，exit 0 | `outputs/baseline/supervisor_check.log` |
| runner 预算结束 | 完成 1 轮后正常保存，状态 `wall_time_budget_complete`，exit 0 | `outputs/baseline/budget_check.log` |
| 原生 CLI 无窗口回放 | 加载 checkpoint，执行 20 步，exit 0 | `outputs/baseline/play_check.log` |
| 完整 ONNX 导出 | 导出 encoder + modulation + actor；单元测试对照 PyTorch 通过 | `outputs/baseline/smoke_onnx/actor.onnx` |
| 源码校验 | 新增 Python 文件 Black 检查通过；编译通过；`git diff --check` 通过 | 命令检查 |

## 可复查的源码与环境

run 目录另外保存了：

- `source_snapshot.tar.gz`：235 个文件，含训练时使用的 Python 源码、测试、启动器、`uv.lock` 和配置。
- `source_sha256.json`：上述每个文件的 SHA-256。
- `runtime_versions.json`：运行版本、仓库 HEAD、论文 SHA-256 和 UMR 数据文件 SHA-256。
- `params/env.yaml` / `params/agent.yaml`：正式运行的实际参数。

版本：Torch 2.10.0；MuJoCo / MuJoCo Warp 3.11.0；Warp 1.14.0；mjlab 1.6.0；
instinct_rl 1.0.2；NumPy 2.4.2。

仓库 HEAD 为 `4334232347656ef476d8f21e21d0128c11203510`，训练使用包含本次实现和
用户既有未提交修改的工作目录，不能只用该 HEAD 还原训练源码；应使用源码快照。

尚未获得本次正式训练的最终指标。完整有符号材料随机化范围、原论文数值对照、三种子统计
和实机验证的限制见 [README.md](README.md)。
