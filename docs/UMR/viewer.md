# UMR 动作查看器

从项目根目录启动：

```bash
.venv/bin/python scripts/umr/viewer.py
```

浏览器打开 <http://127.0.0.1:8080>。默认暂停在第一段入选动作的首帧。
依赖项目环境已有的 `viser` 和 `mjviser`（由 `mjlab` 引入），不需要启动训练、加载策略或使用 GPU。
按 `Ctrl+C` 关闭服务。

## 数据与操作

- 默认数据目录为 `outputs/umr/motions/`，也支持环境变量 `UMR_MOTION_DIR` 或参数 `--motion-dir`。
- 默认列表遵循 `motions.yaml`；“全部候选动作”包含未入选 AMP 的文件。
- 当前本地数据为 18 段候选、13 段入选。界面显示类型、来源片段、采样率、时长和 AMP 入选状态。
- 播放、暂停、循环、倍速（0.25×、0.5×、1×、2×）、逐帧前后移动、拖动帧滑块。
- 切换动作会暂停并回到首帧；拖动和逐帧操作也会暂停。帧编号从 0 开始。
- 正面、侧面、俯视、重置视角；支持鼠标旋转、缩放与平移。预设视角沿世界坐标轴。
- 相机跟随保持水平跟随，不随身体高度上下抖动。
- 外观、碰撞几何及两者叠加；可切换地面、世界坐标轴和根部／左右踝轨迹。
- 原地查看仅消除每帧根部的水平平移，保留高度、朝向及关节动作，轨迹同步变换。

根部轨迹为橙色、左踝为蓝色、右踝为红色。踝轨迹表示踝关节所在 body 的原点，
不代表足底最低点或真实接触状态。原地查看改变了水平运动的显示方式，检查滑脚时应关闭它。
多个浏览器共享播放状态，各自可以调整相机；视角按钮会同步所有已连接浏览器。

## 自定义路径与远程访问

```bash
.venv/bin/python scripts/umr/viewer.py \
  --motion-dir outputs/umr/motions --port 8081
```

远程服务器可使用 SSH 端口转发，在本机执行：

```bash
ssh -L 8080:127.0.0.1:8080 user@server
```

需要局域网直接访问时，启动参数加 `--host 0.0.0.0`，然后打开 `http://服务器IP:8080`。

模型默认来自数据目录上一级的 `manifest.json`。迁移数据后可用 `--robot-xml` 指向匹配的 MJCF；
存在 manifest 哈希时会核对模型，防止以错误模型显示动作。没有 manifest 时使用仓库内的 G1 模型。
文件格式与 AMP 一致：`framerate`、`joint_names`、`joint_pos`、`base_pos_w`、`base_quat_w`，
四元数顺序为 `wxyz`。首版读取 `motions/` 中转换后的 NPZ，不直接读取 UMR 原始 `qpos` 文件。

## 实现与验证

`scripts/umr/viewer_data.py` 负责校验数据、按关节名称映射 MJCF、预计算轨迹和推进播放时间。
`scripts/umr/viewer.py` 使用 MuJoCo 正向运动学与 `mjviser.ViserMujocoScene` 显示姿态。
不执行物理积分，不修正动作、不截断关节角，也不修改数据和训练配置。

```bash
.venv/bin/python -m unittest discover -s scripts/umr -p test_viewer.py -v
```

检查包含关节重排、原地显示坐标、循环边界、倍速时间推进；本地数据存在时，还会逐文件
对照现有 `render.py` 的姿态构造方式，比较首／中／末帧的全部几何体位置和朝向。
