# PRISM-WM 项目进度与概览

> 更新日期：2026-09-11
> 配套文档：`PRISM-WM_项目计划书_v1.0.docx`（总体规划）、`PRISM-WM_可行性调研与校准方案_v2.0.md`（技术方案与坑）
> 本文档回答三个问题：**这个项目在干什么？现在做到哪了？接下来做什么？**

---

## 1. 项目是干什么的（一段话）

用 MuJoCo 算物理、用 UE5 画画面，构造「**同一条物理轨迹、三种渲染保真度**」的数据集，训练
DreamerV3 风格的世界模型（latent dynamics model），量化「**观测画质差异**」对世界模型的影响，
并用图形学手段（观测对齐）把性能找回来。

**一句话**：世界模型失效的原因往往不是动力学没学好，而是"眼睛"没学好——我们用引擎工程师的
能力把这件事量化、归因、修复。对应具身世界模型岗位 JD 的「从数据/模型/感知链路定位失败原因」。

## 2. 为什么这个设计成立

| 前提 | 依赖 | 状态 |
|---|---|---|
| 三档数据**物理完全相同** | 所有档位的位姿来自同一次 MuJoCo 采集（diff=0 已验证） | ✅ |
| **唯一自变量是渲染质量** | L0=MuJoCo 默认渲染，L1=UE5 中画质，L2=UE5 高画质（Lumen/Nanite） | 🔧 进行中 |
| 模型是"真 WM" | 输入是图像观测，**不喂任何物体真值**（本体感知除外） | ✅ 设计已定 |
| 结论可归因 | 3×3 保真度矩阵 + 失败归因（latent 漂移 vs 动力学误差） | ⬜ W5 |

## 3. 数据集三档定义

| 档 | 渲染器 | 状态 | 用途 |
|---|---|---|---|
| **L0** | MuJoCo 默认渲染，64×64 RGB | ✅ **1241 ep / 9.9 万帧**（已质检+修色） | 训练主数据 |
| **L1** | UE5 中等画质（离屏渲染） | 🔧 **链路已打通，首个 episode 47 帧已渲出** | 泛化测试/训练 |
| **L2** | UE5 高画质（Lumen/Nanite/材质） | ⬜ 复用 L1 管线，换画质配置 | 泛化测试 |
| L3(可选) | 3DGS 重建 | ⬜ 复用已有 3DGS 资产 | demo/第四档 |

## 4. 已完成的里程碑

### W1：L0 数据集 ✅（100%）

- 环境：`FetchPush-v4`（推箱任务），启发式策略 + 15% 随机探索 + 物理域随机化
- 规模：**1241 episode × ~80 帧 ≈ 9.9 万帧**，0.10 GB
- 数据**物理分离**：`obs/`（图像+本体10维+动作4维）与 `gt/`（物体真值15维+成功判定）
- 质量指标：成功率 44.3%、掉落率 5.2%、停在边缘 20%、静止帧 57%
- 踩过的坑（全部已修，详见 v2.0 文档第 9 节）：v3 废弃、episode 默认 50 帧截断、
  **body_mass 累积随机游走 bug**、灰方块与灰桌面同色（改青色）、推飞方块（边缘刹车+减速）、
  82% 卡边缘（绕行接近）

### W2：UE5 渲染链路 ✅（核心突破，刚完成）

目标：证明「UE5 能在命令行无窗口状态下，按 MuJoCo 的位姿逐帧渲染」。

**已打通的完整链路**：

```
MuJoCo 跑物理
  ├─ 导出位姿: 24 个几何体 × pos(3)+旋转矩阵(9)，与 L0 数据集逐帧 diff=0
  ├─ 导出相机: 从渲染管线直接读 pos/forward/up/fovy（手推公式会错 3.6 米！）
  ▼
UE5（RaidGame 内置插件 GSWMVerify，命令行 -GSWMReplay 激活）
  ├─ 读 geom_meta.json → 建 24 个对应形状的 Actor（Box/Sphere/Cylinder + MuJoCo 颜色）
  ├─ 读 poses/ep_XXXXX.bin → 跨帧状态机逐帧摆位姿（同 Tick 截图会拿到旧场景！）
  ├─ 读 camera.json → 相机与 MuJoCo 完全对齐（位置/朝向/FOV 45°）
  ├─ 自建光照（方向光+天光+补光）+ 固定曝光（数据集亮度一致性）
  └─ SceneCapture → ReadPixels → PNG 逐帧落盘
```

**验证结果**（首帧对比图见第 8 节）：

- 视角对齐 ✅：UE5 画面里的机械臂结构、桌子、方块布局与 MuJoCo 画面一一对应
- 运动同步 ✅：47 帧逐帧位姿驱动
- 待修：过曝（曝光 bias 调整中）

**零侵入承诺兑现**：全部代码在 `RaidGame/Plugins/GSWMVerify/`，不改 RaidGame 任何现有文件，
不带 `-GSWMReplay` 参数时对游戏零影响（仅一行 idle 日志）。

### W2 踩过的坑（UE5 侧，全部已修或定位）

| # | 坑 | 修法 |
|---|---|---|
| 1 | `AStaticMeshActor::SetStaticMesh` 运行时不生效（GetStaticMesh 恒 null） | Spawn 空 Actor + 手动 `NewObject<UStaticMeshComponent>` + `RegisterComponent` |
| 2 | 同 Tick 改位姿立即 CaptureScene → 渲染线程没更新，画面全空 | 跨帧状态机：本 Tick 摆位姿，下 Tick 拍 |
| 3 | 默认灰白材质与棋盘格地面融为一体 | `BasicShapeMaterial` 的 Color 参数应用 MuJoCo rgba |
| 4 | MuJoCo 辅助 geom（细条）渲成杂乱黑线 | 过滤 size<1cm 的几何体（保持 index 对应） |
| 5 | MuJoCo 相机参数手推公式**错了 3.6 米**（azimuth/elevation 约定反直觉） | 把参数喂回 `mujoco.Renderer`，让引擎算 pos/forward/up |
| 6 | 自动曝光导致帧间亮度漂移 | 固定 AutoExposure Min/Max/Bias |
| 7 | 定制引擎没有 `UnrealEditor-Cmd.exe` | 用 `RaidGameEditor-Cmd.exe`（项目 Binaries 下） |

## 5. 当前文件与工具清单

```
prism/
├── collect/
│   ├── collect_mujoco.py    # L0 采集（含 --export-poses 位姿导出）
│   ├── check_data.py        # 数据体检（--all 全量统计）
│   ├── drop_stat.py         # 掉落/边缘/静止帧统计
│   ├── dump_one.py          # 单条 episode 摊开看 + 预览图
│   ├── make_gif.py          # episode 合成 GIF
│   └── export_camera.py     # 导出 MuJoCo 相机精确位姿
├── engine/                  # UE5 插件源码副本（正式版在 RaidGame/Plugins/GSWMVerify）
├── bridge/verify_frames.py  # UE5 输出帧质检（帧数/分辨率/黑屏）
└── data/
    ├── L0/                  # 正式数据集（obs/ 与 gt/ 物理分离）
    └── _probe/              # 桥接调试用小批量数据

RaidGame/Plugins/GSWMVerify/  # 正式插件（已编译，随 RaidGame 维护）
D:\gswm_poses\                # UE5 回放输入（geom_meta.json + poses/*.bin + camera.json）
D:\gswm_l1\ep_00000\          # UE5 渲染输出（47 帧 PNG）
```

## 6. 剩余工作与计划

| 任务 | 内容 | 预估 | 状态 |
|---|---|---|---|
| W2 收尾 | 曝光校准、L1 画质微调、批量渲染 9 个 probe episode | 1 天 | 🔧 |
| W3 | 训练最小 RSSM（连续高斯版）跑通 L0 | 1 周 | ⬜ |
| W4 | TSSM（Transformer 主干）+ A 组 ablation | 1 周 | ⬜ |
| W5 | 3×3 保真度矩阵 + 失败归因 | 1 周 | ⬜ |
| W6 | 观测对齐（神经渲染）+ latent MPC | 1 周 | ⬜ |
| W7 | 技术报告 / arXiv | 1 周 | ⬜ |

## 8. 效果图

| 图 | 说明 |
|---|---|
| `![首帧对比 L0 vs L1](./_compare_l0_l1.png)` | 左 MuJoCo / 右 UE5，同一位姿（绝对路径见工作区根目录） |
| `![并排动图](./_l0_vs_l1.gif)` | 47 帧逐帧同步播放 |
| `D:\gswm_l1\ep_00000\frame_000~046.png` | UE5 原始输出 |

## 9. 关键经验

1. **数值体检全绿 ≠ 数据可用**：L0 的两个致命问题（方块不可见、79% 掉落）都是"看图"发现的。
   流程必须包含：数值统计 → 异常统计 → **人眼看 GIF**。
2. **跨引擎坐标/单位/手性**是最容易错的地方，必须用"让源引擎自己算"的方式拿到精确值，
   而不是手推公式（相机公式错 3.6 米的教训）。
3. **零侵入原则**：往生产项目里加代码，必须是独立插件 + 命令行激活，默认零影响。
