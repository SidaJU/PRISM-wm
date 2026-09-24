# PRISM-WM 可行性调研与校准方案 v2.0

> 日期：2026-09-10
> 目的：在动手前把 MuJoCo / UE5 / Dreamer 三条基础链路的**实际代码与实测数据**核一遍，
> 有偏差就地校准，避免做到一半发现行不通。
> 结论：**方案主体成立，但有 1 处阻塞级错误、2 处方法学缺陷、3 处实现路线调整。**

---

## 0. TL;DR：改了什么

| # | 原方案（v1.0 计划书） | 校准后（v2.0） | 级别 |
|---|---|---|---|
| 1 | UE5 采集用 `-nullrhi` | **改用 `-RenderOffScreen`** | **阻塞级** |
| 2 | G-buffer 5 通道一次拿到 | 无 objectID 通道 → **后处理材质输出 RGBA16F**，objectID 用 CustomStencil | 阻塞级 |
| 3 | 三档各用不同观测模态（RGB / G-buffer / 3DGS） | **主实验三档统一 RGB**，G-buffer 降为独立 ablation | **方法学** |
| 4 | L2 = 3DGS 渲染同源轨迹 | 3DGS 渲不了多关节机械臂 → **L2 改为 UE5 高保真**，3DGS 降为可选第四档 + demo | **方法学** |
| 5 | 用 sheeprl（完整 DreamerV3） | **自写最小 PyTorch RSSM**（~300 行），有现成参考实现 | 实现 |
| 6 | 直接上 DreamerV3 离散 latent | 分阶段：**连续高斯 RSSM → TSSM →（可选）离散 latent** | 实现 |
| 7 | 环境 `FetchPush-v3` | **`FetchPush-v4`**（v3 已废弃） | 数据 |

好消息：**采集速度远超预期**，Isaac 的并行吞吐确认不需要。

---

## 1. 环境实测结果（已完成安装）

### 1.1 已安装

| 项 | 版本 | 位置 |
|---|---|---|
| Python | 3.11.15 | conda 环境 `llava` |
| torch | 2.7.1+cu128（CUDA 可用） | 同上 |
| mujoco | 3.13.0 | 同上 |
| gymnasium-robotics | 1.4.2 | 同上 |
| gymnasium | 1.3.0 | 同上 |
| dm_control | 1.0.46 | 同上 |

> 安装未破坏既有 torch / numpy（numpy 仍为 2.4.4）。

### 1.2 FetchPush-v4 实测

```
env = gym.make('FetchPush-v4', render_mode='rgb_array', width=64, height=64)

obs  = Dict{ observation:(25,), achieved_goal:(3,), desired_goal:(3,) }
act  = Box(-1.0, 1.0, (4,), float32)
info = { 'is_success': bool }            # 成功判据现成
render() -> (64,64,3) uint8              # 分辨率参数生效
```

**速度（单进程，实测）**

| 模式 | 速度 | 40 万步耗时 |
|---|---|---|
| 无渲染 | **1554 steps/s** | 约 4.3 分钟 |
| 64×64 RGB 渲染 | **194 steps/s** | 约 34 分钟 |

→ 目标数据集 2000 ep × 200 步 = 40 万步，**单进程半小时到一小时**，多进程还可线性加速。
→ **再次确认：Isaac 的 GPU 并行吞吐在本项目规模下完全用不上。**

### 1.3 一个必须处理的数据卫生问题

`observation` 那 25 维**包含物体真值**，构成如下：

| 字段 | 维数 | 归属 |
|---|---|---|
| grip_pos / grip_velp / gripper_state / gripper_vel | 3+3+2+2 = **10** | **本体感知 — 可进模型** |
| object_pos / object_rel_pos / object_rot / object_velp / object_velr | 15 | **他者真值 — 只进评测** |

> ⚠️ `object_rel_pos` 是相对夹爪的位移，虽含本体信息，仍归为真值侧，禁止进模型。
> 工程做法：写 dataset 时就把 25 维切成 `proprio[0:10]` 与 `gt[...]`，落盘成两个文件，
> 训练 dataloader 只挂载前者。

### 1.4 小坑

- `env.unwrapped.sim` **不存在**（新版 mujoco binding 用 `data` / `model`）。
  我们不需要它——25 维 obs 已经给了全部真值。
- 环境名必须是 `v4`，`v3` 会抛 `DeprecatedEnv`。

---

## 2. UE5 源码调研（基于你本地完整源码）

### 2.1 离屏渲染：用 `-RenderOffScreen`，不是 `-nullrhi`

| 开关 | 位置 | 行为 |
|---|---|---|
| `-RenderOffScreen` | `D3D12RHI/Private/Windows/WindowsD3D12Viewport.cpp:33`<br>`Windows/D3D11RHI/.../WindowsD3D11Viewport.cpp:134`<br>`VulkanRHI/Private/VulkanViewport.cpp:125`<br>`Slate/.../SlateApplication.cpp:857` | **有 RHI、无窗口、无 SwapChain** → 能渲染能采集 |
| `-nullrhi` | `RHI/Private/DynamicRHI.cpp:61` `InitNullRHI()` | **完全无渲染** → 只能跑物理 |

> **这是 v1.1 规格书里的错误**：headless 采集 L1 必须保留 RHI。
> 一句话记：`-nullrhi` = 物理能跑、画面没有；`-RenderOffScreen` = 画面有、窗口没有。

### 2.2 SceneCapture 能拿到什么

`Engine/Source/Runtime/Engine/Classes/Engine/EngineTypes.h:524`

```cpp
enum ESceneCaptureSource : int
{
    SCS_SceneColorHDR, SCS_SceneColorHDRNoAlpha, SCS_FinalColorLDR,
    SCS_SceneColorSceneDepth,   // SceneColor + SceneDepth in A
    SCS_SceneDepth,             // SceneDepth in R
    SCS_DeviceDepth,            // DeviceDepth in RGB
    SCS_Normal,                 // Normal in RGB (仅延迟渲染器)
    SCS_BaseColor,              // BaseColor in RGB (仅延迟渲染器)
    SCS_FinalColorHDR, SCS_FinalToneCurveHDR, SCS_MAX
};
```

**结论：depth / normal / baseColor 都有，但一次 capture 只能选一个，且没有 objectID。**

→ 校准做法：挂一个**后处理材质**在 SceneCapture 上，材质内采样
`SceneTexture:WorldNormal` + `SceneTexture:SceneDepth` + `SceneTexture:CustomStencil`，
一次输出 `RGBA16F = (normal.x, normal.y, depth, stencil)`。
objectID 通过给每个物体设 `CustomDepthStencilValue`（0–255）实现。

### 2.3 高精度读回：可行

```
Engine/Source/Runtime/Engine/Private/UnrealClient.cpp:79
    bool FRenderTarget::ReadPixels(TArray<FColor>&, FReadSurfaceDataFlags, FIntRect);
TextureRenderTarget2D / Cube / Volume 均提供
    ReadPixels(TArray<FFloat16Color>&)     ← 关键：支持 float16 读回
```

> 普通 RGBA8 会毁掉 depth 与 normal 的数值精度。
> **必须让 RenderTarget 格式为 RGBA16F，并用 `FFloat16Color` 版本读回。**

### 2.4 固定步长

```
Engine/Source/Runtime/Core/Private/Misc/App.cpp:39-40
    bool  FApp::bUseFixedTimeStep = false;
    double FApp::FixedDeltaTime  = 1/30.0;
UnrealEngine.cpp:2695   bUseFixedTimeStep = FApp::IsBenchmarking() || FApp::UseFixedTimeStep();
```

命令行参数名未在源码中直接确认（`-UseFixedTimeStep` / `-Benchmark -FPS=` 待实测），
**W2 必须实测确认**，否则帧间隔抖动会导致与 MuJoCo 不同步。

---

## 3. Dreamer 侧调研

### 3.1 结论：不适配 sheeprl，自写最小 RSSM

理由：sheeprl 是为**完整 RL** 设计的（actor / critic / GAE / 环境并行 / 日志 / 分布式）。
我们只需要它的 1/5（encoder + dynamics + decoder + KL），适配成本 > 自写成本，
而且自写在改主干（GRU→Transformer）时完全没有束缚。

### 3.2 现成参考实现（已确认存在，可直接照着写）

**datawhalechina / learn-world-model**（2026-08 更新，纯 PyTorch，中文）

| 章节 | 内容 | 对我们的用处 |
|---|---|---|
| P01 | VAE 编码器 | 观测 encoder 起点 |
| **P02** | **构建 RSSM 动力学模型**（GRU / MDN-RNN / RSSM 三对照） | **直接可用，约 120 行核心类** |
| P03 | 训练 Dreamer 智能体 | **跳过**（我们不做 RL） |
| **P04** | **替换动力学骨干网络** | **正是我们的 TSSM 改造** |
| P05 | 世界模型评估仪表盘 | 评测脚本起点 |
| **P06** | **反事实的动作条件世界模型** | 对应我们的反事实评测 |

P02 已含我们需要的全部要素：prior/posterior 分离、`rollout()`（只用先验）、
`horizon_mse()`（1..H 步长误差曲线）——正是我们的主指标。

其他参考：`bit1029public/HRSSM`（ICML-24，PyTorch）。

### 3.3 一个技术差异要心里有数

P02 的 RSSM 用的是**连续高斯 latent**，而 DreamerV3 是**离散 categorical latent（32×16）
+ straight-through + KL balancing + symlog**。

→ 校准：**分三阶段，一次只改一个变量**（也正好是 ablation 结构）

| 阶段 | 内容 | 目的 |
|---|---|---|
| S1 | 连续高斯 RSSM + GRU | 快速跑通全流程，验证管线 |
| S2 | **换 Transformer 主干（TSSM）** | 唯一变量 = 主干，得到 A 组 ablation |
| S3（可选） | 升级离散 latent + KL balancing | 与 DreamerV3 对齐，增强论文说服力 |

---

## 4. 校准后的方案 v2.0

### 4.1 三档保真度（重新定义——这是最关键的方法学修正）

**原问题**：三档用不同观测模态时，"渲染保真度"就不再是唯一自变量，
实验不干净，审稿人一句话就能打死。

**校准后：主实验三档统一输出 64×64 RGB，唯一变量是渲染器与渲染质量。**

| 档 | 渲染 | 配置 | 采集成本 |
|---|---|---|---|
| **L0** | MuJoCo 默认 | 平面光照、纯色材质、无阴影 | 分钟级（已实测 194 fps） |
| **L1** | UE5 中等 | 无 GI、简单材质、直接光照 | 小时级 |
| **L2** | UE5 高保真 | Lumen GI + Nanite + 复杂材质 + 相机噪声/后处理 | 小时级 |

- **3DGS 降级为可选第四档（L3）+ demo**：只做**静态工位 + 刚体方块**，
  不做多关节机械臂（3DGS 处理非刚体关节链成本过高，风险隔离）。
- **G-buffer 降为独立 ablation（D 组）**：三档各自额外存一份 G-buffer，
  单独做一次"RGB vs G-buffer"对照，不混进主矩阵。

### 4.2 观测与状态规格

| 项 | 内容 |
|---|---|
| 视觉观测 | 64×64×3 RGB（三档统一） |
| 本体状态 | 10 维：grip_pos(3) + grip_velp(3) + gripper_state(2) + gripper_vel(2) |
| 上帧动作 | 4 维 |
| **禁止进模型** | object_pos / object_rel / object_rot / object_velp / object_velr（15 维，仅评测） |

### 4.3 模型路线

```
Encoder : 64x64x3 -> Conv x4 (32/64/128/128, stride2) -> 8x8x128 -> 256d   ~0.4M
TSSM    : token_t = [e_t; p_t; a_{t-1}] -> Linear(128)
          2 层 causal Transformer, head4, ctx=32, KV cache                ~1.2M
          prior  : h -> z_logits
          post   : [h, e_t] -> z_logits
Decoder : [h,z] -> 64x64x3 (RGB) + 位姿增量                               ~0.7M
Loss    : 重建(MSE) + KL(post||prior) [+ λ L_phys 可选]
```
总参数 ~2.3M，5060 8GB 训练峰值 4–5 GB（与 v1.1 估算一致）。

### 4.4 周计划校准（8 周 → 7 周，采集提速）

| 周 | 目标 | 变化 |
|---|---|---|
| W1 | MuJoCo 采集 L0（2000 ep）+ 数据拆分（输入/gt 双文件） | 缩短（实测 30 分钟采完） |
| W2 | **MuJoCo→UE5 桥接 + `-RenderOffScreen` 实测 + L1/L2 采集** | **延长**（风险最高） |
| W3 | 最小 RSSM（S1）跑通 + rollout 可视化 | — |
| W4 | TSSM（S2）+ A 组 ablation | — |
| W5 | 3×3 保真度矩阵 + 失败归因 | — |
| W6 | 观测对齐（神经渲染）+ latent MPC | — |
| W7 | 技术报告 / arXiv | — |
| 选做 | 3DGS 静态场景（L3）+ demo 视频 | 不占主线 |

---

## 5. 待实测验证清单（W1–W2 必须确认，否则会翻车）

| # | 验证项 | 不通过的后果 | 降级方案 |
|---|---|---|---|
| 1 | `-RenderOffScreen` 下 SceneCapture 是否正常工作 | L1/L2 全灭 | 用带窗口最小分辨率 + 后台运行 |
| 2 | 后处理材质能否输出 RGBA16F (normal.xy, depth, stencil) | G-buffer 组 ablation 取消 | 主实验不受影响（只用 RGB） |
| 3 | 固定步长命令行参数名 | 帧间隔抖动，与 MuJoCo 不同步 | 改为"逐帧手动步进 + 等待渲染完成" |
| 4 | UE5 采集吞吐（目标 ≥ 30 fps） | 采集过慢 | 降低分辨率 / 关闭 Lumen |
| 5 | 遮挡查询（CustomStencil / 可见性） | 恒常性评测取消 | 主实验不受影响 |

> 建议顺序：**1 → 4 → 3 → 2 → 5**。第 1 项不过就立刻改路线，不要恋战。

---

## 6. 降级预案（按优先级砍）

1. 砍 3DGS（L3）→ 三档矩阵不变
2. 砍 G-buffer ablation（D 组）→ 4 组变 3 组
3. **砍到两档（L0 + L1）** → 2×2 矩阵仍能讲完整故事，论文成立
4. 砍观测对齐（B 组）→ 只剩诊断报告，仍有价值
5. 砍 MPC → 只剩长程 rollout 指标

**底线：L0 + L1 两档 + RSSM/TSSM 对照 + 一张误差-步长曲线 = 一个完整可讲的项目。**

---

## 7. 立即可执行的下一步

1. **W1**：写正式采集脚本（基于本次探针），产出 L0 数据集 2000 episode，
   落盘为 `obs/`（图像 + 本体 10 维 + 动作）与 `gt/`（物体 15 维 + is_success）两个目录。
   预计 **4–6 小时**（含写脚本）。
2. **W2 前置**：在 UE5 里起一个最小工程，**只验证 `-RenderOffScreen` + SceneCapture 读回**，
   先不同步 MuJoCo。预计 **3–4 小时**。
3. 两项都通过，再开始做桥接。

---

## 附录：本次调研引用的关键源码位置

| 主题 | 文件:行 |
|---|---|
| 离屏渲染开关（D3D12） | `Runtime/D3D12RHI/Private/Windows/WindowsD3D12Viewport.cpp:33` |
| 离屏渲染开关（D3D11） | `Runtime/Windows/D3D11RHI/Private/Windows/WindowsD3D11Viewport.cpp:134` |
| 离屏渲染开关（Slate） | `Runtime/Slate/Private/Framework/Application/SlateApplication.cpp:857` |
| NullRHI 入口 | `Runtime/RHI/Private/DynamicRHI.cpp:61` |
| SceneCapture 输出枚举 | `Runtime/Engine/Classes/Engine/EngineTypes.h:524` |
| SceneCapture 组件 | `Runtime/Engine/Classes/Components/SceneCaptureComponent.h:80` |
| 高精度像素读回 | `Runtime/Engine/Private/UnrealClient.cpp:79` |
| 固定步长变量 | `Runtime/Core/Private/Misc/App.cpp:39-40` |
| 固定步长使用点 | `Runtime/Engine/Private/UnrealEngine.cpp:2695` |

---

## 9. W1 执行记录（2026-09-10 已完成）

### 9.1 产物

| 项 | 内容 |
|---|---|
| 采集脚本 | `prism/collect/collect_mujoco.py` |
| 体检脚本 | `prism/collect/check_data.py`（`--all` 全量统计） |
| 数据集 | `prism/data/L0/`（`obs/` 与 `gt/` 物理分离 + `meta.json`） |
| 规模 | 1000 episode × 100 步 = **10 万帧**，磁盘 **0.09 GB**（压缩良好） |

采集耗时 **108 秒**（单进程）。

### 9.2 数据集规格（已落盘）

| 目录 | 字段 | 形状 | 用途 |
|---|---|---|---|
| `obs/` | `images` | (100,64,64,3) uint8 | 视觉观测（唯一图像输入） |
| `obs/` | `proprio` | (100,10) float32 | 本体感知：grip_pos(3)+gripper_state(2)+grip_velp(3)+gripper_vel(2) |
| `obs/` | `actions` | (100,4) float32 | 动作 |
| `gt/` | `object_state` | (100,15) float32 | **仅评测**：obj_pos/rel/rot/velp/velr |
| `gt/` | `achieved_goal` / `desired_goal` / `success` | — | 评测 |

### 9.3 数据质量（全量统计）

| 指标 | 值 | 判断 |
|---|---|---|
| episode 成功率（任意步） | 11.4% | 偏低但可接受 |
| 物体平均位移 | **44.4 cm** | 合理（修复前 174 cm = 被推飞） |
| 到目标最小距离 | 15.8 cm（阈值 5 cm） | 多数未达成 |
| 图像 mean/std | 150 / 51 | 非黑屏，渲染正常 |

> 世界模型训练**不依赖专家数据**，只需要状态-动作覆盖。11% 成功率对 W3 训练足够；
> 会影响到的是 W6 的 MPC 评测基线，届时再优化启发式。

### 9.4 本轮踩到的 3 个坑（已修）

1. **`FetchPush-v3` 已废弃** → 必须用 `v4`。
2. **`max_episode_steps` 默认只有 50**，env 会提前截断，导致 `steps=200` 静默失效。
   → 必须 `gym.make(..., max_episode_steps=args.steps)`。
3. **domain randomization 的累积 bug（最隐蔽）**：
   原写法 `model.body_mass[...] *= scale` 每个 episode 累积一次，
   形成随机游走，几百个 episode 后质量漂移到极端值 → 物体被推飞。
   → 必须保存初始 `body_mass` 副本，每次用 `base * scale` 赋值。
   修复前后物体平均位移：**174 cm → 130 cm → 44 cm**。

### 9.5 第二轮数据质量修复（由"肉眼看图"发现，非常重要）

第一版数据能通过数值体检（位移 44cm、成功率 11%），但**人眼看渲染画面时发现两个致命问题**——
这说明"数值指标正常"不等于"数据可用"，必须实际看图。

| # | 问题 | 根因 | 修复 | 效果 |
|---|---|---|---|---|
| 4 | **画面上看不出方块** | `object0` 默认中灰 (0.5,0.5,0.5)，与灰色桌面几乎同色；红色 (1,0,0) 是 `target0`（目标标记，本就不动），被误认成方块 | 采集时重设 object0 颜色 | 方块在 64×64 下清晰可辨 |
| 4b | **橙/红分不清** | 先用橙色 (1,0.35,0)，但它与红色 (1,0,0) 在 RGB 空间太近，64×64 + 光照后两者都"又红又橙" | 改用**青色 (0,0.95,0.95)**——红/橙的互补色，与灰桌面、蓝白机械臂、红球都能一眼区分 | 颜色语义无歧义 |
| 5 | **79% 的 episode 方块被推下桌面** | 启发式推过头；且掉落后仍继续采集 → 平均第 27 帧掉落，**73% 的帧是"方块躺在地上"的垃圾帧** | ① 加桌面边缘刹车（`TABLE_X/Y` + margin）<br>② 掉落即终止 episode（`--drop-z 0.35`）<br>③ 推速 0.5→0.3，接近速度 6→4 | 掉落率 **79% → 10%**；末帧 z 均值 0.417（桌面 0.425），方块基本全程在桌面上 |

| 6 | **82% 的 episode 停在边缘，77% 的帧方块完全静止** | 夹爪若在物体前方/侧面，直线冲向 approach 点会**撞穿物体**，把它顶向边缘；触发刹车后又不动 | **绕行接近**：先判定夹爪是否在物体后方（`dot(rel,dir_g)`），不在则抬高绕到侧后方再推；再加"静止窗口内无位移即终止 episode" | 成功率 **11%→44%**，停边缘 **82%→20%**，静止帧 **77%→57%** |

**踩坑记录**：安全区一开始设成比采样范围还窄（x[1.19,1.53]），导致一开局就误刹车，
位移从 26cm 掉到 5.8cm。平台比采样范围大（实测推到 y≈1.29 才掉落），
安全区必须放宽到 `TABLE_X=(1.15,1.58) TABLE_Y=(0.52,1.02)`。

### 9.6 最终数据指标（修复后重采）

| 指标 | 第一版 | 第二版 | **最终版** |
|---|---|---|---|
| episodes | 1000 | 928 | **1238**（2000 次尝试） |
| 总帧数 | 10 万 | 8.6 万 | **9.9 万** |
| 方块掉落率 | **79%** | 10% | **6.1%** |
| 停在边缘比例 | — | **82%** | **20%** |
| 静止帧比例（<1mm/步） | — | **77%** | **57%** |
| 成功率（任意步） | 11.4% | 14.3% | **44.3%** |
| 物体平均位移 | 44.4 cm（推飞） | 14.6 cm | **17.6 cm** |
| 到目标最小距离 | 15.6 cm | 15.6 cm | **7.4 cm**（中位 5.5，成功阈值 5） |
| 末帧 z 均值（桌面 0.425） | 0.111（地上） | 0.417 | **0.422** |

> 位移 44cm→17.6cm 不是退步：原来的 44cm 是"推飞+掉落"的伪动力学。
> 真正的转折是**绕行接近**——它让方块第一次被真正地"推向目标"而不是"撞向边缘"。

### 9.7 W1 结论

**L0 档位已完成，可以进入 W2（UE5 离屏渲染验证）。**
W2 是全项目风险最高的一环，建议优先做第 5 章验证清单的第 1 项。

**方法论教训（写进后续每一档的检查流程）**：
采集完必须做三件事——① 跑 `check_data.py --all` 看数值；② 跑 `drop_stat.py` 看异常；
③ **导出 GIF 用眼睛看一遍**。第 ③ 步抓出了前两步都没发现的两个问题。

---

## 10. W2 铺垫：UE5 离屏渲染验证（进行中）

### 10.1 环境确认

| 项 | 值 |
|---|---|
| 引擎版本 | **UE 5.6.1**（`Engine/Build/Build.version`，Branch UE5） |
| 源码位置 | `g:\theozhu_THEOZHU-PC1_4001\UnrealEngine` |
| 离屏开关 | `-RenderOffScreen`（D3D12/D3D11/Vulkan 均已确认支持） |

### 10.2 已写好的代码

| 文件 | 作用 |
|---|---|
| `prism/engine/GSWMCaptureActor.h/.cpp` | 最小采集 Actor：SceneCapture → `ReadPixels` → 存 PNG → 自动退出（源码副本） |
| `prism/bridge/verify_frames.py` | Python 侧验证：帧数、分辨率、是否黑屏 |
| **`RaidGame/Plugins/GSWMVerify/`** | **已安装到实际项目的独立插件**（5 个文件），命令行 `-GSWMVerify` 才激活 |

### 10.2b 为什么做成插件（重要）

RaidGame 是**在研大型项目**（10 个模块 + 140 个插件 + 服务器/客户端分离），
绝不能为了验证去改它的 GameMode 或任何现有代码——否则一启动游戏就变成"采集完退出"。

因此采用**独立插件 + 命令行激活**：

- 插件路径 `RaidGame/Plugins/GSWMVerify/`，模块名 `GSWMVerify`，不依赖 RaidGame 任何模块。
- **没有修改 RaidGame 的任何现有文件**（含 `.uproject`）。
- 模块 `StartupModule()` 里检测 `-GSWMVerify`：
  - **没有该参数** → 只打一行日志就返回，对游戏**零影响**（编辑器 / 游戏 / 服务器都是）。
  - **有该参数** → 注册 `OnPostWorldInitialization`，延迟 0.5s Spawn 采集 Actor，
    采完 `RequestExit`。

代码刻意做成**最小**：只证明"离屏下能拿到画面"，不涉及 MuJoCo 桥接、G-buffer、socket。
桥接是 W2 后半段的事，先确保地基稳。

### 10.3 操作步骤（照做即可）

1. 把 `GSWMCaptureActor.h/.cpp` 拷进项目 `Source/<YourProject>/`。
2. 编译（Development Editor）。
3. 在测试关卡里放一个 `GSWMCaptureActor`，位置对准有物体的地方。
4. 命令行运行（**无窗口**）：

```bat
UnrealEditor-Cmd.exe "D:\Proj\MyProj.uproject" /Game/Maps/TestMap ^
  -game -RenderOffScreen -ResX=320 -ResY=240 ^
  -NoSplash -unattended -GSWMOut=D:/gswm_out
```

> 必须是 `-RenderOffScreen`，**不能**用 `-nullrhi`（那个完全没有渲染）。

5. 验证：

```bat
python prism\bridge\verify_frames.py D:\gswm_out --gif
```

### 10.4 判定标准

| 检查 | 通过条件 |
|---|---|
| 有输出 | `frame_*.png` 数量 = `NumFrames` |
| 非黑屏 | 每帧 std > 5（脚本自动判定） |
| 分辨率 | 全部等于设定值（默认 64×64） |

三项全过 = W2 第 1 项（`-RenderOffScreen` 可用）通过，可以做桥接了。

### 10.5 已知坑 / 排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 全黑屏 | 场景无光照 / 相机对着空气 | 先放个 Cube + 平行光，或改用 `SCS_SceneColorHDR` |
| `ReadPixels` 失败 | 渲染状态被回收 | 已设 `bAlwaysPersistRenderingState = true` |
| 启动即崩 | RHI 未初始化 | 确认没传 `-nullrhi`，且显卡驱动正常 |
| 只有 1 帧 | `NumFrames` 默认值 | 改 Actor 上的 `NumFrames` |

### 10.6 W2 后半段（第 1 项通过后再做）

1. MuJoCo → UE5 状态桥接（先只同步 1 个方块 + 1 个相机）
2. G-buffer 导出（后处理材质 → RGBA16F → `FFloat16Color` 读回）
3. 固定步长确认（`-UseFixedTimeStep` / `-Benchmark -FPS=` 待实测）
