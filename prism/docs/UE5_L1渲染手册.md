# UE5 离屏渲染（L1 档位）使用手册

> 给下一个 agent：看完这份文档，你应该能**直接复现 L1 渲染**，不用重新踩坑。
>
> 目标：把 MuJoCo 算好的物理轨迹（位姿），在 UE5 里用高保真画质重新渲一遍，得到 **L1 数据集**。
> 关键点：**物理完全相同，只有像素不同**（这是三档对比实验成立的前提）。

---

## 1. 路径清单

| 用途 | 路径 |
|---|---|
| **UE 工程** | `G:/theozhu_THEOZHU-PC1_4001/RaidGame/RaidGame.uproject` |
| **源码引擎** | `G:/theozhu_THEOZHU-PC1_4001/UnrealEngine`（UE 5.6.1，源码编译） |
| 编辑器（GUI） | `RaidGame/Binaries/Win64/RaidGameEditor.exe` |
| **命令行版（渲染用）** | `RaidGame/Binaries/Win64/RaidGameEditor-Cmd.exe` |
| **UBT 编译脚本** | `G:/theozhu_THEOZHU-PC1_4001/UnrealEngine/Engine/Build/BatchFiles/Build.bat` |
| **插件目录** | `RaidGame/Plugins/GSWMVerify/` |
| 插件源码 | `GSWMVerify/Source/GSWMVerify/{Public,Private}` |
| 已编译产物 | `RaidGame/Binaries/Win64/RaidGameEditor-GSWMVerify.dll` |

### 插件文件

| 文件 | 作用 |
|---|---|
| `GSWMCaptureActor.h/.cpp` | W2 冒烟测试用（自己生成 Cube 测试场景），**L1 不用它** |
| `GSWMReplayActor.h/.cpp` | **L1 回放渲染的核心**：读位姿、摆几何体、逐帧渲出 PNG |
| `GSWMVerifyModule.cpp` | 模块入口，靠命令行参数决定是否激活 |
| `GSWMVerify.Build.cs` | 编译配置 |

---

## 2. 关卡

**必须用：`/Game/Arts/THEOLEVEL`**（文件 `Content/Arts/THEOLEVEL.umap`，是用户专门搭建的）。

⚠️ 不要用 `LookDev_Raid_Empty` 之类的 LookDev 场景——之前渲出"蓝天+雪地"就是因为用错关卡 + 相机没对准。

关卡内容（编辑器里可见）：`DirectionalLight`、`ExponentialHeightFog`、`SkyAtmosphere`、`SkyLight`、`VolumetricCloud`、`Floor`（棋盘格地面）、`PlayerStart`。

---

## 3. 输入数据准备

### 3.1 数据从哪来

MuJoCo 侧采集脚本：`prism/collect/collect_mujoco.py`，加 `--export-poses` 会额外导出：

```
prism/data/L0/
├── obs/ep_XXXXX.npz        images / proprio / actions
├── gt/  ep_XXXXX.npz       object_state / achieved_goal / ...
├── poses/ep_XXXXX.bin      每帧几何体位姿（裸 float32）
├── geom_meta.json          24 个几何体的静态信息
└── camera.json             相机参数（见第 4 节）
```

**位姿格式**：`poses/ep_XXXXX.bin` = `float32 [T, ngeom, 12]`，每个 geom 是 `pos(3) + 旋转矩阵(9)`。
帧数 `T = 文件大小 / (ngeom × 12 × 4)`。

**geom_meta.json**：每个 geom 的 `name / type / size / rgba`。UE 侧据此建对应形状的 Actor、逐帧只更新 Transform（通用"场景镜像"方案，不需要 UE 复刻 MuJoCo 的运动学链）。

### 3.2 ⚠️ 必须拷到纯英文路径

UE 命令行对**中文路径**可能出问题。当前工作目录含中文（`C:\Users\theozhu\Desktop\拓展\World Model\...`），所以要拷一份到 `D:/gswm_in`：

```python
import os, shutil
src = 'prism/data/L0'; dst = 'D:/gswm_in'
if os.path.exists(dst): shutil.rmtree(dst)
os.makedirs(dst)
for f in ['geom_meta.json', 'camera.json', 'meta.json']:
    shutil.copy(os.path.join(src, f), dst)
shutil.copytree(os.path.join(src, 'poses'), os.path.join(dst, 'poses'))
```

（这两个文件是运行时给 UE 读的，不是最终产物。）

---

## 4. 相机（关键）

### 4.1 用 camera.json，不要手调

`GSWMReplayActor::SetupCapture()` 会**优先**读 `InputDir/camera.json`；读不到才退回手调的 `FocusPoint + CameraOffset`（那个会跑偏，之前"看到天空"就是这么来的）。

camera.json 格式（MuJoCo 世界坐标：米、Z-up、右手系）：

```json
{
  "pos":     [2.8979, -1.0754, 1.1548],
  "forward": [-0.6493, 0.7211, -0.2419],
  "up":      [-0.1619, 0.1798, 0.9703],
  "fovy": 45.0,
  "lookat": [1.3, 0.75, 0.55], "distance": 2.5,
  "azimuth": 132.0, "elevation": -14.0,
  "source": "mujoco scene.camera[0]（渲染实际使用值）"
}
```

⚠️ **不要手算** `lookat/azimuth/elevation → pos/forward/up`（elevation 符号很容易搞错，实测手算误差 18 px）。
正确做法：直接用 MuJoCo 渲染时**内部实际使用**的相机矩阵 `scene.camera[0]`。

生成脚本已写好：`prism/tools/export_camera.py`（会做双重验证）。

---

## 5. 运行渲染（核心命令）

```powershell
$exe  = 'G:\theozhu_THEOZHU-PC1_4001\RaidGame\Binaries\Win64\RaidGameEditor-Cmd.exe'
$proj = 'G:\theozhu_THEOZHU-PC1_4001\RaidGame\RaidGame.uproject'

Start-Process -FilePath $exe -ArgumentList `
  "$proj", "/Game/Arts/THEOLEVEL", "-game", "-RenderOffScreen", "-unattended", "-nosplash", `
  "-GSWMReplay", "-GSWMIn=D:/gswm_in", "-GSWMOut=D:/gswm_l1_out", "-GSWMMaxEp=1", `
  "-GSWMBias=4.0", "-GSWMLightScale=1.0" `
  -RedirectStandardOutput 'D:/gswm_run.log' -RedirectStandardError 'D:/gswm_run.err' -NoNewWindow
```

### 参数表

| 参数 | 默认 | 说明 |
|---|---|---|
| `-GSWMReplay` | — | **必须**，激活回放（插件检测到才 Spawn Actor） |
| `-GSWMIn=` | — | 输入目录（含 geom_meta.json / poses/ / camera.json） |
| `-GSWMOut=` | — | 输出目录 |
| `-GSWMMaxEp=` | 1 | 渲几个 episode（先 1 个验证，再批量） |
| `-GSWMBias=` | 2.0 | 曝光补偿（EV）。数值越大越亮 |
| `-GSWMLightScale=` | 1.0 | 灯光强度总缩放。**偏暗就调大**（如 5、10） |
| `-GSWMLevelLook` | off | 用关卡光照 —— **实测更黑（mean 8.5），别开** |
| `-GSWMFocus=x,y,z` | — | 手动相机注视点（仅在无 camera.json 时生效） |
| `-GSWMCamOff=x,y,z` | — | 手动相机偏移（同上） |

### 输出结构

```
D:/gswm_l1_out/
└── ep_00000/
    ├── frame_000.png
    ├── frame_001.png
    └── ...（每 episode 一个子目录）
```

分辨率默认 64×64（与 L0 一致）。

---

## 6. 日志在哪（重要）

UE 的日志**不会**写到 `-RedirectStandardOutput`（那个文件通常是空的）。
真正的日志在：

```
G:/theozhu_THEOZHU-PC1_4001/RaidGame/Saved/Logs/RaidGame.log
```

（多个实例同时跑时会有 `RaidGame_2.log`，取 mtime 最新的。）

用这个判断进度：

```python
import glob, os
fs = sorted(glob.glob('G:/theozhu_THEOZHU-PC1_4001/RaidGame/Saved/Logs/*.log'), key=os.path.getmtime)
lines = open(fs[-1], encoding='utf-8', errors='ignore').read().splitlines()
print([l for l in lines if '[GSWM]' in l][-10:])
```

看到 `[GSWM] REPLAY DONE` 就是跑完了。

**耗时参考**：工程加载约 2–3 分钟（RaidGame 是大工程），渲染 43 帧只要 1 秒左右。所以一次运行约 **3 分钟**。

---

## 7. 编译（改了 C++ 之后）

```powershell
$bat = 'G:\theozhu_THEOZHU-PC1_4001\UnrealEngine\Engine\Build\BatchFiles\Build.bat'
Start-Process -FilePath $bat -ArgumentList 'RaidGameEditor','Win64','Development',`
  '-project=G:\theozhu_THEOZHU-PC1_4001\RaidGame\RaidGame.uproject','-waitmutex' `
  -RedirectStandardOutput 'D:\ubt.log' -RedirectStandardError 'D:\ubt.err' -NoNewWindow
```

增量编译约 15–30 秒，`Result: Succeeded` 即成功。

⚠️ **坑：Live Coding**。如果 UE 编辑器开着，会报：
```
Unable to build while Live Coding is active
```
**必须先关闭编辑器**（或让用户 Ctrl+Alt+F11 完成 Live Coding）。

---

## 8. 已踩过的坑（务必记住）

| # | 坑 | 解法（代码里已处理 / 需注意） |
|---|---|---|
| 1 | **UE 渲染是异步的**：同一 Tick 内 SetActorLocation 后立刻 Capture，拿到的是**渲染线程上一帧**的画面（空场景） | Tick 状态机：这一 Tick 摆位姿，下一 Tick 才捕获 |
| 2 | **AStaticMeshActor 运行时 SetStaticMesh 不生效** | Spawn 空 Actor + `NewObject<UStaticMeshComponent>` + RegisterComponent |
| 3 | **自动曝光导致亮度逐帧漂移**（实测 79→98，差 23%），对数据集是致命的 | 锁定 AutoExposureMin/Max=1.0 + Bias |
| 4 | **Lumen/GI 需要多帧收敛**（第一帧黑、后面过冲亮），帧间亮度不一致 | `r.DynamicGlobalIlluminationMethod=0` 禁用 GI |
| 5 | 光照/材质首帧未稳定 | 每 episode 预热 12 帧（WarmupLeft=12） |
| 6 | **天空/大气在离屏下不渲染** → 画面大片纯黑（实测 65% 黑） | 已加背景幕（相机前方 40m 的大平面），但**效果有限**，需配合 `-GSWMLightScale` 调亮 |
| 7 | **Live Coding 活跃时无法编译** | 关闭编辑器再编译 |
| 8 | **中文路径** | 输入数据拷到 `D:/gswm_in` |
| 9 | 手算相机（azimuth/elevation → pos）符号易错（误差 18px） | 用 `scene.camera[0]` 的权威值 |
| 10 | 关卡用错（LookDev 场景）→ 渲出"蓝天雪地" | 用 `/Game/Arts/THEOLEVEL` |
| 11 | 关卡光照 `-GSWMLevelLook` 反而更黑（mean 8.5） | 不开；保持自己 SpawnLights |
| 12 | MuJoCo 方块只有 2.5cm，64×64 画面里**不到 1 像素**，颜色质心检测测不准 | 别用"找青色质心"验证相机，改用**渲染一致性**（见 export_camera.py 的验证 1） |

---

## 9. 当前状态 & 下一个 agent 的待办

### 已完成

| 项 | 状态 |
|---|---|
| MuJoCo 采集（带位姿） | ✅ `prism/data/L0`，743 episode，`--export-poses` |
| camera.json | ✅ 已生成并验证（渲染一致性 差 0.10） |
| 位姿正确性 | ✅ 与 gt 物体位置误差 **0.000 mm** |
| GSWMReplayActor 实现 | ✅ 读位姿/摆几何体/相机对齐/异步处理/曝光锁定/禁 GI/预热 |
| 单次渲染跑通 | ✅ 43 帧（与 L0 对齐），输出到 `D:/gswm_l1_v5` |
| 曝光参数 `-GSWMBias` | ✅ 已加（实测 2→4 提升有限：mean 51.6→61.5） |
| 光强参数 `-GSWMLightScale` | ✅ 代码已改（**尚未编译+实测**） |
| 背景幕 | ✅ 代码已加（相机前方 40m 平面，实测 mean 61.5→69） |

### ⚠️ 待办（按顺序）

1. **编译**（代码已改完，但没编译）：
   ```powershell
   Build.bat RaidGameEditor Win64 Development -project=... -waitmutex
   ```
   （编辑器必须已关闭）

2. **试光强**：用不同 `-GSWMLightScale` 渲，找亮度合适的那个（建议试 5、10）。目标：黑像素比例降到 **<20%**，mean 接近 L0（149）的量级。

   每次命令（改输出目录 `D:/gswm_l1_s5` 等），渲完用：
   ```python
   import imageio, numpy as np
   im = imageio.imread('D:/gswm_l1_s5/ep_00000/frame_000.png')[...,:3].astype(np.float32)
   lum = im.mean(axis=2)
   print('mean', im.mean(), '黑%', float((lum<10).mean())*100, '亮%', float((lum>=150).mean())*100)
   ```

3. **生成对比图给用户看**：`prism/tools/compare_l0_l1.py`（左 L0 / 右 L1 并排）。

4. **批量渲染** 743 episode（改 `-GSWMMaxEp=743`）。注意磁盘：每 episode 43 帧 × ~2KB ≈ 90KB，743 个约 65MB（没问题）。但**耗时**：每 episode 有 12 帧预热 + UE 加载一次（3 分钟）→ 743 episode 在**同一进程内**跑（代码已支持自动切下一个 episode），总共约 3 分钟加载 + 743×(12+43) Tick ≈ 几分钟到十几分钟。

5. **组装成数据集**：把 `D:/gswm_l1_out/ep_XXXXX/*.png` 打包成和 L0 一样的 npz 结构（`prism/data/L1/obs/ep_XXXXX.npz`），gt/ 直接复用 L0 的（**同一条轨迹，gt 相同**）。

---

## 10. 配套脚本

| 脚本 | 用途 |
|---|---|
| `prism/tools/export_camera.py` | 导出 camera.json（含渲染一致性验证） |
| `prism/tools/check_poses.py` | 验证位姿与 gt 是否同步（应为 0.000 mm） |
| `prism/tools/compare_l0_l1.py` | 生成 L0 vs L1 对比图 + 亮度统计 |
| `prism/tools/check_input.py` | 检查网络输入（形状/范围/无 NaN） |
| `prism/tools/imagine_demo.py` | imagine() 演示 + rollout 误差曲线 |

---

## 11. 一句话流程

```
采集数据（--export-poses）
  → 拷到 D:/gswm_in（英文路径）
  → 生成 camera.json（export_camera.py）
  → 关闭 UE 编辑器
  → UBT 编译（改了 C++ 才需要）
  → RaidGameEditor-Cmd.exe ... -GSWMReplay -GSWMIn=D:/gswm_in -GSWMOut=D:/gswm_l1_out
  → 看 Saved/Logs/RaidGame.log 里的 [GSWM] 进度
  → 用 compare_l0_l1.py 检查亮度与对齐
  → 批量渲 743 episode → 打包成 prism/data/L1/
```
