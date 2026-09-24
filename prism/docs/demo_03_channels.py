"""笔记 03 的可运行示例：为什么输出是 (32,32,32) 而不是 (32,32,3)。

回答两件事：
  1. 通道数由 kernel 个数决定，跟输入通道数无关
  2. 排列顺序：numpy 是 HWC（通道在最后），PyTorch 是 CHW（通道在最前）

运行:
    python prism/docs/demo_03_channels.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------- 0. 排列顺序：同一个东西的两种写法 ----------
print('=' * 64)
print('0) 排列顺序：numpy 是 HWC，PyTorch 是 CHW')

img_np = np.zeros((64, 64, 3), dtype=np.uint8)          # 高, 宽, 通道
print(f'  磁盘 / numpy 里的图像   (H, W, C) = {img_np.shape}')

t = torch.from_numpy(img_np).permute(2, 0, 1)           # 通道挪到最前
print(f'  PyTorch 里的张量        (C, H, W) = {tuple(t.shape)}')

from prism.model.dataset import L0Dataset               # noqa: E402
ds = L0Dataset(str(ROOT / 'prism/data/L0'), seq_len=32, max_episodes=3)
print(f'  dataset 输出的单帧      (C, H, W) = {tuple(ds[0]["images"][0].shape)}')
print('  -> 所以笔记里的 (32,32,32) 是 (通道32, 高32, 宽32)，不是 (高,宽,通道)')

# ---------- 1. 一个 kernel 是 3D 的 ----------
print('\n' + '=' * 64)
print('1) 一个 kernel 不是 2D，是 3D：形状 (输入通道, k, k)')

# 造一张 3 通道的 4x4 小图
img = torch.zeros(1, 3, 4, 4)
img[0, 0] = torch.tensor([[1, 2, 3, 1],             # R 通道
                          [0, 5, 6, 1],
                          [1, 7, 8, 1],
                          [0, 1, 1, 0]], dtype=torch.float32)
img[0, 1] = torch.ones(4, 4)                        # G 通道：全 1
img[0, 2] = torch.zeros(4, 4)                       # B 通道：全 0

conv = nn.Conv2d(3, 1, kernel_size=2, stride=1, bias=False)
print(f'  conv.weight 形状 = {tuple(conv.weight.shape)}   <- (输出通道1, 输入通道3, 2, 2)')

with torch.no_grad():
    conv.weight[0, 0] = torch.tensor([[1., 0.], [0., 1.]])   # 压在 R 上
    conv.weight[0, 1] = torch.tensor([[0., 1.], [1., 0.]])   # 压在 G 上
    conv.weight[0, 2] = torch.tensor([[1., 1.], [1., 1.]])   # 压在 B 上

out = conv(img)
print(f'  输出形状 = {tuple(out.shape)}   <- 只输出 1 个通道！')

# ---------- 2. 手算左上角 ----------
print('\n' + '=' * 64)
print('2) 手算左上角：3 个通道各自算完，再加起来')

r = 1 * 1 + 2 * 0 + 0 * 0 + 5 * 1
g = 1 * 0 + 1 * 1 + 1 * 1 + 1 * 0
b = 0 * 1 + 0 * 1 + 0 * 1 + 0 * 1
print(f'  R 通道窗口 [[1,2],[0,5]] x kernel_R -> {r}')
print(f'  G 通道窗口 (全 1)       x kernel_G -> {g}')
print(f'  B 通道窗口 (全 0)       x kernel_B -> {b}')
print(f'  三者相加 = {r} + {g} + {b} = {r + g + b}')
print(f'  程序输出[0][0] = {out[0, 0, 0, 0].item()}   <- 一致')
print('\n  关键：3 个输入通道 -> 加在一起 -> 只得到 1 个数（1 个通道）')

# ---------- 3. 那 32 是哪个 32 ----------
print('\n' + '=' * 64)
print('3) 输出 32 个通道 = 用了 32 个不同的 kernel')

# 注意 padding=1：真实 ObsEncoder 就是这么写的。
# 输出尺寸公式: (64 - 4 + 2*1) / 2 + 1 = 32   <- 正好减半
# 若 padding=0:  (64 - 4) / 2 + 1 = 31      <- 不是整齐的一半
c32 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
print(f'  32 个 kernel 的权重形状 = {tuple(c32.weight.shape)}')
print(f'    解读: 32 个 kernel，每个形状 (3,4,4) = {3 * 4 * 4} 个权重')
print(f'    合计 = 32 x 48 = {32 * 3 * 4 * 4}  (+32 个偏置 = {sum(p.numel() for p in c32.parameters())})')

big = torch.zeros(1, 3, 64, 64)
o = c32(big)
print(f'\n  输入 (3, 64, 64) --Conv(3->32,k4,s2)--> 输出 {tuple(o.shape)}')
print(f'    第 1 个 32 = 通道数（kernel 个数，是我们自己定的）')
print(f'    后两个 32  = 空间尺寸（stride=2，64 减半）')

# ---------- 4. 输出通道数是自由选择的 ----------
print('\n' + '=' * 64)
print('4) 输出通道数想设多少就设多少，与输入通道数无关')
for n in (1, 3, 32, 256):
    cc = nn.Conv2d(3, n, kernel_size=4, stride=2, padding=1)
    print(f'  Conv2d(3, {n:3d}, k4 s2)  输入(3,64,64) -> 输出 {tuple(cc(torch.zeros(1, 3, 64, 64)).shape)}')
