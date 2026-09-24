"""笔记 02 的可运行示例：手算卷积 / 拉平 / 全连接。

每一步都能用手验证，数字都是真跑出来的。

运行:
    python prism/docs/demo_02_conv_flatten.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

torch.manual_seed(0)

# ---------- 1. 一张极小的图 ----------
print('=' * 62)
print('1) 一张极小的图：4x4 灰度（真实是 64x64 彩色，原理完全一样）')
img = torch.tensor([[1, 2, 3, 1],
                    [0, 5, 6, 1],
                    [1, 7, 8, 1],
                    [0, 1, 1, 0]], dtype=torch.float32).reshape(1, 1, 4, 4)
for row in img.squeeze().int().tolist():
    print('   ', row)

# ---------- 2. 卷积：小窗口滑动，加权求和 ----------
print('\n' + '=' * 62)
print('2) 一个 2x2 卷积核在图上滑动，每停一次算一个加权和')
conv = nn.Conv2d(1, 1, kernel_size=2, stride=1, bias=False)
with torch.no_grad():
    conv.weight[:] = torch.tensor([[[[-1., 1.], [-1., 1.]]]])   # 左减右加 = 找竖边
print('kernel 权重（2x2，只有 4 个数）:')
for row in conv.weight.squeeze().tolist():
    print('   ', row)

out = conv(img)
print('输出（3x3，因为 2x2 窗口在 4x4 上只能停 3x3 个位置）:')
for row in out.squeeze().tolist():
    print('   ', [round(v, 1) for v in row])

manual = 1 * (-1) + 2 * 1 + 0 * (-1) + 5 * 1
print(f'\n手算左上角: 1*(-1)+2*1+0*(-1)+5*1 = {manual}')
print(f'程序输出[0][0]            = {out.squeeze()[0, 0].item()}   <- 一致')

# ---------- 3. 卷积 vs 全连接 ----------
print('\n' + '=' * 62)
print('3) 参数量对比：为什么用卷积而不是全连接')
c1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)      # 真实 Encoder 第 1 层
n_conv = sum(p.numel() for p in c1.parameters())
n_fc = (64 * 64 * 3) * (32 * 32 * 32)               # 全连接要达到同样输出尺寸
print(f'卷积 Conv2d(3,32,k=4,s=2)     参数 = {n_conv:,}')
print(f'若改成全连接（同样输出尺寸）   参数 = {n_fc:,}')
print(f'相差 {n_fc // n_conv:,} 倍')

# ---------- 4. 拉平 ----------
print('\n' + '=' * 62)
print('4) 拉平 flatten：把多维排成一条直线')
x = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
print('多维 (2,2,2):', x.squeeze().int().tolist())
print('拉平后 (8,) :', x.reshape(1, -1).squeeze().int().tolist())

# ---------- 5. 全连接 ----------
print('\n' + '=' * 62)
print('5) 全连接 fc：每个输出 = 所有输入 x 各自权重 求和')
f = nn.Linear(3, 2, bias=False)
with torch.no_grad():
    f.weight[:] = torch.tensor([[1., 0., -1.],
                                [0., 1., 1.]])
xx = torch.tensor([[2., 3., 4.]])
print('权重 w = [[1, 0, -1], [0, 1, 1]]，输入 x = [2, 3, 4]')
print(f'  输出[0] = 2*1 + 3*0 + 4*(-1) = {2*1 + 3*0 + 4*(-1)}')
print(f'  输出[1] = 2*0 + 3*1 + 4*1    = {2*0 + 3*1 + 4*1}')
print(f'  程序输出 = {f(xx).squeeze().tolist()}')

fc = nn.Linear(4096, 256)
print(f'\nEncoder 最后一步 fc: 4096 -> 256，参数 = {sum(p.numel() for p in fc.parameters()):,}')

# ---------- 6. 真实 Encoder 的参数分布 ----------
print('\n' + '=' * 62)
print('6) 真实 ObsEncoder 里参数都花在哪')
from prism.model.rssm import ObsEncoder      # noqa: E402

enc = ObsEncoder(emb_dim=256)
tot = sum(p.numel() for p in enc.parameters())
for name, p in enc.named_parameters():
    print(f'  {name:14s} {str(tuple(p.shape)):18s} {p.numel():>9,}  ({p.numel() / tot * 100:5.1f}%)')
print(f'  {"合计":14s} {"":18s} {tot:>9,}')
