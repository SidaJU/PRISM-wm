"""笔记 01 的可运行示例：跟着一张 64x64 的帧，看它怎么变成 256 个数字。

运行:
    python prism/docs/demo_01_encoder.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]     # .../World Model
sys.path.insert(0, str(ROOT))

from prism.model.rssm import ObsEncoder        # noqa: E402

# ---------- 1. 一帧长什么样 ----------
# 真实数据里 images 是 uint8 (0~255)；DataLoader 里做了 /255，变成 0~1 的浮点数
frame = torch.zeros(1, 3, 64, 64)
print('输入：一帧画面')
print(f'  形状   {tuple(frame.shape)}')
print(f'  数字个数 {frame.numel()}   (64*64*3，每个像素 RGB 三个值)')

# ---------- 2. 逐层走一遍 Encoder ----------
enc = ObsEncoder(emb_dim=256)

h = enc.conv[1](enc.conv[0](frame))
print(f'\n第1层 Conv+ReLU  {tuple(h.shape)}   画面 64->32，通道 3->32')

h = enc.conv[3](enc.conv[2](h))
print(f'第2层 Conv+ReLU  {tuple(h.shape)}   画面 32->16，通道 32->64')

h = enc.conv[5](enc.conv[4](h))
print(f'第3层 Conv+ReLU  {tuple(h.shape)}   画面 16->8 ，通道 64->128')

h = enc.conv[7](enc.conv[6](h))
print(f'第4层 Conv+ReLU  {tuple(h.shape)}   画面 8->4  ，通道 128->256')

flat = h.reshape(h.size(0), -1)
print(f'\n拉平 (flatten)   {tuple(flat.shape)}   = {flat.numel()} 个数字')

emb = enc.fc(flat)
print(f'全连接 fc        {tuple(emb.shape)}')
print(f'\n最终输出：{emb.numel()} 个数字，压缩比 {frame.numel() // emb.numel()}:1')

# ---------- 3. 用真实数据再走一遍 ----------
print('\n--- 用数据集里真实的一帧 ---')
from prism.model.dataset import L0Dataset       # noqa: E402

ds = L0Dataset(str(ROOT / 'prism/data/L0'), seq_len=32, max_episodes=3)
sample = ds[0]
real = sample['images'][0].unsqueeze(0)         # 取序列第 0 帧 -> (1,3,64,64)
print(f'真实一帧 形状={tuple(real.shape)}  像素值范围=[{real.min():.2f}, {real.max():.2f}]')
e = enc(real)
print(f'编码结果 形状={tuple(e.shape)}  前 8 个数字: {[round(float(v), 3) for v in e[0, :8]]}')
