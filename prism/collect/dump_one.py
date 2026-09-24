"""把一个 npz 的内容摊开看，并导出预览图。

用法: python dump_one.py prism/data/L0 ep_00000
"""
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

d = Path(sys.argv[1] if len(sys.argv) > 1 else 'prism/data/L0')
tag = sys.argv[2] if len(sys.argv) > 2 else 'ep_00000'

o = np.load(d / 'obs' / f'{tag}.npz')
g = np.load(d / 'gt' / f'{tag}.npz')

print(f'=== obs/{tag}.npz ===')
for k in o.files:
    a = o[k]
    print(f'  {k:12s} shape={str(a.shape):18s} dtype={a.dtype}  min={a.min()} max={a.max()}')
print(f'=== gt/{tag}.npz ===')
for k in g.files:
    a = g[k]
    print(f'  {k:14s} shape={str(a.shape):18s} dtype={a.dtype}')

print('\n--- 第 0 帧的具体数值 ---')
np.set_printoptions(precision=3, suppress=True)
print('proprio[0] (本体: 夹爪xyz + 手指开合2 + 夹爪速度3 + 手指速度2):')
print('   ', o['proprio'][0])
print('actions[0] (动作: xyz位移 + 手指):')
print('   ', o['actions'][0])
print('object_state[0] (真值: 方块xyz + 相对位移xyz + 旋转3 + 线速度3 + 角速度3):')
print('   ', g['object_state'][0])
print('desired_goal (本局目标位置):', g['desired_goal'])
print('success[0:10]:', g['success'][:10])
print('randomize:', g['randomize'])

imgs = o['images']
idx = [0, 25, 50, 75, 99]
frames = [imgs[i] for i in idx if i < len(imgs)]
grid = np.concatenate(frames, axis=1)
grid = np.repeat(grid, 3, axis=0)
grid = np.repeat(grid, 3, axis=1)
out = d / f'_preview_{tag}.png'
imageio.imwrite(out, grid)
print(f'\n预览图已导出: {out}  (帧 {idx} 横排, 放大3倍)')

# 视觉太稀，再出一张更密的：每 5 帧取一次，4x5 栅格
dense_idx = list(range(0, len(imgs), 5))[:20]
cells = np.stack([imgs[i] for i in dense_idx])           # (20,64,64,3)
rows, cols = 4, 5
gh, gw = 64, 64
grid2 = cells.reshape(rows, cols, gh, gw, 3).transpose(0, 2, 1, 3, 4).reshape(rows * gh, cols * gw, 3)
grid2 = np.repeat(np.repeat(grid2, 2, axis=0), 2, axis=1)
out2 = d / f'_preview_dense_{tag}.png'
imageio.imwrite(out2, grid2)
print(f'密集预览: {out2}  (20帧 / 4×5栅格, 每帧放大2倍)')

# 用数字证明：每隔 10 帧打印方块位置，看轨迹
op = g['object_state'][:, 0:3]
print('\n方块位置轨迹 (object_state[:, 0:3] 的前两个分量 XY, 米):')
for t in range(0, len(op), 10):
    x, y = op[t, 0], op[t, 1]
    print(f'  t={t:3d}  x={x:.3f}  y={y:.3f}  dx_from_t0={x-op[0,0]:+.3f}  dy_from_t0={y-op[0,1]:+.3f}')
print(f'\n首末位置: {op[0]} -> {op[-1]}')
print(f'位移: {np.linalg.norm(op[-1]-op[0])*100:.1f} cm  移动总路径: '
      f'{np.linalg.norm(np.diff(op,axis=0),axis=1).sum()*100:.1f} cm')
