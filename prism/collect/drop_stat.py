"""统计有多少 episode 的方块被推下桌面。"""
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1] if len(sys.argv) > 1 else 'prism/data/L0')
TABLE_Z = 0.425   # 桌面高度
files = sorted((d / 'gt').glob('*.npz'))

dropped = 0
z_ends, first_drop = [], []
for f in files:
    g = np.load(f)
    op = g['object_state'][:, 0:3]
    z = op[:, 2]
    z_ends.append(float(z[-1]))
    below = np.where(z < TABLE_Z - 0.05)[0]      # 明显低于桌面 = 掉下去
    if len(below):
        dropped += 1
        first_drop.append(int(below[0]))
    else:
        first_drop.append(-1)

# 平台安全区（与 collect_mujoco.py 保持一致）
TABLE_X = (1.15, 1.58)
TABLE_Y = (0.52, 1.02)
EDGE = 0.05      # 距边缘 5cm 内算"停在边缘"
near_edge = 0
static_frames = 0
total_frames = 0
for f in files:
    g = np.load(f)
    op = g['object_state'][:, 0:3]
    p = op[-1]
    if (p[0] < TABLE_X[0] + EDGE or p[0] > TABLE_X[1] - EDGE
            or p[1] < TABLE_Y[0] + EDGE or p[1] > TABLE_Y[1] - EDGE):
        near_edge += 1
    step = np.linalg.norm(np.diff(op, axis=0), axis=1)   # 每步位移
    static_frames += int((step < 0.001).sum())            # < 1mm 视为静止
    total_frames += len(step)

fd = np.array([x for x in first_drop if x >= 0])
print(f'episodes           : {len(files)}')
print(f'方块掉下桌面的比例 : {dropped}/{len(files)} = {dropped/len(files):.3f}')
if len(fd):
    print(f'首次掉落帧         : 平均 {fd.mean():.0f} / 中位 {np.median(fd):.0f}  '
          f'(即后半程 {(100-fd.mean()):.0f}% 的帧是"方块已落地"的垃圾帧)')
print(f'末帧 z             : mean={np.mean(z_ends):.3f}  min={min(z_ends):.3f}  max={max(z_ends):.3f}')
print(f'停在边缘的比例     : {near_edge}/{len(files)} = {near_edge/len(files):.3f}  (距平台边<5cm)')
print(f'静止帧比例         : {static_frames}/{total_frames} = {static_frames/total_frames:.3f}  (每步<1mm)')
