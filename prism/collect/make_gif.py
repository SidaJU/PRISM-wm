"""把一整局 episode 合成 GIF，让"动没动"一目了然。

用法: python make_gif.py prism/data/L0 ep_00000
      python make_gif.py prism/data/L0 ep_00000 --frames 50 --scale 3
"""
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

d = Path(sys.argv[1] if len(sys.argv) > 1 else 'prism/data/L0')
tag = sys.argv[2] if len(sys.argv) > 2 else 'ep_00000'
n_frames = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else None
# 支持 --frames 与 --scale
parts = sys.argv[3:]
n_frames = None
scale = 3
if '--frames' in parts:
    n_frames = int(parts[parts.index('--frames') + 1])
if '--scale' in parts:
    scale = int(parts[parts.index('--scale') + 1])

o = np.load(d / 'obs' / f'{tag}.npz')
imgs = o['images']
if n_frames is None or n_frames > len(imgs):
    n_frames = len(imgs)
imgs = imgs[:n_frames]

# 放大
big = np.repeat(np.repeat(imgs, scale, axis=1), scale, axis=2)

# 自动 fps：100 帧 5s 最舒服
fps = max(4, min(15, 100 // max(1, n_frames // 5)))
out = d / f'_anim_{tag}.gif'
imageio.mimsave(out, list(big), fps=fps, loop=0)
print(f'GIF: {out}  frames={n_frames}  fps={fps}  size={big.shape[1]}x{big.shape[2]}')
print('duration = %.1fs' % (n_frames / fps))