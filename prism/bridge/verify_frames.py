"""W2 验证：检查 UE5 -RenderOffScreen 导出的帧是否可用。

用法:
    python verify_frames.py D:/gswm_out
    python verify_frames.py D:/gswm_out --gif          # 额外合成 gif
"""
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    d = Path(args[0]) if args else Path('.')
    files = sorted(list(d.glob('frame_*.png')) + list(d.glob('*.png')))
    if not files:
        print(f'X 没找到任何 PNG: {d}')
        print('  UE5 是否正确运行？检查日志里有没有 [GSWM] saved ...')
        return 1

    print(f'=== {d} ===')
    print(f'帧数: {len(files)}')

    imgs = []
    ok = True
    for f in files:
        im = imageio.imread(f)
        imgs.append(im)
        std = float(im[..., :3].astype(np.float32).std())
        mean = float(im[..., :3].astype(np.float32).mean())
        flag = 'OK ' if std > 5 else '!! '
        if std <= 5:
            ok = False
        print(f'  {flag}{f.name}  {im.shape}  mean={mean:.1f} std={std:.1f}'
              + ('   <- 疑似黑屏' if std <= 5 else ''))

    shapes = {im.shape for im in imgs}
    print(f'\n分辨率一致: {len(shapes) == 1}  {shapes}')
    print(f'全部非黑屏: {ok}')

    if '--gif' in sys.argv:
        out = d / '_verify.gif'
        big = np.repeat(np.repeat(np.stack([im[..., :3] for im in imgs]), 3, axis=1), 3, axis=2)
        imageio.mimsave(out, list(big), fps=4)
        print(f'预览 gif: {out}')

    print('\n判定：帧数>0 且 全部非黑屏 => W2 第 1 项(-RenderOffScreen 可用) 通过')
    return 0 if (ok and len(files) > 0) else 1


if __name__ == '__main__':
    sys.exit(main())
