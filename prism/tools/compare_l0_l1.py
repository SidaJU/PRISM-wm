"""生成 L0 vs L1 对比图（并排 + 统计）。

用法:
    python prism/tools/compare_l0_l1.py L0目录 L1目录 输出png
"""
import sys
from pathlib import Path

import numpy as np
import imageio

ROOT = Path(__file__).resolve().parents[2]


def main():
    l0_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (ROOT / 'prism/data/L0/obs')
    l1_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('D:/gswm_l1_v2/ep_00000')
    out_png = Path(sys.argv[3]) if len(sys.argv) > 3 else (ROOT / 'prism/preview/l0_vs_l1_v2.png')

    l1_files = sorted(l1_dir.glob('frame_*.png'))
    print(f'L0: {l0_dir}')
    print(f'L1: {l1_dir}  ({len(l1_files)} 帧)')

    # 对齐检查
    import numpy as np_  # noqa
    d = np.load(l0_dir / 'ep_00000.npz')
    T = len(d['images'])
    print(f'L0 ep_00000 帧数: {T}   L1 帧数: {len(l1_files)}   对齐: {T == len(l1_files)}')

    idx = [i for i in [0, 10, 20, 30, 40] if i < T and i < len(l1_files)]

    rows = []
    stats = []
    for i in idx:
        l0 = d['images'][i]                              # (64,64,3) uint8
        l1 = imageio.imread(l1_files[i])[..., :3]        # (64,64,3) uint8
        for name, x in (('L0', l0), ('L1', l1)):
            xf = x.astype(np.float32)
            stats.append((i, name, float(xf.mean()), float(xf.std()),
                          len(np.unique(x.reshape(-1, 3), axis=0))))
        pair = np.concatenate([l0, l1], axis=1)          # 左右拼
        pair = np.repeat(np.repeat(pair, 3, axis=0), 3, axis=1)   # 放大3倍
        rows.append(pair)

    grid = np.concatenate(rows, axis=0)
    imageio.imwrite(out_png, grid)
    print(f'\n对比图已导出: {out_png}  （每行: 左=L0  右=L1，帧 {idx} 从上到下）')

    print('\n量化统计:')
    print(f'  {"帧":>4} {"档":>3} {"mean":>8} {"std":>8} {"颜色数":>8}')
    for i, name, m, s, c in stats:
        print(f'  {i:>4} {name:>3} {m:>8.1f} {s:>8.1f} {c:>8}')

    # 结构相似性粗查：L0 与 L1 的相关性（相同内容应呈正相关）
    i0 = d['images'][idx[0]].astype(np.float32).ravel()
    i1 = imageio.imread(l1_files[idx[0]])[..., :3].astype(np.float32).ravel()
    if i0.std() > 0 and i1.std() > 0:
        corr = float(np.corrcoef(i0, i1)[0, 1])
        print(f'\n首帧 L0-L1 像素相关性: {corr:.3f}（>0.3 表示画面结构大体对应）')


if __name__ == '__main__':
    main()
