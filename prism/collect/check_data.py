"""数据集体检：形状、成功率、动力学信息量、体积。

用法:
    python check_data.py prism/data/L0            # 抽查 5 条
    python check_data.py prism/data/L0 --all      # 全量统计
"""
import sys
from pathlib import Path

import numpy as np


def stats(d, all_files=False):
    d = Path(d)
    files = sorted((d / 'obs').glob('*.npz'))
    if not all_files:
        files = files[:5]
    tot = succ = 0
    disp, mind, end_d, lens, size = [], [], [], [], 0
    for f in files:
        o = np.load(f)
        g = np.load(d / 'gt' / f.name)
        img = o['images']
        op = g['object_state'][:, 0:3]
        dg = np.linalg.norm(g['achieved_goal'] - g['desired_goal'], axis=1)
        tot += len(img)
        succ += bool(g['success'].any())
        lens.append(len(img))
        disp.append(float(np.linalg.norm(op[-1] - op[0])))
        mind.append(float(dg.min()))
        end_d.append(float(dg[-1]))
        size += f.stat().st_size + (d / 'gt' / f.name).stat().st_size
        if not all_files:
            print(f'\n{f.name}')
            print(f'  T={len(img)} img{img.shape}{img.dtype} proprio{o["proprio"].shape} '
                  f'act{o["actions"].shape} gt{g["object_state"].shape}')
            print(f'  img mean={img.mean():.1f} std={img.std():.1f}  (std<5 = 黑屏)')
            print(f'  obj disp={disp[-1]*100:.1f}cm  travel='
                  f'{np.linalg.norm(np.diff(op,axis=0),axis=1).sum()*100:.1f}cm')
            print(f'  dist to goal: start={dg[0]*100:.1f} min={dg.min()*100:.1f} end={dg[-1]*100:.1f}cm')
            print(f'  action std={o["actions"].std(axis=0).round(2)}  success_any={g["success"].any()}')
    if all_files:
        print(f'\n===== {d} =====')
        print(f'episodes      : {len(files)}')
        print(f'total frames  : {tot}')
        print(f'ep length     : mean={np.mean(lens):.0f} min={min(lens)} max={max(lens)}')
        print(f'success(any)  : {succ}/{len(files)} = {succ/max(1,len(files)):.3f}')
        print(f'obj displ(cm) : mean={np.mean(disp)*100:.1f} median={np.median(disp)*100:.1f}')
        print(f'min dist(cm)  : mean={np.mean(mind)*100:.1f} median={np.median(mind)*100:.1f}')
        print(f'end dist(cm)  : mean={np.mean(end_d)*100:.1f}')
        print(f'disk          : {size/1e9:.2f} GB')


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'prism/data/L0'
    stats(path, '--all' in sys.argv)
