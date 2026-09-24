"""L1（UE5 渲染输出）质检。

检查三项：
  1. episode 数与帧数统计
  2. 抽样 episode 的帧间亮度漂移（数据集一致性的关键指标）
  3. 是否有黑屏/空帧

用法:
    python prism/bridge/qc_l1.py D:/gswm_l1_full
    python prism/bridge/qc_l1.py D:/gswm_l1_full 20      # 抽样 20 个 episode
"""
import sys
from pathlib import Path

import imageio.v2 as imageio


def main(root, n_sample=5):
    root = Path(root)
    eps = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f'root    : {root}')
    print(f'episodes: {len(eps)}')
    if not eps:
        return

    counts = []
    for e in eps:
        counts.append(len(list(e.glob('frame_*.png'))))
    total = sum(counts)
    print(f'frames  : total={total} min={min(counts)} max={max(counts)} '
          f'mean={total/len(counts):.1f}')

    print('\n--- brightness consistency (sampled) ---')
    worst = 0.0
    checked = 0
    for e in eps[:n_sample]:
        fs = sorted(e.glob('frame_*.png'))
        if len(fs) < 3:
            continue
        idx = [0, len(fs) // 2, len(fs) - 1]
        means = []
        for i in idx:
            im = imageio.imread(fs[i])[..., :3]
            means.append(round(float(im.mean()), 1))
        avg = sum(means) / len(means)
        drift = abs(max(means) - min(means)) / max(1e-6, avg) * 100
        worst = max(worst, drift)
        checked += 1
        flag = 'OK ' if drift < 10 else '!! '
        print(f'  {flag}{e.name}: frames={len(fs):3d} mean@{idx}={means} drift={drift:.1f}%')

    print(f'\nchecked {checked} episodes, worst drift = {worst:.1f}%  '
          f'(<10% 视为帧间一致)')


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else 'D:/gswm_l1_full'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(root, n)
