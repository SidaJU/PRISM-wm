"""画 RL 训练曲线：想象回报 + 想象终点到目标的距离。

用法: python prism/tools/plot_rl_curve.py [rl_<时间戳>目录]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

if len(sys.argv) < 2:
    p = Path('prism/runs/latest_rl.txt')
    d = Path(p.read_text(encoding='utf-8').strip()) if p.exists() else None
else:
    d = Path(sys.argv[1])
if d is None:
    print('未指定目录且无 latest_rl.txt')
    sys.exit(1)

hist = json.load(open(d / 'rl_history.json', encoding='utf-8'))
it = [r['iter'] for r in hist]
ret = [r['return'] for r in hist]
dist = [r['final_dist_cm'] for r in hist]
cr = [r['critic'] for r in hist]

fig, ax = plt.subplots(1, 3, figsize=(15, 4))


def smooth(x, k=15):
    if len(x) < k:
        return x
    return [sum(x[max(0, i - k + 1):i + 1]) / len(x[max(0, i - k + 1):i + 1])
            for i in range(len(x))]


ax[0].plot(it, ret, alpha=0.35, color='#2d7ff9')
ax[0].plot(it, smooth(ret), color='#2d7ff9', linewidth=2)
ax[0].set_title('Imagined return (higher = better)')
ax[0].set_xlabel('iteration')
ax[0].grid(alpha=0.3)

ax[1].plot(it, dist, alpha=0.35, color='#e74c3c')
ax[1].plot(it, smooth(dist), color='#e74c3c', linewidth=2)
ax[1].set_title('Final dist: object -> goal (cm, lower = better)')
ax[1].set_xlabel('iteration')
ax[1].grid(alpha=0.3)

ax[2].plot(it, cr, alpha=0.4, color='#27ae60')
ax[2].plot(it, smooth(cr), color='#27ae60', linewidth=2)
ax[2].set_title('Critic loss')
ax[2].set_xlabel('iteration')
ax[2].grid(alpha=0.3)

plt.tight_layout()
out = d / 'rl_curve.png'
plt.savefig(out, dpi=110)
print(f'saved: {out}')
print(f'return : {ret[0]:.2f} -> {smooth(ret)[-1]:.2f} (smoothed)')
print(f'dist cm: {dist[0]:.2f} -> {smooth(dist)[-1]:.2f} (smoothed)')
