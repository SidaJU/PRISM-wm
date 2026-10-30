"""对比几种「用世界模型做决策」的方式，看谁能把物体推到目标。

用法: python prism/tools/plot_decision_compare.py [rl_<时间戳>目录]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# 各方式的「想象终点 -> 目标」距离（cm，来自 plan_cem.py 与 RL 训练实测，样本 #57）
#   起点 -> 目标 = 22.72 cm
DATA = [
    ('CEM 规划（最优序列）', 0.40, '#27ae60'),
    ('CEM 规划（均值序列）', 6.42, '#52be80'),
    ('RL 策略 Actor-Critic', 13.00, '#f39c12'),
    ('随机搜索 256 条（最好）', 12.87, '#95a5a6'),
    ('零动作（不动）', 24.18, '#e74c3c'),
    ('随机动作（平均）', 24.83, '#c0392b'),
]

names = [d[0] for d in DATA]
vals = [d[1] for d in DATA]
colors = [d[2] for d in DATA]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(vals)), vals, color=colors)
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(names)
ax.invert_yaxis()
ax.set_xlabel('想象的终点：物体 -> 目标距离 (cm)，越低越好')
ax.set_title('用同一个世界模型做决策：谁的想象结果最接近目标？\n(起点距离 22.72 cm)')
ax.axvline(22.72, color='#2c3e50', linestyle='--', linewidth=1.5, label='起点距离 22.72 cm')
ax.legend()
for i, (b, v) in enumerate(zip(bars, vals)):
    ax.text(v + 0.4, i, f'{v:.2f} cm', va='center', fontsize=10)
ax.set_xlim(0, 28)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('prism/runs')
out = out_dir / 'decision_compare.png'
plt.savefig(out, dpi=110)
print(f'saved: {out}')
for n, v in zip(names, vals):
    print(f'  {n:24s} {v:6.2f} cm')
