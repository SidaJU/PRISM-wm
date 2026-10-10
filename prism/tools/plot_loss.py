"""训练 loss 可视化：每个 epoch 的 4 项 loss 数值与曲线，用于判断收敛。

用法:
    python prism/tools/plot_loss.py                     # 默认读 train_loss_fb01.json
    python prism/tools/plot_loss.py <loss_log.json>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

COLORS = {
    'loss': '#e74c3c', 'val': '#8e44ad', 'recon': '#3498db',
    'emb': '#27ae60', 'proprio': '#f39c12', 'kl': '#16a085',
}


def fmt(v):
    return f'{v:.5f}' if np.isfinite(v) else '   n/a'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', nargs='?', default=str(ROOT / 'prism/preview/train_loss_fb01.json'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f'X 找不到 loss 日志: {path}')
        return 1

    hist = json.loads(path.read_text(encoding='utf-8'))
    epochs = [r['epoch'] for r in hist]
    keys = [k for k in ('loss', 'recon', 'emb', 'proprio', 'kl', 'val') if k in hist[0]]

    # ---------- 1. 数值表 ----------
    print('=== 每个 epoch 的各项 loss ===')
    print(f"{'ep':>3} " + ' '.join(f'{k:>9}' for k in keys))
    for r in hist:
        print(f"{r['epoch']:>3} " + ' '.join(f'{fmt(r.get(k, np.nan)):>9}' for k in keys))

    # ---------- 2. 曲线 ----------
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    for i, k in enumerate(keys):
        ax = axes[i]
        vals = [r.get(k, np.nan) for r in hist]
        ax.plot(epochs, vals, '-o', lw=2, ms=4, color=COLORS.get(k, '#333'))
        ax.set_title(k, fontsize=13, fontweight='bold')
        ax.set_xlabel('epoch')
        ax.set_ylabel(k)
        ax.grid(alpha=0.3)

    # ---------- 3. 收敛判断（写在最后一个子图） ----------
    ax = axes[len(keys)]
    ax.axis('off')
    lines = ['Convergence check', '']
    third = max(1, len(hist) // 3)
    for k in ('loss', 'recon', 'kl'):
        if k not in keys:
            continue
        v = np.array([r.get(k, np.nan) for r in hist], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            continue
        head = float(np.mean(v[:third]))
        tail = float(np.mean(v[-third:]))
        rel = (head - tail) / max(abs(head), 1e-12)
        last_std = float(np.std(v[-third:]) / max(abs(tail), 1e-12))
        trend = 'down' if rel > 0.05 else ('flat' if abs(rel) <= 0.05 else 'up')
        lines.append(f'{k}: {head:.5f} -> {tail:.5f}  ({trend})')
        lines.append(f'    last-1/3 relative std = {last_std*100:.1f}%')
        lines.append('')

    if 'kl' in keys:
        kl = np.array([r.get('kl', np.nan) for r in hist], dtype=float)
        kl = kl[np.isfinite(kl)]
        if len(kl) and kl[-1] > 0.001:
            lines.append('KL alive (not collapsed) -> OK')
        else:
            lines.append('!! KL collapsed to ~0')
    lines.append('')
    lines.append('Note: free_bits=0.1 -> KL floor ~0.1')

    ax.text(0.02, 0.98, '\n'.join(lines), va='top', fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#fff8e6', edgecolor='#f5b041'))
    for j in range(len(keys) + 1, len(axes)):
        axes[j].axis('off')
    fig.suptitle('RSSM training loss (per epoch)', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path(args.out) if args.out else (ROOT / 'prism/preview/train_loss_curves.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f'\n曲线图已保存: {out}')

    # ---------- 4. 结论 ----------
    print('\n=== 收敛判断 ===')
    for line in lines:
        print('  ' + line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
