"""对比两次训练的 loss 曲线（训练日志值 + 真实 KL）。

用法:
    python prism/tools/compare_train_loss.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PAIRS = [
    ('v2    free_bits=0.1 ', ROOT / 'prism/ckpt_fb01/loss_log.json',
     ROOT / 'prism/preview/loss_fb01_real.json'),
    ('fb001 free_bits=0.01', ROOT / 'prism/runs/train_20261009_190129/loss_log.json',
     ROOT / 'prism/runs/train_20261009_190129/loss_real.json'),
]


def load(p):
    try:
        return json.load(Path(p).open(encoding='utf-8'))
    except Exception:
        return None


def norm(d):
    """兼容 list[dict] / dict{key: list} 两种格式，返回 list[dict]"""
    if d is None:
        return None
    if isinstance(d, list):
        return d
    if isinstance(d, dict):
        keys = [k for k in d if isinstance(d[k], list)]
        if keys:
            n = min(len(d[k]) for k in keys)
            return [{k: (d[k][i] if isinstance(d[k], list) else d[k]) for k in d} for i in range(n)]
    return None


def show(rows, tag):
    if not rows:
        print(f'{tag}: loss_real 数据缺失')
        return
    ks = [k for k in rows[0] if k in ('epoch', 'loss', 'recon', 'emb', 'proprio', 'kl')]
    print(f'\n--- {tag} 真实值（eval_curve，KL 未过地板）---')
    print('  ' + ' '.join(f'{k:>9}' for k in ks))
    for r in rows[::4] + [rows[-1]]:
        print('  ' + ' '.join(f'{r[k]:>9.5f}' if isinstance(r[k], (int, float)) else f'{r[k]:>9}'
                              for k in ks))


for tag, log_p, real_p in PAIRS:
    print('=' * 64)
    print(tag)
    log = norm(load(log_p))
    if log:
        ks = [k for k in log[0] if k in ('epoch', 'loss', 'recon', 'emb', 'proprio', 'kl', 'val')]
        print('--- 训练日志（KL 为 clamp 后的值）---')
        print('  ' + ' '.join(f'{k:>9}' for k in ks))
        for r in log[::4] + [log[-1]]:
            print('  ' + ' '.join(f'{r[k]:>9.5f}' if isinstance(r[k], (int, float)) else f'{r[k]:>9}'
                                  for k in ks))
    else:
        print('训练日志缺失:', log_p)
    show(norm(load(real_p)), tag)
    print()
