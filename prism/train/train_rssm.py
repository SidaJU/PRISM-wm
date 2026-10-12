"""W3: 在 L0 上训练最小 RSSM（连续高斯版）。

用法:
    python prism/train/train_rssm.py --epochs 10 --batch 16 --seq 32
    python prism/train/train_rssm.py --max-episodes 50 --epochs 1     # 快速冒烟
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prism.model.dataset import L0Dataset          # noqa: E402
from prism.model.rssm import RSSM, rssm_loss       # noqa: E402


def to_seq_batch(batch, device):
    """DataLoader 给出 (B,T,...)，RSSM 需要 (T,B,...)"""
    images = batch['images'].to(device).permute(1, 0, 2, 3, 4).contiguous()
    proprio = batch['proprio'].to(device).permute(1, 0, 2).contiguous()
    actions = batch['actions'].to(device).permute(1, 0, 2).contiguous()
    # v5：物体位置真值 (T,B,3)，作为预测目标传入 loss
    obj_pos = batch['object_pos'].to(device).permute(1, 0, 2).contiguous()
    return images, proprio, actions, obj_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'prism/config.yaml'), help='配置文件（YAML）')
    ap.add_argument('--data', default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--seq', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--latent', type=int, default=None)
    ap.add_argument('--hidden', type=int, default=None)
    ap.add_argument('--emb', type=int, default=None)
    ap.add_argument('--kl-weight', type=float, default=None)
    ap.add_argument('--free-bits', type=float, default=None)
    ap.add_argument('--max-episodes', type=int, default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--object-weight', type=float, default=None,
                    help='v5 物体位置预测 loss 的权重')
    ap.add_argument('--loss-log', default=None,
                    help='每个 epoch 的各项 loss 记录（JSON），默认写到 --out/loss_log.json')
    args = ap.parse_args()

    # 从 config 填充未显式指定的参数（优先级: 命令行 > config > 内置默认）
    cfg_path = Path(args.config)
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}
        except Exception as e:
            print(f'[warn] config 读取失败({e})，用默认参数')
            cfg = {}
        for k, v in (cfg.get('train', {}) or {}).items():
            if getattr(args, k, None) is None and v is not None:
                setattr(args, k, v)
        print(f'[config] {cfg_path}')
    else:
        print(f'[info] 未找到 config: {cfg_path}，用默认参数')

    # 内置兜底默认值
    defaults = {
        'data': str(ROOT / 'prism/data/L0'),
        'epochs': 10, 'batch': 16, 'seq': 32, 'lr': 1e-3,
        'latent': 32, 'hidden': 200, 'emb': 256,
        'kl_weight': 1.0, 'free_bits': 0.0, 'object_weight': 1.0,
        'out': str(ROOT / 'prism/ckpt'),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 0,
    }
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    device = torch.device(args.device)
    print(f'device={device}')

    train_ds = L0Dataset(args.data, seq_len=args.seq, train=True,
                         max_episodes=args.max_episodes)
    val_ds = L0Dataset(args.data, seq_len=args.seq, train=False,
                       max_episodes=args.max_episodes)
    print(f'train samples={len(train_ds)}  val samples={len(val_ds)}')

    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    drop_last=True)
    dl_val = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0,
                        drop_last=True)

    model = RSSM(latent_dim=args.latent, hidden_dim=args.hidden, emb_dim=args.emb).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 每次运行输出到带时间戳的子目录（<out>/train_<ts>/），并记下 latest 供推理读取
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_root = Path(args.out)
    out_dir = out_root / f'train_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'本次训练输出目录: {out_dir}')

    latest_file = out_root / 'latest_train.txt'
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        rel = out_dir.relative_to(ROOT)
    except Exception:
        rel = out_dir
    latest_file.write_text(str(rel), encoding='utf-8')
    print(f'latest 记录 -> {latest_file}')

    history = []      # 记录每个 epoch 的各项 loss，用于画曲线判断收敛
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        agg = {}
        for batch in dl:
            images, proprio, actions, obj_pos = to_seq_batch(batch, device)
            opt.zero_grad()
            out = model(images, proprio, actions)
            losses = rssm_loss(out, images, proprio, kl_weight=args.kl_weight,
                               free_bits=args.free_bits, object_pos=obj_pos,
                               object_weight=args.object_weight)
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v)

        n = max(1, len(dl))
        row = {'epoch': epoch}
        for k, v in agg.items():
            row[k] = v / n                       # 本 epoch 各项的平均 loss
        msg = ' '.join(f'{k}={v:.4f}' for k, v in row.items() if k != 'epoch')
        print(f'epoch {epoch:3d} | {msg} | {time.time()-t0:.1f}s')

        # 简单验证
        if len(dl_val):
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for batch in dl_val:
                    images, proprio, actions, obj_pos = to_seq_batch(batch, device)
                    out = model(images, proprio, actions)
                    l = rssm_loss(out, images, proprio, kl_weight=args.kl_weight,
                                  object_pos=obj_pos, object_weight=args.object_weight)
                    val_loss += float(l['loss'])
                val_loss = val_loss / max(1, len(dl_val))
                row['val'] = val_loss
                print(f'          val_loss={val_loss:.4f}')
        history.append(row)

        torch.save({'model': model.state_dict(), 'args': vars(args)},
                   out_dir / f'rssm_epoch{epoch}.pt')

    # 保存 loss 历史（供可视化与收敛判断）
    log_path = Path(args.loss_log) if args.loss_log else (out_dir / 'loss_log.json')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(history, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'loss log -> {log_path}')

    print('done ->', out_dir)


if __name__ == '__main__':
    main()
