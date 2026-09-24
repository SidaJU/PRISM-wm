"""W3: 在 L0 上训练最小 RSSM（连续高斯版）。

用法:
    python prism/train/train_rssm.py --epochs 10 --batch 16 --seq 32
    python prism/train/train_rssm.py --max-episodes 50 --epochs 1     # 快速冒烟
"""
import argparse
import sys
import time
from pathlib import Path

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
    return images, proprio, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'prism/data/L0'))
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--seq', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--latent', type=int, default=32)
    ap.add_argument('--hidden', type=int, default=200)
    ap.add_argument('--emb', type=int, default=256)
    ap.add_argument('--kl-weight', type=float, default=1.0)
    ap.add_argument('--free-bits', type=float, default=0.0)
    ap.add_argument('--max-episodes', type=int, default=None)
    ap.add_argument('--out', default=str(ROOT / 'prism/ckpt'))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        agg = {}
        for batch in dl:
            images, proprio, actions = to_seq_batch(batch, device)
            opt.zero_grad()
            out = model(images, proprio, actions)
            losses = rssm_loss(out, images, proprio, kl_weight=args.kl_weight,
                               free_bits=args.free_bits)
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v)

        n = max(1, len(dl))
        msg = ' '.join(f'{k}={v/n:.4f}' for k, v in agg.items())
        print(f'epoch {epoch:3d} | {msg} | {time.time()-t0:.1f}s')

        # 简单验证
        if len(dl_val):
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for batch in dl_val:
                    images, proprio, actions = to_seq_batch(batch, device)
                    out = model(images, proprio, actions)
                    l = rssm_loss(out, images, proprio, kl_weight=args.kl_weight)
                    val_loss += float(l['loss'])
                print(f'          val_loss={val_loss/max(1,len(dl_val)):.4f}')

        torch.save({'model': model.state_dict(), 'args': vars(args)},
                   out_dir / f'rssm_epoch{epoch}.pt')

    print('done ->', out_dir)


if __name__ == '__main__':
    main()
