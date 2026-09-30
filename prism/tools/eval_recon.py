"""加载已训练权重，验证重建效果，并导出「真实 vs 重建」对比图。

用法:
    python prism/tools/eval_recon.py                  # 默认加载 epoch20
    python prism/tools/eval_recon.py --ckpt 5         # 指定 epoch
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import imageio
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prism.model.dataset import L0Dataset        # noqa: E402
from prism.model.rssm import RSSM, rssm_loss     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='20', help='epoch 号')
    ap.add_argument('--dir', default=None, help='权重目录，默认 prism/ckpt')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--seq', type=int, default=32)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = Path(args.dir) if args.dir else (ROOT / 'prism/ckpt')
    ckpt_path = ckpt_dir / f'rssm_epoch{args.ckpt}.pt'
    if not ckpt_path.exists():
        print('找不到权重:', ckpt_path)
        return

    ckpt = torch.load(ckpt_path, map_location='cpu')
    ca = ckpt['args']
    model = RSSM(latent_dim=ca.get('latent', 32),
                 hidden_dim=ca.get('hidden', 200),
                 emb_dim=ca.get('emb', 256)).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'已加载 {ckpt_path.name}  device={device}')

    ds = L0Dataset(str(ROOT / 'prism/data/L0'), seq_len=args.seq, train=False)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, drop_last=True)
    print(f'val samples={len(ds)}  batches={len(dl)}')

    # ---------- 算 val loss ----------
    agg = {}
    with torch.no_grad():
        for batch in dl:
            images = batch['images'].to(device).permute(1, 0, 2, 3, 4).contiguous()
            proprio = batch['proprio'].to(device).permute(1, 0, 2).contiguous()
            actions = batch['actions'].to(device).permute(1, 0, 2).contiguous()
            out = model(images, proprio, actions)
            l = rssm_loss(out, images, proprio)
            for k, v in l.items():
                agg[k] = agg.get(k, 0.0) + float(v)

    n = max(1, len(dl))
    print('\nval loss（越小越好）')
    for k, v in agg.items():
        print(f'   {k:9s} = {v / n:.6f}')

    # ---------- 导出真实 vs 重建 ----------
    with torch.no_grad():
        batch = next(iter(dl))
        images = batch['images'].to(device).permute(1, 0, 2, 3, 4).contiguous()
        proprio = batch['proprio'].to(device).permute(1, 0, 2).contiguous()
        actions = batch['actions'].to(device).permute(1, 0, 2).contiguous()
        out = model(images, proprio, actions)
        recon = out['recon']                       # (T,B,3,64,64)

    real = images[:, 0].cpu().numpy()              # (T,3,64,64)
    pred = recon[:, 0].cpu().numpy()

    idx = [0, 8, 16, 24, 31]
    idx = [i for i in idx if i < real.shape[0]]

    rows = []
    for i in idx:
        r = (np.transpose(real[i], (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
        p = (np.transpose(pred[i], (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
        seg = np.concatenate([r, p], axis=1)
        seg = np.repeat(np.repeat(seg, 3, axis=0), 3, axis=1)
        rows.append(seg)
    grid = np.concatenate(rows, axis=0)

    out_png = ckpt_dir / f'_recon_epoch{args.ckpt}.png'
    imageio.imwrite(out_png, grid)
    print(f'\n对比图已导出: {out_png}')
    print('  每行左边=真实  右边=重建   （帧 %s 从上到下）' % idx)

    mse = float(((pred - real) ** 2).mean())
    print(f'\n像素 MSE = {mse:.6f}')
    print(f'参考：全预测成均值时的 MSE 约 {float(((real.mean() - real) ** 2).mean()):.6f}')


if __name__ == '__main__':
    main()
