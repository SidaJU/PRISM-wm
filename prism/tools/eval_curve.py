"""加载每个 epoch 的权重，算出该轮的 4 项 loss（不重新训练），可选保存 JSON。

重要：这里调用 rssm_loss 时**不传 free_bits**，因此得到的是"真实 KL"（不受训练时的地板 clamp 影响），
可用于判断 KL 到底有没有塌缩、以及真实值与 free_bits 地板的关系。

用法:
    python prism/tools/eval_curve.py --ckpt-dir prism/ckpt_fb01
    python prism/tools/eval_curve.py --ckpt-dir prism/ckpt_fb01 --out prism/preview/loss_fb01_real.json
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from prism.model.dataset import L0Dataset            # noqa: E402
from prism.model.rssm import RSSM, rssm_loss        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='prism/ckpt_fb01')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = L0Dataset(str(ROOT / 'prism/data/L0'), seq_len=32, train=False)
    dl = DataLoader(ds, batch_size=8, shuffle=False, drop_last=True)
    b = next(iter(dl))
    images = b['images'].to(device).permute(1, 0, 2, 3, 4).contiguous()
    proprio = b['proprio'].to(device).permute(1, 0, 2).contiguous()
    actions = b['actions'].to(device).permute(1, 0, 2).contiguous()

    ckpt_dir = ROOT / args.ckpt_dir
    epochs = sorted(int(p.stem.replace('rssm_epoch', '')) for p in ckpt_dir.glob('rssm_epoch*.pt'))
    print(f'读取 {ckpt_dir}   共 {len(epochs)} 个权重    device={device}')
    print('（不传 free_bits → 显示"真实 KL"）\n')
    print(f'{"epoch":>5} | {"loss":>10} | {"recon":>10} | {"emb":>10} '
          f'| {"proprio":>10} | {"kl(真实)":>10}')
    print('-' * 72)

    history = []
    for ep in epochs:
        ckpt = torch.load(ckpt_dir / f'rssm_epoch{ep}.pt', map_location='cpu')
        ca = ckpt['args']
        model = RSSM(latent_dim=ca.get('latent', 32),
                     hidden_dim=ca.get('hidden', 200),
                     emb_dim=ca.get('emb', 256)).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        with torch.no_grad():
            out = model(images, proprio, actions)
            l = rssm_loss(out, images, proprio)          # 不带 free_bits → 真实 KL
        row = {'epoch': ep, 'loss': float(l['loss']), 'recon': float(l['recon']),
               'emb': float(l['emb']), 'proprio': float(l['proprio']), 'kl': float(l['kl'])}
        history.append(row)
        print(f'{ep:>5} | {row["loss"]:>10.6f} | {row["recon"]:>10.6f} | '
              f'{row["emb"]:>10.6f} | {row["proprio"]:>10.6f} | {row["kl"]:>10.6f}')

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(history, indent=1), encoding='utf-8')
        print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
