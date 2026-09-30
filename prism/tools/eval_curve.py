"""不重新训练，直接加载每个 epoch 的权重，画出 loss 随 epoch 的变化。

用来诊断训练过程（尤其是 KL 有没有塌成 0）。

用法:
    python prism/tools/eval_curve.py
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prism.model.dataset import L0Dataset        # noqa: E402
from prism.model.rssm import RSSM, rssm_loss     # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ds = L0Dataset(str(ROOT / 'prism/data/L0'), seq_len=32, train=False)
dl = DataLoader(ds, batch_size=8, shuffle=False, drop_last=True)
batch = next(iter(dl))
images = batch['images'].to(device).permute(1, 0, 2, 3, 4).contiguous()
proprio = batch['proprio'].to(device).permute(1, 0, 2).contiguous()
actions = batch['actions'].to(device).permute(1, 0, 2).contiguous()

ckpt_dir = ROOT / 'prism/ckpt'
epochs = sorted(
    int(p.stem.replace('rssm_epoch', ''))
    for p in ckpt_dir.glob('rssm_epoch*.pt')
)

print(f'{"epoch":>5} | {"loss":>10} | {"recon":>10} | {"emb":>10} | {"proprio":>10} | {"kl":>10}')
print('-' * 68)
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
        l = rssm_loss(out, images, proprio)
    print(f'{ep:>5} | {float(l["loss"]):>10.6f} | {float(l["recon"]):>10.6f} | '
          f'{float(l["emb"]):>10.6f} | {float(l["proprio"]):>10.6f} | {float(l["kl"]):>10.6f}')
