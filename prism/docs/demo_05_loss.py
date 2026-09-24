"""笔记 05：训练时到底在优化什么 —— loss 的四项。

运行:
    python prism/docs/demo_05_loss.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prism.model.dataset import L0Dataset        # noqa: E402
from prism.model.rssm import RSSM, rssm_loss     # noqa: E402

torch.manual_seed(0)

ds = L0Dataset(str(ROOT / 'prism/data/L0'), seq_len=32, max_episodes=3)
s = ds[0]
images = s['images'].unsqueeze(1)        # (T,1,3,64,64)
proprio = s['proprio'].unsqueeze(1)      # (T,1,10)
actions = s['actions'].unsqueeze(1)      # (T,1,4)

model = RSSM()
out = model(images, proprio, actions)
losses = rssm_loss(out, images, proprio)

# ---------- 1. 四项 loss ----------
print('=' * 62)
print('1) 四项 loss（未训练模型，数值只说明量级）')
for k in ('recon', 'emb', 'proprio', 'kl'):
    print(f'   {k:9s} = {float(losses[k]):.6f}')
print(f'   {"loss(合计)":9s} = {float(losses["loss"]):.6f}')

# ---------- 2. 各自在比什么 ----------
print('\n' + '=' * 62)
print('2) 每一项拿什么和什么比')
print(f"   重建   : recon {tuple(out['recon'].shape)}  vs  images {tuple(images.shape)}")
print(f"   emb    : emb_pred {tuple(out['emb_pred'].shape)}  vs  emb {tuple(out['emb'].shape)}")
print(f"   本体   : proprio_pred {tuple(out['proprio_pred'].shape)}  vs  proprio {tuple(proprio.shape)}")
print(f"   KL     : kl {tuple(out['kl'].shape)}   （prior 与 posterior 两个分布的差距）")

# ---------- 3. detach 的作用 ----------
print('\n' + '=' * 62)
print('3) emb loss 里为什么写 .detach()')
emb, emb_pred = out['emb'], out['emb_pred']
mse_detach = ((emb_pred - emb.detach()) ** 2).mean()
mse_full = ((emb_pred - emb) ** 2).mean()
print(f'   用 detach 的 MSE  : {mse_detach.item():.6f}')
print(f'   不用 detach 的 MSE: {mse_full.item():.6f}')
print('   数值完全相同 —— 区别在梯度：')
print('   detach 后，梯度不会流回 Encoder（Encoder 只由重建 loss 训练）')

# ---------- 4. 遮挡后的变化 ----------
print('\n' + '=' * 62)
print('4) 把第 10~19 帧涂黑后，各项 loss 怎么变')
occ = images.clone()
occ[10:20] = 0.0
out_occ = model(occ, proprio, actions)
losses_occ = rssm_loss(out_occ, occ, proprio)
for k in ('recon', 'emb', 'proprio', 'kl'):
    print(f'   {k:9s} : {float(losses[k]):.6f}  ->  {float(losses_occ[k]):.6f}')
print('\n   重建 loss 变化最大 —— 因为它直接比像素')

# ---------- 5. free_bits ----------
print('\n' + '=' * 62)
print('5) free_bits 的作用（防止 KL 塌成 0）')
kl_raw = out['kl']
print(f'   kl 原始: min={kl_raw.min():.4f}  mean={kl_raw.mean():.4f}  max={kl_raw.max():.4f}')
for fb in (0.0, 0.1, 0.5):
    kl_fb = torch.clamp(kl_raw, min=fb)
    print(f'   free_bits={fb:.1f} -> clamp 后 mean={kl_fb.mean():.4f}')
print('   free_bits 给 KL 一个下限，逼模型真的去用 z')
