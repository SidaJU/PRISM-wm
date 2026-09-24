"""笔记 04 的可运行示例：RSSM 的两个状态 h / z，以及"想象"为什么可行。

运行:
    python prism/docs/demo_04_rssm.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prism.model.dataset import L0Dataset   # noqa: E402
from prism.model.rssm import RSSM           # noqa: E402

torch.manual_seed(0)

# ---------- 1. 取一段真实序列 ----------
ds = L0Dataset(str(ROOT / 'prism/data/L0'), seq_len=32, max_episodes=5)
s = ds[0]
images = s['images'].unsqueeze(1)      # (T,1,3,64,64)
proprio = s['proprio'].unsqueeze(1)    # (T,1,10)
actions = s['actions'].unsqueeze(1)    # (T,1,4)
T = images.shape[0]
print(f'序列长度 T = {T}')

# ---------- 2. 维度配置 ----------
model = RSSM()      # latent_dim=32, hidden_dim=200, emb_dim=256
EMB = model.encoder.fc.out_features      # RSSM 本身没存 emb_dim，从 encoder 推断
print(f'\n维度配置')
print(f'  emb 观测嵌入 = {EMB}')
print(f'  h   记忆     = {model.hidden_dim}')
print(f'  z   潜变量   = {model.latent_dim}')
print(f'  GRU 输入     = z({model.latent_dim}) + action(4) + proprio(10) = {model.latent_dim + 4 + 10}')
print(f'  （注意：GRU 的输入里【没有图像】）')

# ---------- 3. 前向 ----------
out = model(images, proprio, actions)
print('\nforward 输出')
for k in ('emb', 'emb_pred', 'proprio_pred', 'recon', 'kl'):
    print(f'  {k:14s} {tuple(out[k].shape)}')

# ---------- 4. 遮挡：图像被遮，观测嵌入被污染 ----------
print('\n' + '=' * 56)
print('遮挡演示：把第 10~19 帧图像涂黑')
occ = images.clone()
occ[10:20] = 0.0
out_occ = model(occ, proprio, actions)
diff = (out['emb'] - out_occ['emb']).abs().mean(-1).squeeze(1)     # (T,)
clean = torch.cat([diff[:10], diff[20:]]).mean().item()
blocked = diff[10:20].mean().item()
print(f'  未遮挡帧的 emb 差异 : {clean:.4f}')
print(f'  遮挡帧的   emb 差异 : {blocked:.4f}')
print(f'  -> 图像一被遮，观测嵌入立刻被污染（差 {blocked / max(clean, 1e-6):.1f} 倍）')
print('  -> 这就是为什么需要"记忆 h"：光看当前帧是不够的')

# ---------- 5. 想象：完全不看图像 ----------
print('\n' + '=' * 56)
print('想象演示：不看任何图像，只用 prior 采样 z 往前推')
h = torch.zeros(1, model.hidden_dim)
z = torch.zeros(1, model.latent_dim)
for t in range(T):
    gru_in = torch.cat([z, actions[t], proprio[t]], dim=-1)
    h = model.gru(gru_in, h)
    mu_p, _ = model.prior_net(h).chunk(2, dim=-1)
    z = mu_p                                   # 用均值，不采样
emb_hat = model.obs_pred(torch.cat([h, z], dim=-1))
recon = model.decoder(emb_hat)
print(f'  推了 {T} 步，全程没看过图像')
print(f'  猜出的 emb  : {tuple(emb_hat.shape)}')
print(f'  解码出的图像: {tuple(recon.shape)}')
print('\n  结构上完全跑得通 —— 只差一个 imagine() 方法')
print('  （现在数值无意义，因为模型还没训练）')
