"""笔记 04 补遗：prior 和 posterior 的区别、为什么要分开、怎么协作。

运行:
    python prism/docs/demo_04f_prior_posterior.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prism.model.dataset import L0Dataset   # noqa: E402
from prism.model.rssm import RSSM           # noqa: E402

torch.manual_seed(0)
model = RSSM()
EMB = model.encoder.fc.out_features

# ---------- 1. 结构对比 ----------
print('=' * 66)
print('1) 两个网络的结构对比')
print('prior_net:'); print(model.prior_net)
print('post_net:');  print(model.post_net)
print(f'\n  prior 输入 = h           = {model.hidden_dim}')
print(f'  post  输入 = cat(h, emb) = {model.hidden_dim} + {EMB} = {model.hidden_dim + EMB}')
print('  -> 唯一区别：posterior 多看了当前帧图像 emb')

# ---------- 2. 真实数据：两条都算，只用 posterior 采样 ----------
ds = L0Dataset(str(ROOT / 'prism/data/L0'), seq_len=32, max_episodes=3)
s = ds[0]
images = s['images'].unsqueeze(1)
proprio = s['proprio'].unsqueeze(1)
actions = s['actions'].unsqueeze(1)

out = model(images, proprio, actions)
print('\n' + '=' * 66)
print('2) forward 跑完后的输出')
print(f"  kl         {tuple(out['kl'].shape)}   范围 [{out['kl'].min():.4f}, {out['kl'].max():.4f}]")
print(f"  emb_pred   {tuple(out['emb_pred'].shape)}   （z 走的是 posterior）")
print(f"  recon      {tuple(out['recon'].shape)}")

# ---------- 3. 手动对比两个分布 ----------
print('\n' + '=' * 66)
print('3) 同一时刻，两个分布各自给出什么')
h = torch.zeros(1, model.hidden_dim)
emb = model.encoder(images[0])                     # 第 0 帧的真实摘要

mu_p, _ = model.prior_net(h).chunk(2, dim=-1)              # 只凭记忆
mu_q, _ = model.post_net(torch.cat([h, emb], -1)).chunk(2, dim=-1)   # 看了图像

print(f'  mu_p（prior，只看记忆）  范围 [{mu_p.min():.4f}, {mu_p.max():.4f}]')
print(f'  mu_q（posterior，看了图）范围 [{mu_q.min():.4f}, {mu_q.max():.4f}]')
print(f'  两者平均差距: {(mu_p - mu_q).abs().mean():.4f}')
print('  -> 这个差距就是 KL 要压小的东西')

# ---------- 4. 遮挡：posterior 被污染，prior 纹丝不动 ----------
print('\n' + '=' * 66)
print('4) 关键演示：图像被遮住时会发生什么')
emb_black = model.encoder(torch.zeros_like(images[0]))     # 全黑帧
mu_qb, _ = model.post_net(torch.cat([h, emb_black], -1)).chunk(2, dim=-1)

print(f'  posterior 的变化: {(mu_qb - mu_q).abs().mean():.4f}   <- 被污染了')
print(f'  prior 的变化:     0.0000（结构决定：prior 根本不看图像）')
print('\n  -> 遮住画面时，posterior 靠不住了，只有 prior 还稳')
print('  -> 这就是为什么必须训练 prior：它是"没有图像时唯一能用的那条路"')

# ---------- 5. 想象：只用 prior 也能推 ----------
print('\n' + '=' * 66)
print('5) 想象模式：全程只用 prior')
h = torch.zeros(1, model.hidden_dim)
z = torch.zeros(1, model.latent_dim)
T = images.shape[0]
for t in range(T):
    h = model.gru(torch.cat([z, actions[t], proprio[t]], dim=-1), h)
    mu_p, _ = model.prior_net(h).chunk(2, dim=-1)
    z = mu_p                                   # 只用 prior，不看图像
emb_hat = model.obs_pred(torch.cat([h, z], dim=-1))
print(f'  推了 {T} 步，全程没看图')
print(f'  最后猜出的 emb: {tuple(emb_hat.shape)}')
print(f'  解码出的图像:   {tuple(model.decoder(emb_hat).shape)}')
print('\n  -> 结构上完全可行；要让猜得准，就得靠 KL 把 prior 训好')
print('  （当前未训练，数值无意义）')
