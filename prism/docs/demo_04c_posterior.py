"""笔记 04 补遗：posterior 到底是什么。

核心：它输出的不是"一个答案"，而是"一个分布"（均值 + 不确定度）。

运行:
    python prism/docs/demo_04c_posterior.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prism.model.rssm import RSSM   # noqa: E402

torch.manual_seed(0)

# ---------- 1. 采样公式手算 ----------
print('=' * 60)
print('1) posterior 输出的是分布参数，采样公式：')
print('     std = exp(0.5 * logvar)')
print('     z   = mu + std * 随机噪声\n')

mu = torch.tensor([[1.0, -0.5]])
logvar = torch.tensor([[-2.0, 0.0]])
noise = torch.tensor([[0.5, -1.2]])

std = (0.5 * logvar).exp()
z = mu + std * noise

print(f'   mu     = {mu.squeeze().tolist()}')
print(f'   logvar = {logvar.squeeze().tolist()}')
print(f'   std    = {[round(v, 4) for v in std.squeeze().tolist()]}   (exp(0.5*logvar))')
print(f'   噪声   = {noise.squeeze().tolist()}')
print(f'   z      = {[round(v, 4) for v in z.squeeze().tolist()]}')
print(f'   手算核对: 1.0 + {std[0,0]:.4f}*0.5 = {1.0 + std[0,0].item()*0.5:.4f}')
print(f'             -0.5 + {std[0,1]:.4f}*(-1.2) = {-0.5 + std[0,1].item()*(-1.2):.4f}')

# ---------- 2. 真实模型里的 posterior ----------
print('\n' + '=' * 60)
print('2) 真实模型里的 post_net 结构')
model = RSSM()
print(model.post_net)
print(f'\n   输入 = cat(h, emb) = 200 + 256 = 456')
print(f'   输出 = 64 = mu(32) + logvar(32)')

h = torch.zeros(1, 200)
emb = torch.randn(1, 256)
post = model.post_net(torch.cat([h, emb], dim=-1))
mu_q, lv_q = post.chunk(2, dim=-1)
std_q = (0.5 * lv_q).exp()

print(f'\n   mu  形状={tuple(mu_q.shape)}  值范围=[{mu_q.min():.4f}, {mu_q.max():.4f}]')
print(f'   std 形状={tuple(std_q.shape)}  值范围=[{std_q.min():.4f}, {std_q.max():.4f}]')
print('   -> 32 维里每一维都有【自己独立的】mu 和 std')

# ---------- 3. 同一个 mu，不同 std 采样出来的差别 ----------
print('\n' + '=' * 60)
print('3) 直观感受 std 的作用：mu 相同，std 不同')
torch.manual_seed(42)
mu0 = torch.zeros(1, 1)
for lv in (-4.0, 0.0, 2.0):
    sd = (0.5 * torch.tensor([[lv]])).exp()
    samples = mu0 + sd * torch.randn(1, 6)
    tag = {-4.0: 'std 很小 → 几乎确定', 0.0: 'std = 1 → 中等', 2.0: 'std 很大 → 很不确定'}[lv]
    print(f'   logvar={lv:+.1f}  std={sd.item():.3f}  6 次采样: '
          f'{[round(v, 2) for v in samples.squeeze().tolist()]}   ({tag})')

print('\n   解读：std 越大，同一次判断采样出来的结果越分散')
print('   -> 这就是"不确定度"：画面清楚时 std 应该小，被遮挡时应该大')
print('   （当前是未训练模型，std 还是随机的；训练后它才会真正学会表达不确定性）')
