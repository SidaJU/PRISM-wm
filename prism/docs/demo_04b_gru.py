"""笔记 04 补遗：GRU 到底是什么。

核心比喻：GRU 就是一个 Lerp 节点，但它自己决定 alpha。

运行:
    python prism/docs/demo_04b_gru.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prism.model.dataset import L0Dataset   # noqa: E402

torch.manual_seed(0)

# ---------- 1. Lerp 视角：GRU 的更新公式 ----------
print('=' * 62)
print('1) GRU 的更新公式，本质就是 Lerp')
print('   h_new = (1 - alpha) * h_old  +  alpha * 候选新记忆')
print('   alpha 由"更新门"根据当前输入算出来（不是常量）\n')

h_old = torch.tensor([[1.0, 0.0, -1.0, 0.5]])
n_new = torch.tensor([[0.0, 2.0, 0.5, -0.5]])
print('   旧记忆 h_old =', h_old.squeeze().tolist())
print('   新候选 n_new =', n_new.squeeze().tolist())
for a in (0.0, 0.2, 0.5, 1.0):
    h_t = (1 - a) * h_old + a * n_new
    tag = {0.0: '完全不更新', 0.2: '小幅修正', 0.5: '一半一半', 1.0: '彻底重写'}[a]
    print(f'   alpha={a:.1f}  ->  {[round(v, 3) for v in h_t.squeeze().tolist()]}   ({tag})')

# ---------- 2. 真实数据上跑一遍 GRU ----------
print('\n' + '=' * 62)
print('2) 用真实数据跑 GRU：nn.GRUCell(46, 200)')
print('   输入 = z(32) + action(4) + proprio(10) = 46\n')

ds = L0Dataset(str(ROOT / 'prism/data/L0'), seq_len=32, max_episodes=3)
s = ds[0]
proprio = s['proprio']      # (T,10)
actions = s['actions']      # (T,4)
T = proprio.shape[0]

gru = nn.GRUCell(46, 200)
h = torch.zeros(1, 200)
z = torch.zeros(1, 32)
print(f'   序列 {T} 帧，记忆 h 维度 200')
print(f'   {"帧":>4}  {"记忆 h 的模长":>14}  {"与上一帧的余弦相似度":>20}')
for t in range(T):
    x = torch.cat([z, actions[t:t + 1], proprio[t:t + 1]], dim=-1)   # (1,46)
    h_new = gru(x, h)
    if t < 6 or t == T - 1:
        sim = torch.nn.functional.cosine_similarity(h_new, h).item()
        print(f'   {t:>4}  {h_new.norm().item():>14.4f}  {sim:>20.4f}')
    h = h_new

print('\n   解读：')
print('   - 模长 = 这条记忆"有多少内容"')
print('   - 余弦相似度 = 这一帧改写了多少；越接近 1 表示记忆越稳定')

# ---------- 3. 为什么不能直接 h = h + x ----------
print('\n' + '=' * 62)
print('3) 为什么需要"门"，不能直接累加？')
h = torch.zeros(1, 4)
for t in range(6):
    h = h + torch.ones(1, 4) * 0.5          # 朴素累加
print(f'   朴素累加 6 步后 h = {[round(v, 2) for v in h.squeeze().tolist()]}')
print('   -> 只会越滚越大（梯度爆炸 / 该忘的忘不掉）')
print('   -> 门的作用：让模型【学会什么时候该忘、什么时候该记】')
