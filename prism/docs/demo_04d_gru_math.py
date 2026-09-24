"""笔记 04 补遗：GRU 里 46 维输入和 200 维记忆到底怎么运算。

结论：不是直接相加，而是各自乘一个矩阵、投影到同一维度(200)后再相加。

运行:
    python prism/docs/demo_04d_gru_math.py
"""
import torch
import torch.nn as nn

# ---------- 1. 参数形状：看清楚"投影矩阵" ----------
print('=' * 64)
print('1) nn.GRUCell(46, 200) 的参数')
g = nn.GRUCell(46, 200)
print(f'   weight_ih  {tuple(g.weight_ih.shape)}   <- (3*200, 46)：把 46 维输入投影到 200')
print(f'   weight_hh  {tuple(g.weight_hh.shape)}   <- (3*200, 200)：把 200 维记忆投影到 200')
print(f'   bias_ih    {tuple(g.bias_ih.shape)}')
print(f'   bias_hh    {tuple(g.bias_hh.shape)}')
print(f'   参数量      {sum(p.numel() for p in g.parameters()):,}')
print('   （3 个门 r / z / n 各占 200 行，所以是 3*200）')

# ---------- 2. 手算一个小 GRU，验证公式 ----------
print('\n' + '=' * 64)
print('2) 用一个极小的 GRU 手算验证：input=2, hidden=3')
torch.manual_seed(0)
gs = nn.GRUCell(2, 3)
with torch.no_grad():
    gs.bias_ih.zero_()
    gs.bias_hh.zero_()

x = torch.tensor([[1.0, 2.0]])          # 输入 (1,2)
h = torch.tensor([[0.5, -0.5, 0.0]])    # 旧记忆 (1,3)
out = gs(x, h)

# 拆出三个门的权重
W_ir, W_iz, W_in = gs.weight_ih[0:3], gs.weight_ih[3:6], gs.weight_ih[6:9]
W_hr, W_hz, W_hn = gs.weight_hh[0:3], gs.weight_hh[3:6], gs.weight_hh[6:9]

# 手写公式
r = torch.sigmoid(W_ir @ x[0] + W_hr @ h[0])          # 重置门
z = torch.sigmoid(W_iz @ x[0] + W_hz @ h[0])          # 更新门
n = torch.tanh(W_in @ x[0] + r * (W_hn @ h[0]))       # 候选新记忆
h_new = (1 - z) * n + z * h[0]                        # 混合

print(f'   x(2)      = {x.squeeze().tolist()}')
print(f'   h(3)      = {h.squeeze().tolist()}')
print(f'\n   重置门 r  = sigmoid(W_ir·x + W_hr·h) = {[round(v,4) for v in r.tolist()]}')
print(f'   更新门 z  = sigmoid(W_iz·x + W_hz·h) = {[round(v,4) for v in z.tolist()]}')
print(f'   候选   n  = tanh(W_in·x + r⊙(W_hn·h)) = {[round(v,4) for v in n.tolist()]}')
print(f'\n   手算 h_new = (1-z)⊙n + z⊙h')
print(f'             = {[round(v,4) for v in h_new.tolist()]}')
print(f'   程序输出   = {[round(v,4) for v in out[0].tolist()]}')
print(f'   一致?      = {torch.allclose(h_new, out[0], atol=1e-5)}')

# ---------- 3. 维度变化的关键一步 ----------
print('\n' + '=' * 64)
print('3) 维度是怎么对齐的（关键点）')
xv, hv = x[0], h[0]
print(f'   x:  {tuple(xv.shape)}  --W_iz{tuple(W_iz.shape)}-->  {tuple((W_iz @ xv).shape)}')
print(f'   h:  {tuple(hv.shape)}  --W_hz{tuple(W_hz.shape)}-->  {tuple((W_hz @ hv).shape)}')
print(f'   两者维度相同，才能相加 -> {tuple((W_iz @ xv + W_hz @ hv).shape)}')
print('\n   所以"46 维和 200 维怎么运算"的答案是：')
print('   各自乘一个矩阵，都变成 200 维，然后相加。')

# ---------- 4. 6 个权重矩阵藏在哪 ----------
print('\n' + '=' * 64)
print('4) 6 个权重矩阵在 PyTorch 里怎么存（竖着堆起来）')
print('   weight_ih  (600, 46)：')
print('     [0:200]   = W_ir   输入 → 重置门')
print('     [200:400] = W_iz   输入 → 更新门')
print('     [400:600] = W_in   输入 → 候选')
print('   weight_hh  (600, 200)：')
print('     [0:200]   = W_hr   记忆 → 重置门')
print('     [200:400] = W_hz   记忆 → 更新门')
print('     [400:600] = W_hn   记忆 → 候选')
print('\n   命名规则：W_i? 都吃输入 x（列数 = 46）；W_h? 都吃记忆 h（列数 = 200）')
print('   上面第 2 节的手算就是按这个切片取权重算的，结果与程序一致 -> 规则已验证')

# ---------- 5. 参数是学出来的 ----------
print('\n' + '=' * 64)
print('5) 这些矩阵是学出来的吗')
print(f'   requires_grad = {g.weight_ih.requires_grad}')
print(f'   参数量         = {sum(p.numel() for p in g.parameters()):,}')
print('   -> 全部参与梯度下降，跟着 loss 一起被更新')
print('   -> 它们是"全连接层"(y = W·x + b)，但只有一层，不是完整 MLP')
