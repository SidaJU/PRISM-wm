"""笔记 04 补遗：从 46 维输入开始，完整走一遍 GRU 的计算，不跳步。

运行:
    python prism/docs/demo_04e_gru_full.py
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
g = nn.GRUCell(46, 200)

# 第 t 帧的输入：z(32) + action(4) + proprio(10) = 46
x = torch.randn(1, 46)
h = torch.zeros(1, 200)          # 第一帧，记忆还是 0

print('=' * 66)
print('输入')
print(f'  x  {tuple(x.shape)}   前 5 个: {[round(v, 3) for v in x[0, :5].tolist()]}')
print(f'  h  {tuple(h.shape)}  前 5 个: {[round(v, 3) for v in h[0, :5].tolist()]}   (第一帧全 0)')

# 拆权重（三个一组竖着堆）
W_ir, W_iz, W_in = g.weight_ih[0:200], g.weight_ih[200:400], g.weight_ih[400:600]
W_hr, W_hz, W_hn = g.weight_hh[0:200], g.weight_hh[200:400], g.weight_hh[400:600]
b_ir, b_iz, b_in = g.bias_ih[0:200], g.bias_ih[200:400], g.bias_ih[400:600]
b_hr, b_hz, b_hn = g.bias_hh[0:200], g.bias_hh[200:400], g.bias_hh[400:600]

# ---- 步骤 1：投影 ----
print('\n' + '=' * 66)
print('步骤 1  投影：各自乘矩阵，都变成 200 维')
px = W_iz @ x[0] + b_iz
ph = W_hz @ h[0] + b_hz
print(f'  x(46)  × W_iz(200×46)   -> {tuple(px.shape)}   前5: {[round(v, 3) for v in px[:5].tolist()]}')
print(f'  h(200) × W_hz(200×200)  -> {tuple(ph.shape)}   前5: {[round(v, 3) for v in ph[:5].tolist()]}')
print(f'  数值范围: px[{px.min():.3f}, {px.max():.3f}]   ph[{ph.min():.3f}, {ph.max():.3f}]')

# ---- 步骤 2：相加 ----
print('\n' + '=' * 66)
print('步骤 2  相加（维度相同了，可以加）')
raw_z = px + ph
print(f'  raw_z  {tuple(raw_z.shape)}   范围 [{raw_z.min():.3f}, {raw_z.max():.3f}]')
print(f'  前 5 个: {[round(v, 3) for v in raw_z[:5].tolist()]}')
print('  [!] 这是任意实数 —— 可能是负数、也可能大于 1，不能直接当插值系数')

# ---- 步骤 3：sigmoid ----
print('\n' + '=' * 66)
print('步骤 3  用 sigmoid 压到 0~1')
z = torch.sigmoid(raw_z)
print(f'  z  {tuple(z.shape)}   范围 [{z.min():.4f}, {z.max():.4f}]')
print(f'  前 5 个: {[round(v, 4) for v in z[:5].tolist()]}')
print('  [OK] 现在全部落在 0~1，可以直接当插值系数了')

# ---- r 和 n 同理 ----
raw_r = (W_ir @ x[0] + b_ir) + (W_hr @ h[0] + b_hr)
r = torch.sigmoid(raw_r)
raw_n = (W_in @ x[0] + b_in) + r * (W_hn @ h[0] + b_hn)
n = torch.tanh(raw_n)
print(f'\n  同一个套路再算两个：')
print(f'    r = sigmoid(...)  范围 [{r.min():.4f}, {r.max():.4f}]   前5: {[round(v, 4) for v in r[:5].tolist()]}')
print(f'    n = tanh(...)     范围 [{n.min():.4f}, {n.max():.4f}]   前5: {[round(v, 4) for v in n[:5].tolist()]}')

# ---- 步骤 4：混合 ----
print('\n' + '=' * 66)
print('步骤 4  混合出新记忆')
h_new = (1 - z) * n + z * h[0]
print(f'  h_new = (1−z)⊙n + z⊙h')
print(f'        {tuple(h_new.shape)}   前 5 个: {[round(v, 4) for v in h_new[:5].tolist()]}')

out = g(x, h)
print(f'  程序输出          前 5 个: {[round(v, 4) for v in out[0][:5].tolist()]}')
print(f'  一致? {torch.allclose(h_new, out[0], atol=1e-5)}')

# ---- sigmoid / tanh 手算 ----
print('\n' + '=' * 66)
print('附：sigmoid 到底做了什么（1 / (1 + e^-x)）')
for v in (0, 1, -1, 2, -2, 10, -10):
    print(f'  sigmoid({v:>3}) = {torch.sigmoid(torch.tensor(float(v))):.4f}')
print('  -> 无论输入多大/多小，输出永远在 0~1 之间；输入 0 时正好 0.5')

print('\n附：tanh 做了什么')
for v in (0, 1, -1, 2, -2):
    print(f'  tanh({v:>3})    = {torch.tanh(torch.tensor(float(v))):.4f}')
print('  -> 输出永远在 -1~1 之间；输入 0 时正好 0')
print('  -> 用于"内容"（可正可负），而 sigmoid 用于"比例"（必须 0~1）')
