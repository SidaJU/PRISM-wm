"""笔记 03 补遗：nn.Conv2d(3, 32, 4, stride=2, padding=1) 每个参数在干什么。

运行:
    python prism/docs/demo_03b_conv_params.py
"""
import torch
import torch.nn as nn

# ---------- 1. 这一层的权重长什么样 ----------
print('=' * 62)
print('1) nn.Conv2d(3, 32, 4, stride=2, padding=1) 的参数')
c = nn.Conv2d(3, 32, 4, stride=2, padding=1)
print(f'   weight 形状 = {tuple(c.weight.shape)}   <- (输出通道 32, 输入通道 3, 4, 4)')
print(f'   bias   形状 = {tuple(c.bias.shape)}       <- 每个输出通道一个偏置')
print(f'   参数量     = {sum(p.numel() for p in c.parameters()):,}')
print('   解读：32 个 kernel，每个形状 (3,4,4) = 48 个权重；32×48 + 32 = 1568')

# ---------- 2. 输出尺寸 ----------
print('\n' + '=' * 62)
print('2) 输入 (1,3,64,64) 经过这一层')
x = torch.zeros(1, 3, 64, 64)
print(f'   输出形状 = {tuple(c(x).shape)}   <- (batch, 通道 32, 高 32, 宽 32)')

# ---------- 3. stride / padding 各起什么作用 ----------
print('\n' + '=' * 62)
print('3) 单独改 stride / padding，看输出尺寸怎么变')
print(f'   {"stride":>6} {"padding":>7}   输出高/宽   公式 (n - k + 2p)/s + 1')
for s in (1, 2):
    for p in (0, 1):
        cc = nn.Conv2d(3, 32, 4, stride=s, padding=p)
        out = cc(x)
        formula = f'(64 - 4 + 2×{p})/{s} + 1 = {(64 - 4 + 2 * p) // s + 1}'
        print(f'   {s:>6} {p:>7}   {out.shape[2]:>4}        {formula}')
print('   -> stride=2 让画面减半；padding=1 让尺寸正好是整数的一半(32)')

# ---------- 4. 手算：窗口怎么滑、怎么求和 ----------
print('\n' + '=' * 62)
print('4) 手算验证：kernel 权重全设为 1（此时输出 = 窗口内求和）')
torch.manual_seed(0)
c1 = nn.Conv2d(1, 1, 2, stride=2, padding=0)
with torch.no_grad():
    c1.weight[:] = 1.0
    c1.bias[:] = 0.0
xx = torch.arange(16.).reshape(1, 1, 4, 4)
out = c1(xx)
print('   输入 4×4:')
for row in xx.squeeze().tolist():
    print('     ', [int(v) for v in row])
print(f'   输出 2×2: {[[int(v) for v in r] for r in out.squeeze().tolist()]}')
print('   验证：')
print('     左上块 [[0,1],[4,5]]      = 10')
print('     右上块 [[2,3],[6,7]]      = 18   <- 跳了 2 格(stride=2)')
print('     左下块 [[8,9],[12,13]]    = 42')
print('     右下块 [[10,11],[14,15]]  = 50')

# ---------- 5. ReLU ----------
print('\n' + '=' * 62)
print('5) nn.ReLU() 做了什么')
relu = nn.ReLU()
vals = torch.tensor([-2.0, -0.5, 0.0, 0.5, 3.0])
print(f'   输入 {vals.tolist()}')
print(f'   输出 {relu(vals).tolist()}')
print('   -> 负数归零，正数不变。作用是给网络引入"非线性"')
