"""笔记 05 补遗：KL 公式里每个符号是什么，逐项手算。

运行:
    python prism/docs/demo_05b_kl.py
"""
import torch

torch.manual_seed(0)

# ---------- 0. 四个符号 ----------
print('=' * 64)
print('0) 公式里的 4 个符号（各自是 32 维向量，这里用 2 维举例）')
mu_p = torch.tensor([1.0, -0.5])   # prior 的均值
lv_p = torch.tensor([-2.0, 0.0])   # prior 的 logvar
mu_q = torch.tensor([1.2, -0.3])   # posterior 的均值
lv_q = torch.tensor([-1.5, 0.2])   # posterior 的 logvar
print(f'   mu_p  = {mu_p.tolist()}   prior 的 mu      （闭卷，猜的中心）')
print(f'   lv_p  = {lv_p.tolist()}   prior 的 logvar  （闭卷，猜的不确定度）')
print(f'   mu_q  = {mu_q.tolist()}   posterior 的 mu      （开卷，看过图的中心）')
print(f'   lv_q  = {lv_q.tolist()}   posterior 的 logvar  （开卷，看过图的不确定度）')
print('   命名：下标 p = prior（闭卷），q = posterior（开卷）')

var_p, var_q = lv_p.exp(), lv_q.exp()
print(f'\n   var_p = lv_p.exp() = {var_p.tolist()}   （方差 = exp(logvar)）')
print(f'   var_q = lv_q.exp() = {var_q.tolist()}')

# ---------- 1. 解析公式，逐维手算 ----------
print('\n' + '=' * 64)
print('1) 解析公式（代码里的），逐维算')
kl_formula = 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2)
                    / lv_p.exp().clamp(min=1e-4) - 1)
for i in range(2):
    term1 = (lv_p - lv_q)[i].item()                       # log(var_p/var_q)
    term2 = ((var_q + (mu_q - mu_p) ** 2) / var_p)[i].item()
    print(f'\n   维 {i}:')
    print(f'     mu 差     = {mu_q[i].item()} - {mu_p[i].item()} = {(mu_q-mu_p)[i].item():+.2f}')
    print(f'     var_p     = {var_p[i].item():.4f}')
    print(f'     var_q     = {var_q[i].item():.4f}')
    print(f'     项1 lv_p-lv_q                        = {term1:+.4f}   (= log(var_p/var_q)，宽度比)')
    print(f'     项2 (var_q + (mu_q-mu_p)²)/var_p - 1 = {term2 - 1:+.4f}   (位置差+宽度差的组合)')
    print(f'     该维 KL = 0.5 * ({term1:+.4f} + {term2:.4f} - 1) = {kl_formula[i].item():.4f}')
print(f'\n   两维相加 = {kl_formula.sum().item():.4f}   （真实代码对 32 维求和）')

# ---------- 2. 用采样近似验证公式没写错 ----------
print('\n' + '=' * 64)
print('2) 验证：从 q 采 20 万个点，用 log q - log p 的平均近似 KL')
N = 200_000
std_q = (0.5 * lv_q).exp()
x = mu_q + std_q * torch.randn(N, 2)              # 从 posterior 采样

def log_prob(x, mu, lv):
    std = (0.5 * lv).exp()
    return -0.5 * ((x - mu) / std) ** 2 - torch.log(std) - 0.5 * torch.log(
        torch.tensor(2 * torch.pi))

kl_mc = (log_prob(x, mu_q, lv_q) - log_prob(x, mu_p, lv_p)).sum(-1).mean()
print(f'   采样近似 KL  = {kl_mc.item():.4f}   （先对 2 维求和，再对样本平均）')
print(f'   解析公式 KL  = {kl_formula.sum().item():.4f}')
print(f'   一致?        = {abs(kl_mc.item() - kl_formula.sum().item()) < 0.01}')

# ---------- 3. 直觉 ----------
print('\n' + '=' * 64)
print('3) 公式的直觉：KL 测的是「位置差 + 宽度差」')
print('   (mu_q - mu_p)²        -> 两个分布的中心差多远（位置差）')
print('   lv_p - lv_q (log比)   -> 两个分布的宽度差多少（形状差）')
print('   两者相同时 KL = 0     -> 这就是训练目标：prior 猜得和 posterior 一样')
