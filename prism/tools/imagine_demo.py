"""imagine() 演示：给前几帧"看清现在"，然后不看图往前推，画出想象画面。

同时给出 rollout 误差曲线（想象帧 vs 真实帧，随步数的变化）—— 长程稳定性评测。

用法:
    python prism/tools/imagine_demo.py
    python prism/tools/imagine_demo.py --ckpt-dir prism/ckpt_fb01 --ckpt 20 --context 8 --horizon 20
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import imageio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prism.model.dataset import L0Dataset        # noqa: E402
from prism.model.rssm import RSSM                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='prism/ckpt_fb01')
    ap.add_argument('--ckpt', default='20')
    ap.add_argument('--context', type=int, default=8, help='前几帧用于建立记忆')
    ap.add_argument('--horizon', type=int, default=20, help='往前想象几步')
    ap.add_argument('--seq', type=int, default=32)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = ROOT / args.ckpt_dir / f'rssm_epoch{args.ckpt}.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ca = ckpt['args']
    model = RSSM(latent_dim=ca.get('latent', 32), hidden_dim=ca.get('hidden', 200),
                 emb_dim=ca.get('emb', 256)).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'已加载 {ckpt_path.name}  device={device}')

    ds = L0Dataset(str(ROOT / 'prism/data/L0'), seq_len=args.seq, train=False)
    s = ds[0]
    images = s['images'].unsqueeze(1).to(device)          # (T,1,3,64,64)
    proprio = s['proprio'].unsqueeze(1).to(device)        # (T,1,10)
    actions = s['actions'].unsqueeze(1).to(device)        # (T,1,4)
    T = images.shape[0]
    C, H = args.context, args.horizon
    if C + H > T:
        H = T - C
        print(f'  (裁剪 horizon 到 {H})')

    # 1) 用前 C 帧观测建立记忆
    with torch.no_grad():
        h, z = model.observe(images[:C], proprio[:C], actions[:C])
    print(f'已用前 {C} 帧建立记忆: h={tuple(h.shape)} z={tuple(z.shape)}')

    # 2) 不看图，两种想象模式对比：
    #    sample : z 从 prior 按分布采样（有随机噪声 → 画面可能出现噪点）
    #    mu     : z 直接用 prior 均值（确定性推演 → 若稳定则证明问题出在采样）
    real = images[C:C + H, 0].cpu().numpy()          # (H,3,64,64)
    pred = {}
    with torch.no_grad():
        for mode, sample in (('sample', True), ('mu', False)):
            emb_hat, prop_hat = model.imagine(h, z, actions[C:C + H], proprio[C - 1], H, sample=sample)
            img_hat = model.decoder(emb_hat.reshape(H, -1)).reshape(H, 1, 3, 64, 64)
            pred[mode] = img_hat[:, 0].cpu().numpy()      # (H,3,64,64)
    print(f'想象了 {H} 步 × 2 种模式（全程没看图像）')

    # 3) 与真实未来帧比
    mse = {m: [float(((pred[m][t] - real[t]) ** 2).mean()) for t in range(H)]
           for m in ('sample', 'mu')}

    print('\nrollout 误差（想象帧 vs 真实帧，像素 MSE）')
    print(f'  {"step":>5} {"sample(采样)":>14} {"mu(均值)":>12}')
    for t in range(0, H, max(1, H // 10)):
        bar_s = '#' * int(mse['sample'][t] * 300)
        bar_m = '#' * int(mse['mu'][t] * 300)
        print(f'  {t:>5} {mse["sample"][t]:>9.5f} {bar_s}')
        print(f'  {"":>5} {mse["mu"][t]:>9.5f} {bar_m}')
    print(f'\n  sample 第 1 步 {mse["sample"][0]:.5f} -> 第 {H} 步 {mse["sample"][-1]:.5f}')
    print(f'  mu     第 1 步 {mse["mu"][0]:.5f} -> 第 {H} 步 {mse["mu"][-1]:.5f}')

    # 参考：什么算差（预测成均值）
    baseline = float(((real.mean(axis=(0, 1, 2)) - real) ** 2).mean())
    print(f'  参考"全预测成均值"的 MSE = {baseline:.5f}')
    for m in ('sample', 'mu'):
        verdict = 'OK 好于基线' if mse[m][-1] < baseline else '!! 比基线还差'
        print(f'  判定[{m}]: {verdict}')

    # 4) 导出对比图：三列（真实 | 采样想象 | 均值想象）
    idx = list(range(0, H, max(1, H // 6)))
    rows = []
    for t in idx:
        r = real[t].transpose(1, 2, 0)                          # HWC
        ps = pred['sample'][t].transpose(1, 2, 0)
        pm = pred['mu'][t].transpose(1, 2, 0)
        trip = np.concatenate([r, ps, pm], axis=1)
        trip = np.repeat(np.repeat(trip, 3, axis=0), 3, axis=1)
        rows.append(trip)
    grid = np.concatenate(rows, axis=0) * 255.0        # 数据是 0~1 浮点，必须 ×255！
    grid = grid.clip(0, 255).astype(np.uint8)
    out_png = ROOT / 'prism/preview/imagine_real_vs_pred.png'
    imageio.imwrite(out_png, grid)
    print(f'\n对比图已导出: {out_png}   （每行: 左=真实  中=采样想象  右=均值想象，step {idx}）')

    # 5) 存误差曲线数据
    curve = {'mse_sample': mse['sample'], 'mse_mu': mse['mu'], 'baseline': baseline,
             'context': C, 'horizon': H}
    (ROOT / 'prism/preview/imagine_curve.json').write_text(json.dumps(curve, indent=1), encoding='utf-8')
    print(f'误差曲线数据: prism/preview/imagine_curve.json')


if __name__ == '__main__':
    main()
