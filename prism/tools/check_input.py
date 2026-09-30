"""检查网络真正的输入：Dataset 吐出来的那一批数据。

用法:
    python prism/tools/check_input.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import imageio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prism.model.dataset import L0Dataset          # noqa: E402
from prism.model.rssm import RSSM                  # noqa: E402


def main():
    ds = L0Dataset(str(ROOT / 'prism/data/L0'), seq_len=32, train=True)
    print(f'train samples = {len(ds)}')

    s = ds[0]
    x = s['images']          # (32, 3, 64, 64)
    p = s['proprio']         # (32, 10)
    a = s['actions']         # (32, 4)

    print('\n--- images（网络输入）---')
    print(f'   形状      : {tuple(x.shape)}')
    print(f'   类型      : {x.dtype}')
    print(f'   值范围    : [{x.min():.4f}, {x.max():.4f}]   （期望 0~1，因为除以了 255）')
    print(f'   均值/标准差: {x.mean():.4f} / {x.std():.4f}')
    print(f'   各通道均值 : R={x[:,0].mean():.4f}  G={x[:,1].mean():.4f}  B={x[:,2].mean():.4f}')
    print(f'   有 NaN?   : {bool(torch.isnan(x).any())}')

    diff = (x[1:] - x[:-1]).abs().mean().item()
    print(f'   相邻帧差异 : {diff:.5f}   （太小说明画面几乎不动）')

    print('\n--- proprio（本体）---')
    print(f'   形状 {tuple(p.shape)}   范围 [{p.min():.3f}, {p.max():.3f}]')
    print('\n--- actions（动作）---')
    print(f'   形状 {tuple(a.shape)}   范围 [{a.min():.3f}, {a.max():.3f}]')

    # 转置成 RSSM 要的格式，确认形状
    print('\n--- 打包后（模拟 DataLoader batch=4 再转置）---')
    batch = torch.stack([ds[i]['images'] for i in range(4)])       # (4,32,3,64,64)
    to_rssm = batch.permute(1, 0, 2, 3, 4).contiguous()            # (32,4,3,64,64)
    print(f'   DataLoader 形状 (B,T,C,H,W): {tuple(batch.shape)}')
    print(f'   转置后      (T,B,C,H,W): {tuple(to_rssm.shape)}')

    # 确认 RSSM 能吃进去
    model = RSSM()
    with torch.no_grad():
        pr = batch[:1].permute(1, 0, 2, 3, 4).contiguous()          # (32,1,...)
        out = model(pr,
                    torch.stack([ds[0]['proprio']]).permute(1, 0, 2).contiguous(),
                    torch.stack([ds[0]['actions']]).permute(1, 0, 2).contiguous())
    print(f'   RSSM 输出的 recon 形状: {tuple(out["recon"].shape)}  （应等于输入形状）')

    # 导出真实输入帧
    idx = [0, 5, 10, 15, 20, 25]
    frames = [x[i] for i in idx]
    grid = np.concatenate([np.transpose(f.numpy(), (1, 2, 0)) for f in frames], axis=1)
    grid = np.repeat(np.repeat(grid, 3, axis=0), 3, axis=1)
    grid = (grid * 255).clip(0, 255).astype(np.uint8)
    out_png = ROOT / 'prism/ckpt/_input_check.png'
    imageio.imwrite(out_png, grid)
    print(f'\n真实输入图已导出: {out_png}')
    print(f'   帧 {idx} 横排，放大 3 倍')


if __name__ == '__main__':
    main()
