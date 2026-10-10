"""imagine() 推理：读 config -> 建立记忆 -> 闭眼推演 -> 结果全部收进"本次运行目录"。

每次运行都新建一个带时间戳的目录:
    prism/runs/imagine_<时间戳>/
        curve.png / compare.png / steps.csv / curve.json / observe/frame_XX.png
并写 prism/runs/latest_imagine.txt 记录本次目录（bat 读它来打开结果）。

权重来源: config 的 use_latest=true 时，自动读 prism/runs/latest_train.txt 指向的最新训练目录。

用法:
    python prism/tools/imagine_demo.py                # 全部用 config
    python prism/tools/imagine_demo.py --sample-index 0 --frame-stride 1
"""
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import imageio
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prism.model.dataset import L0Dataset        # noqa: E402
from prism.model.rssm import RSSM                # noqa: E402

DEFAULTS = {
    'ckpt_dir': 'prism/ckpt_fb01', 'ckpt': '20', 'context': 8, 'horizon': 20, 'seq': 48,
    'sample_index': 57, 'frame_stride': 3,
    'use_latest': True,
    'out_root': 'prism/runs',
    'modes': ['sample', 'mu'],
}


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'prism/config.yaml'))
    ap.add_argument('--ckpt-dir', default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--context', type=int, default=None)
    ap.add_argument('--horizon', type=int, default=None)
    ap.add_argument('--seq', type=int, default=None)
    ap.add_argument('--sample-index', type=int, default=None)
    ap.add_argument('--frame-stride', type=int, default=None)
    ap.add_argument('--use-latest', default=None)
    ap.add_argument('--out-root', default=None)
    ap.add_argument('--modes', nargs='+', default=None)
    args = ap.parse_args()

    cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        try:
            cfg = (yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}).get('imagine', {}) or {}
            print(f'[config] {cfg_path}')
        except Exception as e:
            print(f'[warn] config 读取失败({e})')
    for k, v in DEFAULTS.items():
        if getattr(args, k, None) is None:
            setattr(args, k, cfg.get(k, v))

    # use_latest: 读 latest_train.txt 覆盖 ckpt_dir
    if str(args.use_latest).lower() in ('true', '1', 'yes'):
        latest = ROOT / 'prism/runs' / 'latest_train.txt'
        if latest.exists():
            p = (latest.read_text(encoding='utf-8').strip())
            cand = ROOT / p if not Path(p).is_absolute() else Path(p)
            if cand.exists():
                args.ckpt_dir = str(cand.relative_to(ROOT)) if str(cand).startswith(str(ROOT)) else str(cand)
                print(f'[use_latest] 使用最新训练目录: {args.ckpt_dir}')
    return args


def main():
    args = build_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- 本次运行目录 ----------
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_dir = ROOT / args.out_root / f'imagine_{ts}'
    run_dir.mkdir(parents=True, exist_ok=True)
    obs_dir = run_dir / 'observe'
    obs_dir.mkdir(parents=True, exist_ok=True)
    print(f'本次推理结果目录: {run_dir}')

    # ---------- 模型 ----------
    ckpt_path = ROOT / args.ckpt_dir / f'rssm_epoch{args.ckpt}.pt'
    if not ckpt_path.exists():
        print(f'X 找不到权重: {ckpt_path}')
        print('  提示: 先跑训练，或把 config 的 imagine.use_latest 设为 false 并指定 ckpt_dir')
        return 1
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ca = ckpt['args']
    model = RSSM(latent_dim=ca.get('latent', 32), hidden_dim=ca.get('hidden', 200),
                 emb_dim=ca.get('emb', 256)).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'已加载 {ckpt_path.name}   device={device}')

    # ---------- 数据 ----------
    ds = L0Dataset(str(ROOT / 'prism/data/L0'), seq_len=int(args.seq), train=False)
    si = int(args.sample_index)
    if si >= len(ds):
        print(f'[warn] sample_index={si} 超出范围（验证集 {len(ds)} 个），改用 0')
        si = 0

    s = ds[si]
    images = s['images'].unsqueeze(1).to(device)
    proprio = s['proprio'].unsqueeze(1).to(device)
    actions = s['actions'].unsqueeze(1).to(device)
    T = images.shape[0]

    C = int(args.context)
    stride = max(1, int(args.frame_stride))
    sel = [k * stride for k in range(C)]
    if sel[-1] >= T:
        print(f'X observe 需要到第 {sel[-1]} 帧，但样本只有 {T} 帧 —— 减小 context/frame_stride')
        return 1

    obs_img, obs_pro, obs_act = images[sel], proprio[sel], actions[sel]
    print(f'observe: 样本#{si} 的帧 {sel}（步长{stride}，覆盖 {sel[-1]+1} 帧）')

    # 存档 observe 帧到本次目录
    for k in range(C):
        arr = (np.transpose(obs_img[k, 0].cpu().numpy(), (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
        imageio.imwrite(obs_dir / f'frame_{k:02d}.png', arr)
    print(f'  observe 帧已存档 -> {obs_dir}（{C} 张）')

    with torch.no_grad():
        h, z = model.observe(obs_img, obs_pro, obs_act)
    print(f'  已用这 {C} 帧建立记忆')

    # ---------- 推演 ----------
    H = int(args.horizon)
    start_future = sel[-1] + 1
    fut_act = actions[start_future:start_future + H]
    if len(fut_act) < H:
        H = len(fut_act)
        print(f'  (裁剪 horizon 到 {H})')

    real = images[start_future:start_future + H, 0].cpu().numpy()
    modes = list(args.modes)
    pred, prop = {}, {}
    with torch.no_grad():
        for m in modes:
            emb_hat, prop_hat = model.imagine(h.clone(), z.clone(), fut_act, obs_pro[-1], H,
                                              sample=(m != 'mu'))
            img_hat = model.decoder(emb_hat.reshape(H, -1)).reshape(H, 1, 3, 64, 64)
            pred[m] = img_hat[:, 0].cpu().numpy()
            prop[m] = prop_hat[:, 0].cpu().numpy()

    mse = {m: [float(((pred[m][t] - real[t]) ** 2).mean()) for t in range(H)] for m in modes}
    baseline = float(((real.mean(axis=(0, 1, 2)) - real) ** 2).mean())

    # ---------- 每步 CSV ----------
    real_prop = proprio[start_future:start_future + H, 0].cpu().numpy()
    rows = []
    for t in range(H):
        r = {'step': t}
        for m in modes:
            r[f'mse_{m}'] = round(mse[m][t], 8)
        rp, pp = real_prop[t], prop[modes[0]][t]
        r['grip_real_x'], r['grip_real_y'], r['grip_real_z'] = float(rp[0]), float(rp[1]), float(rp[2])
        r['grip_pred_x'], r['grip_pred_y'], r['grip_pred_z'] = float(pp[0]), float(pp[1]), float(pp[2])
        r['grip_err'] = float(np.linalg.norm(pp[0:3] - rp[0:3]))
        rows.append(r)

    csv_path = run_dir / 'steps.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\n每步数据: {csv_path}')
    cols = [c for c in rows[0].keys() if c != 'step']
    print('  step ' + ' '.join(f'{c:>11}' for c in cols))
    for r in rows[:6]:
        print(f'  {r["step"]:>4} ' + ' '.join(f'{r[c]:>11.5f}' for c in cols))

    # ---------- 对比图 ----------
    idx = list(range(0, H, max(1, H // 6)))
    grid_rows = []
    for t in idx:
        trip = [real[t].transpose(1, 2, 0)] + [pred[m][t].transpose(1, 2, 0) for m in modes]
        grid_rows.append(np.repeat(np.repeat(np.concatenate(trip, axis=1), 3, axis=0), 3, axis=1))
    png_path = run_dir / 'compare.png'
    imageio.imwrite(png_path, (np.concatenate(grid_rows, axis=0) * 255).clip(0, 255).astype(np.uint8))
    print(f'\n对比图: {png_path}')

    # ---------- 曲线 ----------
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    for m in modes:
        ax.plot(range(H), mse[m], '-o', lw=2, ms=4, label=m)
    ax.axhline(baseline, color='gray', ls='--', label=f'baseline {baseline:.5f}')
    ax.set_xlabel('imagine step')
    ax.set_ylabel('pixel MSE')
    ax.set_title('imagine rollout error')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    curve_path = run_dir / 'curve.png'
    fig.savefig(curve_path)
    plt.close(fig)
    print(f'曲线图: {curve_path}')

    jd = {f'mse_{m}': mse[m] for m in modes}
    jd['baseline'] = baseline
    jd['sample_index'] = si
    jd['frame_stride'] = stride
    jd['horizon'] = H
    jd['ckpt'] = str(ckpt_path)
    (run_dir / 'curve.json').write_text(json.dumps(jd, indent=1), encoding='utf-8')

    # ---------- 记录 latest & 写一份摘要 ----------
    latest = ROOT / args.out_root / 'latest_imagine.txt'
    latest.parent.mkdir(parents=True, exist_ok=True)
    try:
        latest.write_text(str(run_dir.relative_to(ROOT)), encoding='utf-8')
    except Exception:
        latest.write_text(str(run_dir), encoding='utf-8')

    summary = [
        'imagine 运行摘要',
        f'时间      : {ts}',
        f'权重      : {ckpt_path}',
        f'样本      : 验证集 #{si}，帧步长 {stride}，observe {C} 帧 {sel}',
        f'想象步数  : {H}',
        f'基线 MSE  : {baseline:.5f}',
    ]
    for m in modes:
        summary.append(f'{m:>7} MSE: 第1步 {mse[m][0]:.5f} -> 第{H}步 {mse[m][-1]:.5f} '
                      f'({"好于基线" if mse[m][-1] < baseline else "差于基线"})')
    (run_dir / 'README.txt').write_text('\n'.join(summary), encoding='utf-8')

    print('\n' + '\n'.join(summary))
    print(f'\n全部结果已收入: {run_dir}')
    print(f'latest 记录 -> {latest}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
