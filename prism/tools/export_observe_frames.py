"""把 imagine 的 observe context（8 帧）固化成一个"规范文件夹"，供 config 指定、bat 直接调用。

文件夹内格式（固定）:
    frame_00.png ~ frame_07.png    8 帧观测图（64×64，模型实际输入尺寸）
    context.npz                    images(8,3,64,64) / proprio(8,10) / actions(8,4)
                                   —— imagine_demo 直接读它做 observe，不再依赖数据集索引
    info.csv                       每帧的末端/物体坐标（给人看）

用法:
    python prism/tools/export_observe_frames.py --sample-index 57 --frame-stride 3 \
        --out prism/observe_context/active
    python prism/tools/export_observe_frames.py --auto-active --frame-stride 3
    python prism/tools/export_observe_frames.py --scan 494        # 只看运动量排名
    python prism/tools/export_observe_frames.py --big             # 额外生成放大版
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import imageio
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prism.model.dataset import L0Dataset        # noqa: E402


def motion_of(ds, data_dir, idx, seq_len):
    """样本运动量 = 物体移动路径长度（米）。"""
    try:
        fi, start = ds.samples[idx]
        gt_path = ROOT / data_dir / 'gt' / Path(ds.files[fi]).name
        if not gt_path.exists():
            return None
        d = np.load(gt_path)
        if 'object_state' not in d:
            return None
        obj = d['object_state'][start:start + seq_len, 0:3]
        if len(obj) < 2:
            return None
        return float(np.linalg.norm(np.diff(obj, axis=0), axis=1).sum())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'prism/config.yaml'))
    ap.add_argument('--sample-index', type=int, default=None)
    ap.add_argument('--context', type=int, default=None)
    ap.add_argument('--seq', type=int, default=None)
    ap.add_argument('--data', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--frame-stride', type=int, default=1)
    ap.add_argument('--auto-active', action='store_true', help='自动选运动量最大的样本')
    ap.add_argument('--scan', type=int, default=0, help='>0 只扫描排名，不导出')
    ap.add_argument('--big', action='store_true', help='额外生成放大 5 倍的预览图')
    args = ap.parse_args()

    cfg = {}
    p = Path(args.config)
    if p.exists():
        cfg = (yaml.safe_load(p.read_text(encoding='utf-8')) or {}).get('imagine', {}) or {}
    ctx = int(args.context or cfg.get('context', 8))
    seq = int(args.seq or cfg.get('seq', 32))
    data = args.data or 'prism/data/L0'
    out_dir = Path(args.out) if args.out else (ROOT / 'prism/observe_context/active')
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = L0Dataset(str(ROOT / data), seq_len=seq, train=False)
    n = len(ds)

    if args.scan > 0:
        k = min(args.scan, n)
        scores = [(m, i) for i in range(k) if (m := motion_of(ds, data, i, seq)) is not None]
        scores.sort(reverse=True)
        print(f'扫描前 {k} 个验证集样本（运动量=物体移动路径长度，米）')
        print('  rank  sample   motion')
        for r, (m, i) in enumerate(scores[:15]):
            print(f'  {r:>4}  {i:>6}   {m:.4f}')
        return 0

    if args.auto_active:
        best, best_i = -1.0, 0
        for i in range(n):
            m = motion_of(ds, data, i, seq)
            if m is not None and m > best:
                best, best_i = m, i
        idx = best_i
        print(f'[auto-active] 选中样本 #{idx}，运动量={best:.4f}')
    else:
        idx = int(args.sample_index) if args.sample_index is not None else 0

    if idx >= n:
        print(f'X 样本索引 {idx} 超出范围（验证集共 {n} 个）')
        return 1

    s = ds[idx]
    images = s['images'].numpy()
    proprio = s['proprio'].numpy()
    actions = s['actions'].numpy()
    stride = max(1, int(args.frame_stride))

    need = (ctx - 1) * stride + 1
    if need > len(images):
        print(f'X 需要 {need} 帧，样本只有 {len(images)} 帧 —— 减小 --context 或 --frame-stride')
        return 1

    # 按步长抽帧
    sel = [k * stride for k in range(ctx)]
    ctx_img = images[sel]        # (8,3,64,64)
    ctx_pro = proprio[sel]       # (8,10)
    ctx_act = actions[sel]       # (8,4)

    obj = None
    try:
        fi, start = ds.samples[idx]
        gt_path = ROOT / data / 'gt' / Path(ds.files[fi]).name
        if gt_path.exists():
            d = np.load(gt_path)
            obj = d['object_state'][start:start + len(images), 0:3]
    except Exception:
        obj = None

    # ---------- 写盘 ----------
    for k in range(ctx):
        img = (np.transpose(ctx_img[k], (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
        imageio.imwrite(out_dir / f'frame_{k:02d}.png', img)
        if args.big:
            imageio.imwrite(out_dir / f'frame_{k:02d}_big.png',
                            np.repeat(np.repeat(img, 5, axis=0), 5, axis=1))

    npz_path = out_dir / 'context.npz'
    np.savez(npz_path, images=ctx_img.astype(np.float32),
             proprio=ctx_pro.astype(np.float32), actions=ctx_act.astype(np.float32))

    rows = []
    for k in range(ctx):
        t = sel[k]
        r = {'ctx_idx': k, 'src_frame': t, 'file': f'frame_{k:02d}.png',
             'grip_x': float(ctx_pro[k][0]), 'grip_y': float(ctx_pro[k][1]),
             'grip_z': float(ctx_pro[k][2])}
        if obj is not None and t < len(obj):
            r['obj_x'], r['obj_y'], r['obj_z'] = float(obj[t][0]), float(obj[t][1]), float(obj[t][2])
        rows.append(r)

    csv_path = out_dir / 'info.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f'\n样本 #{idx}  context={ctx}  帧步长={stride}（时间跨度 {(ctx-1)*stride} 帧）')
    print(f'规范文件夹: {out_dir}')
    print('  frame_00.png ~ frame_%02d.png   观测图（64×64）' % (ctx - 1))
    print(f'  context.npz                     模型直接读（images/proprio/actions）')
    print(f'  info.csv                        每帧坐标')

    print('\n每帧状态:')
    for r in rows:
        line = (f"  frame_{r['ctx_idx']:02d} (源帧{r['src_frame']:>2})  "
                f"grip=({r['grip_x']:.4f}, {r['grip_y']:.4f})")
        if 'obj_x' in r:
            line += f"   obj=({r['obj_x']:.4f}, {r['obj_y']:.4f})"
        print(line)

    d_grip = float(np.linalg.norm(ctx_pro[-1][0:2] - ctx_pro[0][0:2]))
    print(f'\n首尾末端位移 = {d_grip*100:.2f} cm')
    if obj is not None:
        d_obj = float(np.linalg.norm(obj[sel[-1]][0:2] - obj[sel[0]][0:2]))
        print(f'首尾物体位移 = {d_obj*100:.2f} cm')
    return 0


if __name__ == '__main__':
    sys.exit(main())
