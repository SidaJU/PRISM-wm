"""检查 --export-poses 导出的位姿是否可用（UE5 回放渲染用）。

核心验证：位姿里的物体位置，必须和 gt/ 里记录的物体位置一致。
（两者若对不上，说明位姿与渲染帧不同步，L1 重渲就会错位。）

用法:
    python prism/tools/check_poses.py
    python prism/tools/check_poses.py prism/data/L0 ep_00000
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (ROOT / 'prism/data/L0')
    tag = sys.argv[2] if len(sys.argv) > 2 else 'ep_00000'

    meta_path = data_dir / 'geom_meta.json'
    if not meta_path.exists():
        print(f'X 缺少 geom_meta.json —— 采集时有没有加 --export-poses？\n  {meta_path}')
        return 1

    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    ngeom = meta['ngeom']
    geoms = meta['geoms']
    print(f'=== {data_dir} ===')
    print(f'几何体数量 ngeom = {ngeom}')
    print('前几个几何体:')
    for g in geoms[:6]:
        print(f"   {g['name']:20s} type={g['type']} size={[round(s,3) for s in g['size']]}")

    pose_path = data_dir / 'poses' / f'{tag}.bin'
    if not pose_path.exists():
        print(f'X 缺少位姿文件: {pose_path}')
        return 1

    raw = np.fromfile(pose_path, dtype=np.float32)
    T = raw.size // (ngeom * 12)
    poses = raw[:T * ngeom * 12].reshape(T, ngeom, 12)
    print(f'\n位姿文件 {pose_path.name}: 帧数 T={T}  形状={poses.shape}  （每帧每 geom: pos3 + rot9）')

    # 找 object0 的索引
    idx_obj = next((i for i, g in enumerate(geoms) if g['name'] == 'object0'), None)
    if idx_obj is None:
        print('X 几何体里没有 object0')
        return 1
    print(f'object0 的索引 = {idx_obj}')

    # 与 gt 里的物体位置对比
    gt_path = data_dir / 'gt' / f'{tag}.npz'
    g = np.load(gt_path)
    obj_gt = g['object_state'][:, 0:3]          # (T,3) 物体世界位置
    obj_pose = poses[:, idx_obj, 0:3]           # (T,3) 位姿里的物体位置

    n = min(len(obj_gt), T)
    err = np.linalg.norm(obj_gt[:n] - obj_pose[:n], axis=1)
    print(f'\n--- 同步性验证（关键）---')
    print(f'   比较帧数       : {n}')
    print(f'   平均位置误差   : {err.mean()*1000:.3f} mm')
    print(f'   最大位置误差   : {err.max()*1000:.3f} mm')

    ok = err.max() < 0.01      # 1 cm 以内
    print(f'\n判定: {"OK 位姿与 gt 一致，可用于 UE5 回放" if ok else "!! 不一致，检查采集顺序"}')

    # 简单统计：物体是否真的动了
    disp = np.linalg.norm(obj_gt[-1] - obj_gt[0])
    print(f'\n物体总位移: {disp*100:.1f} cm   （太小说明这一局几乎没推动）')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
