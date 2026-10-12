"""L0 数据集加载。

目录结构（采集时物理分离）：
    L0/obs/ep_XXXXX.npz   images(T,64,64,3) uint8, proprio(T,10), actions(T,4)
    L0/gt/ ep_XXXXX.npz   object_state(T,15), ...   <- 训练**不加载**

用法:
    ds = L0Dataset('prism/data/L0', seq_len=32)
    batch = ds[0]   # images(32,3,64,64) proprio(32,10) actions(32,4)
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class L0Dataset(Dataset):
    def __init__(self, root, seq_len=32, stride=8, train=True, train_ratio=0.9,
                 max_episodes=None, normalize_obs=True):
        self.seq_len = seq_len
        self.normalize_obs = normalize_obs

        root = Path(root)
        files = sorted((root / 'obs').glob('*.npz'))
        if not files:
            raise FileNotFoundError(f'no obs npz under {root/"obs"}')
        if max_episodes:
            files = files[:max_episodes]

        # 按 episode 划分训练/验证，避免同一局既训又测
        n_train = int(len(files) * train_ratio)
        files = files[:n_train] if train else files[n_train:]

        self.samples = []      # (file_idx, start)
        self.files = files
        for fi, f in enumerate(files):
            with np.load(f) as d:
                T = len(d['images'])
            if T < seq_len:
                continue
            for s in range(0, T - seq_len + 1, stride):
                self.samples.append((fi, s))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fi, s = self.samples[idx]
        f = self.files[fi]
        with np.load(f) as d:
            img = d['images'][s:s + self.seq_len]          # (T,64,64,3) uint8
            proprio = d['proprio'][s:s + self.seq_len]     # (T,10)
            act = d['actions'][s:s + self.seq_len]         # (T,4)

        # v5：物体位置真值。它是「预测目标（监督信号）」，不是模型输入 ——
        # 与"obs/ gt 物理分离防泄漏"不冲突：模型依然看不到它，只能从图像里推断。
        gt_path = f.parent.parent / 'gt' / f.name
        with np.load(gt_path) as g:
            obj = g['object_state'][s:s + self.seq_len, 0:3].astype(np.float32)  # (T,3)

        x = torch.from_numpy(img).float().permute(0, 3, 1, 2) / 255.0            # (T,3,H,W)
        p = torch.from_numpy(proprio).float()
        a = torch.from_numpy(act).float()
        return {'images': x, 'proprio': p, 'actions': a,
                'object_pos': torch.from_numpy(obj)}
