"""用世界模型做 CEM（交叉熵方法）规划：在「想象」里搜索最优动作序列。

这是 model-based 决策最直接的用法（MPC 的核心）：
    采样一堆候选动作序列 -> 在世界模型里推演 -> 选回报最高的执行

用来回答：既然模型可用，规划能到多好？（与随机动作 / RL 策略对比）

用法: python prism/tools/plan_cem.py [样本序号]
"""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prism.model.dataset import L0Dataset          # noqa: E402
from prism.model.rssm import RSSM                  # noqa: E402


def batch_rollout(model, h, z, prop, act_seq):
    """批量推演：act_seq (N,H,4) -> 终点物体位置 (N,3)。"""
    N, H, _ = act_seq.shape
    h_t = h.expand(N, -1).contiguous()
    z_t = z.expand(N, -1).contiguous()
    prop_t = prop.expand(N, -1).contiguous()
    for t in range(H):
        a = act_seq[:, t]
        gru_in = torch.cat([z_t, a, prop_t], dim=-1)
        h_next = model.gru(gru_in, h_t)
        mu_p, lv_p = model.prior_net(h_next).chunk(2, dim=-1)
        z_next = model._rsample(mu_p, lv_p)
        hz = torch.cat([h_next, z_next], dim=-1)
        prop_t = model.proprio_pred(hz)
        last_obj = model.object_pred(hz)
        h_t, z_t = h_next, z_next
    return last_obj


def main():
    si = int(sys.argv[1]) if len(sys.argv) > 1 else 57
    cfg = yaml.safe_load(open(ROOT / 'prism' / 'config.yaml', encoding='utf-8'))
    tr, im = cfg['train'], cfg['imagine']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_dir = Path(im['ckpt_dir'])
    if im.get('use_latest', True) and (ROOT / 'prism/runs/latest_train.txt').exists():
        ckpt_dir = ROOT / (ROOT / 'prism/runs/latest_train.txt').read_text(encoding='utf-8').strip()
    ckpt = torch.load(ckpt_dir / f"rssm_epoch{im.get('ckpt', 20)}.pt", map_location=device)
    ca = ckpt['args']
    model = RSSM(latent_dim=int(ca.get('latent', 32)), hidden_dim=int(ca.get('hidden', 200)),
                 emb_dim=int(ca.get('emb', 256))).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    ds = L0Dataset(str(ROOT / tr['data']), seq_len=int(im.get('seq', 48)), train=False)
    s = ds[si]
    ctx, stride = int(im.get('context', 8)), int(im.get('frame_stride', 3))
    sel = [stride * i for i in range(ctx)]
    images = s['images'][sel].unsqueeze(1).to(device)
    proprio = s['proprio'][sel].unsqueeze(1).to(device)
    actions = s['actions'][sel].unsqueeze(1).to(device)
    with torch.no_grad():
        h, z = model.observe(images, proprio, actions)
    prop = proprio[-1]
    goal = s['goal'].numpy()
    obj0 = s['object_pos'][sel[-1]].numpy()

    H = 10
    print(f'样本 #{si}   起点->目标 = {np.linalg.norm(obj0 - goal) * 100:.2f} cm'
          f'   (目标方向: {np.round(goal - obj0, 3).tolist()})')
    print()

    def evaluate(seqs):
        with torch.no_grad():
            ends = batch_rollout(model, h, z, prop, torch.from_numpy(seqs).float().to(device))
        return np.linalg.norm(ends.cpu().numpy() - goal, axis=-1) * 100   # cm

    # 基线 1：随机动作序列
    N = 256
    rnd = np.random.uniform(-1, 1, (N, H, 4)).astype(np.float32)
    d_rnd = evaluate(rnd)
    # 基线 2：零动作
    zero = np.zeros((1, H, 4), dtype=np.float32)
    d_zero = evaluate(zero)[0]

    print(f'  随机动作序列 (N={N}) : 终点->目标 平均 {d_rnd.mean():.2f} cm   最好 {d_rnd.min():.2f} cm')
    print(f'  零动作               : 终点->目标 {d_zero:.2f} cm')
    print()

    # CEM 迭代优化
    mean = np.zeros((H, 4), dtype=np.float32)
    std = np.ones((H, 4), dtype=np.float32) * 0.6
    print('  CEM 规划迭代:')
    for it in range(6):
        seqs = np.random.normal(mean, std, (N, H, 4)).astype(np.float32)
        seqs = np.clip(seqs, -1, 1)
        d = evaluate(seqs)
        idx = np.argsort(d)[:32]                      # 选距离最小的 32 条
        elite = seqs[idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 0.05
        print(f'    iter {it}: 最好 {d.min():6.2f} cm   elite 平均 {d[idx].mean():6.2f} cm')

    d_best = evaluate(mean[None, ...].astype(np.float32))[0]
    print()
    print(f'  CEM 规划出的动作序列 : 终点->目标 {d_best:.2f} cm')
    print()
    print(f'  最优动作序列（前 3 步）: {np.round(mean[:3], 3).tolist()}')
    print(f'  主导方向: {np.round(mean.mean(axis=0), 3).tolist()}   '
          f'(目标相对物体方向: {np.round(goal - obj0, 3).tolist()})')


if __name__ == '__main__':
    main()
