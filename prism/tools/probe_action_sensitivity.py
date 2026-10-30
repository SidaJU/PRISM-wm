"""诊断：世界模型想象的物体轨迹，对**不同动作**是否敏感？

这是判断「能不能用这个模型做 RL」的关键前提：
  若不同的动作序列推出几乎相同的物体轨迹 -> reward 对动作没有梯度信号 -> RL 不可能学会。

用法: python prism/tools/probe_action_sensitivity.py [样本序号]
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


def rollout(model, h, z, prop, act_seq):
    """用给定动作序列在世界模型里推演，返回想象的物体位置轨迹。"""
    h_t, z_t, prop_t = h, z, prop
    traj = []
    for t in range(act_seq.shape[0]):
        a = act_seq[t]
        if a.dim() == 1:                      # (4,) -> (1,4)，补 batch 维
            a = a.unsqueeze(0)
        gru_in = torch.cat([z_t, a, prop_t], dim=-1)
        h_next = model.gru(gru_in, h_t)
        mu_p, lv_p = model.prior_net(h_next).chunk(2, dim=-1)
        z_next = model._rsample(mu_p, lv_p)
        hz = torch.cat([h_next, z_next], dim=-1)
        traj.append(model.object_pred(hz)[0].detach().cpu().numpy())
        prop_t = model.proprio_pred(hz)
        h_t, z_t = h_next, z_next
    return np.array(traj)


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
    images = s['images'][sel].unsqueeze(1).to(device)      # (T,1,3,H,W)
    proprio = s['proprio'][sel].unsqueeze(1).to(device)
    actions = s['actions'][sel].unsqueeze(1).to(device)
    with torch.no_grad():
        h, z = model.observe(images, proprio, actions)
    prop = proprio[-1]

    H = 10
    obj0 = s['object_pos'][sel[-1]].numpy()
    goal = s['goal'].numpy()
    print(f'样本 #{si}')
    print(f'  起点物体位置: {obj0.round(4).tolist()}')
    print(f'  目标位置    : {goal.round(4).tolist()}')
    print(f'  起点->目标距离: {np.linalg.norm(obj0 - goal) * 100:.2f} cm')
    print()

    # 几组截然不同的动作序列
    proto = {
        '零动作     ': np.zeros((H, 4), dtype=np.float32),
        '持续 +x    ': np.tile([0.8, 0, 0, 0], (H, 1)).astype(np.float32),
        '持续 -x    ': np.tile([-0.8, 0, 0, 0], (H, 1)).astype(np.float32),
        '持续 +y    ': np.tile([0, 0.8, 0, 0], (H, 1)).astype(np.float32),
        '持续 -y    ': np.tile([0, -0.8, 0, 0], (H, 1)).astype(np.float32),
        '持续 +z(抬) ': np.tile([0, 0, 0.8, 0], (H, 1)).astype(np.float32),
    }

    print('  动作序列        | 想象终点物体位置            | 相对起点位移 | 终点->目标')
    print('  ' + '-' * 76)
    ends = {}
    for name, seq in proto.items():
        with torch.no_grad():
            traj = rollout(model, h, z, prop, torch.from_numpy(seq).to(device))
        end = traj[-1]
        ends[name] = end
        move = np.linalg.norm(end - obj0)
        d_goal = np.linalg.norm(end - goal)
        print('  %s | %s | %8.2f cm | %8.2f cm' % (
            name, np.round(end, 4).tolist(), move * 100, d_goal * 100))

    arr = np.array([ends[k] for k in proto])
    spread = np.linalg.norm(arr.max(axis=0) - arr.min(axis=0))
    print()
    print(f'  不同动作导致的终点差异（最大-最小）: {spread * 100:.2f} cm')
    real_move = 7.0     # 实测：真实数据里 20 步物体位移约 7 cm
    print(f'  参考：真实数据里物体 20 步的真实位移 ≈ {real_move:.1f} cm')
    print()
    if spread * 100 < 1.0:
        print('  结论: [NG] 模型对动作几乎不敏感 -> reward 对动作没有梯度信号，RL 学不到东西')
    elif spread * 100 < real_move:
        print('  结论: [弱] 有反应但幅度小于真实位移 -> 信号被噪声淹没，RL 很难学')
    else:
        print('  结论: [OK] 模型对动作敏感且幅度足够 -> 具备做 RL 的前提')


if __name__ == '__main__':
    main()
