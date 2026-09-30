"""MPC（模型预测控制）：用 imagine() 在"脑内"推演候选动作，选最优的执行。

流程（每步）:
  1. 用最近 C 帧观测 observe() 建立记忆
  2. 随机采样 N 条长度为 H 的候选动作序列
  3. imagine() 推演每条序列的后果（不碰真实环境）
  4. 代价 = 想象中末端位置到目标的距离（纯模型输出）
  5. 执行最优序列的第一个动作，然后重新规划

用法:
    python prism/tools/mpc_demo.py --episodes 10 --policy mpc
    python prism/tools/mpc_demo.py --episodes 10 --policy heuristic   # 基线
    python prism/tools/mpc_demo.py --episodes 10 --policy random       # 基线
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import mujoco
import gymnasium as gym
import gymnasium_robotics

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prism.model.rssm import RSSM       # noqa: E402

# FetchPush observation(25) 里本体所在的切片
PROPRIO_SLICES = [(0, 3), (9, 11), (20, 23), (23, 25)]


def split_proprio(o25):
    return np.concatenate([o25[a:b] for a, b in PROPRIO_SLICES]).astype(np.float32)


def set_object_color(env, rgba=(0.0, 0.95, 0.95, 1.0)):
    """方块改青色 —— 必须与训练数据一致。"""
    m = env.unwrapped.model
    for i in range(m.ngeom):
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) == 'object0':
            m.geom_rgba[i] = list(rgba)
            return True
    return False


def heuristic_push(o25, goal):
    """采集时的启发式（用作预热动作和基线策略）。"""
    grip_pos = o25[0:3]
    object_pos = o25[3:6]
    to_goal = goal - object_pos
    d_goal = float(np.linalg.norm(to_goal))
    if d_goal < 0.05:
        return np.zeros(4, dtype=np.float32)
    dir_g = to_goal / d_goal if d_goal > 1e-6 else np.zeros(3)
    approach = object_pos - dir_g * 0.05
    if np.linalg.norm(approach - grip_pos) > 0.05:
        act = np.clip((approach - grip_pos) * 5.0, -1, 1)
    else:
        speed = float(np.clip(d_goal / 0.15, 0, 1)) * 0.3
        act = dir_g * speed
    return np.concatenate([act, [0.0]]).astype(np.float32)


@torch.no_grad()
def mpc_action(model, h, z, proprio_now, goal, H, N, device):
    """随机射击：采样 N 条序列 -> imagine 推演 -> 按代价选最优首动作。"""
    act = (torch.rand(H, N, 4, device=device) * 2 - 1)
    act[:, :, 3] = 0.0                      # 夹爪固定（推任务）

    h_rep = h.repeat(N, 1)
    z_rep = z.repeat(N, 1)
    p_rep = proprio_now.repeat(N, 1)

    _, prop_hat = model.imagine(h_rep, z_rep, act, p_rep, H, sample=False)
    grip = prop_hat[:, :, 0:3]              # 预测的末端位置
    goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device).view(1, 1, 3)
    dist = torch.norm(grip - goal_t, dim=-1)
    cost = dist.mean(dim=0)                 # 整段平均距离作代价
    best = int(cost.argmin())
    return act[0, best].cpu().numpy()


def run_episode(env, model, policy, C, H, N, device, max_steps=100):
    obs, _ = env.reset()
    goal = obs['desired_goal'].astype(np.float32)
    buf_img, buf_prop, buf_act = [], [], []
    h = z = None
    n_mpc = 0

    for t in range(max_steps):
        o25 = obs['observation'].astype(np.float32)
        img = env.render()
        prop = split_proprio(o25)

        if policy == 'mpc':
            buf_img.append(img)
            buf_prop.append(prop)
            # 攒够 C 帧才第一次建立记忆
            if h is None and len(buf_img) >= C:
                imgs = torch.from_numpy(np.stack(buf_img[-C:])).float().permute(0, 3, 1, 2).div(255).unsqueeze(1).to(device)
                props = torch.from_numpy(np.stack(buf_prop[-C:])).float().unsqueeze(1).to(device)
                acts = torch.from_numpy(np.stack(buf_act[-C:])).float().unsqueeze(1).to(device) if len(buf_act) >= C \
                    else torch.zeros(C, 1, 4, device=device)
                h, z = model.observe(imgs, props, acts)

            if h is not None:
                a = mpc_action(model, h, z,
                               torch.from_numpy(prop).float().unsqueeze(0).to(device),
                               goal, H, N, device)
                n_mpc += 1
            else:
                a = heuristic_push(o25, goal)     # 预热期
        elif policy == 'heuristic':
            a = heuristic_push(o25, goal)
        else:
            a = env.action_space.sample()
            a[3] = 0.0

        buf_act.append(a)
        obs, reward, term, trunc, info = env.step(a)
        if term or trunc:
            break

    o = obs['observation']
    final_dist = float(np.linalg.norm(o[3:6] - goal))     # 物体到目标（合法观测）
    return bool(info.get('is_success', False)), final_dist, n_mpc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='prism/ckpt_fb01')
    ap.add_argument('--ckpt', default='20')
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--policy', default='mpc', choices=['mpc', 'heuristic', 'random'])
    ap.add_argument('--context', type=int, default=4)
    ap.add_argument('--horizon', type=int, default=6)
    ap.add_argument('--samples', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('FetchPush-v4', render_mode='rgb_array', width=64, height=64,
                   max_episode_steps=100)
    set_object_color(env)

    model = None
    if args.policy == 'mpc':
        ckpt_path = ROOT / args.ckpt_dir / f'rssm_epoch{args.ckpt}.pt'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ca = ckpt['args']
        model = RSSM(latent_dim=ca.get('latent', 32), hidden_dim=ca.get('hidden', 200),
                     emb_dim=ca.get('emb', 256)).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        print(f'已加载 {ckpt_path.name}')

    print(f'策略={args.policy}  episodes={args.episodes}  context={args.context}  '
          f'horizon={args.horizon}  samples={args.samples}')

    rng = np.random.default_rng(args.seed)
    succs, dists, steps = [], [], []
    for ep in range(args.episodes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        s, d, n = run_episode(env, model, args.policy, args.context, args.horizon,
                              args.samples, device)
        succs.append(s); dists.append(d); steps.append(n)
        print(f'  ep {ep}: success={s}  物体到目标={d*100:.1f}cm  MPC步数={n}')

    succs = np.array(succs); dists = np.array(dists)
    print(f'\n=== 结果（{args.episodes} episode）===')
    print(f'  成功率  : {succs.mean()*100:.1f}%')
    print(f'  末态距离: 平均 {dists.mean()*100:.1f} cm')
    env.close()


if __name__ == '__main__':
    main()
