"""在世界模型的「想象」中训练 Actor-Critic（model-based RL，Dreamer 风格）。

思路（不需要任何真实环境交互）：
    1. 用真实数据的一段观测 observe() 得到起点状态 (h, z)
    2. 在想象中逐步 rollout：actor 出动作 -> RSSM 推演 -> object_pred 给物体位置 -> 算 reward
    3. Critic 用 λ-return 训练（监督 "这个状态值多少"）
    4. Actor 直接最大化累积回报 —— 世界模型是可微的，梯度能穿过想象轨迹回传到策略

世界模型全程**冻结**（只当模拟器），只训练 actor / critic。
reward 用 v5 的物体位置预测头：r = -||预测物体位置 - 目标位置||（越近越好）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prism.model.agent import Actor, Critic            # noqa: E402
from prism.model.dataset import L0Dataset              # noqa: E402
from prism.model.rssm import RSSM                      # noqa: E402


def build_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    ca = ckpt['args']
    model = RSSM(latent_dim=int(ca.get('latent', 32)),
                 hidden_dim=int(ca.get('hidden', 200)),
                 emb_dim=int(ca.get('emb', 256))).to(device)
    missing, _ = model.load_state_dict(ckpt['model'], strict=False)
    if missing:
        print(f'  [警告] 权重缺少层: {list(missing)} -> 物体位置预测不可用，RL 无法训练')
        sys.exit(1)
    model.eval()
    for p in model.parameters():          # 冻结：世界模型只当模拟器
        p.requires_grad_(False)
    return model


def imagine_rollout(model, actor, critic, h, z, prop, goal, H, gamma, lam, reward_scale):
    """在世界模型里逐步推演 H 步，返回 (returns, values, rewards)。

    returns/values: (H, B) —— returns 对 actor 可导（梯度穿过世界模型）
    """
    h_t, z_t, prop_t = h, z, prop
    log_probs, rewards, values = [], [], []
    last_obj = None
    for _ in range(H):
        # 关键：状态 detach —— 梯度**不穿过世界模型**，避免模型 OOD 时给出错误梯度
        hz = torch.cat([h_t, z_t], dim=-1).detach()
        a_t, lp_t = actor.sample(hz)                      # 随机策略采样（对策略可导）
        v_t = critic(hz)
        with torch.no_grad():                             # 世界模型只当模拟器
            gru_in = torch.cat([z_t, a_t, prop_t], dim=-1)
            h_next = model.gru(gru_in, h_t)
            mu_p, lv_p = model.prior_net(h_next).chunk(2, dim=-1)
            z_next = model._rsample(mu_p, lv_p)
            hz_next = torch.cat([h_next, z_next], dim=-1)
            obj_next = model.object_pred(hz_next)         # v5：想象的物体位置
            prop_next = model.proprio_pred(hz_next)       # 闭环：下一步用自己的预测
            r_t = -torch.norm(obj_next - goal, dim=-1) * reward_scale
        log_probs.append(lp_t)
        rewards.append(r_t)
        values.append(v_t)
        last_obj = obj_next
        h_t, z_t, prop_t = h_next.detach(), z_next.detach(), prop_next.detach()

    rewards = torch.stack(rewards)                        # (H,B) 无梯度（标量信号）
    log_probs = torch.stack(log_probs)                    # (H,B) 对策略可导
    # bootstrap：用最后状态的价值收尾
    last_v = critic(torch.cat([h_t, z_t], dim=-1).detach())
    values.append(last_v)
    values = torch.stack(values)                          # (H+1,B)

    # λ-return：混合蒙特卡洛回报与 bootstrap
    rets, ret = [], values[-1].detach()
    for t in reversed(range(H)):
        v_next = values[t + 1].detach()
        ret = rewards[t] + gamma * ((1.0 - lam) * v_next + lam * ret)
        rets.insert(0, ret)
    rets = torch.stack(rets)                              # (H,B)
    # 想象终点时物体离目标多远（米）—— "策略有没有把物体推向目标"的直接指标
    last_dist = torch.norm(last_obj - goal, dim=-1)
    return rets, values[:-1], log_probs, last_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'prism' / 'config.yaml'))
    ap.add_argument('--iters', type=int, default=None)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    rl = cfg.get('rl', {})
    tr, im = cfg['train'], cfg['imagine']

    device = torch.device(args.device or tr.get('device', 'cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')

    iters = int(args.iters or rl.get('iters', 300))
    H = int(rl.get('horizon', 15))
    gamma = float(rl.get('gamma', 0.99))
    lam = float(rl.get('lambda', 0.95))
    reward_scale = float(rl.get('reward_scale', 10.0))
    lr = float(rl.get('lr', 1e-4))
    batch = int(rl.get('batch', 16))
    ctx, stride = int(im.get('context', 8)), int(im.get('frame_stride', 3))

    # ---- 世界模型 ----
    ckpt_dir = Path(im['ckpt_dir'])
    if im.get('use_latest', True) and (ROOT / 'prism/runs/latest_train.txt').exists():
        ckpt_dir = ROOT / (ROOT / 'prism/runs/latest_train.txt').read_text(encoding='utf-8').strip()
    ckpt_path = ckpt_dir / f"rssm_epoch{im.get('ckpt', 20)}.pt"
    print(f'世界模型: {ckpt_path}')
    model = build_model(ckpt_path, device)

    state_dim = model.hidden_dim + model.latent_dim
    hidden = int(rl.get('hidden', 256))
    actor = Actor(state_dim, model.action_dim, hidden=hidden,
                  action_scale=float(rl.get('action_scale', 1.0))).to(device)
    critic = Critic(state_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    ds = L0Dataset(str(ROOT / tr['data']), seq_len=int(tr.get('seq', 32)), train=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0,
                                         drop_last=True)
    print(f'数据集: {len(ds)} 样本   想象步数 H={H}   iters={iters}')

    sel = [stride * i for i in range(ctx)]
    hist = []
    it = 0
    print()
    print('  iter |  actor_loss | critic_loss |  回报均值 | 终点距离(cm)')
    print('-' * 62)
    while it < iters:
        for batch_data in loader:
            if it >= iters:
                break
            images = batch_data['images'].to(device).permute(1, 0, 2, 3, 4)   # (T,B,3,H,W)
            proprio = batch_data['proprio'].to(device).permute(1, 0, 2)
            actions = batch_data['actions'].to(device).permute(1, 0, 2)
            goal = batch_data['goal'].to(device)

            with torch.no_grad():                      # observe 不需要梯度
                h, z = model.observe(images[sel], proprio[sel], actions[sel])
            prop = proprio[sel[-1]]

            rets, vals, log_probs, last_dist = imagine_rollout(
                model, actor, critic, h, z, prop, goal, H, gamma, lam, reward_scale)

            # REINFORCE with baseline：用「回报 - 价值」作为优势加权动作的对数概率
            adv = rets - vals.detach()
            loss_actor = -(log_probs * adv).mean()                     # 策略梯度
            loss_critic = F.mse_loss(vals, rets.detach())              # 价值回归
            opt.zero_grad()
            (loss_actor + loss_critic).backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), 10.0)
            opt.step()

            mean_ret = float(rets.mean().detach())
            fd_cm = float(last_dist.mean().detach()) * 100     # 米 -> cm
            hist.append({'iter': it, 'return': mean_ret,
                         'actor': float(loss_actor.detach()),
                         'critic': float(loss_critic.detach()),
                         'final_dist_cm': fd_cm})
            if it % 25 == 0 or it == iters - 1:
                print('%6d | %10.4f | %11.4f | %9.3f | %10.2f' % (
                    it, float(loss_actor.detach()), float(loss_critic.detach()),
                    mean_ret, fd_cm))
            it += 1

    # ---- 保存 ----
    stamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'prism' / 'runs' / f'rl_{stamp}'
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict(),
                'args': {'state_dim': state_dim, 'action_dim': model.action_dim,
                         'hidden': hidden, 'horizon': H,
                         'action_scale': float(rl.get('action_scale', 1.0))},
                'config': rl}, out / 'actor_critic.pt')
    json.dump(hist, open(out / 'rl_history.json', 'w', encoding='utf-8'), indent=1)
    (ROOT / 'prism/runs/latest_rl.txt').write_text(str(out.relative_to(ROOT)).replace('\\', '/'),
                                                   encoding='utf-8')
    print()
    print(f'训练完成 -> {out}')
    print(f'  权重: actor_critic.pt')
    print(f'  历史: rl_history.json')
    return str(out)


if __name__ == '__main__':
    main()
