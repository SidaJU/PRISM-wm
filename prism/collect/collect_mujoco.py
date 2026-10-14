"""W1: MuJoCo L0 数据采集.

产出同源数据集的一档（L0 = MuJoCo 默认渲染，低保真）。
工程约束：观测侧与真值侧**物理分离**落盘，训练 dataloader 只挂载 obs/。

用法:
    python collect_mujoco.py --episodes 20 --out ../../data/L0
    python collect_mujoco.py --episodes 500 --out ../../data/L0 --randomize
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import mujoco
import gymnasium as gym
import gymnasium_robotics

# FetchPush observation(25) 布局:
#   [0:3]   grip_pos        <- proprio
#   [3:6]   object_pos      <- gt
#   [6:9]   object_rel_pos  <- gt
#   [9:11]  gripper_state   <- proprio
#   [11:14] object_rot      <- gt
#   [14:17] object_velp     <- gt
#   [17:20] object_velr     <- gt
#   [20:23] grip_velp       <- proprio
#   [23:25] gripper_vel     <- proprio
PROPRIO_SLICES = [(0, 3), (9, 11), (20, 23), (23, 25)]   # 10 维
GT_SLICES = [(3, 6), (6, 9), (11, 14), (14, 17), (17, 20)]  # 15 维


def split_obs(obs25):
    p = np.concatenate([obs25[a:b] for a, b in PROPRIO_SLICES]).astype(np.float32)
    g = np.concatenate([obs25[a:b] for a, b in GT_SLICES]).astype(np.float32)
    return p, g


# 桌面有效区（实测 FetchPush 的采样范围）：x[1.213,1.512] y[0.599,0.898] z=0.425
# 方块被推出这个平台就会掉到地面（z 0.425 -> 0.025），后续帧全是垃圾数据。
# 平台比采样范围大：采样 x[1.213,1.512] y[0.599,0.898]，实测推到 y≈1.29 才掉落。
# 安全区必须比采样范围宽，否则一开局就误刹车（位移会掉到 5cm）。
TABLE_X = (1.15, 1.58)
TABLE_Y = (0.52, 1.02)
EDGE_MARGIN = 0.02


def heuristic_push(obs25, goal, reach=0.05, tol=0.05):
    """两阶段启发式：先绕到物体背后，再朝目标推；接近目标时减速，避免推飞。"""
    grip_pos = obs25[0:3]
    object_pos = obs25[3:6]

    to_goal = goal - object_pos
    d_goal = float(np.linalg.norm(to_goal))
    if d_goal > 1e-6:
        dir_g = to_goal / d_goal
    else:
        dir_g = np.zeros(3)

    if d_goal < tol:
        return np.zeros(4, dtype=np.float32)          # 已到位，停手

    approach = object_pos - dir_g * reach              # 物体背后的接触点
    d_obj = float(np.linalg.norm(approach - grip_pos))

    # 边缘刹车：方块接近平台边缘就停手，绝不能把它推下去
    if (object_pos[0] < TABLE_X[0] + EDGE_MARGIN or object_pos[0] > TABLE_X[1] - EDGE_MARGIN
            or object_pos[1] < TABLE_Y[0] + EDGE_MARGIN or object_pos[1] > TABLE_Y[1] - EDGE_MARGIN):
        return np.zeros(4, dtype=np.float32)

    # 关键：判断夹爪是否在"物体后方"（目标反方向）。
    # 若夹爪在前方或侧面，直接朝 approach 点直线移动会**撞穿物体**，把它顶向边缘
    # （这正是 82% episode 停在边缘、77% 帧静止的根因）。必须先绕行。
    rel = grip_pos - object_pos
    proj = float(np.dot(rel, dir_g))          # >0 表示夹爪在物体靠目标那一侧

    if proj > -0.02:
        side = np.cross(dir_g, [0.0, 0.0, 1.0])
        n = float(np.linalg.norm(side))
        side = side / n if n > 1e-3 else np.array([0.0, 1.0, 0.0])
        if float(np.dot(side, rel)) < 0:      # 选离夹爪近的一侧绕
            side = -side
        waypoint = object_pos - dir_g * 0.09 + side * 0.08
        waypoint[2] = object_pos[2] + 0.05    # 抬高，避免水平撞到物体
        act = np.clip((waypoint - grip_pos) * 5.0, -1.0, 1.0)
    else:
        d_obj = float(np.linalg.norm(approach - grip_pos))
        if d_obj > 0.04:
            act = np.clip((approach - grip_pos) * 4.0, -1.0, 1.0)
        else:
            # 接触后减速推。0.3（0.5 会推过头，把方块顶下桌面）
            speed = float(np.clip(d_goal / 0.15, 0.0, 1.0)) * 0.3
            act = dir_g * speed
    return np.concatenate([act, [0.0]]).astype(np.float32)  # 夹爪保持


def set_object_color(env, rgba=(0.0, 0.95, 0.95, 1.0)):
    """把被推的方块改成醒目颜色。

    FetchPush 默认 object0 是中灰 (0.5,0.5,0.5)，而桌面也是灰的 —— 在 64x64
    下两者几乎无法分辨，模型（和人眼）都看不出方块在哪、有没有被推动。

    颜色选择（踩过坑）：
      - 不能用红色 (1,0,0)：已被 target0（目标标记）占用。
      - 不能用橙色 (1,0.35,0)：与红色在 RGB 空间太近，64x64 + 光照后
        两者看起来都"又红又橙"，分不清哪个是方块哪个是目标。
      - 用青色 (0,0.95,0.95)：红/橙的互补色，与灰色桌面、蓝白机械臂、
        红色目标球都能一眼区分。
    """
    try:
        m = env.unwrapped.model
    except Exception:
        return False
    for i in range(m.ngeom):
        try:
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) == 'object0':
                m.geom_rgba[i] = list(rgba)
                return True
        except Exception:
            break
    return False


def set_geom_color(env, geom_name, rgba):
    """改任意几何体的颜色（OOD 泛化测试用，例如改地面 floor0 造出不同背景）。"""
    try:
        m = env.unwrapped.model
    except Exception:
        return False
    for i in range(m.ngeom):
        try:
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) == geom_name:
                m.geom_rgba[i] = list(rgba)
                return True
        except Exception:
            break
    return False


def collect_geom_meta(model):
    """一次性导出所有几何体的静态信息（名字/类型/尺寸/颜色）。

    UE5 侧据此为每块几何体建一个对应形状的 Actor，逐帧只更新 Transform。
    这是通用的"MuJoCo 场景镜像"方案，与具体机器人无关。
    """
    metas = []
    for i in range(model.ngeom):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        metas.append({
            'name': str(n) if n else f'geom_{i}',
            'type': int(model.geom_type[i]),
            'size': [float(x) for x in model.geom_size[i]],
            'rgba': [float(x) for x in model.geom_rgba[i]],
        })
    return metas


def collect_frame_poses(data, ngeom):
    """每帧导出所有几何体的世界变换：pos(3) + 3x3 旋转矩阵(9) = 12 个 float。

    用旋转矩阵而不是四元数，避免 mujoco 与 UE 的坐标系/四元数约定差异。
    """
    # 向量化：全量采集时每帧都循环 ngeom 次会拖慢数倍
    out = np.zeros((ngeom, 12), dtype=np.float32)
    out[:, 0:3] = data.geom_xpos[:ngeom]
    out[:, 3:12] = data.geom_xmat[:ngeom].reshape(ngeom, 9)
    return out


def maybe_randomize(env, rng, base):
    """物理 domain randomization: 摩擦 / 质量。

    注意：质量必须基于**初始副本**缩放。直接 `body_mass *= scale` 会在每个 episode
    累积，形成随机游走，几百个 episode 后质量会漂移到极端值（物体被推飞）。
    """
    try:
        model = env.unwrapped.model
    except Exception:
        return None
    info = {}
    try:
        f = rng.uniform(0.7, 1.3)
        model.geom_friction[:, 0] = f
        info['friction'] = float(f)
    except Exception:
        pass
    try:
        if 'mass' not in base:
            base['mass'] = model.body_mass.copy()
        scale = rng.uniform(0.8, 1.2)
        model.body_mass[:] = base['mass'] * scale
        info['mass_scale'] = float(scale)
    except Exception:
        pass
    return info or None


def collect(args):
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / 'obs').mkdir(parents=True, exist_ok=True)
    (out / 'gt').mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env, render_mode='rgb_array',
                   width=args.size, height=args.size,
                   max_episode_steps=args.steps)

    obj_colored = set_object_color(env, rgba=args.obj_color)
    print(f'  object0 recolored: {obj_colored} -> rgba={args.obj_color}')

    if args.bg_color:
        bg_ok = set_geom_color(env, 'floor0', args.bg_color)
        print(f'  floor0 (背景) recolored: {bg_ok} -> rgba={args.bg_color}')

    # ---- 位姿导出（UE5 回放渲染用）----
    pose_model = None
    geom_meta = None
    if args.export_poses:
        try:
            pose_model = env.unwrapped.model
            geom_meta = collect_geom_meta(pose_model)
            (out / 'geom_meta.json').write_text(
                json.dumps({'ngeom': len(geom_meta), 'geoms': geom_meta}, indent=1),
                encoding='utf-8')
            (out / 'poses').mkdir(parents=True, exist_ok=True)
            print(f'  pose export ON: {pose_model.ngeom} geoms')
        except Exception as e:
            print(f'  [warn] pose export disabled: {e}')
            args.export_poses = False

    t_start = time.time()
    n_saved = 0
    success_rate_sum = 0.0
    rand_base = {}   # 保存物理参数基线，避免随机化累积
    drop_count = 0

    for ep in range(args.episodes):
        rand_info = maybe_randomize(env, rng, rand_base) if args.randomize else None
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        goal = obs['desired_goal'].astype(np.float32)

        imgs, proprios, gts, acts, succs, goals = [], [], [], [], [], []
        dropped = False
        stalled = False
        poses = []
        for t in range(args.steps):
            o25 = obs['observation'].astype(np.float32)
            imgs.append(env.render())
            if args.export_poses:
                # 与 render() 同状态：渲染的是什么，位姿就是什么
                poses.append(collect_frame_poses(env.unwrapped.data, pose_model.ngeom))
            p, g = split_obs(o25)
            proprios.append(p)
            gts.append(g)

            if rng.random() < args.epsilon:
                a = env.action_space.sample()
            else:
                a = heuristic_push(o25, goal)
                if args.action_noise > 0:
                    a = a + rng.normal(0, args.action_noise, size=a.shape).astype(np.float32)
                    a = np.clip(a, -1.0, 1.0).astype(np.float32)
            acts.append(a)

            obs, reward, term, trunc, info = env.step(a)
            succs.append(bool(info.get('is_success', False)))
            goals.append(obs['achieved_goal'].astype(np.float32))

            if term or trunc:
                break
            # 方块掉下桌面 -> 立即终止。否则后半程全是"方块躺在地上"的垃圾帧
            if obs['observation'][5] < args.drop_z:
                dropped = True
                break

            # 方块卡住不动 -> 终止。静止帧对动力学训练是纯噪声（占 77% 时模型只学到"世界不变"）
            if len(gts) >= args.stall_window:
                recent = np.asarray(gts[-args.stall_window:])[:, 0:3]
                if float(np.linalg.norm(recent[-1] - recent[0])) < args.stall_dist:
                    stalled = True
                    break

        success_rate_sum += float(np.any(succs)) if succs else 0.0

        # 太短的 episode 丢弃（序列训练需要足够长度）
        if len(imgs) < args.min_len:
            continue

        tag = f'ep_{n_saved:05d}'
        np.savez_compressed(
            out / 'obs' / f'{tag}.npz',
            images=np.asarray(imgs, dtype=np.uint8),
            proprio=np.asarray(proprios, dtype=np.float32),
            actions=np.asarray(acts, dtype=np.float32),
        )
        np.savez_compressed(
            out / 'gt' / f'{tag}.npz',
            object_state=np.asarray(gts, dtype=np.float32),
            achieved_goal=np.asarray(goals, dtype=np.float32),
            desired_goal=goal,
            success=np.asarray(succs, dtype=bool),
            dropped=bool(dropped),
            randomize=json.dumps(rand_info or {}),
        )
        if args.export_poses and poses:
            # 裸二进制 float32 [T, ngeom, 12]，UE5 直接 fread，不必解析 npy 格式。
            # 帧数 T = 文件大小 / (ngeom * 12 * 4)
            np.asarray(poses, dtype=np.float32).tofile(out / 'poses' / f'{tag}.bin')

        n_saved += 1
        drop_count += int(dropped)

        if (ep + 1) % max(1, args.episodes // 10) == 0:
            el = time.time() - t_start
            print(f'  [{ep+1}/{args.episodes}] saved={n_saved} '
                  f'elapsed={el:.0f}s eta={el/(ep+1)*(args.episodes-ep-1):.0f}s')

    env.close()

    meta = {
        'tier': 'L0',
        'renderer': 'mujoco-default',
        'env': args.env,
        'episodes_saved': n_saved,
        'steps_per_ep': args.steps,
        'image_size': [args.size, args.size, 3],
        'proprio_dim': 10,
        'action_dim': 4,
        'gt_dim': 15,
        'policy': f'heuristic_push(eps_random={args.epsilon})',
        'action_noise': args.action_noise,
        'domain_randomization': bool(args.randomize),
        'object_recolor': bool(obj_colored),
        'drop_rate': round(drop_count / max(1, n_saved), 3),
        'drop_z_threshold': args.drop_z,
        'stall_window': args.stall_window,
        'stall_dist': args.stall_dist,
        'seed': args.seed,
        'note': 'gt/ 仅供评测，禁止作为模型输入',
        'libs': {'gymnasium': gym.__version__, 'numpy': np.__version__},
        'collect_seconds': round(time.time() - t_start, 1),
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'\nDone: {n_saved} episodes -> {out}')
    if n_saved:
        print(f'episode success rate (any step): {success_rate_sum/max(1,n_saved):.3f}')
    print(json.dumps(meta, indent=2)[:600])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='FetchPush-v4')
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--steps', type=int, default=100, help='同时作为 max_episode_steps')
    ap.add_argument('--size', type=int, default=64)
    # 默认路径基于脚本位置解析，避免受当前工作目录影响
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[2] / 'prism' / 'data' / 'L0'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--obj-color', type=float, nargs=4, default=[0.0, 0.95, 0.95, 1.0],
                    metavar=('R', 'G', 'B', 'A'),
                    help='方块颜色 rgba（默认青色 0 0.95 0.95 1；'
                         '改颜色可用于测试模型对颜色的泛化能力 / OOD 评测）')
    ap.add_argument('--bg-color', type=float, nargs=4, default=None,
                    metavar=('R', 'G', 'B', 'A'),
                    help='背景（地面 floor0）颜色 rgba；不给则用环境默认。'
                         '【实测未生效】floor0 的颜色由材质/渲染器缓存控制，'
                         '改 geom_rgba 与 mat_rgba 均无效（与 --obj-color 不同）。'
                         '若要测背景泛化，需改用后期处理或直接换场景')
    ap.add_argument('--epsilon', type=float, default=0.15, help='随机动作比例')
    ap.add_argument('--action-noise', type=float, default=0.05)
    ap.add_argument('--min-len', type=int, default=30, help='短于此长度的 episode 丢弃')
    ap.add_argument('--drop-z', type=float, default=0.35,
                    help='方块 z 低于此值视为掉下桌面(桌面高 0.425)，立即终止该 episode')
    ap.add_argument('--stall-window', type=int, default=20, help='连续这么多帧用于判定"卡住"')
    ap.add_argument('--stall-dist', type=float, default=0.004,
                    help='窗口内方块位移小于此值(米)视为卡住，终止 episode')
    ap.add_argument('--randomize', action='store_true')
    ap.add_argument('--export-poses', action='store_true',
                    help='同时导出每帧几何体位姿，供 UE5 侧回放渲染（L1/L2 档位）')
    collect(ap.parse_args())
