"""导出 MuJoCo 渲染相机参数 -> camera.json（供 UE5 回放用，保证视角与 L0 一致）。

关键：不用手算 lookat/azimuth -> pos/forward/up（容易错），
而是直接从 MuJoCo 渲染时**实际使用**的相机矩阵读（scene.camera[0]）。
再用"方块投影到图像的位置"验证正确性。

用法:
    python prism/tools/export_camera.py [输出目录]
"""
import json
import sys
from pathlib import Path

import numpy as np
import mujoco
import gymnasium as gym
import gymnasium_robotics

ROOT = Path(__file__).resolve().parents[2]


def find_cyan_centroid(img):
    """找青色方块 (0,0.95,0.95) 的像素质心"""
    a = np.asarray(img).astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (r < 90) & (g > 150) & (b > 150)
    if mask.sum() < 3:
        return None, int(mask.sum())
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()), float(ys.mean())), int(mask.sum())


def project(P, pos, fwd, up, right, fovy_deg, W, H):
    d = np.asarray(P, dtype=np.float64) - pos
    z = float(np.dot(d, fwd))
    if z <= 1e-6:
        return None
    x = float(np.dot(d, right))
    y = float(np.dot(d, up))
    ty = np.tan(np.deg2rad(fovy_deg) / 2.0)
    tx = ty * (W / H)
    u = W / 2.0 + (x / z) / tx * (W / 2.0)
    v = H / 2.0 - (y / z) / ty * (H / 2.0)
    return u, v


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (ROOT / 'prism/data/L0')

    env = gym.make('FetchPush-v4', render_mode='rgb_array', width=64, height=64)
    obs, _ = env.reset(seed=0)
    model = env.unwrapped.model
    data = env.unwrapped.data

    # 1) 拿到 gymnasium 用的相机配置
    mr = env.unwrapped.mujoco_renderer
    viewer = mr._get_viewer('rgb_array')
    cam = viewer.cam
    fovy = float(model.vis.global_.fovy)
    print('=== MuJoCo 相机配置 ===')
    print(f'  lookat={np.array(cam.lookat).tolist()} distance={cam.distance} '
          f'azimuth={cam.azimuth} elevation={cam.elevation} fovy={fovy}')

    # 2) 用同一个相机渲一次，读 MuJoCo 内部实际使用的相机位姿
    renderer = mujoco.Renderer(model, height=64, width=64)
    renderer.update_scene(data, camera=cam)
    glc = renderer.scene.camera[0]
    pos = np.array(glc.pos, dtype=np.float64)
    fwd = np.array(glc.forward, dtype=np.float64)
    up = np.array(glc.up, dtype=np.float64)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)

    print('\n=== 实际相机位姿（MuJoCo 内部） ===')
    print(f'  pos     = {np.round(pos,4).tolist()}')
    print(f'  forward = {np.round(fwd,4).tolist()}')
    print(f'  up      = {np.round(up,4).tolist()}')

    # 3) 验证一：渲染是否与原 env.render() 一致
    img_ref = env.render()                       # gymnasium 的渲染
    img_ours = renderer.render()                 # 我们用同相机的渲染
    diff = np.abs(img_ref.astype(np.int16) - img_ours.astype(np.int16)).mean()
    print(f'\n验证1 渲染一致性: 平均像素差 = {diff:.2f} (0 表示完全相同)')

    # 4) 验证二：方块投影是否落在图上青色位置
    obj_pos = obs['observation'][3:6].astype(np.float64)
    gt, npx = find_cyan_centroid(img_ref)
    uv = project(obj_pos, pos, fwd, up, right, fovy, 64, 64)
    print(f'验证2 方块投影: 世界坐标={np.round(obj_pos,3).tolist()}')
    print(f'        图上青色质心 = {gt} (像素数 {npx})')
    print(f'        投影预测     = ({uv[0]:.1f}, {uv[1]:.1f})' if uv else '        投影失败')
    if uv and gt:
        err = float(np.hypot(uv[0] - gt[0], uv[1] - gt[1]))
        print(f'        误差 = {err:.1f} px')
        ok = err < 8
    else:
        err, ok = -1, False

    cam_json = {
        'pos': [round(float(v), 6) for v in pos],
        'forward': [round(float(v), 6) for v in fwd],
        'up': [round(float(v), 6) for v in up],
        'fovy': fovy,
        'lookat': [round(float(v), 6) for v in np.array(cam.lookat)],
        'distance': float(cam.distance),
        'azimuth': float(cam.azimuth),
        'elevation': float(cam.elevation),
        'image_size': [64, 64],
        'source': 'mujoco scene.camera[0]（渲染实际使用值）',
        'note': 'MuJoCo 世界坐标(米, Z-up)。UE 侧: 位置×100, Y 取反。',
    }
    (out_dir / 'camera.json').write_text(
        json.dumps(cam_json, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\n已写入 {out_dir / "camera.json"}')
    print(f'判定: {"OK 相机正确" if ok else "!! 仍不对，误差 " + str(round(err,1)) + " px"}')
    env.close()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
