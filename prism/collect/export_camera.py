"""导出 MuJoCo 渲染相机的**精确**世界位姿，供 UE5 对齐视角。

不要用 lookat/distance/azimuth/elevation 手推球坐标——MuJoCo 的 azimuth/elevation
约定与常规不同，手推会错好几米（实测差 3.6m）。正确做法：把参数喂回 MuJoCo 自己的
Renderer，让它算出 pos/forward/up。

用法: python export_camera.py [out_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import mujoco
import gymnasium as gym
import gymnasium_robotics


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('D:/gswm_poses')
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make('FetchPush-v4', render_mode='rgb_array', width=64, height=64)
    env.reset()
    env.render()                       # viewer 要渲染一次才初始化

    model = env.unwrapped.model
    data = env.unwrapped.data
    src = env.unwrapped.mujoco_renderer.viewer.cam

    # 把 gymnasium 实际使用的相机参数喂回 MuJoCo，让引擎自己算世界位姿
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array(src.lookat)
    cam.distance = float(src.distance)
    cam.azimuth = float(src.azimuth)
    cam.elevation = float(src.elevation)

    rend = mujoco.Renderer(model, height=64, width=64)
    rend.update_scene(data, camera=cam)
    gl = rend.scene.camera[0]

    info = {
        'pos': [float(v) for v in np.array(gl.pos)],
        'forward': [float(v) for v in np.array(gl.forward)],
        'up': [float(v) for v in np.array(gl.up)],
        'fovy': float(model.vis.global_.fovy),
        'lookat': [float(v) for v in np.array(cam.lookat)],
        'note': ('MuJoCo 世界坐标(米, 右手 Z-up)。UE 侧: 位置与方向向量的 Y 取反并 x100 转厘米; '
                 '旋转用 FRotationMatrix::MakeFromXZ(forward_ue, up_ue)。'),
    }

    path = out_dir / 'camera.json'
    path.write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'camera -> {path}')
    print(json.dumps(info, indent=1))
    return info


if __name__ == '__main__':
    main()
