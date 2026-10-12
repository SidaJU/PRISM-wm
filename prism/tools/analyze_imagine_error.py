"""分析一次 imagine 运行的"真实能力"：位置误差 vs 静止基线。

为什么要这个工具：
    像素 MSE 会被背景淹没（背景占 70% 且恒定，预测静止就能赢），
    所以"好于基线"可能是假象。真正该看的是：
        1) 模型预测的位置 vs 真实位置（误差多少厘米）
        2) 这个误差 vs "假设世界不动"（静止基线）谁更小
    如果模型误差 > 静止基线，说明想象不但没用、还有害。

支持两类位置：
    - 末端（grip）：机械臂末端 xyz（一直都有）
    - 物体（obj） ：箱子/物体 xyz（v5 加了物体预测头后才有）

用法:
    python prism/tools/analyze_imagine_error.py <imagine结果目录>
"""
import csv
import json
import math
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print('用法: python prism/tools/analyze_imagine_error.py <imagine结果目录>')
        return 1
    d = Path(sys.argv[1])
    csv_path = d / 'steps.csv'
    if not csv_path.exists():
        print('找不到 steps.csv:', csv_path)
        return 1

    rows = list(csv.DictReader(csv_path.open(encoding='utf-8-sig')))

    def xyz(r, pre):
        return [float(r[f'{pre}_x']), float(r[f'{pre}_y']), float(r[f'{pre}_z'])]

    curve = json.loads((d / 'curve.json').read_text(encoding='utf-8')) if (d / 'curve.json').exists() else {}
    baseline = curve.get('baseline')
    has_obj = 'obj_real_x' in rows[0]

    print('=' * 66)
    print('imagine 真实能力分析：' + str(d))
    print('=' * 62 if False else '=' * 66)

    for tag, pre in (('末端 grip', 'grip'), ('物体 obj', 'obj')):
        if pre == 'obj' and not has_obj:
            continue
        p0, pN = xyz(rows[0], f'{pre}_real'), xyz(rows[-1], f'{pre}_real')
        q0, qN = xyz(rows[0], f'{pre}_pred'), xyz(rows[-1], f'{pre}_pred')
        real_move = math.dist(p0, pN)
        pred_move = math.dist(q0, qN)
        static_err = [math.dist(xyz(r, f'{pre}_real'), q0) for r in rows]
        final_err = math.dist(xyz(rows[-1], f'{pre}_real'), xyz(rows[-1], f'{pre}_pred'))

        print()
        print(f'【{tag}】')
        print(f'  真实 {len(rows)} 步总位移 : {real_move * 100:7.2f} cm')
        print(f'  预测 {len(rows)} 步总位移 : {pred_move * 100:7.2f} cm')
        print(f'  真实方向 (dx, dy) : ({pN[0] - p0[0]:+.4f}, {pN[1] - p0[1]:+.4f})')
        print(f'  预测方向 (dx, dy) : ({qN[0] - q0[0]:+.4f}, {qN[1] - q0[1]:+.4f})')
        print()
        print(f'  {"step":>5} {"模型误差":>11} {"静止基线":>11} {"谁更好":>7}')
        for i in range(0, len(rows), max(1, len(rows) // 6)):
            e = math.dist(xyz(rows[i], f'{pre}_real'), xyz(rows[i], f'{pre}_pred')) * 100
            s = static_err[i] * 100
            print(f'  {i:>5} {e:>9.2f}cm {s:>9.2f}cm {"模型" if e < s else "静止":>7}')
        i = len(rows) - 1
        e, s = final_err * 100, static_err[i] * 100
        print(f'  {i:>5} {e:>9.2f}cm {s:>9.2f}cm {"模型" if e < s else "静止":>7}')
        print()
        ratio = final_err / static_err[-1] if static_err[-1] > 0 else float('nan')
        if final_err < static_err[-1]:
            print(f'  结论[{tag}]: [OK] 优于静止假设（误差是静止基线的 {ratio:.2f} 倍）')
        else:
            print(f'  结论[{tag}]: [NG] 差于静止假设（误差是静止基线的 {ratio:.2f} 倍）-- 不如认为世界不动')

    print()
    print('【像素指标（对照，可能是假象）】')
    ms = curve.get('mse_sample', [])
    if ms:
        print(f'  像素 MSE 第 1 步  : {ms[0]:.5f}')
        print(f'  像素 MSE 最后一步: {ms[-1]:.5f}')
    if baseline:
        print(f'  基线(全猜均值)   : {baseline:.5f}')
    print('=' * 66)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
