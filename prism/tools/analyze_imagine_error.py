"""分析一次 imagine 运行的"真实能力"：位置误差 vs 静止基线。

为什么要这个工具：
    像素 MSE 会被背景淹没（背景占 70% 且恒定，预测静止就能赢），
    所以"好于基线"可能是假象。真正该看的是：
        1) 模型预测的末端位置 vs 真实末端位置（误差多少厘米）
        2) 这个误差 vs "假设世界不动"（静止基线）谁更小
    如果模型误差 > 静止基线，说明想象不但没用、还有害。

用法:
    python prism/tools/analyze_imagine_error.py prism/runs/imagine_20261009_194251
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

    p0, pN = xyz(rows[0], 'grip_real'), xyz(rows[-1], 'grip_real')
    q0, qN = xyz(rows[0], 'grip_pred'), xyz(rows[-1], 'grip_pred')

    real_move = math.dist(p0, pN)
    pred_move = math.dist(q0, qN)

    # 静止基线：假设世界完全静止（保持第 0 步的预测位置不动）
    static_err = [math.dist(xyz(r, 'grip_real'), q0) for r in rows]

    curve = json.loads((d / 'curve.json').read_text(encoding='utf-8')) if (d / 'curve.json').exists() else {}
    baseline = curve.get('baseline')

    print('=' * 62)
    print('imagine 真实能力分析：' + str(d))
    print('=' * 62)
    print()
    print('【运动量】')
    print(f'  真实末端 {len(rows)} 步总位移 : {real_move * 100:6.2f} cm')
    print(f'  预测末端 {len(rows)} 步总位移 : {pred_move * 100:6.2f} cm')
    print(f'  真实方向 (dx, dy) : ({pN[0] - p0[0]:+.4f}, {pN[1] - p0[1]:+.4f})')
    print(f'  预测方向 (dx, dy) : ({qN[0] - q0[0]:+.4f}, {qN[1] - q0[1]:+.4f})')
    print()
    print('【末端位置误差 vs 静止基线】  (静止基线 = 假设世界不动)')
    print(f'  {"step":>5} {"模型误差":>10} {"静止基线":>10} {"谁更好":>8}')
    for i in range(0, len(rows), max(1, len(rows) // 8)):
        e = float(rows[i]['grip_err']) * 100
        s = static_err[i] * 100
        win = '模型' if e < s else '静止'
        print(f'  {i:>5} {e:>8.2f}cm {s:>8.2f}cm {win:>8}')
    i = len(rows) - 1
    e, s = float(rows[i]['grip_err']) * 100, static_err[i] * 100
    print(f'  {i:>5} {e:>8.2f}cm {s:>8.2f}cm {"模型" if e < s else "静止":>8}')
    print()
    print('【像素指标（对照，注意它可能是假象）】')
    print(f'  像素 MSE 第 1 步  : {curve.get("mse_sample", [float("nan")])[0]:.5f}')
    print(f'  像素 MSE 最后一步: {curve.get("mse_sample", [float("nan")])[-1]:.5f}')
    if baseline:
        print(f'  基线(全猜均值)   : {baseline:.5f}')
    print()
    print('=' * 62)
    ratio = float(rows[-1]['grip_err']) / static_err[-1] if static_err[-1] > 0 else float('nan')
    if float(rows[-1]['grip_err']) < static_err[-1]:
        verdict = f'[OK] 模型优于静止假设（误差是静止基线的 {ratio:.2f} 倍）'
    else:
        verdict = f'[NG] 模型差于静止假设（误差是静止基线的 {ratio:.2f} 倍）-- 想象不如假设世界不动'
    print('结论: ' + verdict)
    print('=' * 62)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
