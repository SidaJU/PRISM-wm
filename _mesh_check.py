import json

meta = json.load(open(r'D:\gswm_poses\geom_meta.json'))['geoms']
print('=== MESH type=7 geoms (mechanical arm) ===')
zero = 0
for m in meta:
    if m['type'] == 7:
        s = m['size']
        flag = ' <-- size all zero!' if all(abs(v) < 1e-6 for v in s) else ''
        if all(abs(v) < 1e-6 for v in s):
            zero += 1
        print(f"  {m['name']:30s} size={s}{flag}")

print(f'\nMESH geoms with zero size: {zero}')
print('\n=== summary by type ===')
from collections import Counter
print(Counter(m['type'] for m in meta))
