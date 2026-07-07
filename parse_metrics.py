import json

with open('baseline_audit.json', 'r') as f:
    data = json.load(f)

for k, v in data['variables'].items():
    if 'IA' not in v or 'REAL' not in v: continue
    ia = v['IA']
    real = v['REAL']
    if not ia or not real: continue
    print(f"--- {k} ---")
    print(f"IA   -> Mean: {ia['mean']:.4f}, Median: {ia['median']:.4f}")
    print(f"REAL -> Mean: {real['mean']:.4f}, Median: {real['median']:.4f}")

