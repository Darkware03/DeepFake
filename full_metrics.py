import json
with open('baseline_audit.json', 'r') as f:
    data = json.load(f)

for k, v in data['variables'].items():
    if 'IA' not in v or 'REAL' not in v: continue
    ia = v['IA']
    real = v['REAL']
    if not ia or not real: continue
    print(f"{k}: IA(mean={ia['mean']:.4f}, p50={ia['p50']:.4f}) | REAL(mean={real['mean']:.4f}, p50={real['p50']:.4f})")
