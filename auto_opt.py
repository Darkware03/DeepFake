import json
import glob
import math

def get_data():
    data = []
    ia_paths = glob.glob('dataset/01_videos_ia/*/analysis/response.json')
    for p in ia_paths:
        with open(p, 'r') as f:
            j = json.load(f)
            data.append((0, j))
    real_paths = glob.glob('dataset/02_videos_reales/*/analysis/response.json')
    for p in real_paths:
        with open(p, 'r') as f:
            j = json.load(f)
            data.append((1, j))
    return data

def median_feat(features, name):
    vals = [v for k, v in features.items() if name in k and isinstance(v, (int, float))]
    if not vals: return 0.0
    vals.sort()
    return vals[len(vals)//2]

data = get_data()

parsed = []
for label, resp in data:
    features = resp.get('challenge', {}).get('features', {})
    deepfake_prob = resp.get('deepfake_probability', 0.5)
    de_l = features.get("left_cheek_deltaE76", 0.0)
    de_r = features.get("right_cheek_deltaE76", 0.0)
    sym = 0.0
    if max(de_l, de_r, 1.0) > 0:
        sym = max(0.0, 1.0 - abs(de_l - de_r) / max(de_l, de_r, 1.0))
        
    refl = median_feat(features, 'reflection_strength')
    cm = median_feat(features, 'expected_color_match')
    q = features.get("global_quality_score", 1.0)
    lbp = median_feat(features, 'lbp') / 255.0
    psd = median_feat(features, 'psd') / 100.0
    
    parsed.append({
        'label': label,
        'sym': sym,
        'refl': refl,
        'refl_inv': max(0.0, 1.0 - refl * 3.0),
        'df': deepfake_prob,
        'cm': cm,
        'q': q,
        'q_inv': max(0.0, 1.0 - q),
        'lbp': lbp,
        'psd': psd
    })

import random

best_acc = 0
best_w = None

for _ in range(50000):
    w_sym = random.uniform(0, 1)
    w_refl = random.uniform(-1, 1)
    w_df = random.uniform(0, 2)
    w_q = random.uniform(0, 1)
    w_lbp = random.uniform(-1, 1)
    thresh = random.uniform(0, 2)
    
    tp, tn, fp, fn = 0, 0, 0, 0
    for p in parsed:
        score = p['sym'] * w_sym + p['refl'] * w_refl + p['df'] * w_df + p['q_inv'] * w_q + p['lbp'] * w_lbp
        is_live = score >= thresh
        if p['label'] == 1:
            if is_live: tp += 1
            else: fn += 1
        else:
            if is_live: fp += 1
            else: tn += 1
            
    acc = (tp + tn) / 72.0
    if acc > best_acc:
        best_acc = acc
        best_w = (w_sym, w_refl, w_df, w_q, w_lbp, thresh)
        if acc == 1.0: break

print(f"Acc: {best_acc:.4f}, W: {best_w}")
