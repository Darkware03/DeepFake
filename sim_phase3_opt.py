import json, glob, math, random
import numpy as np

data = []
for t, p in [(0, 'dataset/01_videos_ia/*/analysis/response.json'), 
             (1, 'dataset/02_videos_reales/*/analysis/response.json')]:
    for path in glob.glob(p):
        with open(path, 'r') as f:
            j = json.load(f)
            data.append((t, j))

parsed = []
for label, resp in data:
    f = resp.get('challenge', {}).get('features', {})
    
    def avg_feat(name):
        vals = [v for k, v in f.items() if name in k and isinstance(v, (int, float))]
        if not vals: return 0.0
        vals.sort()
        return vals[len(vals)//2]
    
    de_l = f.get("left_cheek_deltaE76", 0.0)
    de_r = f.get("right_cheek_deltaE76", 0.0)
    sym = max(0.0, 1.0 - abs(de_l - de_r) / max(de_l, de_r, 1.0)) if max(de_l, de_r, 1.0) > 0 else 0.0
    
    cm = avg_feat('expected_color_match') 
    refl = avg_feat('reflection_strength')
    lbp = min(1.0, avg_feat('lbp')/255.0)
    psd = min(1.0, avg_feat('psd')/100.0)
    q = max(0.0, 1.0 - f.get('global_quality_score', 1.0))
    df_prob = resp.get('deepfake_probability', 0.5)
    
    parsed.append({
        'label': label,
        'cm': cm,
        'refl': refl,
        'lbp': lbp,
        'psd': psd,
        'sym': sym,
        'q': q,
        'df': max(0.0, 1.0 - df_prob),
        'skin_resp': refl * cm
    })

best_acc = 0
best_mcc = -1
best_w = None

for _ in range(50000):
    weights = [random.uniform(0, 1) for _ in range(8)]
    s = sum(weights)
    w_cm, w_refl, w_tx, w_psd, w_sym, w_q, w_df, w_sr = [x/s for x in weights]
    
    thresh = random.uniform(0, 1)
    
    tp, tn, fp, fn = 0, 0, 0, 0
    for p in parsed:
        score = (p['cm'] * w_cm + 
                 p['refl'] * w_refl + 
                 p['lbp'] * w_tx + 
                 p['psd'] * w_psd + 
                 p['sym'] * w_sym + 
                 p['skin_resp'] * w_sr + 
                 p['q'] * w_q +
                 p['df'] * w_df)
                 
        if score >= thresh:
            if p['label'] == 1: tp += 1
            else: fp += 1
        else:
            if p['label'] == 1: fn += 1
            else: tn += 1
            
    acc = (tp + tn) / 72.0
    mcc_num = tp*tn - fp*fn
    mcc_den = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0 else 1
    mcc = mcc_num / mcc_den
    
    if mcc > best_mcc:
        best_mcc = mcc
        best_acc = acc
        best_w = {
            'w_cm': w_cm, 'w_refl': w_refl, 'w_tx': w_tx, 
            'w_psd': w_psd, 'w_sym': w_sym, 'w_sr': w_sr, 'w_q': w_q, 'w_df': w_df,
            'thresh': thresh, 'acc': acc, 'mcc': mcc, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }

print("Phase 3 Optimal Config:")
print(json.dumps(best_w, indent=2))
