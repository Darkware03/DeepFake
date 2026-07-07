import json, glob, random, math
import numpy as np
from sklearn.model_selection import KFold

def get_data():
    data = []
    for t, p in [(0, 'dataset/01_videos_ia/*/analysis/response.json'), 
                 (1, 'dataset/02_videos_reales/*/analysis/response.json')]:
        for path in glob.glob(p):
            with open(path, 'r') as f:
                j = json.load(f)
                data.append((t, j))
    return data

data = get_data()

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
    
    row = {
        'label': label,
        'deepfake_prob': resp.get('deepfake_probability', 0.5),
        'psd': min(1.0, avg_feat('psd')/100.0),
        'lbp': min(1.0, avg_feat('lbp')/255.0),
        'reflection': avg_feat('reflection_strength'),
        'color_match': avg_feat('expected_color_match'), # Using original for now
        'symmetry': sym,
        'quality': f.get('global_quality_score', 1.0),
    }
    parsed.append(row)

# Define evaluation
def eval_engine(w_df, w_cm, w_refl, w_tx, w_psd, w_sym, w_sr, w_q, thresh, df_alert, fold_data):
    tp, tn, fp, fn = 0, 0, 0, 0
    for p in fold_data:
        df_score = max(0.0, 1.0 - p['deepfake_prob'])
        skin_resp = p['reflection'] * p['color_match']
        
        physical_score = (p['color_match'] * w_cm + 
                          p['reflection'] * w_refl + 
                          p['lbp'] * w_tx + 
                          p['psd'] * w_psd + 
                          p['symmetry'] * w_sym + 
                          skin_resp * w_sr + 
                          p['quality'] * w_q +
                          df_score * w_df)
                          
        penalty = min(1.0, p['deepfake_prob']) if p['deepfake_prob'] > df_alert else (p['deepfake_prob'] * 0.5)
        final_score = max(0.0, physical_score * (1.0 - penalty))
        
        if final_score >= thresh:
            if p['label'] == 1: tp += 1
            else: fp += 1
        else:
            if p['label'] == 1: fn += 1
            else: tn += 1
            
    acc = (tp + tn) / len(fold_data) if len(fold_data) > 0 else 0
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    mcc_num = tp*tn - fp*fn
    mcc_den = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0 else 1
    mcc = mcc_num / mcc_den
    return acc, f1, mcc

# K-Fold optimization
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_mcc = -1
best_w = None

for _ in range(50000):
    weights = [random.uniform(0, 1) for _ in range(8)]
    s = sum(weights)
    w_df, w_cm, w_refl, w_tx, w_psd, w_sym, w_sr, w_q = [x/s for x in weights]
    
    thresh = random.uniform(0, 1)
    df_alert = random.uniform(0, 1)
    
    fold_mccs = []
    for train_idx, val_idx in kf.split(parsed):
        train_data = [parsed[i] for i in train_idx]
        val_data = [parsed[i] for i in val_idx]
        
        # We don't train, we just evaluate the random config on val
        acc, f1, mcc = eval_engine(w_df, w_cm, w_refl, w_tx, w_psd, w_sym, w_sr, w_q, thresh, df_alert, val_data)
        fold_mccs.append(mcc)
        
    avg_mcc = np.mean(fold_mccs)
    if avg_mcc > best_mcc:
        best_mcc = avg_mcc
        best_w = {
            'w_df': w_df, 'w_cm': w_cm, 'w_refl': w_refl, 'w_tx': w_tx, 
            'w_psd': w_psd, 'w_sym': w_sym, 'w_sr': w_sr, 'w_q': w_q,
            'thresh': thresh, 'df_alert': df_alert
        }

print(f"Best MCC: {best_mcc:.4f}")
print(json.dumps(best_w, indent=2))
