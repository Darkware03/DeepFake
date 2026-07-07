import json, glob, math
import numpy as np
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

data = []
for t, p in [(0, 'dataset/01_videos_ia/*/analysis/response.json'), 
             (1, 'dataset/02_videos_reales/*/analysis/response.json')]:
    for path in glob.glob(p):
        with open(path, 'r') as f:
            j = json.load(f)
            data.append((t, j))

tp, tn, fp, fn = 0, 0, 0, 0

w_cm = 0.15
w_refl = 0.10
w_tx = 0.05
w_psd = 0.20
w_sym = 0.35
w_sr = 0.05
w_q = 0.10
w_df = 0.0

thresh = 0.25
df_alert = 0.60

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
    
    # We can't perfectly recover cosine_sim without running the image.
    # We'll just use the old expected_color_match + 0.5 to simulate the mathematical fix roughly for evaluation.
    # Actually, ColorPlugin does: cosine_sim = np.dot... / ...
    # old_color_match = max(0.0, float(cosine_sim))
    # We don't have the negative values. We'll just use old_color_match for now in simulation, 
    # but we'll apply the fix in the engine anyway.
    cm = avg_feat('expected_color_match') 
    
    refl = avg_feat('reflection_strength')
    lbp = min(1.0, avg_feat('lbp')/255.0)
    psd = min(1.0, avg_feat('psd')/100.0)
    q = max(0.0, 1.0 - f.get('global_quality_score', 1.0))
    df_prob = resp.get('deepfake_probability', 0.5)
    
    skin_resp = refl * cm
    
    physical_score = (cm * w_cm + 
                      refl * w_refl + 
                      lbp * w_tx + 
                      psd * w_psd + 
                      sym * w_sym + 
                      skin_resp * w_sr + 
                      q * w_q +
                      max(0.0, 1.0 - df_prob) * w_df)
                      
    penalty = 0.0
    final_score = max(0.0, physical_score * (1.0 - penalty))
    
    if final_score >= thresh:
        if label == 1: tp += 1
        else: fp += 1
    else:
        if label == 1: fn += 1
        else: tn += 1

acc = (tp + tn) / 72.0
prec = tp / (tp + fp) if tp + fp > 0 else 0
rec = tp / (tp + fn) if tp + fn > 0 else 0
f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
mcc_num = tp*tn - fp*fn
mcc_den = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0 else 1
mcc = mcc_num / mcc_den

print(f"Accuracy: {acc:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
