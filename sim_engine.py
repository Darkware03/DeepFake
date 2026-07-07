import os
import json
import glob
import math

def get_data():
    data = []
    ia_paths = glob.glob(os.path.join('dataset/01_videos_ia', '*', 'analysis', 'response.json'))
    for p in ia_paths:
        with open(p, 'r') as f:
            j = json.load(f)
            data.append((0, j))
            
    real_paths = glob.glob(os.path.join('dataset/02_videos_reales', '*', 'analysis', 'response.json'))
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

for W_SYM in [0.0, 0.4, 0.6, 0.8, 1.0]:
    for W_REFL in [0.0, -0.4, -0.6]: # negative weight for reflection
        for W_DF in [0.0, -0.5, 0.5]:
            # we can test combinations
            pass

def test_engine(w_sym, w_refl, w_df, w_cm, thresh):
    tp, tn, fp, fn = 0, 0, 0, 0
    for label, resp in data:
        features = resp.get('challenge', {}).get('features', {})
        deepfake_prob = resp.get('deepfake_probability', 0.5)
        
        # calculate
        sym = 0.0
        de_l = features.get("left_cheek_deltaE76", 0.0)
        de_r = features.get("right_cheek_deltaE76", 0.0)
        if max(de_l, de_r, 1.0) > 0:
            sym = max(0.0, 1.0 - abs(de_l - de_r) / max(de_l, de_r, 1.0))
            
        refl = median_feat(features, 'reflection_strength')
        cm = median_feat(features, 'expected_color_match')
        
        score = sym * w_sym + refl * w_refl + deepfake_prob * w_df + cm * w_cm
        
        is_live = score >= thresh
        if label == 1:
            if is_live: tp += 1
            else: fn += 1
        else:
            if is_live: fp += 1
            else: tn += 1
            
    acc = (tp + tn) / max(1, tp+tn+fp+fn)
    return acc, tp, tn, fp, fn

best_acc = 0
best_params = None

for w_sym in [0.3, 0.5, 0.7, 0.9]:
    for w_refl in [-0.5, -0.3, 0.0, 0.3]:
        for w_df in [-0.5, 0.0, 0.5]:
            for w_cm in [-0.3, 0.0, 0.3]:
                for thresh in [-0.5, 0.0, 0.2, 0.4, 0.6, 0.8]:
                    acc, tp, tn, fp, fn = test_engine(w_sym, w_refl, w_df, w_cm, thresh)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (w_sym, w_refl, w_df, w_cm, thresh, tp, tn, fp, fn)

print(f"Best Accuracy: {best_acc:.4f}")
print(f"W_SYM: {best_params[0]}, W_REFL: {best_params[1]}, W_DF: {best_params[2]}, W_CM: {best_params[3]}, THRESH: {best_params[4]}")
print(f"TP: {best_params[5]}, TN: {best_params[6]}, FP: {best_params[7]}, FN: {best_params[8]}")
