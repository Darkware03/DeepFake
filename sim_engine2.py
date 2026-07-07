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

def test_engine():
    # We will compute 
    # sym: REAL=0.56, IA=0.46
    # quality: REAL=0.89, IA=0.94 -> (1 - quality) or just quality with negative weight
    # deepfake: REAL=0.57, IA=0.43 -> deepfake
    # reflection: REAL=0.09, IA=0.17 -> (1 - reflection)
    
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for label, resp in data:
        features = resp.get('challenge', {}).get('features', {})
        deepfake_prob = resp.get('deepfake_probability', 0.5)
        
        de_l = features.get("left_cheek_deltaE76", 0.0)
        de_r = features.get("right_cheek_deltaE76", 0.0)
        sym = 0.0
        if max(de_l, de_r, 1.0) > 0:
            sym = max(0.0, 1.0 - abs(de_l - de_r) / max(de_l, de_r, 1.0))
            
        refl = median_feat(features, 'reflection_strength')
        refl_inv = max(0.0, 1.0 - refl * 2.0)
        
        q = features.get("global_quality_score", 1.0)
        q_inv = max(0.0, 1.0 - q)
        
        # We can construct score:
        w_sym = 0.35
        w_refl = 0.25
        w_df = 0.30
        w_q = 0.10
        
        score = sym * w_sym + refl_inv * w_refl + deepfake_prob * w_df + q_inv * w_q
        
        thresh = 0.45
        
        is_live = score >= thresh
        if label == 1:
            if is_live: tp += 1
            else: fn += 1
        else:
            if is_live: fp += 1
            else: tn += 1
            
    acc = (tp + tn) / max(1, tp+tn+fp+fn)
    print(f"Acc: {acc:.4f}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

test_engine()
