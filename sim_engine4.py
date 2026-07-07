import os
import json
import glob
import math
import statistics

ENGINE_CONFIG = {
    "weights": {
        "deepfake": 0.0,
        "color_match": 0.35,
        "reflection": 0.25,
        "texture": 0.15,
        "psd": 0.05,
        "symmetry": 0.05,
        "skin_response": 0.10,
        "quality": 0.05
    },
    "heuristics": {
        "psd_normalization_divisor": 100.0,
        "lbp_normalization_divisor": 255.0,
        "reflection_normalization_divisor": 15.0,
        "blur_threshold": 50.0,
        "brightness_min": 40.0,
        "brightness_max": 230.0
    },
    "thresholds": {
        "live_score": 0.60,
        "reflection_min": 0.50,
        "color_match_min": 0.70,
        "symmetry_min": 0.70,
        "quality_min": 0.80,
        "deepfake_alert": 0.60
    }
}

class WeightedDecisionEngine:
    def decide(self, features, deepfake_prob):
        w = ENGINE_CONFIG["weights"]
        div = ENGINE_CONFIG["heuristics"]
        thresh = ENGINE_CONFIG["thresholds"]
        
        df_score = max(0.0, 1.0 - deepfake_prob)
        
        def avg_feat(name):
            vals = [v for k, v in features.items() if name in k and isinstance(v, (int, float))]
            if not vals: return 0.0
            vals.sort()
            return vals[len(vals)//2]
            
        color_match = avg_feat("expected_color_match")
        reflection = avg_feat("reflection_strength")
        texture_lbp = min(1.0, avg_feat("lbp") / div["lbp_normalization_divisor"])
        psd = min(1.0, avg_feat("psd") / div["psd_normalization_divisor"]) 
        
        de_l = features.get("left_cheek_deltaE76", 0.0)
        de_r = features.get("right_cheek_deltaE76", 0.0)
        symmetry = max(0.0, 1.0 - abs(de_l - de_r) / max(de_l, de_r, 1.0))
        
        skin_resp = reflection * color_match
        quality = features.get("global_quality_score", 1.0)
        
        df_cont = 0.0
        cm_cont = color_match * w["color_match"]
        re_cont = reflection * w["reflection"]
        tx_cont = texture_lbp * w["texture"]
        ps_cont = psd * w["psd"]
        sy_cont = symmetry * w["symmetry"]
        sr_cont = skin_resp * w["skin_response"]
        qu_cont = quality * w["quality"]
        
        physical_score = cm_cont + re_cont + tx_cont + ps_cont + sy_cont + sr_cont + qu_cont
        
        penalty = min(1.0, deepfake_prob) if deepfake_prob > thresh["deepfake_alert"] else (deepfake_prob * 0.5)
        final_score = max(0.0, physical_score * (1.0 - penalty))
        
        is_live = final_score >= thresh["live_score"]
        
        return {"is_live": is_live, "final_score": final_score}

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
engine = WeightedDecisionEngine()

tp, tn, fp, fn = 0, 0, 0, 0
for label, resp in data:
    features = resp.get('challenge', {}).get('features', {})
    deepfake_prob = resp.get('deepfake_probability', 0.5)
    
    result = engine.decide(features, deepfake_prob)
    is_live = result['is_live']
    
    if label == 1:
        if is_live: tp += 1
        else: fn += 1
    else:
        if is_live: fp += 1
        else: tn += 1

acc = (tp + tn) / 72.0
print(f"Engine Test Acc: {acc:.4f} (TP:{tp} TN:{tn} FP:{fp} FN:{fn})")
