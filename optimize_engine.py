import json, glob, random

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

best_acc = 0
best_cfg = None

for _ in range(50000):
    w_cm = random.uniform(0, 0.5) if random.random() > 0.5 else 0.0
    w_refl = random.uniform(-1.0, 1.0)
    w_tx = random.uniform(-0.5, 0.5)
    w_psd = random.uniform(-0.5, 0.5)
    w_sym = random.uniform(0.0, 1.0)
    w_sr = random.uniform(0.0, 0.5) if random.random() > 0.5 else 0.0
    w_q = random.uniform(-1.0, 1.0)
    w_df = random.uniform(-1.0, 1.0)
    
    thresh = random.uniform(0, 2)
    use_penalty = random.choice([True, False])
    
    tp, tn, fp, fn = 0, 0, 0, 0
    for label, resp in data:
        features = resp.get('challenge', {}).get('features', {})
        deepfake_prob = resp.get('deepfake_probability', 0.5)
        
        def avg_feat(name):
            vals = [v for k, v in features.items() if name in k and isinstance(v, (int, float))]
            if not vals: return 0.0
            vals.sort()
            return vals[len(vals)//2]
            
        color_match = avg_feat("expected_color_match")
        reflection = avg_feat("reflection_strength")
        texture_lbp = min(1.0, avg_feat("lbp") / 255.0)
        psd = min(1.0, avg_feat("psd") / 100.0)
        
        de_l = features.get("left_cheek_deltaE76", 0.0)
        de_r = features.get("right_cheek_deltaE76", 0.0)
        symmetry = max(0.0, 1.0 - abs(de_l - de_r) / max(de_l, de_r, 1.0))
        
        skin_resp = reflection * color_match
        quality = features.get("global_quality_score", 1.0)
        
        physical_score = (color_match * w_cm + 
                          reflection * w_refl + 
                          texture_lbp * w_tx + 
                          psd * w_psd + 
                          symmetry * w_sym + 
                          skin_resp * w_sr + 
                          quality * w_q +
                          deepfake_prob * w_df)
                          
        penalty = min(1.0, deepfake_prob) if (use_penalty and deepfake_prob > 0.6) else 0.0
        final_score = max(0.0, physical_score * (1.0 - penalty))
        
        if final_score >= thresh:
            if label == 1: tp += 1
            else: fp += 1
        else:
            if label == 1: fn += 1
            else: tn += 1
            
    acc = (tp + tn) / 72.0
    if acc > best_acc:
        best_acc = acc
        best_cfg = {
            'w_cm': w_cm, 'w_refl': w_refl, 'w_tx': w_tx, 'w_psd': w_psd, 
            'w_sym': w_sym, 'w_sr': w_sr, 'w_q': w_q, 'w_df': w_df,
            'thresh': thresh, 'use_penalty': use_penalty
        }
        if acc > 0.95: break

print(f"Best Acc: {best_acc:.4f}")
print(json.dumps(best_cfg, indent=2))
