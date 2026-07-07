import json, glob

data = []
for t, p in [(0, 'dataset/01_videos_ia/*/analysis/response.json'), 
             (1, 'dataset/02_videos_reales/*/analysis/response.json')]:
    for path in glob.glob(p):
        with open(path, 'r') as f:
            j = json.load(f)
            data.append((t, j))

def avg_feat(f, name):
    vals = [v for k, v in f.items() if name in k and isinstance(v, (int, float))]
    if not vals: return 0.0
    vals.sort()
    return vals[len(vals)//2]

best_acc = 0
best_params = None

for w_sym in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    for w_refl in [0.0, 0.2, 0.4, 0.6, 0.8]:
        for w_q in [0.0, 0.2, 0.4, 0.6]:
            for w_df in [0.0, 0.2, 0.4, 0.6]:
                for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                    
                    tp, tn, fp, fn = 0, 0, 0, 0
                    for label, resp in data:
                        f = resp.get('challenge', {}).get('features', {})
                        df = resp.get('deepfake_probability', 0.5)
                        
                        sym = 0.0
                        de_l = f.get("left_cheek_deltaE76", 0.0)
                        de_r = f.get("right_cheek_deltaE76", 0.0)
                        if max(de_l, de_r, 1.0) > 0:
                            sym = max(0.0, 1.0 - abs(de_l - de_r) / max(de_l, de_r, 1.0))
                            
                        refl = avg_feat(f, "reflection_strength")
                        q = f.get("global_quality_score", 1.0)
                        
                        # Apply new logic
                        re_cont = max(0.0, 1.0 - refl) * w_refl
                        sy_cont = sym * w_sym
                        qu_cont = max(0.0, 1.0 - q) * w_q
                        df_cont = df * w_df  # Because REAL has higher df prob
                        
                        score = sy_cont + re_cont + qu_cont + df_cont
                        
                        if score >= thresh:
                            if label == 1: tp += 1
                            else: fp += 1
                        else:
                            if label == 1: fn += 1
                            else: tn += 1
                            
                    acc = (tp + tn) / 72.0
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (w_sym, w_refl, w_q, w_df, thresh)

print(f"Best Acc: {best_acc:.4f}")
print(f"w_sym: {best_params[0]}, w_refl: {best_params[1]}, w_q: {best_params[2]}, w_df: {best_params[3]}, thresh: {best_params[4]}")
