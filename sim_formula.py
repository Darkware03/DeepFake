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

scores_ia = []
scores_real = []

for label, resp in data:
    f = resp.get('challenge', {}).get('features', {})
    sym = 0.0
    de_l = f.get("left_cheek_deltaE76", 0.0)
    de_r = f.get("right_cheek_deltaE76", 0.0)
    if max(de_l, de_r, 1.0) > 0:
        sym = max(0.0, 1.0 - abs(de_l - de_r) / max(de_l, de_r, 1.0))
        
    refl = avg_feat(f, "reflection_strength")
    q = f.get("global_quality_score", 1.0)
    df = resp.get('deepfake_probability', 0.5)
    
    score = sym * 1.0 + (1.0 - refl) * 0.5 + (1.0 - q) * 0.5
    
    if label == 1: scores_real.append(score)
    else: scores_ia.append(score)

scores_ia.sort()
scores_real.sort()

print("IA: ", scores_ia)
print("REAL:", scores_real)
