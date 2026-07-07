import os
import json
import glob
from app.liveness_engine import WeightedDecisionEngine

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
    
    # Run the engine
    result = engine.decide(features, deepfake_prob)
    is_live = result['is_live']
    
    if label == 1:
        if is_live: tp += 1
        else: fn += 1
    else:
        if is_live: fp += 1
        else: tn += 1

acc = (tp + tn) / 72.0
print(f"Original Engine Acc via script: {acc:.4f} (TP:{tp} TN:{tn} FP:{fp} FN:{fn})")

