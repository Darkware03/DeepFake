import json, glob
import numpy as np
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

data = []
for t, p in [(0, 'dataset/01_videos_ia/*/analysis/response.json'), 
             (1, 'dataset/02_videos_reales/*/analysis/response.json')]:
    for path in glob.glob(p):
        with open(path, 'r') as f:
            j = json.load(f)
            data.append((t, j))

y_true = []
y_q = []
y_q_inv = []

for label, resp in data:
    f = resp.get('challenge', {}).get('features', {})
    q = f.get('global_quality_score', 1.0)
    y_true.append(label)
    y_q.append(q)
    y_q_inv.append(1.0 - q)

fpr, tpr, _ = roc_curve(y_true, y_q)
auc_q = auc(fpr, tpr)

fpr, tpr, _ = roc_curve(y_true, y_q_inv)
auc_q_inv = auc(fpr, tpr)

print(f"Quality AUC: {auc_q:.4f}")
print(f"Inverse Quality AUC: {auc_q_inv:.4f}")
