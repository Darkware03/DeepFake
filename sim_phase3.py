import json, glob, random
import numpy as np

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
    
    # Simulate the Color Match fix without re-running OpenCV, by approximating:
    # Actually we can't recalculate cosine_sim because we only have expected_color_match which is max(0, cos_sim).
    # We can't recover negative cos_sim. We would need to run ColorPlugin again!
    # Luckily I can read the original response to see if we have deltaE.
    pass

