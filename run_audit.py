import os
import json
import glob
import math

def load_data(base_path):
    data = []
    ia_paths = glob.glob(os.path.join(base_path, '01_videos_ia', '*', 'analysis'))
    for p in ia_paths:
        data.append(parse_folder(p, label=0, type_label='IA'))
    
    real_paths = glob.glob(os.path.join(base_path, '02_videos_reales', '*', 'analysis'))
    for p in real_paths:
        data.append(parse_folder(p, label=1, type_label='REAL'))
        
    return [d for d in data if d is not None]

def parse_folder(analysis_path, label, type_label):
    response_file = os.path.join(analysis_path, 'response.json')
    if not os.path.exists(response_file): return None
    
    try:
        with open(response_file, 'r') as f:
            resp = json.load(f)
            
        record = {
            'label': label,
            'type': type_label,
            'response': resp,
            'path': analysis_path
        }
        return record
    except Exception as e:
        return None

def calc_stats(values):
    if not values: return {}
    n = len(values)
    values.sort()
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std_dev = math.sqrt(variance)
    
    def percentile(p):
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c: return values[int(k)]
        d0 = values[int(f)] * (c - k)
        d1 = values[int(c)] * (k - f)
        return d0 + d1

    return {
        'mean': mean,
        'median': percentile(0.5),
        'min': values[0],
        'max': values[-1],
        'std_dev': std_dev,
        'variance': variance,
        'p25': percentile(0.25),
        'p50': percentile(0.50),
        'p75': percentile(0.75),
        'p95': percentile(0.95),
    }

def run_analysis():
    data = load_data('dataset')
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    variables = {
        'IA': {},
        'REAL': {}
    }
    
    def add_var(t, name, val):
        if val is None: return
        if name not in variables[t]:
            variables[t][name] = []
        variables[t][name].append(val)
    
    for row in data:
        t = row['type']
        resp = row['response']
        
        # Base variables
        add_var(t, 'Deepfake Probability', resp.get('deepfake_probability'))
        add_var(t, 'Average Probability', resp.get('average_probability'))
        add_var(t, 'Max Probability', resp.get('max_probability'))
        add_var(t, 'P95', resp.get('p95_probability'))
        
        liveness = resp.get('liveness', {})
        add_var(t, 'Confidence', liveness.get('confidence'))
        add_var(t, 'Attack Probability', liveness.get('attack_probability'))
        
        bd = liveness.get('decision_breakdown', {})
        add_var(t, 'Final Score', bd.get('final_score'))
        
        # Breakdown raw values
        for k in ['deepfake', 'color_match', 'reflection', 'lbp', 'psd', 'symmetry', 'skin_response', 'quality']:
            if k in bd and isinstance(bd[k], dict):
                add_var(t, k.capitalize(), bd[k].get('raw'))
                add_var(t, k.capitalize() + ' Weight', bd[k].get('weight'))
                add_var(t, k.capitalize() + ' Contribution', bd[k].get('contribution'))
                
        # Also let's extract the actual metrics to do confusion matrix
        is_live = liveness.get('is_live', False)
        
        if row['label'] == 1: # REAL
            if is_live:
                TP += 1
            else:
                FN += 1
        else: # IA
            if is_live:
                FP += 1
            else:
                TN += 1

    P = TP + FN
    N = TN + FP
    
    accuracy = (TP + TN) / (P + N) if (P + N) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / P if P > 0 else 0
    specificity = TN / N if N > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    far = FP / N if N > 0 else 0
    frr = FN / P if P > 0 else 0
    balanced_acc = (recall + specificity) / 2
    
    try:
        mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    except:
        mcc = 0

    results = {
        'count': len(data),
        'metrics': {
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'Balanced_Accuracy': balanced_acc,
            'F1': f1,
            'FAR': far,
            'FRR': frr,
            'MCC': mcc
        },
        'variables': {}
    }
    
    all_var_names = set(list(variables['IA'].keys()) + list(variables['REAL'].keys()))
    for v_name in all_var_names:
        results['variables'][v_name] = {
            'IA': calc_stats(variables['IA'].get(v_name, [])),
            'REAL': calc_stats(variables['REAL'].get(v_name, []))
        }

    with open('baseline_audit.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(json.dumps(results['metrics'], indent=2))

if __name__ == '__main__':
    run_analysis()
