import os, json, glob, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc, mutual_info_score
from sklearn.feature_selection import mutual_info_classif, f_classif
import warnings
warnings.filterwarnings('ignore')

os.makedirs("plots_phase2", exist_ok=True)

# 1. LOAD DATA
data = []
for t, p in [(0, 'dataset/01_videos_ia/*/analysis/response.json'), 
             (1, 'dataset/02_videos_reales/*/analysis/response.json')]:
    for path in glob.glob(p):
        with open(path, 'r') as f:
            j = json.load(f)
            data.append((t, j))

rows = []
for label, resp in data:
    f = resp.get('challenge', {}).get('features', {})
    
    def avg_feat(name):
        vals = [v for k, v in f.items() if name in k and isinstance(v, (int, float))]
        if not vals: return 0.0
        vals.sort()
        return vals[len(vals)//2]
    
    de_l = f.get("left_cheek_deltaE76", 0.0)
    de_r = f.get("right_cheek_deltaE76", 0.0)
    sym = max(0.0, 1.0 - abs(de_l - de_r) / max(de_l, de_r, 1.0)) if max(de_l, de_r, 1.0) > 0 else 0.0
    
    liveness = resp.get('liveness', {})
    bd = liveness.get('decision_breakdown', {})
    
    row = {
        'label': label,
        'deepfake_probability': resp.get('deepfake_probability', 0.5),
        'average_probability': resp.get('average_probability', 0.5),
        'p95_probability': resp.get('p95_probability', 0.5),
        'max_probability': resp.get('max_probability', 0.5),
        'psd': avg_feat('psd'),
        'lbp': avg_feat('lbp'),
        'reflection': avg_feat('reflection_strength'),
        'color_match': avg_feat('expected_color_match'),
        'deltaE76': avg_feat('deltaE76'),
        'symmetry': sym,
        'quality_score': f.get('global_quality_score', 1.0),
        'confidence': liveness.get('confidence', 0.0),
        'attack_probability': liveness.get('attack_probability', 0.0),
        'final_score': bd.get('final_score', 0.0),
    }
    # We can also get breakdown raw values if needed
    rows.append(row)

df = pd.DataFrame(rows)
features = [c for c in df.columns if c != 'label']

# 2. STATS & SEPARABILITY
def compute_stats(s):
    n = len(s)
    mean = np.mean(s)
    sem = stats.sem(s)
    ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem) if sem > 0 else (mean, mean)
    return {
        'Mean': mean,
        'Median': np.median(s),
        'Std': np.std(s),
        'Min': np.min(s),
        'Max': np.max(s),
        'P5': np.percentile(s, 5),
        'P25': np.percentile(s, 25),
        'P75': np.percentile(s, 75),
        'P95': np.percentile(s, 95),
        'P99': np.percentile(s, 99),
        'CI95': ci
    }

def cohend(d1, d2):
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (np.mean(d1) - np.mean(d2)) / s if s > 0 else 0

results = {}
for col in features:
    s_ia = df[df['label'] == 0][col].values
    s_real = df[df['label'] == 1][col].values
    
    fpr, tpr, _ = roc_curve(df['label'], df[col])
    auc_val = auc(fpr, tpr)
    if auc_val < 0.5:
        fpr, tpr, _ = roc_curve(df['label'], -df[col])
        auc_val = auc(fpr, tpr)
        
    mi = mutual_info_classif(df[[col]], df['label'], discrete_features=False, random_state=42)[0]
    f_val, p_val = f_classif(df[[col]], df['label'])
    
    pb_corr, _ = stats.pointbiserialr(df['label'], df[col])
    pearson, _ = stats.pearsonr(df['label'], df[col])
    spearman, _ = stats.spearmanr(df['label'], df[col])
    
    cd = cohend(s_real, s_ia)
    
    if auc_val > 0.8: strength = "Muy discriminativa"
    elif auc_val > 0.7: strength = "Discriminativa"
    elif auc_val > 0.6: strength = "Moderada"
    elif auc_val > 0.55: strength = "Débil"
    else: strength = "Ruido"
    
    results[col] = {
        'IA': compute_stats(s_ia),
        'REAL': compute_stats(s_real),
        'AUC': auc_val,
        'Cohen_d': cd,
        'Mutual_Info': mi,
        'F_score': f_val[0],
        'PB_Corr': pb_corr,
        'Spearman': spearman,
        'Strength': strength
    }
    
    # Visuals
    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1)
    sns.kdeplot(s_ia, label="IA", fill=True)
    sns.kdeplot(s_real, label="REAL", fill=True)
    plt.title(f"{col} KDE")
    
    plt.subplot(1,3,2)
    sns.boxplot(x=df['label'], y=df[col])
    plt.title("Boxplot")
    
    plt.subplot(1,3,3)
    plt.plot(fpr, tpr, label=f'AUC={auc_val:.2f}')
    plt.plot([0,1],[0,1],'--')
    plt.title("ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots_phase2/{col}_plot.png")
    plt.close()

# Correlaciones
corr = df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.savefig("plots_phase2/correlation_heatmap.png")
plt.close()

# Generate Markdown
md = ["# 1. Resumen Ejecutivo\nEste reporte presenta la auditoría matemática rigurosa exigida para evaluar la separabilidad de las features, sin asumir prejuicios previos.\n"]

# Feature Ranking
ranking = sorted([(k, v['AUC'], v['Strength']) for k, v in results.items()], key=lambda x: x[1], reverse=True)
md.append("# 2. Ranking completo de Features (Separabilidad)\n")
md.append("| Feature | AUC | Cohen's d | Mutual Info | Categoría |")
md.append("|---|---|---|---|---|")
for k, v in results.items():
    md.append(f"| {k} | {v['AUC']:.4f} | {v['Cohen_d']:.4f} | {v['Mutual_Info']:.4f} | {v['Strength']} |")
md.append("\n")

# Deep Dive Sections
def format_stats(st):
    return f"Media: {st['Mean']:.4f} | Mediana: {st['Median']:.4f} | Min: {st['Min']:.4f} | Max: {st['Max']:.4f} | CI95: ({st['CI95'][0]:.4f}, {st['CI95'][1]:.4f})"

md.append("# 3. Verificación de Xception (Deepfake Probability)\n")
md.append("¿Está Xception invertido?\n")
md.append(f"**IA Stats:** {format_stats(results['deepfake_probability']['IA'])}\n")
md.append(f"**REAL Stats:** {format_stats(results['deepfake_probability']['REAL'])}\n")
md.append(f"AUC: {results['deepfake_probability']['AUC']:.4f}\n")
md.append("Demostración: Si la media de REAL es MAYOR que la de IA, entonces el modelo está clasificando a los reales como más 'fake' que la propia IA.\n")

md.append("# 4. Verificación de Color Match\n")
cm_ia = df[df['label']==0]['color_match']
cm_real = df[df['label']==1]['color_match']
md.append(f"Ceros en IA: {(cm_ia == 0).sum()} de {len(cm_ia)} ({(cm_ia == 0).mean()*100:.1f}%)\n")
md.append(f"Ceros en REAL: {(cm_real == 0).sum()} de {len(cm_real)} ({(cm_real == 0).mean()*100:.1f}%)\n")
md.append(f"AUC: {results['color_match']['AUC']:.4f}. Esto demuestra si realmente sirve o si el exceso de ceros destruye la señal.\n")

md.append("# 5. Verificación de PSD y LBP\n")
md.append(f"PSD AUC: {results['psd']['AUC']:.4f}. Categoría: {results['psd']['Strength']}\n")
md.append(f"LBP AUC: {results['lbp']['AUC']:.4f}. Categoría: {results['lbp']['Strength']}\n")

md.append("# 6. Verificación de Reflection y Symmetry\n")
md.append(f"Reflection AUC: {results['reflection']['AUC']:.4f}. IA Media: {results['reflection']['IA']['Mean']:.4f} | REAL Media: {results['reflection']['REAL']['Mean']:.4f}\n")
md.append(f"Symmetry AUC: {results['symmetry']['AUC']:.4f}. IA Media: {results['symmetry']['IA']['Mean']:.4f} | REAL Media: {results['symmetry']['REAL']['Mean']:.4f}\n")

md.append("# 7. Verificación de Quality\n")
md.append(f"Quality AUC: {results['quality_score']['AUC']:.4f}. IA Media: {results['quality_score']['IA']['Mean']:.4f} | REAL Media: {results['quality_score']['REAL']['Mean']:.4f}\n")

with open('analysis_report_phase2.md', 'w') as f:
    f.write("\n".join(md))

print("Phase 2 Audit completed.")
