import cv2
import numpy as np
import mediapipe as mp
import math
import time
from abc import ABC, abstractmethod

# ==========================================
# CONSTANTES Y VERSIONADO
# ==========================================
ENGINE_VERSION = "2.1.1"
FEATURE_SCHEMA_VERSION = "2.0.1"

ENGINE_CONFIG = {
    "weights": {
        "deepfake": 0.25,
        "color_match": 0.25,
        "reflection": 0.15,
        "texture": 0.10,
        "psd": 0.10,
        "symmetry": 0.05,
        "skin_response": 0.05,
        "quality": 0.05
    },
    "heuristics": {
        "psd_normalization_divisor": 100.0,
        "lbp_normalization_divisor": 255.0,
        "reflection_normalization_divisor": 15.0, # Delta L esperado razonable para un flash (15 puntos de luminosidad)
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

# ==========================================
# UTILS & ALGORITHMS
# ==========================================
def numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return [numpy_to_python(item) for item in obj.tolist()]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_python(item) for item in obj)
    return obj

def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return 0, 0, 0
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def deltaE_76(lab1, lab2):
    """
    Distancia Euclideana en el espacio LAB (CIE76).
    Nota: Esto NO es CIEDE2000. Es una métrica de distancia de color básica.
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    return math.sqrt((L2 - L1)**2 + (a2 - a1)**2 + (b2 - b1)**2)

from skimage.feature import local_binary_pattern

def compute_lbp(gray, mask=None):
    """
    Implementación nativa optimizada de LBP utilizando scikit-image (C-backend).
    Acelera el proceso original en ~13x conservando la estructura analítica.
    """
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)
        
    # Radius = 1, Neighbors = 8, default method
    lbp = local_binary_pattern(gray, 8, 1, method='default')
    
    if mask is not None:
        valid_pixels = lbp[mask > 0]
        return float(np.mean(valid_pixels)) if valid_pixels.size > 0 else 0.0
    return float(np.mean(lbp))

# ==========================================
# INTERFACES
# ==========================================
class DecisionEngine(ABC):
    @abstractmethod
    def decide(self, features, deepfake_prob):
        pass

class Plugin(ABC):
    @abstractmethod
    def process(self, img_n, img_c, mask_n, mask_c, expected_hex):
        pass

# ==========================================
# PLUGINS REALES
# ==========================================
class TexturePlugin(Plugin):
    def process(self, img_n, img_c, mask_n, mask_c, expected_hex):
        start = time.perf_counter()
        if img_n is None or img_n.size == 0 or img_c is None or img_c.size == 0:
            return {"valid": False}
        
        gray_n = cv2.cvtColor(img_n, cv2.COLOR_BGR2GRAY)
        
        # PSD
        f_n = np.fft.fft2(gray_n)
        fshift_n = np.fft.fftshift(f_n)
        mag_n = np.abs(fshift_n)
        h, w = mag_n.shape
        cy, cx = h//2, w//2
        mag_n[cy-5:cy+5, cx-5:cx+5] = 0
        psd_n = np.sum(np.log(mag_n + 1)**2) / (h*w)
        
        # LBP
        lbp_n = compute_lbp(gray_n, mask_n)
        
        elapsed = (time.perf_counter() - start) * 1000
        
        # Eliminadas métricas sin uso real (laplacian_var, fft_energy)
        return {
            "valid": True,
            "psd": float(psd_n),
            "lbp": float(lbp_n),
            "time_ms": elapsed
        }

class ColorPlugin(Plugin):
    def process(self, img_n, img_c, mask_n, mask_c, expected_hex):
        start = time.perf_counter()
        if img_n is None or img_n.size == 0 or img_c is None or img_c.size == 0:
            return {"valid": False}
            
        exp_r, exp_g, exp_b = hex_to_rgb(expected_hex or "#FFFFFF")
        
        mean_rgb_n = cv2.mean(img_n, mask=mask_n)[:3]
        mean_rgb_c = cv2.mean(img_c, mask=mask_c)[:3]
        
        lab_n = cv2.cvtColor(img_n, cv2.COLOR_BGR2LAB)
        lab_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2LAB)
        mean_lab_n = cv2.mean(lab_n, mask=mask_n)[:3]
        mean_lab_c = cv2.mean(lab_c, mask=mask_c)[:3]
        
        de76 = deltaE_76(mean_lab_n, mean_lab_c)
        
        # Cosine Similarity para Color Match
        # Vectores [R, G, B]
        v_exp = np.array([exp_r, exp_g, exp_b], dtype=np.float64)
        v_obs = np.array([
            mean_rgb_c[2] - mean_rgb_n[2],
            mean_rgb_c[1] - mean_rgb_n[1],
            mean_rgb_c[0] - mean_rgb_n[0]
        ], dtype=np.float64)
        
        norm_exp = np.linalg.norm(v_exp)
        norm_obs = np.linalg.norm(v_obs)
        
        if norm_exp == 0 or norm_obs == 0:
            color_match = 0.0
        else:
            cosine_sim = np.dot(v_exp, v_obs) / (norm_exp * norm_obs)
            color_match = max(0.0, float(cosine_sim))
            
        # Reflection Strength (Basado en Delta L de LAB normalizado a 15 unidades como ideal)
        delta_l = abs(mean_lab_c[0] - mean_lab_n[0])
        reflection_str = min(1.0, float(delta_l) / ENGINE_CONFIG["heuristics"]["reflection_normalization_divisor"])
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return {
            "valid": True,
            "deltaE76": float(de76),
            "expected_color_match": float(color_match),
            "reflection_strength": float(reflection_str),
            "time_ms": elapsed
        }

# ==========================================
# MOTOR DE DECISIÓN PONDERADO (HEURÍSTICO)
# ==========================================
class WeightedDecisionEngine(DecisionEngine):
    def decide(self, features, deepfake_prob):
        start = time.perf_counter()
        
        w = ENGINE_CONFIG["weights"]
        div = ENGINE_CONFIG["heuristics"]
        thresh = ENGINE_CONFIG["thresholds"]
        
        df_score = max(0.0, 1.0 - deepfake_prob)
        
        def avg_feat(name):
            vals = [v for k, v in features.items() if name in k and isinstance(v, (int, float))]
            return np.mean(vals) if vals else 0.0
            
        color_match = avg_feat("expected_color_match")
        reflection = avg_feat("reflection_strength")
        texture_lbp = min(1.0, avg_feat("lbp") / div["lbp_normalization_divisor"])
        psd = min(1.0, avg_feat("psd") / div["psd_normalization_divisor"]) 
        
        de_l = features.get("left_cheek_deltaE76", 0.0)
        de_r = features.get("right_cheek_deltaE76", 0.0)
        symmetry = max(0.0, 1.0 - abs(de_l - de_r) / max(de_l, de_r, 1.0))
        
        skin_resp = reflection * color_match
        quality = features.get("global_quality_score", 1.0)
        
        # Ponderación
        df_cont = df_score * w["deepfake"]
        cm_cont = color_match * w["color_match"]
        re_cont = reflection * w["reflection"]
        tx_cont = texture_lbp * w["texture"]
        ps_cont = psd * w["psd"]
        sy_cont = symmetry * w["symmetry"]
        sr_cont = skin_resp * w["skin_response"]
        qu_cont = quality * w["quality"]
        
        final_score = df_cont + cm_cont + re_cont + tx_cont + ps_cont + sy_cont + sr_cont + qu_cont
        is_live = final_score >= thresh["live_score"]
        
        evidence = []
        if df_score > thresh["deepfake_alert"]: evidence.append("✓ DeepFake model pass")
        else: evidence.append("✗ DeepFake anomalies detected")
            
        if color_match > thresh["color_match_min"]: evidence.append("✓ Expected challenge color detected")
        else: evidence.append("✗ Expected challenge color not matched")
            
        if reflection > thresh["reflection_min"]: evidence.append("✓ Natural reflection intensity")
        else: evidence.append("✗ Low reflection strength (Possible Screen)")
            
        if symmetry > thresh["symmetry_min"]: evidence.append("✓ Bilateral symmetry verified (Note: Beards/Shadows may affect this)")
        else: evidence.append("✗ Illumination asymmetric")
            
        elapsed = (time.perf_counter() - start) * 1000
        
        breakdown = {
            "deepfake": {"raw": round(df_score, 4), "weight": w["deepfake"], "contribution": round(df_cont, 4)},
            "color_match": {"raw": round(color_match, 4), "weight": w["color_match"], "contribution": round(cm_cont, 4)},
            "reflection": {"raw": round(reflection, 4), "weight": w["reflection"], "contribution": round(re_cont, 4)},
            "lbp": {"raw": round(texture_lbp, 4), "weight": w["texture"], "contribution": round(tx_cont, 4)},
            "psd": {"raw": round(psd, 4), "weight": w["psd"], "contribution": round(ps_cont, 4)},
            "symmetry": {"raw": round(symmetry, 4), "weight": w["symmetry"], "contribution": round(sy_cont, 4)},
            "skin_response": {"raw": round(skin_resp, 4), "weight": w["skin_response"], "contribution": round(sr_cont, 4)},
            "quality": {"raw": round(quality, 4), "weight": w["quality"], "contribution": round(qu_cont, 4)},
            "final_score": round(final_score, 4)
        }
        
        return {
            "is_live": is_live,
            "confidence": round(final_score, 4),
            "attack_probability": round(1.0 - final_score, 4),
            "decision": "LIVE" if is_live else "SPOOF",
            "risk": "LOW" if is_live else "HIGH",
            "classification": "NONE" if is_live else "UNKNOWN",
            "evidence": evidence,
            "decision_breakdown": breakdown,
            "time_ms": elapsed
        }

# ==========================================
# CALIDAD DE CAPTURA
# ==========================================
class QualityModule:
    @staticmethod
    def check_quality(img):
        div = ENGINE_CONFIG["heuristics"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        
        issues = []
        if blur < div["blur_threshold"]: issues.append("blur")
        if brightness < div["brightness_min"]: issues.append("underexposed")
        if brightness > div["brightness_max"]: issues.append("overexposed")
        
        q_score = 1.0
        if "blur" in issues: q_score -= 0.5
        if "underexposed" in issues or "overexposed" in issues: q_score -= 0.3
        
        return len(issues) == 0, issues, max(0.0, q_score)

# ==========================================
# PIPELINE PRINCIPAL
# ==========================================
class PADPipeline:
    def __init__(self):
        self.decision_engine = WeightedDecisionEngine()
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        self.regions = ["forehead", "nose", "left_cheek", "right_cheek", "chin"]
        
    def _extract_rois(self, img, results):
        if not results.multi_face_landmarks: return {}
        h, w = img.shape[:2]
        lms = results.multi_face_landmarks[0]
        
        def get_bbox(indices):
            pts = np.array([(int(lms.landmark[i].x * w), int(lms.landmark[i].y * h)) for i in indices])
            if len(pts) == 0: return None, None
            
            # Sanitizar bounding box
            x, y, bw, bh = cv2.boundingRect(pts)
            
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            
            if x2 <= x1 or y2 <= y1:
                return None, None
                
            roi = img[y1:y2, x1:x2]
            mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
            
            # Ajustar coordenadas relativas a la ROI y limitar para evitar crash
            roi_pts = pts - [x1, y1]
            roi_pts = np.clip(roi_pts, [0, 0], [roi.shape[1]-1, roi.shape[0]-1])
            
            hull = cv2.convexHull(roi_pts)
            cv2.fillConvexPoly(mask, hull, 255)
            return roi, mask
            
        return {
            "forehead": get_bbox([10, 338, 297, 332, 284, 251, 389, 356, 454, 323]),
            "left_cheek": get_bbox([234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]),
            "right_cheek": get_bbox([284, 251, 389, 356, 454, 323, 361, 288, 397, 365]),
            "nose": get_bbox([1, 2, 98, 327, 168, 197, 195, 5, 4]),
            "chin": get_bbox([152, 148, 176, 149, 150, 136, 172, 58, 132, 93])
        }

    def analyze(self, normal_img, challenge_img, expected_hex_color):
        ok_n, iss_n, q_n = QualityModule.check_quality(normal_img)
        ok_c, iss_c, q_c = QualityModule.check_quality(challenge_img)
        
        res_n = self.mp_face_mesh.process(cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB))
        res_c = self.mp_face_mesh.process(cv2.cvtColor(challenge_img, cv2.COLOR_BGR2RGB))
        
        if not res_n.multi_face_landmarks or not res_c.multi_face_landmarks:
            return {"valid": False, "reason": "face_not_found"}
            
        rois_n = self._extract_rois(normal_img, res_n)
        rois_c = self._extract_rois(challenge_img, res_c)
        
        plugins = [TexturePlugin(), ColorPlugin()]
        flat_features = {
            "global_quality_score": (q_n + q_c) / 2.0
        }
        
        for region in self.regions:
            r_img_n, r_mask_n = rois_n.get(region, (None, None))
            r_img_c, r_mask_c = rois_c.get(region, (None, None))
            
            for p in plugins:
                res = p.process(r_img_n, r_img_c, r_mask_n, r_mask_c, expected_hex_color)
                if res.get("valid"):
                    for k, v in res.items():
                        if k != "valid" and k != "time_ms" and isinstance(v, (int, float)):
                            flat_features[f"{region}_{k}"] = v

        res = {
            "valid": True,
            "features": flat_features
        }
        return numpy_to_python(res)

# ==========================================
# EXPORTACIONES PARA main.py
# ==========================================
pipeline = PADPipeline()

def analyze_reflection(normal_img, challenge_img, expected_hex_color):
    return pipeline.analyze(normal_img, challenge_img, expected_hex_color)

def decide_liveness(deepfake_prob, challenge_data):
    if challenge_data and challenge_data.get("valid"):
        features = challenge_data.get("features", {})
        result = pipeline.decision_engine.decide(features, deepfake_prob)
        
        result["engine_version"] = ENGINE_VERSION
        result["feature_schema_version"] = FEATURE_SCHEMA_VERSION
        result["decision_engine"] = pipeline.decision_engine.__class__.__name__
        result["analysis"] = result.get("evidence", [])
        result["recommendations"] = ["Manual review"] if result["risk"] != "LOW" else []
        
        return numpy_to_python(result)
    else:
        is_live = deepfake_prob <= 0.7
        res = {
            "is_live": is_live,
            "confidence": round(1.0 - deepfake_prob, 4),
            "attack_probability": round(deepfake_prob, 4),
            "decision": "LIVE" if is_live else "SPOOF",
            "risk": "HIGH" if deepfake_prob > 0.7 else "LOW",
            "classification": "DEEPFAKE" if deepfake_prob > 0.7 else "NONE",
            "analysis": ["Only DeepFake analysis performed. No physical challenge provided."],
            "recommendations": ["Execute a physical challenge for better security."],
            "engine_version": ENGINE_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "decision_engine": pipeline.decision_engine.__class__.__name__
        }
        return numpy_to_python(res)
