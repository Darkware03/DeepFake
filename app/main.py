import os
import sys
import shutil
import cv2
import numpy as np
import torch
from torchvision import transforms

# ==========================================================
# PYTHON PATH
# ==========================================================
RAIZ_WORKSPACE = "/workspace"

if RAIZ_WORKSPACE not in sys.path:
    sys.path.insert(0, RAIZ_WORKSPACE)

# ==========================================================
# FASTAPI
# ==========================================================
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

# ==========================================================
# DEEPFAKEBENCH
# ==========================================================
from training.networks.xception import Xception

# ==========================================================
# APP
# ==========================================================
app = FastAPI(
    title="DeepFake Detection API",
    description="API basada en DeepFakeBench - Inferencia real con Xception",
    version="1.0.0"
)

# ==========================================================
# CONFIG
# ==========================================================
PATH_PESOS = "/workspace/weights/xception.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = None
preprocess_transform = None

# ==========================================================
# STARTUP
# ==========================================================
@app.on_event("startup")
async def load_model():
    global model, preprocess_transform

    print("=" * 80)
    print(f"[*] DEVICE: {DEVICE}")
    print(f"[*] PESOS: {PATH_PESOS}")
    print("=" * 80)

    if not os.path.exists(PATH_PESOS):
        raise RuntimeError(
            f"No existe el archivo de pesos: {PATH_PESOS}"
        )

    try:
        # Configuración real extraída de xception.yaml
        xception_config = {
            "mode": "original",
            "num_classes": 2,
            "inc": 3,
            "dropout": False
        }

        print("[*] Creando arquitectura Xception...")
        model = Xception(xception_config)

        print("[*] Cargando checkpoint...")
        checkpoint = torch.load(PATH_PESOS, map_location=DEVICE)

        # Si el checkpoint fue guardado desde el XceptionDetector, las llaves empiezan con "backbone."
        # Removemos "backbone." para poder cargarlo directamente en la red Xception.
        if isinstance(checkpoint, dict):
            # Extraer state_dict si viene envuelto
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "model" in checkpoint:
                checkpoint = checkpoint["model"]
            elif "net" in checkpoint:
                checkpoint = checkpoint["net"]

            # Limpiar "backbone." si existe
            first_key = list(checkpoint.keys())[0]
            if first_key.startswith("backbone."):
                print("[*] Limpiando prefijo 'backbone.' de las llaves del checkpoint...")
                checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}

        print("[*] Aplicando pesos...")
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)

        print(f"[*] Missing keys: {len(missing)}")
        print(f"[*] Unexpected keys: {len(unexpected)}")
        if unexpected:
            print(f"[*] Unexpected keys sample: {unexpected[:5]}")

        model.to(DEVICE)
        model.eval()

        print("[*] Inicializando transformaciones de inferencia...")
        # Transformación idéntica a DeepFakeBench: ToTensor() + Normalize(0.5, 0.5)
        preprocess_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        print("[+] MODELO CARGADO CORRECTAMENTE")

    except Exception as e:
        print("=" * 80)
        print("ERROR CARGANDO MODELO")
        print(str(e))
        print("=" * 80)
        raise

# ==========================================================
# HEALTH
# ==========================================================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": model is not None
    }

# ==========================================================
# LIGHT CHALLENGE VALIDATION
# ==========================================================
def validate_light_challenge(challenge_path: str, normal_path: str) -> dict:
    normal_img = cv2.imread(normal_path)
    challenge_img = cv2.imread(challenge_path)

    if normal_img is None or challenge_img is None:
        return {"valid": False, "reason": "image_not_loaded"}

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return {"valid": False, "reason": "face_not_found"}

    x, y, w, h = faces[0]

    # Reducir automáticamente 15% por cada lado
    x_new = int(x + 0.15 * w)
    y_new = int(y + 0.15 * h)
    w_new = int(w * 0.70)
    h_new = int(h * 0.70)

    # Validar boundaries
    if x_new < 0: x_new = 0
    if y_new < 0: y_new = 0
    if x_new + w_new > normal_img.shape[1]: w_new = normal_img.shape[1] - x_new
    if y_new + h_new > normal_img.shape[0]: h_new = normal_img.shape[0] - y_new

    w_half = w_new // 2

    normal_hsv = cv2.cvtColor(normal_img, cv2.COLOR_BGR2HSV)
    challenge_hsv = cv2.cvtColor(challenge_img, cv2.COLOR_BGR2HSV)

    left_normal_roi = normal_hsv[y_new:y_new+h_new, x_new:x_new+w_half]
    right_normal_roi = normal_hsv[y_new:y_new+h_new, x_new+w_half:x_new+w_new]

    left_challenge_roi = challenge_hsv[y_new:y_new+h_new, x_new:x_new+w_half]
    right_challenge_roi = challenge_hsv[y_new:y_new+h_new, x_new+w_half:x_new+w_new]

    def get_mean(roi):
        if roi.size > 0:
            m = cv2.mean(roi)
            return m[0], m[1], m[2]
        return 0.0, 0.0, 0.0

    m_nl_h, m_nl_s, m_nl_v = get_mean(left_normal_roi)
    m_nr_h, m_nr_s, m_nr_v = get_mean(right_normal_roi)

    m_cl_h, m_cl_s, m_cl_v = get_mean(left_challenge_roi)
    m_cr_h, m_cr_s, m_cr_v = get_mean(right_challenge_roi)

    dl_h = abs(m_cl_h - m_nl_h)
    dl_s = abs(m_cl_s - m_nl_s)
    dl_v = abs(m_cl_v - m_nl_v)

    dr_h = abs(m_cr_h - m_nr_h)
    dr_s = abs(m_cr_s - m_nr_s)
    dr_v = abs(m_cr_v - m_nr_v)

    score = min(((dl_h + dr_h) / 180.0 + (dl_s + dr_s) / 255.0 + (dl_v + dr_v) / 255.0) * 0.5, 1.0)

    print("====================================================")
    print("LIGHT CHALLENGE")
    print("====================================================")
    print("Face detected")
    print(f"ROI Left HSV Normal: [{m_nl_h:.2f}, {m_nl_s:.2f}, {m_nl_v:.2f}]")
    print(f"ROI Right HSV Normal: [{m_nr_h:.2f}, {m_nr_s:.2f}, {m_nr_v:.2f}]")
    print(f"ROI Left HSV Challenge: [{m_cl_h:.2f}, {m_cl_s:.2f}, {m_cl_v:.2f}]")
    print(f"ROI Right HSV Challenge: [{m_cr_h:.2f}, {m_cr_s:.2f}, {m_cr_v:.2f}]")
    print(f"Delta Hue (L, R): {dl_h:.2f}, {dr_h:.2f}")
    print(f"Delta Saturation (L, R): {dl_s:.2f}, {dr_s:.2f}")
    print(f"Delta Value (L, R): {dl_v:.2f}, {dr_v:.2f}")
    print(f"Challenge Score: {score:.2f}")
    print("====================================================")

    return {
        "valid": True,
        "score": round(score, 2),
        "delta_hue": round((dl_h + dr_h)/2, 2),
        "delta_saturation": round((dl_s + dr_s)/2, 2),
        "delta_value": round((dl_v + dr_v)/2, 2),
        "left_roi": {
            "normal_hsv": [round(m_nl_h, 2), round(m_nl_s, 2), round(m_nl_v, 2)],
            "challenge_hsv": [round(m_cl_h, 2), round(m_cl_s, 2), round(m_cl_v, 2)],
            "delta": [round(dl_h, 2), round(dl_s, 2), round(dl_v, 2)]
        },
        "right_roi": {
            "normal_hsv": [round(m_nr_h, 2), round(m_nr_s, 2), round(m_nr_v, 2)],
            "challenge_hsv": [round(m_cr_h, 2), round(m_cr_s, 2), round(m_cr_v, 2)],
            "delta": [round(dr_h, 2), round(dr_s, 2), round(dr_v, 2)]
        }
    }

# ==========================================================
# DETECT
# ==========================================================
@app.post("/api/v1/detect")
async def detect(
    file: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    challenge_image: Optional[UploadFile] = File(None),
    normal_image: Optional[UploadFile] = File(None),
    challenge_id: Optional[str] = Form(None)
):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado"
        )

    main_file = file if file is not None else video
    if main_file is None:
        raise HTTPException(status_code=400, detail="Se requiere un archivo de video/imagen")

    temp_dir = "/tmp/analisis"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, main_file.filename)

    challenge_result = None

    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(main_file.file, buffer)

        if challenge_image and normal_image:
            ch_path = os.path.join(temp_dir, "challenge_" + challenge_image.filename)
            nm_path = os.path.join(temp_dir, "normal_" + normal_image.filename)
            with open(ch_path, "wb") as buffer:
                shutil.copyfileobj(challenge_image.file, buffer)
            with open(nm_path, "wb") as buffer:
                shutil.copyfileobj(normal_image.file, buffer)
            
            challenge_result = validate_light_challenge(ch_path, nm_path)
            #print(f"validate_light_challenge: {validate_light_challenge}")

        ext = os.path.splitext(main_file.filename)[1].lower()
        VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
        
        scores = []
        
        if ext in VIDEO_EXTENSIONS:
            cap = cv2.VideoCapture(temp_file)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Video inválido")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                fps = 30.0
            
            frame_interval = max(int(fps / 2), 1)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_CUBIC)
                    from PIL import Image
                    img_pil = Image.fromarray(img_resized)
                    img_tensor = preprocess_transform(img_pil).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        out, _ = model(img_tensor)
                        prob = float(torch.softmax(out, dim=1)[:, 1].item())
                    scores.append(prob)
                
                frame_count += 1
                
            cap.release()
            
            if len(scores) == 0:
                raise HTTPException(status_code=400, detail="No se pudieron extraer frames")
                
            average_score = float(np.mean(scores))
            max_score = float(np.max(scores))
            
            print(f"Frames analizados: {len(scores)}")
            print(f"Average score: {average_score}")
            print(f"Max score: {max_score}")
            
            content_resp = {
                "filename": main_file.filename,
                "frames_analyzed": len(scores),
                "average_probability": round(average_score, 4),
                "max_probability": round(max_score, 4),
                "deepfake_probability": round(max_score, 4),
                "is_manipulated": max_score >= 0.50
            }
            if challenge_result is not None:
                content_resp["challenge"] = challenge_result

            return JSONResponse(
                status_code=200,
                content=content_resp
            )
            
        else:
            # 1. Leer imagen con OpenCV
            img = cv2.imread(temp_file)
            if img is None:
                raise HTTPException(status_code=400, detail=f"No se pudo leer la imagen: {main_file.filename}")

            # 2. Transformaciones DeepFakeBench (abstract_dataset.py)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_CUBIC)
            
            # Convertir a PIL
            from PIL import Image
            img_pil = Image.fromarray(img_resized)
            img_tensor = preprocess_transform(img_pil).unsqueeze(0).to(DEVICE)

            # 3. Inferencia Real
            with torch.no_grad():
                out, _ = model(img_tensor)
                prob = float(torch.softmax(out, dim=1)[:, 1].item())

            content_resp_img = {
                "filename": main_file.filename,
                "deepfake_probability": round(prob, 4),
                "is_manipulated": prob > 0.50,
                "message": "Inferencia real completada"
            }
            if challenge_result is not None:
                content_resp_img["challenge"] = challenge_result

            return JSONResponse(
                status_code=200,
                content=content_resp_img
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
