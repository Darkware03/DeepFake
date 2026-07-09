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

from app.liveness_engine import analyze_reflection, decide_liveness
from app.dataset_logger import DatasetLogger

# ==========================================================
# LIGHT CHALLENGE VALIDATION
# ==========================================================
def validate_light_challenge(challenge_path: str, normal_path: str, expected_hex_color: str = "#FFFFFF", logger=None) -> dict:
    normal_img = cv2.imread(normal_path)
    challenge_img = cv2.imread(challenge_path)

    if normal_img is None or challenge_img is None:
        return {"valid": False, "reason": "image_not_loaded"}

    return analyze_reflection(normal_img, challenge_img, expected_hex_color, logger=logger)

# ==========================================================
# DETECT
# ==========================================================
@app.post("/api/v1/detect_colors", summary="Valida múltiples colores para Liveness (Pigmentación)", description="Recibe una imagen normal y 'n' imágenes de desafío con sus 'n' colores respectivos para validar la pigmentación y el reflejo en la piel.")
async def detect_colors(
    normal_image: UploadFile = File(..., description="Imagen en condiciones normales sin flash"),
    challenge_images: list[UploadFile] = File(..., description="Lista de n imágenes con el flash activo"),
    challenge_colors: list[str] = Form(..., description="Lista de n colores hexadecimales correspondientes a cada imagen de desafío")
):
    import tempfile
    import os
    import shutil
    from pathlib import Path

    # Si viene como un solo string separado por comas
    if len(challenge_colors) == 1 and "," in challenge_colors[0]:
        challenge_colors = [c.strip() for c in challenge_colors[0].split(",")]

    if len(challenge_images) != len(challenge_colors):
        raise HTTPException(
            status_code=400, 
            detail=f"El número de challenge_images ({len(challenge_images)}) debe coincidir con el número de challenge_colors ({len(challenge_colors)})"
        )

    resultados_resumidos = []
    logger = DatasetLogger()

    with tempfile.TemporaryDirectory(dir="/tmp", prefix="analisis_multi_") as temp_dir:
        nm_safe = Path(normal_image.filename).name
        nm_path = os.path.join(temp_dir, "normal_" + nm_safe)
        
        with open(nm_path, "wb") as buffer:
            shutil.copyfileobj(normal_image.file, buffer)
            
        try:
            logger.save_normal_image(nm_path)
        except Exception as e:
            logger.log_error(e)

        # Evaluar la probabilidad de deepfake en la imagen normal
        deepfake_prob = 0.0
        img_n = cv2.imread(nm_path)
        if img_n is not None and model is not None:
            try:
                img_rgb = cv2.cvtColor(img_n, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_CUBIC)
                from PIL import Image
                img_pil = Image.fromarray(img_resized)
                img_tensor = preprocess_transform(img_pil).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    out, _ = model(img_tensor)
                    deepfake_prob = float(torch.softmax(out, dim=1)[:, 1].item())
            except Exception as e:
                print(f"Error evaluando deepfake en detect_colors: {e}")

        for idx, (ch_img, ch_color) in enumerate(zip(challenge_images, challenge_colors)):
            ch_safe = Path(ch_img.filename).name
            ch_path = os.path.join(temp_dir, f"challenge_{idx}_{ch_safe}")
            
            with open(ch_path, "wb") as buffer:
                shutil.copyfileobj(ch_img.file, buffer)
                
            try:
                logger.save_challenge_image(ch_path)
            except Exception as e:
                pass

            resultado = validate_light_challenge(ch_path, nm_path, expected_hex_color=ch_color, logger=logger)
            
            # Usamos la probabilidad de deepfake calculada para que el endpoint sea seguro
            resultado_liveness = decide_liveness(deepfake_prob, resultado, logger=logger)
            
            # Construir el objeto resumido para la respuesta
            color_match = 0.0
            reflection = 0.0
            
            breakdown = resultado_liveness.get("decision_breakdown", {})
            color_match = breakdown.get("color_match", {}).get("raw", 0.0)
            reflection = breakdown.get("reflection", {}).get("raw", 0.0)

            # Para la API devolvemos valores simplificados
            res_item = {
                "color": ch_color,
                "is_valid": resultado_liveness.get("is_live", False),
                "pigmentation_match": round(color_match, 4),
                "reflection_strength": round(reflection, 4),
                "risk_level": resultado_liveness.get("risk", "UNKNOWN")
            }
            
            # Si el motor falló en procesar la imagen, incluir la razón para debug
            if not resultado.get("valid", False):
                res_item["error"] = resultado.get("reason", "unknown_error")

            resultados_resumidos.append(res_item)

    return {"resultados": resultados_resumidos}

@app.post("/api/v1/detect")
async def detect(
    file: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    challenge_image: Optional[UploadFile] = File(None),
    normal_image: Optional[UploadFile] = File(None),
    challenge_id: Optional[str] = Form(None),
    challenge_color: Optional[str] = Form(None)
):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado"
        )

    main_file = file if file is not None else video
    if main_file is None:
        raise HTTPException(status_code=400, detail="Se requiere un archivo de video/imagen")

    import tempfile
    import re
    from pathlib import Path
    import time
    from datetime import datetime

    # Validar challenge_color si fue proveído
    valid_color = "#FFFFFF"
    if challenge_color is not None:
        if not re.match(r"^#[0-9A-Fa-f]{6}$", challenge_color):
            raise HTTPException(status_code=400, detail="Formato de challenge_color inválido. Debe ser HEX (#RRGGBB).")
        valid_color = challenge_color

    challenge_result = None
    logger = DatasetLogger()
    start_time = time.time()

    try:
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="analisis_") as temp_dir:
            # Sanitizar nombres de archivo
            safe_filename = Path(main_file.filename).name
            temp_file = os.path.join(temp_dir, safe_filename)
            with open(temp_file, "wb") as buffer:
                shutil.copyfileobj(main_file.file, buffer)
                
            try:
                logger.save_video(temp_file)
            except Exception as e:
                logger.log_error(e)

            if challenge_image and normal_image:
                ch_safe = Path(challenge_image.filename).name
                nm_safe = Path(normal_image.filename).name
                ch_path = os.path.join(temp_dir, "challenge_" + ch_safe)
                nm_path = os.path.join(temp_dir, "normal_" + nm_safe)
                with open(ch_path, "wb") as buffer:
                    shutil.copyfileobj(challenge_image.file, buffer)
                with open(nm_path, "wb") as buffer:
                    shutil.copyfileobj(normal_image.file, buffer)
                    
                try:
                    logger.save_normal_image(nm_path)
                    logger.save_challenge_image(ch_path)
                except Exception as e:
                    logger.log_error(e)

                challenge_result = validate_light_challenge(ch_path, nm_path, expected_hex_color=valid_color, logger=logger)

            ext = os.path.splitext(safe_filename)[1].lower()
            VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

            scores = []

            if ext in VIDEO_EXTENSIONS:
                cap = cv2.VideoCapture(temp_file)
                if not cap.isOpened():
                    raise HTTPException(status_code=400, detail="Video inválido")

                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0 or np.isnan(fps):
                    fps = 30.0

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                file_size = os.path.getsize(temp_file)

                frame_interval = max(int(fps / 2), 1)
                frame_count = 0
                timeline = {}
                csv_rows = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    logger.save_original_frame(frame_count, frame)

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
                        
                        ts_frame = datetime.now().isoformat()
                        timeline[f"frame_{frame_count:06d}"] = prob
                        csv_rows.append({
                            "frame": frame_count,
                            "timestamp": ts_frame,
                            "deepfake_probability": round(prob, 4),
                            "classification": "FAKE" if prob >= 0.70 else "REAL"
                        })
                        
                        frame_meta = {
                            "frame": frame_count,
                            "timestamp": ts_frame,
                            "score": prob,
                            "probabilidad": round(prob, 4),
                            "modelo": "Xception",
                            "clasificación": "FAKE" if prob >= 0.70 else "REAL",
                            "preprocesamiento": "Resize 256x256, RGB, Normalize(-0.5, 0.5)",
                            "resolución": f"{frame.shape[1]}x{frame.shape[0]}"
                        }
                        logger.save_analyzed_frame(frame_count, frame, frame_meta)

                    frame_count += 1

                cap.release()

                if len(scores) == 0:
                    raise HTTPException(status_code=400, detail="No se pudieron extraer frames")

                average_score = float(np.mean(scores))
                max_score = float(np.max(scores))
                p95_score = float(np.percentile(scores, 95))

                print(f"Frames analizados: {len(scores)}")
                print(f"Average score: {average_score}")
                print(f"Max score: {max_score}")
                print(f"P95 score: {p95_score}")

                content_resp = {
                    "filename": safe_filename,
                    "frames_analyzed": len(scores),
                    "average_probability": round(average_score, 4),
                    "max_probability": round(max_score, 4),
                    # NOTA: p95_probability se reporta únicamente como referencia estadística (Temporal Consistency),
                    # max_probability (max_score) sigue siendo la métrica utilizada obligatoriamente para la decisión.
                    "p95_probability": round(p95_score, 4),
                    "deepfake_probability": round(max_score, 4),
                    "is_manipulated": max_score >= 0.70
                }
                if challenge_result is not None:
                    content_resp["challenge"] = challenge_result

                from app.liveness_engine import numpy_to_python

                # Agregamos Liveness Engine usando max_score como dicta la compatibilidad
                liveness_result = decide_liveness(max_score, challenge_result, logger=logger)
                content_resp["liveness"] = liveness_result

                # Compatibilidad para los campos sueltos pedidos
                content_resp["analysis"] = liveness_result.get("analysis", [])
                content_resp["risk_level"] = liveness_result.get("risk", "LOW")
                if "recommendations" in liveness_result:
                    content_resp["recommendations"] = liveness_result["recommendations"]
                    
                process_time = time.time() - start_time
                try:
                    video_metadata = {
                        "nombre archivo": safe_filename,
                        "timestamp": datetime.now().isoformat(),
                        "hash sha256": logger.hashes.get("video", ""),
                        "fps": fps,
                        "duración": duration,
                        "resolución": f"{width}x{height}",
                        "tamaño": file_size,
                        "frames totales": total_frames,
                        "frames analizados": len(scores),
                        "average_probability": round(average_score, 4),
                        "max_probability": round(max_score, 4),
                        "p95_probability": round(p95_score, 4),
                        "deepfake_probability": round(max_score, 4),
                        "is_manipulated": max_score >= 0.70,
                        "tiempo total de procesamiento": process_time,
                        "modelo utilizado": "Xception",
                        "pesos utilizados": PATH_PESOS,
                        "device CPU/GPU": str(DEVICE)
                    }
                    
                    logger.save_video_metadata(video_metadata)
                    logger.save_timeline(timeline)
                    logger.save_scores_csv(csv_rows)
                    
                    import mediapipe as mp
                    meta = {
                        "engine_version": "2.1.1",
                        "feature_schema_version": "2.0.1",
                        "opencv_version": cv2.__version__,
                        "mediapipe_version": mp.__version__,
                        "python_version": sys.version,
                        "modelo": "Xception",
                        "checkpoint": PATH_PESOS,
                        "device": str(DEVICE)
                    }
                    logger.save_metadata(meta)
                    logger.save_response(numpy_to_python(content_resp))
                    logger.save_checksums()
                except Exception as e:
                    logger.log_error(e)

                return JSONResponse(
                    status_code=200,
                    content=numpy_to_python(content_resp)
                )

            else:
                # 1. Leer imagen con OpenCV
                img = cv2.imread(temp_file)
                if img is None:
                    raise HTTPException(status_code=400, detail=f"No se pudo leer la imagen: {safe_filename}")

                logger.save_original_frame(0, img)

                # 2. Transformaciones DeepFakeBench (abstract_dataset.py)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_CUBIC)
                
                frame_meta = {
                    "frame": 0,
                    "timestamp": datetime.now().isoformat(),
                    "score": 0.0,
                    "probabilidad": 0.0,
                    "modelo": "Xception",
                    "clasificación": "UNKNOWN",
                    "preprocesamiento": "Resize 256x256, RGB, Normalize(-0.5, 0.5)",
                    "resolución": f"{img.shape[1]}x{img.shape[0]}"
                }

                # Convertir a PIL
                from PIL import Image
                img_pil = Image.fromarray(img_resized)
                img_tensor = preprocess_transform(img_pil).unsqueeze(0).to(DEVICE)

                # 3. Inferencia Real
                with torch.no_grad():
                    out, _ = model(img_tensor)
                    prob = float(torch.softmax(out, dim=1)[:, 1].item())
                    
                frame_meta["score"] = prob
                frame_meta["probabilidad"] = round(prob, 4)
                frame_meta["clasificación"] = "FAKE" if prob >= 0.70 else "REAL"
                logger.save_analyzed_frame(0, img, frame_meta)
                
                logger.save_timeline({"frame_000000": prob})
                logger.save_scores_csv([{
                    "frame": 0,
                    "timestamp": datetime.now().isoformat(),
                    "deepfake_probability": round(prob, 4),
                    "classification": "FAKE" if prob >= 0.70 else "REAL"
                }])

                content_resp_img = {
                    "filename": safe_filename,
                    "deepfake_probability": round(prob, 4),
                    "is_manipulated": prob > 0.70,
                    "message": "Inferencia real completada"
                }
                if challenge_result is not None:
                    content_resp_img["challenge"] = challenge_result

                from app.liveness_engine import numpy_to_python

                # Agregamos Liveness Engine
                liveness_result = decide_liveness(prob, challenge_result, logger=logger)
                content_resp_img["liveness"] = liveness_result

                # Compatibilidad para los campos sueltos pedidos
                content_resp_img["analysis"] = liveness_result.get("analysis", [])
                content_resp_img["risk_level"] = liveness_result.get("risk", "LOW")
                if "recommendations" in liveness_result:
                    content_resp_img["recommendations"] = liveness_result["recommendations"]
                    
                process_time = time.time() - start_time
                try:
                    video_metadata = {
                        "nombre archivo": safe_filename,
                        "timestamp": datetime.now().isoformat(),
                        "hash sha256": logger.hashes.get("video", ""),
                        "fps": 0,
                        "duración": 0,
                        "resolución": f"{img.shape[1]}x{img.shape[0]}",
                        "tamaño": os.path.getsize(temp_file),
                        "frames totales": 1,
                        "frames analizados": 1,
                        "average_probability": round(prob, 4),
                        "max_probability": round(prob, 4),
                        "p95_probability": round(prob, 4),
                        "deepfake_probability": round(prob, 4),
                        "is_manipulated": prob >= 0.70,
                        "tiempo total de procesamiento": process_time,
                        "modelo utilizado": "Xception",
                        "pesos utilizados": PATH_PESOS,
                        "device CPU/GPU": str(DEVICE)
                    }
                    
                    logger.save_video_metadata(video_metadata)
                    
                    import mediapipe as mp
                    meta = {
                        "engine_version": "2.1.1",
                        "feature_schema_version": "2.0.1",
                        "opencv_version": cv2.__version__,
                        "mediapipe_version": mp.__version__,
                        "python_version": sys.version,
                        "modelo": "Xception",
                        "checkpoint": PATH_PESOS,
                        "device": str(DEVICE)
                    }
                    logger.save_metadata(meta)
                    logger.save_response(numpy_to_python(content_resp_img))
                    logger.save_checksums()
                except Exception as e:
                    logger.log_error(e)

                return JSONResponse(
                    status_code=200,
                    content=numpy_to_python(content_resp_img)
                )

    except HTTPException:
        raise
    except Exception as e:
        if logger:
            logger.log_error(e)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
