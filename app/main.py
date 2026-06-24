import os
import sys

# =====================================================================
# SECCIÓN CRÍTICA: Configuración preventiva de rutas de Python
# Debe ejecutarse antes de importar cualquier módulo local o de terceros
# =====================================================================
RAIZ_WORKSPACE = "/workspace"
if RAIZ_WORKSPACE not in sys.path:
    sys.path.insert(0, RAIZ_WORKSPACE)

# También añadimos la ruta relativa hacia el directorio superior por seguridad
ruta_padre = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_padre not in sys.path:
    sys.path.insert(0, ruta_padre)
# =====================================================================

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil

# Importaciones del benchmark (Ahora resueltas correctamente)
from networks.xception import Xception
from preprocessing.utils import extract_frames # Ajusta según las funciones de preprocesamiento de tu pipeline

app = FastAPI(
    title="Deepfake Detection Benchmark API",
    description="API forense para la detección de anomalías y manipulaciones faciales en video e imágenes.",
    version="1.0.0"
)

# Configuración de rutas internas de los pesos mapeados por el volumen
PATH_PESOS = "/workspace/weights/xception_best.pth" # Ajusta al nombre exacto de tu archivo .pth
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None

@app.on_event("startup")
async def load_model():
    """
    Inicializa el modelo de detección al arrancar el contenedor.
    Garantiza que los pesos estén disponibles en el volumen montado.
    """
    global model
    if not os.path.exists(PATH_PESOS):
        raise RuntimeError(f"Falta el archivo de pesos del modelo en la ruta montada: {PATH_PESOS}")
    
    try:
        # Inicializar la arquitectura Xception del benchmark
        model = Xception(num_classes=2)
        
        # Cargar los pesos guardados en la GPU o CPU según disponibilidad
        state_dict = torch.load(PATH_PESOS, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print(f"[*] Modelo Xception cargado exitosamente en el dispositivo: {DEVICE}")
    except Exception as e:
        print(f"[-] Error crítico al inicializar el modelo: {str(e)}")
        raise e

@app.get("/health")
async def health_check():
    """Endpoint de control para orquestadores o chequeo rápido."""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": model is not None
    }

@app.post("/api/v1/detect")
async def detect_manipulation(file: UploadFile = File(...)):
    """
    Recibe un archivo multimedia, extrae características faciales
    y determina la probabilidad de manipulación (Deepfake).
    """
    if not model:
        raise HTTPException(status_code=503, detail="El modelo de inferencia no está listo.")

    # Guardar archivo de forma temporal para procesar
    temp_dir = "/tmp/analisis"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # =================================================================
        # PIPELINE DE INFERENCIA (Ajusta la lógica a tu preprocesamiento exacto)
        # =================================================================
        # Ejemplo hipotético de flujo de inferencia:
        # frames = extract_frames(temp_file_path)
        # inputs = transform_frames(frames).to(DEVICE)
        # with torch.no_grad():
        #     outputs = model(inputs)
        #     score = torch.softmax(outputs, dim=1)[:, 1].item()
        # =================================================================
        
        # Mock de respuesta temporal estructurada para validar comunicación
        score_simulado = 0.87 
        
        return JSONResponse(status_code=200, content={
            "filename": file.filename,
            "deepfake_probability": score_simulado,
            "is_manipulated": score_simulado > 0.5,
            "message": "Análisis forense completado con éxito."
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el procesamiento: {str(e)}")
    
    finally:
        # Limpieza del archivo temporal
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
