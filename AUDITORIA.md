# Auditoría Completa del Sistema de Detección (PAD & DeepFake)

## 1. Auditoría de main.py y FastAPI
* **Arquitectura de FastAPI:** La inicialización de la API en `main.py` está correcta. El modelo se carga en el evento de `startup`. Sin embargo, `app.on_event("startup")` está obsoleto en versiones recientes de FastAPI (recomiendan el uso de `lifespan` context managers), aunque funcional por compatibilidad.
* **Manejo de Video y Procesamiento Temporal (main.py):**
  * **Problemas de precisión/estadísticos en temporalidad:** El análisis de video extrae frames en intervalos `fps / 2` (2 frames por segundo aproximadamente). En cada frame se realiza inferencia real con el modelo Xception (`model(img_tensor)`).
  * **Falta de Temporal Consistency:** Se recopilan los scores de todos los frames y se toma el promedio `np.mean(scores)` y el máximo `np.max(scores)`. `is_manipulated` se define como `max_score >= 0.70`. Utilizar `max_score` de *cualquier* frame aislado puede provocar Falsos Positivos severos. Si un solo frame sufre una alteración o ruido visual que la red confunda con un deepfake, todo el video es clasificado como spoof. Un approach de consistencia temporal, suavizado de medias móviles o un percentil (e.g. 95th percentile) sería más robusto.
  * **Integración DeepFakeBench:** La carga de pesos elimina `backbone.` si existe. El pipeline de inferencia aplica `transforms.Normalize([0.5]*3, [0.5]*3)`. Esto concuerda con DeepFakeBench.
  * **Uso de Memoria/Race Conditions:** `model` y `preprocess_transform` son variables globales. En un entorno multi-hilo como `uvicorn` (dependiendo de la cantidad de workers), esto en modo de CPU/GPU concurrente podría ser un bottleneck pero PyTorch es thread-safe en inferencia `no_grad()`. Se lee el video escribiéndolo al disco `/tmp/analisis/...`. Esto puede causar race conditions si dos usuarios suben un archivo con exactamente el mismo nombre (ej. `video.mp4`), ya que ambos escribirían y leerían `os.path.join(temp_dir, main_file.filename)`. ¡CUIDADO! Se debe utilizar un nombre único (ej. UUID).

## 2. Auditoría del Liveness Engine (liveness_engine.py)
* **Código Muerto y Dummy Data:** En `PADPipeline.analyze` (Líneas 420-430), se retorna un diccionario `res` que ignora por completo la variable `flat_features` extraída de las regiones, y devuelve campos dummy: `"score": 0.8, "delta_hue": 0.0`. Esto corrompe la conexión con `WeightedDecisionEngine` o al menos introduce campos inútiles que confunden.
* **Pipeline del ColorPlugin:**
  * Calcula Delta E usando CIE76 (`deltaE_76`), lo cual es computacionalmente eficiente.
  * Realiza similitud coseno del vector de color observado vs esperado. 
* **Pipeline de TexturePlugin (LBP):**
  * `compute_lbp`: Actualmente es un cuello de botella terrible al usar un doble ciclo `for` en Python puro. En `requirements_api.txt` existe la dependencia `scikit-image`, por lo que se debe reemplazar esta implementación propia con `skimage.feature.local_binary_pattern`, tal como el mismo comentario del código sugiere. La implementación actual frena el tiempo de respuesta del endpoint.
* **Extracción de ROIs (FaceMesh y boundingRect):**
  * Utiliza `cv2.boundingRect` y `cv2.fillConvexPoly` para mascarar los puntos clave de MediaPipe. 
  * Riesgos arquitectónicos en el manejo de coordenadas: En rostros muy pegados al borde (recortes parciales), los cálculos de bounding box pueden salir del límite de la imagen. Aunque hace `max(0, y):min(h, y+bh)`, el padding puede generar que los polígonos del ConvexHull tengan desajustes con las dimensiones de la ROI final (se debe asegurar que la máscara siempre sea del tamaño exacto de la ROI recortada `roi.shape[0], roi.shape[1]`). En el código de `get_bbox` se calcula:
    `roi_pts = pts - [x, y]`
    Si `x` o `y` fueron ajustados a `0`, pero los puntos originales eran negativos (ej. cara fuera de la pantalla), `roi_pts` sería negativo, lo que crashea el `cv2.fillConvexPoly`. En MediaPipe esto rara vez ocurre con imágenes normales, pero es un bug potencial.
* **Weighted Decision Engine:**
  * La normalización del divisor de LBP (`lbp_normalization_divisor: 255.0`) y PSD (`100.0`) está definida. 
  * Los pesos suman 0.95 (25+25+15+10+10+5+5+5 = 100... espere: 0.25+0.25+0.15+0.10+0.10+0.05+0.05+0.05 = 1.00. Es correcto).

## 3. Auditoría de Docker y Dependencias
* **Dockerfile:** Clona el repositorio DeepFakeBench pero no existe un control de versiones de commit, lo que expone a problemas de compatibilidad futura (es un riesgo si DeepFakeBench actualiza rompiendo retrocompatibilidad). Utiliza una imagen base muy pesada de PyTorch Runtime. Expone el puerto 8000.
* **Flujo Docker:** Se usa `--rm -v` y hay un problema de warning de arquitectura (el Dockerfile asume AMD64 y se ejecuta en ARM64, Mac M1/M2/M3). Esto se puede mitigar, pero el runtime funciona sobre xnnpack/CPU en inferencia.

## 4. Problemas de Rendimiento y Diseño
1. **LBP Performance:** Algoritmo Python en lugar de implementaciones en C (scikit-image).
2. **Race condition en nombres de archivos:** Usar UUID en `temp_file = os.path.join(temp_dir, uuid.uuid4().hex + "_" + main_file.filename)`.
3. **Agregación de Score DeepFake:** El uso de `np.max` genera falsos positivos con ruidos aislados. Consistencia temporal de puntajes es un approach validado (Temporal Consistency Filter) que simplemente descarta outliers o suaviza.
4. **Data Dummy en API:** La respuesta del pipeline emite "delta_hue" estáticos en lugar de lo que saca el plugin.

## Listado de Mejoras Propuestas (En Resumen)
1. **Prevención de Race Conditions (Archivos Temporales):**
   - *Riesgo:* Crítico en concurrencia (FastAPI).
   - *Mejora:* Agregar `uuid` al guardado del archivo en `/tmp`.
2. **Consistencia Temporal para Scores (Falsos Positivos):**
   - *Riesgo:* Falsos Positivos en la agregación máxima del DeepFake detector (main.py).
   - *Técnica:* Temporal Consistency (suavizado / top-k percentile en lugar de max bruto).
3. **Optimización de Textura LBP (Rendimiento):**
   - *Riesgo:* Tiempos altísimos de proceso en Python puro.
   - *Técnica:* Local Variance Analysis/LBP utilizando `scikit-image.feature.local_binary_pattern` que ya está en dependencias.
4. **Limpieza del Data Dummy del Pipeline de Liveness:**
   - *Riesgo:* Acoplamiento malo y valores falsos que pasan a frontend/cliente.
   - *Técnica:* Devolver el feature-set original.
5. **Seguridad en ConvexHull Bounding Rects (FaceMesh):**
   - *Riesgo:* Crash `cv2.fillConvexPoly` en frames recortados.
   - *Técnica:* Sanitización estricta de ROI coords (`np.clip`).

Esperando aprobación para establecer el `PLAN_DE_EJECUCIÓN.md`.
