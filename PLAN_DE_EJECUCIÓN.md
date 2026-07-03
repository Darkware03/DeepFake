# PLAN DE EJECUCIÓN

## Auditoría Completada

Se ha realizado una auditoría completa del código base, detallada en `AUDITORIA.md`. Los problemas principales encontrados que se abordarán en la implementación sin modificar la arquitectura o endpoints son:

1. **Rendimiento:** `compute_lbp` en `liveness_engine.py` utiliza bucles `for` nativos en Python. (Solución: usar `skimage.feature.local_binary_pattern` que ya se incluye en `requirements_api.txt`).
2. **Race Conditions:** `main.py` guarda los uploads bajo `/tmp/analisis/filename`, provocando choques en peticiones concurrentes con el mismo nombre. (Solución: agregar UUID).
3. **Consistencia Temporal:** En `main.py`, si un solo frame puntúa muy alto (e.g. por ruido), condena todo el video. (Solución: Usar percentil (ej. P95) en lugar de `max` absoluto para `max_score` cuando se evalúa `is_manipulated`).
4. **Campos Dummy en Liveness Engine:** `PADPipeline.analyze` en la línea 420 retorna variables en duro en el diccionario (ej. `score: 0.8`) e ignora `flat_features` procesado, lo que corrompe la toma de decisión posterior. (Solución: retornar únicamente features reales).
5. **Seguridad ROI FaceMesh:** `liveness_engine.py` `get_bbox` no sanitiza las coordenadas extraídas, lo que podría provocar valores negativos y fallos en OpenCV. (Solución: aplicar `np.clip`).

## Archivos a Modificar

1. `app/main.py`:
   - Motivo: Evitar race conditions en guardado de archivos temporales mediante UUIDs.
   - Motivo: Mejorar precisión Deepfake cambiando `max` crudo por `np.percentile(scores, 95)` para descartar outliers (Temporal Consistency).
   - Riesgo: Bajo.
   - Método de validación: `docker run`, prueba funcional de envío simultáneo y score.

2. `app/liveness_engine.py`:
   - Motivo: Reemplazar el bucle lento LBP por la implementación madura y correcta `skimage.feature.local_binary_pattern`.
   - Motivo: Eliminar los datos dummy del `PADPipeline.analyze`.
   - Motivo: Sanitizar arrays de MediaPipe (FaceMesh).
   - Riesgo: Bajo/Medio (depende de validación LBP).
   - Método de validación: Se creará un pequeño script de validación que compare la ejecución original Python vs `skimage` para demostrar equivalencia funcional.

## Criterios de Aceptación
1. Demostrar la equivalencia funcional del LBP mediante un test script.
2. Construir la imagen Docker (`docker build`).
3. Levantar API y ejecutar endpoint (`docker run`).
4. Verificar que el JSON retornado no cambie su estructura original, solo que devuelva los valores calculados de forma más robusta.
5. Verificaciones de sintaxis completadas exitosamente.