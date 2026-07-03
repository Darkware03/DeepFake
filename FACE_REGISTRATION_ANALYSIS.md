# Análisis Conceptual: Registro Facial y Sistemas de Referencia para ROIs

Este documento evalúa el impacto de migrar de una extracción de Regiones de Interés (ROIs) basada en coordenadas de Bounding Box crudas hacia un **sistema de registro facial continuo mediante transformación afín**.

## 1. Conceptos Fundamentales

### Enfoque Actual (ROI Dinámico / Bounding Box)
MediaPipe extrae los landmarks para cada frame y se recorta un rectángulo (Bounding Box) englobando esos puntos. 
- **Problema:** Si el usuario gira el rostro 5 grados, la "Mejilla Izquierda" en la imagen `Challenge` capturará piel que en la imagen `Normal` estaba bajo diferente incidencia lumínica. OpenCV arrojará dimensiones distintas debido al cambio proyectivo de la topología facial.

### Enfoque Propuesto (ROI Registrado / Alineación Afín)
Se establecen landmarks "ancla" que no sufren deformación elástica (ej. centros oculares, puente de la nariz).
- Se calcula una **Transformación Afín** (o Matriz Homográfica) entre el frame `Normal` y el frame `Challenge` utilizando estos anclajes.
- Se "des-rota" y "escala" (warping) el frame `Challenge` para que coincida perfectamente con el sistema de coordenadas espaciales del frame `Normal`.
- **Ventaja:** La cámara pasa a evaluar la cara de forma estabilizada; un píxel (x,y) en el frame alineado representa exactamente el mismo poro de piel que en el frame normal.

---

## 2. Comparativa: ROI Fijo vs ROI Registrado

| Característica | ROI Fijo (Congelar Coordenadas) | ROI Registrado (Alineación Afín) |
|---|---|---|
| **Definición** | Recortar el frame Challenge usando el (x,y,w,h) exacto detectado en el frame Normal. | Alinear geométricamente ambos rostros usando anclas estables antes de recortar. |
| **Ventajas** | Computacionalmente instantáneo. Matrices garantizadas a tener el mismo shape. | Máxima pureza espectral y anatómica. Corrige rotación, pitch, yaw y zoom. |
| **Inconvenientes** | Si el usuario se mueve, el recorte tomará fondo u otras partes de la cara, arruinando el delta. | Requiere cómputo de matriz homográfica y *cv2.warpAffine*, agregando ~5-10ms por frame. |
| **Robustez** | Muy Baja. | Muy Alta. |

---

## 3. Impacto Esperado sobre las Métricas PAD

La adopción de una arquitectura de registro facial transformará el desempeño matemático de las métricas clave:

### Expected Color Match (Cosine Similarity)
- **Impacto:** Positivo extremo.
- **Razón:** Actualmente, si el rostro rota levemente y la mejilla capta una sombra preexistente en lugar de luz, el Delta RGB (`mean_rgb_c - mean_rgb_n`) sufre un colapso en alguno de sus canales (ej. delta negativo en verde). Al registrar la cara, aseguramos que el vector Delta medirá puramente la adición de luz del Challenge, estabilizando el ángulo de la Similitud Coseno cerca a 1.0 para colores reales.

### Reflection Strength (Magnitude)
- **Impacto:** Alta estabilización.
- **Razón:** El ruido introducido por la pérdida o ganancia de píxeles oscuros/claros debido a traslación del Bounding Box se elimina. El módulo del vector Delta medirá estrictamente la ganancia lumínica de los LEDs de la pantalla.

### DeltaE76 (CIE LAB)
- **Impacto:** Reducción de falsos positivos en el cálculo de Simetría (Symmetry).
- **Razón:** El DeltaE entre el lado izquierdo y derecho es altamente dependiente del ángulo de la fuente lumínica ambiente. Si el registro afín estabiliza el rostro, la comparación de ambos pómulos podrá abstraer rotaciones sutiles y medir con alta precisión el gradiente simétrico real que se espera de un ataque de pantalla recta (Screen Attack).

### Estabilidad Temporal
- **Impacto:** Crucial para la fase de Series Temporales.
- **Razón:** Extraer curvas temporales de N frames (Media, Mediana, Std) es matemáticamente inviable sin registro facial. El jitter del Bounding Box destruiría la señal de alta frecuencia introduciendo picos de ruido. El registro facial asegura que la serie temporal actúe como un osciloscopio limpio monitoreando el decaimiento fotónico sobre la piel.

## 4. Conclusión Estratégica
La arquitectura de **Registro Facial mediante Transformación Afín** es el estándar de oro (Gold Standard) en la investigación fotopletismográfica (rPPG) y detección de vida (PAD). Congelar las coordenadas es un parche inestable; alinear el sistema de referencia es la solución definitiva que permitirá que el motor híbrido brille matemáticamente frente al ruido de usuarios poco cooperativos.
