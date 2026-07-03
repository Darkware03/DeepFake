# Análisis Matemático: Métricas de Similitud de Color (Color Match)

## 1. El Problema Actual (Cosine Similarity sobre RGB)

Actualmente, el `ColorPlugin` calcula el `expected_color_match` usando la Similitud Coseno entre dos vectores en el espacio RGB:
- **Vector Esperado (`v_exp`):** `[R_exp, G_exp, B_exp]` (el color hexadecimal inyectado).
- **Vector Observado (`v_obs`):** `[ΔR, ΔG, ΔB]` (la diferencia bruta BGR->RGB entre la imagen Challenge y Normal).

### Por qué falla:
1. **Sensibilidad a Auto-Exposición / Auto-White-Balance (AWB):** Si el teléfono compensa la iluminación bajando el brillo general, los canales que no deberían cambiar mucho (por ejemplo, verde y azul cuando se proyecta un challenge rojo) pueden arrojar deltas negativos (`ΔG < 0`, `ΔB < 0`).
2. **Ortogonalidad Rígida:** En el espacio RGB, un delta negativo drástico mueve el vector `v_obs` hacia cuadrantes negativos. Como `v_exp` siempre está en el cuadrante positivo `[0-255]`, el ángulo entre ambos supera rápidamente los 90 grados, arrojando una **similitud coseno de cero o negativa**.
3. **Dependencia de Luminancia:** RGB acopla estrechamente el color (crominancia) con el brillo (luminancia). Un simple movimiento de cabeza que introduzca sombra penalizará la métrica cromática.

---

## 2. Alternativa A: Cosine Similarity sobre ΔLAB (L*, a*, b*)

El espacio CIELAB (L*, a*, b*) desacopla la luminancia (`L*`) de la crominancia (`a*`, `b*`).
- **`a*`:** Eje Verde-Rojo.
- **`b*`:** Eje Azul-Amarillo.

En este enfoque, el color esperado HEX se proyectaría a un vector LAB ideal. Luego, compararíamos el vector `[ΔL*, Δa*, Δb*]` observado contra el esperado.

### Pros:
- Reduce parcialmente el impacto de las sombras al modelar el color de manera perceptualmente uniforme.

### Contras:
- Seguimos incluyendo `ΔL*` (brillo) en la ecuación direccional. El AWB de la cámara puede suprimir drásticamente la ganancia de luminancia `ΔL*`, torciendo el vector de respuesta lejos de la dirección ideal esperada, volviendo a causar el mismo problema de ángulos abiertos que en RGB.

---

## 3. Alternativa B: Comparación Cromática Pura (Ejes a* y b*)

Este enfoque descarta por completo el eje de luminancia `L*` y la información RGB acoplada. Trabaja *estrictamente* sobre el plano cromático bidimensional `[a*, b*]`.

- **Vector Esperado:** `[a*_exp - a*_neutro, b*_exp - b*_neutro]`
- **Vector Observado:** `[Δa*_obs, Δb*_obs]` (donde `Δa* = a*_challenge - a*_normal`)

### Matemáticas de Robustez:
1. **Inmunidad a Exposición y Sombras:** Si el usuario mueve la cabeza y entra en sombra, `L*` caerá en picada. Sin embargo, la proyección del color sobre la piel (por ejemplo, la inyección del canal rojo) seguirá empujando el valor `a*` hacia el extremo positivo independientemente de si la escena se oscureció.
2. **Inmunidad a Variación de Piel:** Todos los tonos de piel humana se agrupan en una región muy estrecha del plano `[a*, b*]` (pieles más oscuras simplemente tienen menor `L*`). El delta cromático `[Δa*, Δb*]` mide la luz incidente reflejada, no el color intrínseco de la piel.
3. **Distancia Euclidiana Cromática:** En lugar de Similitud Coseno estricta (que castiga duramente pequeños desvíos angulares), se puede usar la magnitud de proyección del `Δa*, Δb*` observado sobre el vector esperado.

### Conclusión sobre Espacios de Color:
**La evaluación cromática pura sobre los ejes `a*` y `b*` es matemáticamente la opción más robusta.** Ignora la respuesta defectuosa de luminancia de los teléfonos (AWB/AE) y se concentra en la pureza de la perturbación espectral introducida.

---

## 4. Evolución Temporal (Múltiples Frames)

La actual comparación "Un Frame vs Un Frame" (Single-Shot) asume un entorno estacionario perfecto, lo cual es irreal (micro-expresiones, ruido del sensor ISO alto, latencia de codificación).

### La Solución: Ventana Temporal Estadística
Dado un video a 30 FPS con un challenge de 2 segundos (60 frames):
1. **Extracción de Señal Continua:** Se extraen 60 muestras de `[a*, b*]` por cada ROI.
2. **Derivada Temporal:** Se calcula la derivada o el diferencial respecto al promedio pre-challenge (frames 0-10).
3. **Agregación por Mediana:** El uso de la **mediana** (en lugar del promedio) de los `Δa*` y `Δb*` máximos filtra por completo los outliers causados por un pestañeo, movimiento súbito o desajuste de un frame corrupto por compresión inter-frame (B-frames en MP4).
4. **Análisis de Decaimiento:** Analizar toda la curva permite detectar "ataques de inyección" (Virtual Cameras), ya que un LCD o un DeepFake inyectado exhibe un salto de función escalón en el color (0ms), mientras que el AWB de un teléfono real y la luz física reflejada sobre piel producen una curva RC (crecimiento amortiguado).

### Recomendación Definitiva
No modificar el código actual hacia un nuevo algoritmo Single-Shot. El salto real a Nivel de Producción requiere rediseñar `ColorPlugin` para que ingiera un array `List[Frames]`, procese la curva temporal `a*, b*` en cada ROI mediante agregación estadística (mediana) y evalúe la coherencia de la función de crecimiento cromático frente al vector esperado proyectado bidimensionalmente.
