# Plan de Recalibración del WeightedDecisionEngine

Este documento justifica técnicamente y matemáticamente la propuesta de recalibración del motor de decisión multicapa del sistema PAD, en respuesta a que el modelo Xception no separa los videos reales de IA por sí solo de forma confiable, sino que aporta un score intermedio (0.36 a 0.45).

## 1. DeepFakeBench como factor de penalización (Señal de Riesgo)
Actualmente, el motor hace una suma ponderada lineal (línea 227):
`final_score = df_cont + cm_cont + re_cont + tx_cont + ps_cont + sy_cont + sr_cont + qu_cont`
donde `df_cont = (1.0 - deepfake_prob) * 0.25`.
Si `deepfake_prob` es 0.45 (video real), el aporte es `(1 - 0.45) * 0.25 = 0.1375`.
Si es 0.38 (video IA), el aporte es `(1 - 0.38) * 0.25 = 0.155`.
Esto diluye por completo el valor del score y, de hecho, le da más puntuación al video de IA porque su probabilidad "fake" generada por Xception resultó ligeramente menor.

**Propuesta Matemática:**
El Xception debe actuar como **evidencia de riesgo asimétrico (penalización) o multiplicador de confianza** en lugar de sumar linealmente con los factores físicos.
De este modo, la autenticidad dependerá del Challenge Físico (Color, Reflexión, etc).
Cálculo de evidencia física neta:
`physical_score = cm_cont + re_cont + tx_cont + ps_cont + sy_cont + sr_cont + qu_cont`
(Ajustando sus pesos para sumar 1.0 en total).
Luego, el score de DeepFake (Xception) penalizará este `physical_score` de forma que si el modelo de red neuronal está extremadamente seguro de un ataque (ej. > 0.8), se castiga la confianza general. Si el modelo está en rangos intermedios (0.3 a 0.7), la decisión final depende casi por completo del `physical_score`.

*Nueva Ecuación:*
`final_score = physical_score * (1.0 - (deepfake_prob * penalty_factor))`
Si el challenge físico es contundente, compensará el ruido de la IA.

## 2. Agregación Robusta de Características (Mediana en lugar de Media)
Actualmente (líneas 202-203), `avg_feat` usa `np.mean(vals)`.
El promedio simple en 5 ROIs (forehead, nose, left_cheek, right_cheek, chin) es extremadamente susceptible a outliers. Por ejemplo, si en un frame el usuario gira la cara y "left_cheek" queda oscurecida o saturada de blanco, la reflexión o el deltaE se disparan o se anulan, arruinando la media.

**Justificación Técnica:**
Al haber 5 regiones, una **mediana (`np.median`)** es estadísticamente la métrica de agregación más robusta. Descarta los 2 valores extremos superior e inferior, evaluando únicamente la tendencia central real de la piel frente al challenge de luz (pantalla).
- Si el brillo de la pantalla afecta bien a la frente y las mejillas, pero no a la nariz por la sombra y la barbilla por la barba, la mediana reflejará el impacto real de las áreas exitosas sin ser arrastrada a 0 por el vello facial (barba).

## 3. Limpieza de Normalizaciones Arbitrarias
`div["lbp_normalization_divisor"] = 255.0`
`div["psd_normalization_divisor"] = 100.0`
`div["reflection_normalization_divisor"] = 15.0`

**Análisis:**
- `lbp / 255.0`: El LBP usando `skimage` con radius=1, neighbors=8 produce valores en el rango [0, 255]. Por ende, dividirlo entre 255 para min-max scaler `[0, 1]` está matemáticamente justificado.
- `psd / 100.0`: El Power Spectral Density logarítmico suele rondar valores entre 10 y 200 dependiendo del brillo. Un divisor estático de 100 truncando a 1.0 (`min(1.0, psd / 100.0)`) es arbitrario pero funcional. Se mantendrá, documentando su naturaleza empírica.
- `reflection_normalization_divisor = 15.0`: Delta L en CIE-LAB mide luminosidad. Un flash de pantalla en la oscuridad puede producir +15 unidades fácilmente, pero a la luz del día puede producir +3 unidades. Si requerimos estáticamente 15 para un score de 1.0, en condiciones de luz diurna el liveness fallará sistemáticamente (Falsos Negativos).
**Propuesta:** No modificar el divisor (por las reglas), pero ajustar la evaluación del `physical_score` considerando que no siempre se alcanza el máximo.

## 4. Prioridad a la Evidencia Física
En la nueva recalibración de los `ENGINE_CONFIG["weights"]` internos del código:
Como sacaremos a `deepfake` de la suma (para que sea penalizador/multiplicador), los pesos de la evidencia física se reescalarán proporcionalmente para sumar 1.0:
- Color Match: 0.35
- Reflection: 0.25
- Texture (LBP): 0.15
- Skin Response: 0.10
- PSD: 0.05
- Symmetry: 0.05
- Quality: 0.05

*Total = 1.00*
Esto garantiza que la física manda.

## Plan de Modificación y Aislamiento en `app/liveness_engine.py`
1. Reemplazar `np.mean` por `np.median` en la función interna `avg_feat(name)` de la clase `WeightedDecisionEngine`.
2. Remover "deepfake" del listado base de pesos que se suman linealmente.
3. Repartir el peso original de `deepfake` en las señales dependientes del challenge (color y reflexión).
4. Usar `deepfake_prob` como un factor modulador sobre el `physical_score` final.
5. Mantener los nombres, clases y JSON idénticos. El cambio afectará sólo a cómo se calcula el `confidence`, `is_live` y `decision_breakdown`.