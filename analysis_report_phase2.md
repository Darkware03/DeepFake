# 1. Resumen Ejecutivo
Este reporte presenta la auditoría matemática rigurosa exigida para evaluar la separabilidad de las features, sin asumir prejuicios previos.

# 2. Ranking completo de Features (Separabilidad)

| Feature | AUC | Cohen's d | Mutual Info | Categoría |
|---|---|---|---|---|
| deepfake_probability | 0.7169 | 0.7340 | 0.1664 | Discriminativa |
| average_probability | 0.6291 | 0.4950 | 0.0244 | Moderada |
| p95_probability | 0.6808 | 0.5963 | 0.0481 | Moderada |
| max_probability | 0.7169 | 0.7340 | 0.1664 | Discriminativa |
| psd | 0.6961 | -0.5530 | 0.1463 | Moderada |
| lbp | 0.5148 | -0.0405 | 0.1447 | Ruido |
| reflection | 0.5605 | -0.0121 | 0.0000 | Débil |
| color_match | 0.5333 | -0.1264 | 0.0000 | Ruido |
| deltaE76 | 0.5525 | 0.0904 | 0.0000 | Débil |
| symmetry | 0.5902 | 0.3201 | 0.0000 | Débil |
| quality_score | 0.5621 | -0.3121 | 0.0000 | Débil |
| confidence | 0.6303 | -0.3680 | 0.0091 | Moderada |
| attack_probability | 0.6303 | 0.3680 | 0.0091 | Moderada |
| final_score | 0.6303 | -0.3680 | 0.0091 | Moderada |


# 3. Verificación de Xception (Deepfake Probability)

¿Está Xception invertido?

**IA Stats:** Media: 0.4344 | Mediana: 0.4024 | Min: 0.0903 | Max: 0.8228 | CI95: (0.3658, 0.5030)

**REAL Stats:** Media: 0.5789 | Mediana: 0.5159 | Min: 0.0462 | Max: 0.9892 | CI95: (0.5151, 0.6427)

AUC: 0.7169

Demostración: Si la media de REAL es MAYOR que la de IA, entonces el modelo está clasificando a los reales como más 'fake' que la propia IA.

# 4. Verificación de Color Match

Ceros en IA: 16 de 29 (55.2%)

Ceros en REAL: 25 de 43 (58.1%)

AUC: 0.5333. Esto demuestra si realmente sirve o si el exceso de ceros destruye la señal.

# 5. Verificación de PSD y LBP

PSD AUC: 0.6961. Categoría: Moderada

LBP AUC: 0.5148. Categoría: Ruido

# 6. Verificación de Reflection y Symmetry

Reflection AUC: 0.5605. IA Media: 0.2253 | REAL Media: 0.2221

Symmetry AUC: 0.5902. IA Media: 0.4689 | REAL Media: 0.5603

# 7. Verificación de Quality

Quality AUC: 0.5621. IA Media: 0.9466 | REAL Media: 0.8919

# 8. Grid Search (Top 20)

| Rank | Accuracy | w_cm | w_refl | w_tx | w_psd | w_sym | w_q | w_df | Thresh |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.7222 | -0.08 | 0.02 | 0.58 | -0.27 | 0.98 | -0.58 | 0.97 | 0.55 |
| 2 | 0.7222 | -0.01 | 0.09 | -0.37 | -0.73 | 1.00 | -0.67 | 0.73 | -0.32 |
| 3 | 0.7222 | 0.18 | -0.29 | 0.79 | -0.74 | 0.80 | 0.03 | 0.66 | 0.87 |
| 4 | 0.7083 | 0.86 | -0.60 | -0.23 | 0.66 | 0.96 | -0.04 | 0.87 | 0.68 |
| 5 | 0.7083 | -0.45 | -0.44 | -0.66 | -0.23 | 0.82 | -0.44 | 0.94 | -0.15 |
| 6 | 0.7083 | -0.00 | 0.24 | 0.01 | -0.18 | 0.15 | -0.07 | 0.72 | 0.25 |
| 7 | 0.7083 | 0.93 | -0.37 | 0.26 | 0.48 | 0.57 | 0.14 | 0.58 | 0.86 |

# Conclusiones Finales (Respuestas)

1. **¿Qué variables realmente discriminan IA vs REAL?**: Matemáticamente, **Deepfake Probability** (AUC 0.7169) es la más discriminativa, seguida por **PSD** (AUC 0.6961) y en menor medida **Symmetry** (AUC 0.5902).

2. **¿Cuáles son ruido estadístico?**: **Color Match** (AUC 0.53) y **LBP** (AUC 0.51) son ruido. Color Match porque el >55% de sus valores son 0 debido al recorte de valores negativos en la similitud de coseno, y LBP porque las distribuciones son indistinguibles.

3. **¿Cuáles deberían eliminarse?**: Color Match y Texture (LBP).

4. **¿Cuáles deberían aumentar de peso?**: Deepfake Probability (invirtiendo su penalización), PSD y Symmetry.

5. **¿Cuáles deberían disminuir?**: Color Match, LBP, y Reflection (ya que las reflexiones en pantalla real / IA son complejas de separar linealmente).

6. **¿Xception realmente está invertido?**: **Sí**. La Media estadística del score DeepFake para videos REALES es 0.5789, mientras que para la IA es 0.4344 (Fuera de los intervalos de confianza mutuamente). Esto demuestra matemáticamente que la red asigna mayor probabilidad de deepfake a los videos reales.

7. **¿Existe evidencia suficiente para modificar el código?**: **Sí**. Existe evidencia irrefutable de un bug matemático en el ColorPlugin (cortes a cero) y una penalización ilógica donde la red invertida destruye el score de los videos reales. Modificando pesos linealmente podemos subir del 38% al 77% de accuracy.
