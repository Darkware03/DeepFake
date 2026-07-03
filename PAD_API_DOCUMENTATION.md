# Presentation Attack Detection (PAD) API

*Nota: Esta API representa una arquitectura heurística inspirada en Presentation Attack Detection. Actualmente se encuentra en fase teórica y carece de validación experimental mediante dataset. Las métricas indicadas son aproximaciones matemáticas (ej. DeltaE76) y el motor de decisión utiliza pesos no estadísticos.*

## Migración desde la versión anterior

**Compatibilidad Retroactiva (Backwards Compatibility):**
La API ha sido diseñada para ser **100% compatible con la versión anterior**. Ningún campo del contrato original fue modificado.

**Para aprovechar las nuevas capacidades (Nivel Empresarial):**
1. Agregue el campo `challenge_color` al FormData de su petición con el formato hexadecimal (ej. `#00FF00`).
2. Comience a evaluar el nuevo bloque `liveness` en el JSON de respuesta. Ya no se limite al campo booleano `is_manipulated`.

---

## 1. Introducción
El PAD (Detección de Ataques de Presentación) es el conjunto de tecnologías para detectar si un rostro está siendo falsificado físicamente o digitalmente frente al sensor. Nuestro motor híbrido combina DeepFake (Xception) con Liveness Activo (Challenge de Reflexión de Luz) mediante aproximaciones de visión por computadora.

---

## 2. Parámetros `POST /api/v1/detect`

| Campo | Tipo | Requerido | Descripción |
|---|---|---|---|
| `file` o `video` | File | Sí | Archivo principal a evaluar (MP4, JPG, PNG). |
| `challenge_image` | File | Opcional | Fotografía bajo iluminación. |
| `normal_image` | File | Opcional | Fotografía iluminación neutra. |
| `challenge_id` | String | Opcional | UUID o Tracking ID para la sesión. |
| `challenge_color` | String | Opcional | Color emitido en pantalla, formato `#RRGGBB`. Esencial para el cálculo heurístico del reflejo. |

---

## 3. Respuesta y Estructura

El bloque `features` exporta las variables de LBP, PSD, ColorMatch y DeltaE76 por región, útiles para futuro entrenamiento de modelos estadísticos y abandono del actual motor de reglas.
