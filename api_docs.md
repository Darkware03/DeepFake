# Documentación de la API de DeepFake y Liveness

Esta API fue desarrollada en **FastAPI**, lo que significa que la documentación interactiva (Swagger UI) se autogenera y está disponible por defecto en:

**👉 `http://<tuservidor>:<puerto>/docs`**

## Endpoints Disponibles

### 1. Detección de Colores y Pigmentación (Liveness)
**POST** `/api/v1/detect_colors`

Este endpoint permite validar el reflejo de la piel humana al iluminar la pantalla con "n" colores sucesivos (Active Flash Liveness).

**Parámetros (multipart/form-data):**
- `normal_image` (File): Una imagen de referencia del usuario con iluminación normal (sin flash).
- `challenge_images` (Array de Archivos): Una lista de imágenes correspondientes al momento en que la pantalla destelló los colores.
- `challenge_colors` (Array de Strings): Una lista de colores en hexadecimal (ej. `#FF0000`, `#0000FF`) con el mismo orden e índice que `challenge_images`.

**Ejemplo de Respuesta:**
```json
{
  "resultados": [
    {
      "color": "#FF0000",
      "is_valid": true,
      "pigmentation_match": 0.8523,
      "reflection_strength": 0.6512,
      "risk_level": "LOW"
    },
    {
      "color": "#0000FF",
      "is_valid": false,
      "pigmentation_match": 0.1234,
      "reflection_strength": 0.2011,
      "risk_level": "HIGH"
    }
  ]
}
```

### 2. Detección de Video / Imagen Simple
**POST** `/api/v1/detect`

Este es el endpoint original para analizar un archivo multimedia (imagen o video) completo, con capacidad opcional de enviar un solo color de desafío y pasar el contenido por la red neuronal Xception.

**Parámetros (multipart/form-data):**
- `video` o `file` (File): El archivo principal a analizar (MP4, JPG, PNG).
- `challenge_image` (File, Opcional): Imagen con destello de flash (para una sola prueba).
- `normal_image` (File, Opcional): Imagen sin flash.
- `challenge_color` (String, Opcional): Color enviado en hexadecimal.

### 3. Health Check
**GET** `/health`

Endpoint de diagnóstico para revisar el estado del motor y de la carga de los pesos del modelo PyTorch.

## ¿Cómo probar la API?
Te recomiendo acceder directamente a la URL de Swagger:
`http://localhost:8000/docs`

Desde ahí podrás ver las estructuras, parámetros requeridos y subir las imágenes (`normal_image` y varios `challenge_images`) haciendo clic en el botón **"Try it out"**.