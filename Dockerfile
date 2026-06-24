FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clonar DeepFakeBench original
RUN git clone https://github.com/SCLBD/DeepfakeBench.git .

# Instalar dependencias del API
COPY requirements_api.txt /tmp/requirements_api.txt
RUN pip install --no-cache-dir -r /tmp/requirements_api.txt

# Instalar torchvision si no estuviera disponible (por si acaso, aunque pytorch base suele incluirlo)
RUN pip install --no-cache-dir torchvision

# Copiar el código del usuario y pesos
COPY . /workspace

# Configurar PYTHONPATH para que DeepFakeBench sea importable directamente
ENV PYTHONPATH=/workspace

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
