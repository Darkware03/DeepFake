
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
ENV PYTHONPATH=/workspace
# Clonar repositorio
RUN git clone https://github.com/sclbd/deepfakebench.git .

# Copiar e instalar el archivo de requerimientos consolidado
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY ./app ./app

EXPOSE 8000

ENV PYTHONPATH="/workspace:${PYTHONPATH}"

COPY . /workspace

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/workspace"]
