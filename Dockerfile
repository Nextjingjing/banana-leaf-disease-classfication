# ใช้ 3.11 เพื่อรองรับ contourpy>=1.3.3 (ถ้าจะคง 3.10 ให้ลด contourpy เป็น 1.3.2)
FROM python:3.11-slim

ARG TORCH_CHANNEL=cpu     # cpu | cu121
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates tzdata \
    libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# ใช้ PyPI เป็นหลักเสมอ แล้ว “เพิ่ม” ช่องทางของ PyTorch
RUN python -m pip install --upgrade pip && \
    if [ "$TORCH_CHANNEL" = "cpu" ]; then \
        python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /app/requirements.txt ; \
    else \
        python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r /app/requirements.txt ; \
    fi

COPY . /app
CMD ["/bin/bash"]
