# ---- Build args (ปรับได้ตอน build) -----------------------------------------
ARG DEVICE=cpu            # cpu | gpu
ARG TORCH_VERSION=2.3.1   # PyTorch เวอร์ชัน
ARG TV_VERSION=0.18.1     # torchvision
ARG TA_VERSION=2.3.1      # torchaudio

# ---- Base image --------------------------------------------------------------
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ไลบรารีระบบพื้นฐาน
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates tzdata \
    libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# ---- ติดตั้ง PyTorch --------------------------------------------------------
ARG DEVICE
ARG TORCH_VERSION
ARG TV_VERSION
ARG TA_VERSION
RUN python -m pip install --upgrade pip
RUN if [ "$DEVICE" = "gpu" ]; then \
      python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==${TORCH_VERSION} torchvision==${TV_VERSION} torchaudio==${TA_VERSION}; \
    else \
      python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==${TORCH_VERSION} torchvision==${TV_VERSION} torchaudio==${TA_VERSION}; \
    fi

# ---- ติดตั้ง dependencies อื่น ๆ --------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN (grep -viE '^(torch|torchvision|torchaudio)' requirements.txt > /tmp/req.no-torch.txt || true) \
 && python -m pip install -r /tmp/req.no-torch.txt

# ---- คัดลอกโค้ด -------------------------------------------------------------
COPY . /app
RUN mkdir -p /app/runs

CMD ["/bin/bash"]
