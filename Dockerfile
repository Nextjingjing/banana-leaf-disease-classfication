# ใช้ BASE_IMAGE แบบยืดหยุ่น: ค่าเริ่มต้นเป็น CPU; ถ้าอยากใช้ GPU ให้เปลี่ยนตอน build
ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cpu
FROM ${BASE_IMAGE}

# ปรับ timezone/locale ได้ตามต้องการ (ข้ามได้)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ติดตั้งเครื่องมือพื้นฐาน (ถ้าต้องการ)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates tzdata \
 && rm -rf /var/lib/apt/lists/*

# คัดลอก requirements ก่อนเพื่อ cache layer
COPY requirements.txt /app/requirements.txt

# ถ้า base image มี torch/torchvision อยู่แล้ว:
# เพื่อหลีกเลี่ยง version clash แนะนำให้ "ไม่" ติดตั้ง torch ซ้ำจาก requirements.txt
# ดังนั้น เราจะกรองบรรทัดที่ขึ้นต้นด้วย torch/torchvision/torchaudio ออกตอน install
RUN python -m pip install --upgrade pip \
 && (grep -viE '^(torch|torchvision|torchaudio)' requirements.txt > /tmp/req.no-torch.txt || true) \
 && python -m pip install -r /tmp/req.no-torch.txt

# คัดลอกซอร์สโค้ดทั้งหมด
COPY . /app

# โฟลเดอร์สำหรับผลลัพธ์/โมเดล
RUN mkdir -p /app/runs

# ค่าเริ่มต้นให้เปิด shell; ค่อยกำหนดคำสั่งตอน docker-compose หรือ docker run
CMD ["/bin/bash"]
