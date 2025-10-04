# 🍌 Banana Leaf Disease Classification

โปรเจกต์นี้เป็นระบบจำแนกโรคใบกล้วยด้วย Deep Learning (PyTorch)  
รองรับการ train / evaluate / test โมเดล และสามารถรันได้ทั้ง **CPU** และ **GPU (NVIDIA)** ผ่าน Docker

---

## 📂 โครงสร้างโปรเจกต์

```
banana-leaf-disease-classfication/
├─ train.py
├─ test.py
├─ evaluate.py
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ .dockerignore
├─ data/                # dataset ที่ใช้เทรน/ทดสอบ
└─ runs/                # เก็บผลลัพธ์และโมเดล
```

---

## 🚀 การใช้งานด้วย Docker

### 1. สร้างอิมเมจ

#### CPU
```bash
docker build -t banana-leaf:cpu .
```

#### GPU (CUDA)
```bash
docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime -t banana-leaf:gpu .
```

---

### 2. ใช้ `docker-compose`

#### เปิด shell (CPU)
```bash
docker compose run --rm banana-cpu
```

#### เทรนโมเดล (CPU)
```bash
docker compose run --rm banana-cpu \
  python train.py --data /data --out /app/runs
```

#### ประเมินผล (CPU)
```bash
docker compose run --rm banana-cpu \
  python evaluate.py --data /data --out /app/runs
```

#### ทดสอบ (CPU)
```bash
docker compose run --rm banana-cpu \
  python test.py --data /data --out /app/runs
```

#### ใช้งาน GPU
```bash
docker compose run --rm banana-gpu python train.py --data /data --out /app/runs
```

---

## ⚙️ Dataset

- Dataset (Roboflow): [Banana Leaf Disease](https://app.roboflow.com/mango-0rmdb/banana-leaf-disease-yxrhe/1)  
- Original Dataset (Kaggle): [Banana Disease Recognition Dataset](https://www.kaggle.com/datasets/sujaykapadnis/banana-disease-recognition-dataset)

โปรเจกต์นี้คาดหวังให้มี dataset อยู่ใน `./data` เช่น:
```
data/
├─ train/
│   ├─ class_0/ ...
│   ├─ class_1/ ...
│   └─ ...
├─ valid/
│   ├─ class_0/ ...
│   └─ ...
└─ test/
    ├─ class_0/ ...
    └─ ...
```

---

## 📌 หมายเหตุ

- ไฟล์ `requirements.txt` จะถูกติดตั้งใน container โดยเว้น **torch/torchvision** ให้ใช้จาก base image เพื่อหลีกเลี่ยงปัญหาเวอร์ชันชนกัน
- หากต้องการระบุเวอร์ชัน PyTorch เอง สามารถแก้ `Dockerfile` ได้
- ใช้ `runs/` เพื่อเก็บ checkpoint และผลลัพธ์ทั้งหมด
- ต้องติดตั้ง [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) หากจะรัน GPU บน Docker

---

## 🖊️ License
MIT
