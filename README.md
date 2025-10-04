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
├─ banana_cnn.pth        # ไฟล์โมเดลที่ได้หลังการเทรน
└─ dataset/              # dataset สำหรับ train/valid/test
```

---

## 🚀 การใช้งานด้วย Docker

### 1. สร้างอิมเมจ

#### CPU
```bash
docker compose build banana-cpu
```

#### GPU (CUDA)
```bash
docker compose build banana-gpu
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
  python train.py --data dataset --out .
```

#### ประเมินผล (CPU)
```bash
docker compose run --rm banana-cpu \
  python evaluate.py --data dataset --out .
```

#### ทดสอบ (CPU)
```bash
docker compose run --rm banana-cpu \
  python test.py --data dataset --out .
```

#### ใช้งาน GPU
```bash
docker compose run --rm banana-gpu python train.py --data dataset --out .
```

---

## ⚙️ Dataset

- Dataset (Roboflow): [Banana Leaf Disease](https://app.roboflow.com/mango-0rmdb/banana-leaf-disease-yxrhe/1)  
- Original Dataset (Kaggle): [Banana Disease Recognition Dataset](https://www.kaggle.com/datasets/sujaykapadnis/banana-disease-recognition-dataset)

โปรเจกต์นี้คาดหวังให้มี dataset อยู่ใน `./dataset` เช่น:
```
dataset/
├── train/
│   ├── Banana Black Sigatoka Disease/
│   ├── Banana Bract Mosaic Virus Disease/
│   ├── Banana Healthy Leaf/
│   ├── Banana Insect Pest Disease/
│   ├── Banana Moko Disease/
│   ├── Banana Panama Disease/
│   └── Banana Yellow Sigatoka Disease/
├── valid/ ...
└── test/ ...
```

---

## 📌 หมายเหตุ

- ไฟล์ `requirements.txt` จะถูกติดตั้งใน container โดยเว้น **torch/torchvision** ให้ใช้จาก base image เพื่อหลีกเลี่ยงปัญหาเวอร์ชันชนกัน
- หากต้องการระบุเวอร์ชัน PyTorch เอง สามารถแก้ `Dockerfile` ได้
- ไฟล์โมเดล `.pth` จะถูกบันทึกไว้ที่ root ของโปรเจกต์
- ต้องติดตั้ง [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) หากจะรัน GPU บน Docker

---

## 🖊️ License
MIT
