# 🍌 Banana Leaf Disease Classification

This project is a Deep Learning (PyTorch) system for **banana leaf disease classification**.  
It supports **training / evaluation / testing** the model, and can run on both **CPU** and **GPU (NVIDIA)** using Docker.  
Cross-platform builds are supported (**x86 / ARM**).

---

## 📂 Project Structure

```
banana-leaf-disease-classfication/
├─ train.py
├─ test.py
├─ evaluate.py
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ .dockerignore
├─ banana_cnn.pth        # Trained model file (after training)
├─ output/               # Output results (e.g. test images, logs)
└─ dataset/              # Dataset for train/valid/test
```

---

## 🚀 Usage with Docker

### 1. Build Images

#### CPU (PyTorch CPU wheels)
```bash
docker compose build --build-arg TORCH_CHANNEL=cpu banana-cpu
```

#### GPU (CUDA 12.1, NVIDIA only)
```bash
docker compose build --build-arg TORCH_CHANNEL=cu121 banana-gpu
```

---

### 2. Run with `docker compose`

> ⚠️ Every command below uses `--rm` so the container will be removed automatically after finishing.  
> This keeps the system clean and avoids wasting disk space.

#### Train model (CPU)
```bash
docker compose run --rm banana-cpu python train.py --data dataset --out output
```

#### Evaluate model (CPU)
```bash
docker compose run --rm banana-cpu python evaluate.py --data dataset --out output
```

#### Test model (CPU)
```bash
docker compose run --rm banana-cpu python test.py --data dataset --out output
```

#### Train model (GPU)
```bash
docker compose run --rm banana-gpu python train.py --data dataset --out output
```

#### Evaluate model (GPU)
```bash
docker compose run --rm banana-gpu python evaluate.py --data dataset --out output
```

#### Test model (GPU)
```bash
docker compose run --rm banana-gpu python test.py --data dataset --out output
```

---

## ⚙️ Dataset

- Roboflow Dataset: [Banana Leaf Disease](https://app.roboflow.com/mango-0rmdb/banana-leaf-disease-yxrhe/1)  
- Original Dataset: [Kaggle - Banana Disease Recognition Dataset](https://www.kaggle.com/datasets/sujaykapadnis/banana-disease-recognition-dataset)

Expected dataset folder structure:

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
├── valid/
└── test/
```

---

## 📌 Notes

- All dependencies are installed from `requirements.txt` during Docker build.  
- PyTorch wheels are resolved via `TORCH_CHANNEL` build arg (`cpu` or `cu121`).  
- The trained model is saved as `banana_cnn.pth` in the project root.  
- Test results and generated images are saved under the `/output` folder.  
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is required for GPU support.  
- Compatible with both **x86_64** and **ARM64 (Apple Silicon)** architectures.

---

## 🖊️ License
MIT
