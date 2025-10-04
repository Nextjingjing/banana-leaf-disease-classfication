import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# 1. Data Transformations
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# 2. Dataset & DataLoader (test set)
# ===============================
data_dir = "dataset"
test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0)

class_names = test_dataset.classes
print("Classes:", class_names)

# ===============================
# 3. Load Model
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on: {device}")

model = models.resnet18(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

model.load_state_dict(torch.load("banana_cnn.pth", map_location=device))
model = model.to(device)
model.eval()

# ===============================
# 4. Helper Function
# ===============================
def imshow(inp, title=None):
    """แปลง tensor -> image และแสดงผล"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title, fontsize=9)
    plt.axis('off')

# ===============================
# 5. Run Test & Save Results
# ===============================
n_batches_to_show = 3  # จำนวน batch ที่อยาก save
output_dir = "output/test"
os.makedirs(output_dir, exist_ok=True)

for batch_idx, (inputs, labels) in enumerate(test_loader):
    if batch_idx >= n_batches_to_show:
        break

    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(12, 8))
    for i in range(min(8, inputs.size(0))):  # โชว์ 8 รูปต่อ batch
        ax = plt.subplot(2, 4, i + 1)
        pred_class = class_names[preds[i]]
        true_class = class_names[labels[i]]
        
        if preds[i] == labels[i]:
            title = f"✅ Correct: {pred_class}"
        else:
            title = f"❌ Pred: {pred_class}\nTrue: {true_class}"
        
        imshow(inputs[i].cpu(), title=title)

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"batch_{batch_idx}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {save_path}")

print(f"All test results saved to {output_dir}")
