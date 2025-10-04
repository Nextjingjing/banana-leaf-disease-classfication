import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ===============================
# 1. Data Transformations (no augmentation)
# ===============================
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "valid": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# ===============================
# 2. Dataset & DataLoader
# ===============================
data_dir = "dataset"
image_datasets = {
    x: datasets.ImageFolder(root=f"{data_dir}/{x}", transform=transform[x])
    for x in ["train", "valid", "test"]
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0)  # <<< เปลี่ยน num_workers=0 ถ้าไม่อยากใช้ multi-process
    for x in ["train", "valid", "test"]
}

class_names = image_datasets["train"].classes
print("Classes:", class_names)

# ===============================
# 3. Model (ResNet18 pretrained)
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

model = models.resnet18(weights="IMAGENET1K_V1")  # แก้ deprecated pretrained=True

# freeze feature extractor
for param in model.parameters():
    param.requires_grad = False

# replace final layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

# ===============================
# 4. Loss & Optimizer
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ===============================
# 5. Training Loop
# ===============================
def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return model

# ===============================
# 6. Main Execution (Windows Safe)
# ===============================
if __name__ == "__main__":
    trained_model = train_model(model, criterion, optimizer, num_epochs=10)
    torch.save(trained_model.state_dict(), "banana_cnn.pth")
    print("Model saved as banana_cnn.pth")
