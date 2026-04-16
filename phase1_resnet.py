import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ── 1. DATASET ────────────────────────────────────────────────────────────────

class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        good_dir = os.path.join(root_dir, 'good')
        for fname in os.listdir(good_dir):
            if fname.endswith('.png'):
                self.images.append(os.path.join(good_dir, fname))
                self.labels.append(0)

        defect_dirs = [d for d in os.listdir(root_dir) if d != 'good']
        for defect in defect_dirs:
            defect_path = os.path.join(root_dir, defect)
            for fname in os.listdir(defect_path):
                if fname.endswith('.png'):
                    self.images.append(os.path.join(defect_path, fname))
                    self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# ── 2. DATA LOADING ───────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# overall ratio = train ratio = test ratio (stratified split)
all_dataset = MVTecDataset('data/metal_nut/test', transform=transform)
indices = list(range(len(all_dataset)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42,
                                       stratify=all_dataset.labels)
train_dataset = torch.utils.data.Subset(all_dataset, train_idx)
test_dataset = torch.utils.data.Subset(all_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train: {len(train_dataset)} images, Test: {len(test_dataset)} images")


# ── 3. MODEL ──────────────────────────────────────────────────────────────────

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model = models.resnet18(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 2)  # replace final layer: 1000 → 2
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)


# ── 4. TRAINING ───────────────────────────────────────────────────────────────

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(20):
    # --- train ---
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)

    # --- test ---
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_losses.append(running_loss / len(test_loader))
    test_accuracies.append(100 * correct / total)

    print(f"Epoch {epoch+1:2d} | Train Loss: {train_losses[-1]:.4f} Acc: {train_accuracies[-1]:.1f}% | Test Loss: {test_losses[-1]:.4f} Acc: {test_accuracies[-1]:.1f}%")


# ── 5. EVALUATION ─────────────────────────────────────────────────────────────

model.eval()
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

print(f"\nAUROC: {roc_auc_score(all_labels, all_probs):.4f}")
print(classification_report(all_labels, (all_probs > 0.5).astype(int),
                             target_names=['normal', 'defective']))


# ── 6. PLOTS ──────────────────────────────────────────────────────────────────

epochs = range(1, 21)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train')
plt.plot(epochs, test_losses, label='Test')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train')
plt.plot(epochs, test_accuracies, label='Test')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy %')
plt.legend()

plt.tight_layout()
plt.savefig('resnet_curves.png')
print("Curves saved to resnet_curves.png")

# confusion matrix
cm = confusion_matrix(all_labels, (all_probs > 0.5).astype(int))
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['normal', 'defective'],
            yticklabels=['normal', 'defective'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('ResNet-18 Confusion Matrix')
plt.savefig('resnet_confusion.png')
print("Confusion matrix saved to resnet_confusion.png")