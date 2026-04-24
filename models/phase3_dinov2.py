import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

# ── 1. LOAD DINOV2 ────────────────────────────────────────────────────────────

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()
model = model.to(device)

# ── 2. TRANSFORMS ─────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── 3. FEATURE EXTRACTION ─────────────────────────────────────────────────────

def extract_embeddings(folder, label):
    embeddings = []
    labels = []
    for fname in os.listdir(folder):
        if fname.endswith('.png'):
            img = Image.open(os.path.join(folder, fname)).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(tensor)
            embeddings.append(embedding.cpu().numpy())
            labels.append(label)
    return np.vstack(embeddings), labels

# ── 4. EXTRACT ALL EMBEDDINGS ─────────────────────────────────────────────────

print("Extracting training embeddings (normal only)...")
train_embeddings, _ = extract_embeddings('../data/metal_nut/train/good', label=0)

print("Extracting test embeddings (normal + defective)...")
test_folders = {
    'good': 0,
    'bent': 1,
    'color': 1,
    'flip': 1,
    'scratch': 1
}

test_embeddings = []
test_labels = []
for folder_name, label in test_folders.items():
    path = os.path.join('../data/metal_nut/test', folder_name)
    emb, lbl = extract_embeddings(path, label)
    test_embeddings.append(emb)
    test_labels.extend(lbl)

test_embeddings = np.vstack(test_embeddings)
test_labels = np.array(test_labels)

print(f"Train embeddings: {train_embeddings.shape}")
print(f"Test embeddings: {test_embeddings.shape}")

# ── 5. KNN ANOMALY SCORING ────────────────────────────────────────────────────

k = 5
knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
knn.fit(train_embeddings)

distances, _ = knn.kneighbors(test_embeddings)
anomaly_scores = distances.mean(axis=1)

auroc = roc_auc_score(test_labels, anomaly_scores)
print(f"AUROC (k={k}): {auroc:.4f}")

for k_val in [1, 3, 5, 10, 20]:
    knn = NearestNeighbors(n_neighbors=k_val, metric='euclidean')
    knn.fit(train_embeddings)
    distances, _ = knn.kneighbors(test_embeddings)
    scores = distances.mean(axis=1)
    print(f"k={k_val:2d} → AUROC: {roc_auc_score(test_labels, scores):.4f}")

# ── 6. T-SNE VISUALIZATION ────────────────────────────────────────────────────

print("Running t-SNE...")
all_embeddings = np.vstack([train_embeddings, test_embeddings])
all_labels = np.array([0] * len(train_embeddings) + list(test_labels))

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
reduced = tsne.fit_transform(all_embeddings)

plt.figure(figsize=(8, 6))
colors = {0: 'blue', 1: 'red'}
label_names = {0: 'normal', 1: 'defective'}
for label in [0, 1]:
    mask = all_labels == label
    plt.scatter(reduced[mask, 0], reduced[mask, 1],
                c=colors[label], label=label_names[label], alpha=0.6, s=20)

plt.title('DINOv2 Embeddings — t-SNE Visualization')
plt.legend()
plt.savefig('../outputs/dinov2_tsne.png')
print("t-SNE saved to dinov2_tsne.png")

