import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Defect Detector", layout="wide")
st.title("Industrial Defect Detection")
st.caption("Upload a metal nut image — DINOv2 and CLIP will predict if it's defective.")

# ── LOAD MODELS (cached — runs once at startup) ───────────────────────────────

@st.cache_resource
def load_dinov2():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
    model.eval()
    model = model.to(device)
    return model, device

@st.cache_resource
def load_clip():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    model = model.to(device)
    return model, processor, device

@st.cache_resource
def load_knn():
    embeddings = np.load("outputs/dinov2_train_embeddings.npy")
    knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    knn.fit(embeddings)
    # use k=2 to skip self-match (each point's nearest neighbor is itself, distance ~0)
    knn2 = NearestNeighbors(n_neighbors=2, metric='euclidean')
    knn2.fit(embeddings)
    distances, _ = knn2.kneighbors(embeddings)
    # take second neighbor (index 1) — first is self
    threshold = float(np.percentile(distances[:, 1], 95))
    return knn, threshold

dinov2, dino_device = load_dinov2()
clip_model, clip_processor, clip_device = load_clip()
knn, dino_threshold = load_knn()

# ── TRANSFORMS ────────────────────────────────────────────────────────────────

dino_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLIP_PROMPTS = ["good", "defective"]

# ── BENCHMARK TABLE ───────────────────────────────────────────────────────────

with st.expander("Full Benchmark — All 6 Models", expanded=False):
    st.table({
        "Model":          ["ResNet-18", "EfficientNet-B0", "ViT-Base", "Swin-Tiny", "DINOv2 + kNN", "CLIP zero-shot"],
        "Type":           ["Supervised", "Supervised", "Supervised", "Supervised", "Unsupervised", "Zero-shot"],
        "AUROC":          [0.8947, 0.9737, 1.0000, 0.8000, 0.9413, 0.6628],
        "Labels needed?": ["Yes", "Yes", "Yes", "Yes", "No", "No"],
    })

# ── IMAGE UPLOAD ──────────────────────────────────────────────────────────────

uploaded = st.file_uploader("Upload a metal nut image (.png or .jpg)", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col_img, col_dino, col_clip = st.columns(3)

    with col_img:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    # ── DINOV2 PREDICTION ─────────────────────────────────────────────────────
    with col_dino:
        st.subheader("DINOv2 + kNN")
        st.caption("Unsupervised — no defect labels used")

        tensor = dino_transform(image).unsqueeze(0).to(dino_device)
        with torch.no_grad():
            embedding = dinov2(tensor).cpu().numpy()

        distance, _ = knn.kneighbors(embedding)
        score = float(distance[0][0])
        is_defective = score > dino_threshold

        st.metric("Anomaly Distance", f"{score:.2f}", help=f"Threshold: {dino_threshold:.2f}")
        if is_defective:
            st.error("DEFECTIVE")
        else:
            st.success("NORMAL")

        st.progress(min(score / (dino_threshold * 2), 1.0), text="Distance from normal")

    # ── CLIP PREDICTION ───────────────────────────────────────────────────────
    with col_clip:
        st.subheader("CLIP Zero-Shot")
        st.caption("Zero-shot — no training, no examples")

        inputs = clip_processor(text=CLIP_PROMPTS, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(clip_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = clip_model(**inputs)
        probs = outputs.logits_per_image[0].softmax(dim=0).cpu().numpy()

        good_pct = float(probs[0]) * 100
        defective_pct = float(probs[1]) * 100
        is_defective_clip = defective_pct > 50

        st.metric("Defective confidence", f"{defective_pct:.1f}%")
        if is_defective_clip:
            st.error("DEFECTIVE")
        else:
            st.success("NORMAL")

        st.progress(defective_pct / 100, text=f"Normal {good_pct:.1f}% / Defective {defective_pct:.1f}%")
