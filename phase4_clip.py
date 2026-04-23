import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import roc_auc_score
import os

# ── 1. LOAD CLIP ──────────────────────────────────────────────────────────────

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
model = model.to(device)

# ── 2. PROMPT SETS ────────────────────────────────────────────────────────────

prompt_sets = {
    "photo context":    ["a photo of a good metal nut",    "a photo of a defective metal nut"],
    "damage language":  ["a metal nut with no damage",     "a damaged metal nut"],
    "industrial":       ["a normal industrial part",        "a defective industrial part"],
    "minimal":          ["good",                            "defective"],
}

# ── 3. SCORING FUNCTION ───────────────────────────────────────────────────────

def score_folder(folder, prompts):
    scores = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.png'):
            continue
        img = Image.open(os.path.join(folder, fname)).convert('RGB')
        inputs = processor(text=prompts, images=img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # logits_per_image shape: (1, num_prompts) — similarity of image to each prompt
        logits = outputs.logits_per_image[0]          # shape: (2,)
        probs = logits.softmax(dim=0).cpu().numpy()   # normalize to probabilities
        scores.append(probs[1])                        # index 1 = "defective" prompt score
    return scores

# ── 4. COLLECT SCORES FOR ALL TEST IMAGES ────────────────────────────────────

test_folders = {
    'good':    0,
    'bent':    1,
    'color':   1,
    'flip':    1,
    'scratch': 1,
}

# ── 5. AUROC PER PROMPT SET ───────────────────────────────────────────────────

print(f"\n{'Prompt Set':<20} {'AUROC':>8}")
print("-" * 30)

for prompt_name, prompts in prompt_sets.items():
    all_scores = []
    all_labels = []
    for folder_name, label in test_folders.items():
        path = os.path.join('data/metal_nut/test', folder_name)
        folder_scores = score_folder(path, prompts)
        all_scores.extend(folder_scores)
        all_labels.extend([label] * len(folder_scores))

    auroc = roc_auc_score(all_labels, all_scores)
    print(f"{prompt_name:<20} {auroc:>8.4f}")

print()
print("Reference — DINOv2 + kNN (k=1): 0.9413")
print("Reference — EfficientNet (supervised): 0.9737")
