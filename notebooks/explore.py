import os
from pyexpat import model
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch import device

train_dir = "../data/metal_nut/train/good"

all_images = os.listdir(train_dir)
selected = random.sample(all_images, 5)

fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i, filename in enumerate(selected):
    img_path = os.path.join(train_dir, filename)
    img = Image.open(img_path)
    img_array = np.array(img)

    axes[i].imshow(img_array)
    axes[i].set_title(f"{filename}\n{img_array.shape}")
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("../outputs/output.png")
plt.show()





