import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from unet import  UNet
from FcnModel import SimpleFCN

# ==================== 1. Charger le modèle ==================== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleFCN(in_channels=3, out_channels=4).to(device)
model.load_state_dict(torch.load("best_FCN_model.pth", map_location=device))
model.eval()

# ==================== 2. Palette des classes ==================== #
class_rgb = {
    0: ((34, 139, 34), "Vegetation"),
    1: ((30, 144, 255), "Water"),
    2: ((255, 215, 0), "Urban"),
    3: ((169, 169, 169), "Unlabeled"),
}

# ==================== 3. Fonction de prédiction ==================== #
def predict(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
    predicted_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
    return predicted_mask

# ==================== 4. Colorisation du masque ==================== #
def colorize_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, (rgb, _) in class_rgb.items():
        color_mask[mask == class_idx] = rgb
    return color_mask

# ==================== 5. Charger le masque réel ==================== #
def load_true_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256))
    mask = mask.astype(np.uint8)
    return colorize_mask(mask)

# ==================== 6. Tester une image ==================== #
image_path = "D:/Documents/telechargement/dataset_split/test/images/seychelles_S2_20211125_CF_v7_patch_0_5376.png"
mask_path = "D:/Documents/telechargement/dataset_split/test/binary masks/seychelles_S2_20211125_CF_v7_patch_0_5376.png"

"""image_path = "D:/Documents/telechargement/dataset_split/images_256_1/kunshan_S2_20240731_10m_CF_crop_a3_v2_9.png"
mask_path = "D:/Documents/telechargement/dataset_split/masks_256_1/kunshan_S2_20240731_10m_CF_crop_a3_v2_9.png"
"""

import os
import random

# ==================== 6. Choisir une image et un masque aléatoirement ==================== #
"""image_dir = "D:/Documents/telechargement/dataset_splitV3/test/images"
mask_dir = "D:/Documents/telechargement/dataset_splitV3/test/binary masks"

image_files = os.listdir(image_dir)
selected_image = random.choice(image_files)
image_path = os.path.join(image_dir, selected_image)
mask_path = os.path.join(mask_dir, selected_image)"""


original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
pred_mask = predict(image_path)
colored_pred = colorize_mask(pred_mask)
true_mask = load_true_mask(mask_path)


save_dir = "./test_results_fcn"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Enregistrer les images et masques
cv2.imwrite(os.path.join(save_dir, f"{os.path.basename(image_path)}_original.png"), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(save_dir, f"{os.path.basename(image_path)}_true_mask.png"), cv2.cvtColor(true_mask, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(save_dir, f"{os.path.basename(image_path)}_pred_mask.png"), cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR))


# ==================== 7. Affichage ==================== #
fig, axes = plt.subplots(1, 5, figsize=(22, 5))

axes[0].imshow(original_image)
axes[0].set_title("Image originale")
axes[0].axis("off")

axes[1].imshow(true_mask, interpolation='none')
axes[1].set_title("Masque réel")
axes[1].axis("off")

axes[2].imshow(colored_pred, interpolation='none')
axes[2].set_title("Masque prédit")
axes[2].axis("off")

legend_patches = [plt.Line2D([0], [0], color=np.array(rgb)/255, lw=4, label=name)
                  for _, (rgb, name) in class_rgb.items()]
axes[3].legend(handles=legend_patches, loc="center", fontsize=10)
axes[3].set_title("Légende")
axes[3].axis("off")

axes[4].imshow(pred_mask, cmap='gray')
axes[4].set_title("Masque brut")
axes[4].axis("off")

plt.tight_layout()
plt.show()
