import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib as plt

def visualize_predictions(model, dataset, device, num_images=5):
    model.eval()
    with torch.no_grad():
        for i in range(num_images):
            image, mask = dataset[i]
            image_input = image.unsqueeze(0).to(device)
            output = model(image_input)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
            axes[0].set_title("Image")
            axes[1].imshow(mask.numpy(), cmap='gray')
            axes[1].set_title("Masque Réel")
            axes[2].imshow(pred_mask, cmap='gray')
            axes[2].set_title("Prédiction")
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            plt.show()
