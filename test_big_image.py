import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from torchvision import transforms
from smooth_tiled_predictions import predict_img_with_smooth_windowing
import matplotlib.pyplot as plt

# Charger ton modèle U-Net++ en PyTorch
from FcnModel import SimpleFCN
from unet import LightNestedUNet , UNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=4).to(device)
model.load_state_dict(torch.load("unet_v3/unet16_cds.pth", map_location=device))
#C:/Users/meriem/Downloads/Chm_Unet/
def model_inference(model, batch, device):
    model.eval()
    with torch.no_grad():
        batch_tensor = torch.tensor(batch, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        outputs = model(batch_tensor)
        return outputs.cpu().numpy().transpose(0, 2, 3, 1)


# Paramètres
patch_size = 256
n_classes = 4

# Lire et normaliser l'image
img = cv2.imread("D:/Documents/telechargement/landcoverv2/images/N-33-60-D-d-1-2.tif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
scaler = MinMaxScaler()
input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)

# Prédiction avec smooth tiling
prediction = predict_img_with_smooth_windowing(
    input_img,
    window_size=patch_size,
    subdivisions=2,
    nb_classes=n_classes,
    pred_func=lambda batch: model_inference(model, batch, device)

)

# Convertir en classes
final_prediction = np.argmax(prediction, axis=2)

def label_to_rgb(predicted_image):
    class_rgb = {
        0: ((34, 139, 34), "Vegetation"),    # Vert
        1: ((30, 144, 255), "Water"),        # Bleu
        2: ((255, 215, 0), "Urban"),         # Jaune
        3: ((169, 169, 169), "Unlabeled"),   # Gris
    }

    segmented_img = np.zeros((predicted_image.shape[0], predicted_image.shape[1], 3), dtype=np.uint8)

    for label, (color, _) in class_rgb.items():
        segmented_img[predicted_image == label] = color

    return segmented_img

########################
#Plot and save results
prediction_with_smooth_blending=label_to_rgb(final_prediction)


plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.title('Testing Image')
plt.imshow(img)
plt.subplot(222)
plt.title('Prediction with smooth blending')
plt.imshow(prediction_with_smooth_blending)
plt.show()
