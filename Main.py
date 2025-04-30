import torch
import torch.nn as nn
import torch.multiprocessing as mp
from Modeltrain import train
from unet import UNet
import os
# ======================== 7. Main Protection ========================= #
if __name__ == '__main__':
    

    mp.set_start_method('spawn')
    NUM_CLASSES = 4
    output = "./unet_datasplit_newimage"
    os.makedirs(output, exist_ok=True)
    # ======================== 3. Modèle U-Net ========================= #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=NUM_CLASSES).to(device)
    #model = LightNestedUNet(in_channels=3, out_channels=NUM_CLASSES).to(device)
    train(model,NUM_CLASSES,output)