import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from Metrics import SegmentationMetrics
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

#======================== My files =========================
from unet import UNet
from SatelliteDataClass import SatelliteDataset
from utilities import visualize_predictions

def train():
    # ======================== 2. Train / Validation Split ========================= #
    images_path = "D:/Documents/telechargement/dubai_2/images_256"
    masks_path = "D:/Documents/telechargement/dubai_2/masks_256"

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        #A.ElasticTransform(p=0.2, alpha=1, sigma=50),
        A.RandomBrightnessContrast(p=0.2),
        A.CLAHE(p=0.3),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
        ])

    val_transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    full_dataset = SatelliteDataset(images_path, masks_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

    #train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset = torch.utils.data.Subset(
        SatelliteDataset(images_path, masks_path, transform=train_transform),
        train_indices
    )
    val_dataset = torch.utils.data.Subset(
        SatelliteDataset(images_path, masks_path, transform=val_transform),
        val_indices
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=6, pin_memory=True)


    # ======================== 3. Modèle U-Net ========================= #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=6).to(device)

    # ======================== 4. Optimiseur & Loss ========================= #
    class DiceLoss(nn.Module):
        def __init__(self):
            super(DiceLoss, self).__init__()

        def forward(self, preds, targets):
            smooth = 1.0
            preds = torch.softmax(preds, dim=1)
            targets_one_hot = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2)
            intersection = torch.sum(preds * targets_one_hot, dim=(2, 3))
            union = torch.sum(preds + targets_one_hot, dim=(2, 3))
            dice = (2. * intersection + smooth) / (union + smooth)
            return 1 - dice.mean()

    class CombinedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.dice = DiceLoss()
            self.ce = nn.CrossEntropyLoss()

        def forward(self, preds, targets):
            return 0.5 * self.dice(preds, targets) + 0.5 * self.ce(preds, targets)

    criterion = CombinedLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # ======================== 5. Entraînement & Validation ========================= #
    num_epochs = 50
    scaler = torch.amp.GradScaler()

    # === Early stopping === #
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    metrics = SegmentationMetrics(num_classes=7)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_dice = 0
        train_iou = 0
        train_pixel_acc = 0


        #bar 
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (images, masks) in enumerate(tepoch):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()

                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                train_dice += metrics.calculate_dice(outputs, masks).item()
                train_iou += metrics.calculate_iou(outputs, masks).item()
                #train_pixel_acc += metrics.calculate_pixel_accuracy(outputs, masks).item()
                tepoch.set_postfix(train_loss=train_loss/(batch_idx+1), 
                                   train_dice=train_dice/(batch_idx+1), 
                                   train_iou=train_iou/(batch_idx+1)
                                   ) #train_pixel_acc=train_pixel_acc/(batch_idx+1)

        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        #val_pixel_acc = 0

        with torch.no_grad():

            with tqdm(val_loader, unit="batch") as tepoch:
                for batch_idx, (images, masks) in enumerate(tepoch):
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    val_dice += metrics.calculate_dice(outputs, masks).item()
                    val_iou += metrics.calculate_iou(outputs, masks).item()
                    #val_pixel_acc += metrics.calculate_pixel_accuracy(outputs, masks).item()

                    tepoch.set_postfix(val_loss=val_loss/(batch_idx+1), 
                                       val_dice=val_dice/(batch_idx+1), 
                                       val_iou=val_iou/(batch_idx+1)
                                       ) #val_pixel_acc=val_pixel_acc/(batch_idx+1)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        with open('training_logs_land.txt', 'a') as log_file:
            log_file.write(f"Epoch {epoch+1}/{num_epochs}, "
                           f"Train Loss: {avg_train_loss:.4f}, "
                           f"Train Dice: {train_dice/len(train_loader):.4f}, "
                           f"Train IoU: {train_iou/len(train_loader):.4f}, "
                        
                           f"Val Loss: {avg_val_loss:.4f}, "
                           f"Val Dice: {val_dice/len(val_loader):.4f}, "
                           f"Val IoU: {val_iou/len(val_loader):.4f}, "
                           ) #f"Val Pixel Accuracy: {val_pixel_acc/len(val_loader):.4f}\n"f"Train Pixel Accuracy: {train_pixel_acc/len(train_loader):.4f},

        #console
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Dice: {train_dice/len(train_loader):.4f}, "
              f"Train IoU: {train_iou/len(train_loader):.4f}, "
               
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Dice: {val_dice/len(val_loader):.4f}, "
              f"Val IoU: {val_iou/len(val_loader):.4f}, "
              ) #f"Val Pixel Accuracy: {val_pixel_acc/len(val_loader):.4f}"

        scheduler.step(avg_val_loss)

        # === Early Stopping === #
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best6_unet_model_dubai.pth")
            print("! Nouveau meilleur modèle sauvegardé.")
        else:
            patience_counter += 1
            print(f"!! Pas d'amélioration. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("!!! Early stopping activé.")
            break

    print(" Entraînement terminé. Le meilleur modèle a été sauvegardé sous 'best5_unet_model_dubai.pth'.")

# ======================== 7. Main Protection ========================= #
if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    train()

    # gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=6).to(device)
    model.load_state_dict(torch.load("best5_unet_model_dubai.pth", map_location=device))

    # dataset validation
    images_path = "D:/Documents/telechargement/dubai_2/images_256"
    masks_path = "D:/Documents/telechargement/dubai_2/masks_256"
    full_dataset = SatelliteDataset(images_path, masks_path)
    _, val_dataset = random_split(full_dataset, [int(0.8 * len(full_dataset)), len(full_dataset) - int(0.8 * len(full_dataset))])

    # Visualisation
    visualize_predictions(model, val_dataset, device)
