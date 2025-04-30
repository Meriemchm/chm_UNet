import torch
from Metrics import SegmentationMetrics
from SatelliteDataClass import SatelliteDataset
from torch.utils.data import DataLoader
#from unet import UNet
from FcnModel import SimpleFCN

def evaluate_model(model, dataloader, metrics, device):
    model.eval()
    total_accuracy = 0.0
    total_f1 = 0.0
    total_dice = 0.0
    total_iou = 0.0
    count = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            acc = metrics.calculate_pixel_accuracy(outputs, masks)
            f1 = metrics.calculate_f1_score(outputs, masks)
            dice = metrics.calculate_dice(outputs, masks)
            iou = metrics.calculate_iou(outputs, masks)

            total_accuracy += acc.item()
            total_f1 += f1.item()
            total_dice += dice.item()
            total_iou += iou.item()
            count += 1

    return {
        "Accuracy": total_accuracy / count,
        "F1": total_f1 / count,
        "Dice": total_dice / count,
        "mIoU": total_iou / count
    }

def main():
    # === Chemins vers le dossier test ===
    images_path_test = "D:/Documents/telechargement/dataset_split/test/images"
    masks_path_test = "D:/Documents/telechargement/dataset_split/test/binary masks"

    # === Dataset et DataLoader ===
    test_dataset = SatelliteDataset(images_path_test, masks_path_test)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # === Modèle ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleFCN(in_channels=3, out_channels=4).to(device)
    model.load_state_dict(torch.load("Fcn/best_FCN_model.pth", map_location=device))
    model.eval()

    # === Métriques ===
    metrics = SegmentationMetrics(num_classes=5)

    # === Évaluation ===
    test_results = evaluate_model(model, test_loader, metrics, device)

    # === Sauvegarde dans un fichier ===
    with open("Fcn/test_metrics_results.txt", "w") as f:
        f.write("Résultats sur le jeu de test :\n")
        for key, value in test_results.items():
            f.write(f"{key}: {value:.4f}\n")

    print("✅ Résultats enregistrés dans test_metrics_results.txt")

if __name__ == '__main__':
    main()
