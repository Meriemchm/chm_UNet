import glob
import os
import cv2

IMGS_DIR = "D:/Documents/telechargement/cds/images"
OUTPUT_DIR = "D:/Documents/telechargement/cds/output_all"
TARGET_SIZE = 512

img_paths = sorted(glob.glob(os.path.join(IMGS_DIR, "*.tif")))

if not img_paths:
    print("Aucune image trouvée.")
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)

for i, img_path in enumerate(img_paths):
    img_filename = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)

    if img is None:
        print(f"Impossible de charger {img_filename}")
        continue

    # Découper en tiles de 512x512, et redimensionner les dernières tiles si nécessaire
    k = 0
    for y in range(0, img.shape[0], TARGET_SIZE):
        for x in range(0, img.shape[1], TARGET_SIZE):
            img_tile = img[y:y + TARGET_SIZE, x:x + TARGET_SIZE]

            if img_tile.shape[0] != TARGET_SIZE or img_tile.shape[1] != TARGET_SIZE:
                img_tile = cv2.resize(img_tile, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)

            out_img_path = os.path.join(OUTPUT_DIR, f"{img_filename}_{k}.jpg")
            cv2.imwrite(out_img_path, img_tile)

            k += 1

    print(f"{img_filename} traité avec {k} splits.\n")
