import os
import shutil
import random

from pathlib import Path


# Ratios for train / val / test
RATIOS = {
    'train': 0.7,
    'val':   0.15,
    'test':  0.15
}

# Adjust this to your actual “raw” folder
RAW_DIR = Path('data/raw/covid19_radiography_dataset')
TARGET_DIR = Path('data/processed')

# These must exactly match the folder names under RAW_DIR
CLASSES = [
    'COVID',
    'Lung_Opacity',
    'Normal',
    'Viral Pneumonia'
]

random.seed(42)

def split_and_copy():
    for cls in CLASSES:
        # Source images live under <RAW_DIR>/<cls>/images/
        src_folder = RAW_DIR / cls / 'images'
        if not src_folder.exists():
            raise FileNotFoundError(f"Expected to find images under {src_folder}, but it does not exist.")

        # List all .png/.jpg/.jpeg in that folder
        all_images = list(src_folder.glob('*.png')) + list(src_folder.glob('*.jpg')) + list(src_folder.glob('*.jpeg'))
        random.shuffle(all_images)

        n = len(all_images)
        n_train = int(RATIOS['train'] * n)
        n_val = int(RATIOS['val'] * n)

        train_imgs = all_images[:n_train]
        val_imgs   = all_images[n_train:n_train + n_val]
        test_imgs  = all_images[n_train + n_val:]

        splits = {
            'train': train_imgs,
            'val':   val_imgs,
            'test':  test_imgs
        }

        for split_name, images_list in splits.items():
            out_dir = TARGET_DIR / split_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in images_list:
                # Copy each image into e.g. data/processed/train/COVID/<filename>.png
                dest_path = out_dir / img_path.name
                shutil.copy(img_path, dest_path)

        print(f"Class '{cls}': total={n}, train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

if __name__ == "__main__":
    split_and_copy()
    print("All classes split into train/val/test under data/processed/.")
