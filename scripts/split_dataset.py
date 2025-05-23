import os
import shutil
import random

from pathlib import Path

RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}

RAW_DIR = Path('data/raw/COVID-19 Radiography Database')
TARGET_DIR = Path('data/processed')
CLASSES = ['COVID', 'Viral Pneumonia', 'Normal']

random.seed(42)

for cls in CLASSES:
    images = list((RAW_DIR / cls).glob('*.png'))
    random.shuffle(images)
    n = len(images)
    train_end = int(RATIOS['train'] * n)
    val_end = train_end + int(RATIOS['val'] * n)

    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:],
    }

    for split, files in splits.items():
        out_dir = TARGET_DIR / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in files:
            shutil.copy(img_path, out_dir / img_path.name)
