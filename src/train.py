import argparse
import yml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset import XRayDataset
from src.model import SimpleCNN, get_transfer_model
from src.utils import save_checkpoint, compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, lbls in tqdm(loader, desc='Train', leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    v_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc='Val', leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            v_loss += criterion(out, lbls).item() * imgs.size(0)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(lbls.cpu().tolist())
    acc, prec, rec, f1 = compute_metrics(all_preds, all_labels)
    return (v_loss / len(loader.dataset), acc, prec, rec, f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = yml.safe_load(open(args.config))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    augment_cfg = cfg['data'].get('augment', {})

    # Build train‐time transforms
    tf_list = [transforms.Resize(cfg['data']['img_size'])]
    if augment_cfg.get('random_horizontal_flip', False):
        tf_list.append(transforms.RandomHorizontalFlip())
    if augment_cfg.get('random_rotation', 0) > 0:
        tf_list.append(transforms.RandomRotation(augment_cfg['random_rotation']))
    if augment_cfg.get('brightness_jitter', 0) > 0 or augment_cfg.get('contrast_jitter', 0) > 0:
        tf_list.append(transforms.ColorJitter(
            brightness=augment_cfg.get('brightness_jitter', 0),
            contrast=augment_cfg.get('contrast_jitter', 0)
        ))
    tf_list += [
        transforms.ToTensor(),
        transforms.Normalize(cfg['data']['mean'], cfg['data']['std'])
    ]
    tf_train = transforms.Compose(tf_list)

    # Validation transforms (no augmentation)
    tf_val = transforms.Compose([
        transforms.Resize(cfg['data']['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['data']['mean'], cfg['data']['std'])
    ])

    # === DataLoaders ===
    ds_train = XRayDataset(cfg['data']['train_dir'], tf_train)
    ds_val   = XRayDataset(cfg['data']['val_dir'],   tf_val)

    loader_train = DataLoader(
        ds_train,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers']
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers']
    )

    # === Model setup ===

    num_classes = len(ds_train.classes)  # should be 4: ["COVID","Lung_Opacity","Normal","Viral Pneumonia"]

    if cfg['model']['type'].lower() == 'simple':
        model = SimpleCNN(num_classes)
    else:
        model = get_transfer_model(cfg['model']['type'], num_classes)

    model = model.to(device)

    # === Loss, optimizer, scheduler ===
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg['training']['lr'])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=cfg['training']['patience'],
        factor=0.5
    )

    best_acc = 0.0
    for epoch in range(1, cfg['training']['epochs'] + 1):
        train_loss = train_one_epoch(model, loader_train, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, loader_val, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # Save checkpoint if this is the best validation accuracy so far
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict()
                },
                cfg['training']['checkpoint_path']
            )

if __name__ == '__main__':
    main()
