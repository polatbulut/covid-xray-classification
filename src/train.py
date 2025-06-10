import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from tqdm import tqdm

from src.dataset import XRayDataset
from src.model import SimpleCNN, get_transfer_model
from src.utils import save_checkpoint, compute_metrics

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0.0
    for imgs, lbls in tqdm(loader, desc='Train', leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        with autocast():
            out = model(imgs)
            loss = criterion(out, lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    v_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc='Val', leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            with autocast():
                out = model(imgs)
                v_loss += criterion(out, lbls).item() * imgs.size(0)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(lbls.cpu().tolist())
    return (v_loss / len(loader.dataset), *compute_metrics(all_preds, all_labels))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transforms
    aug = cfg['data'].get('augment', {})
    tf_list = [transforms.Resize(cfg['data']['img_size'])]
    if aug.get('random_horizontal_flip', False):    tf_list.append(transforms.RandomHorizontalFlip())
    if aug.get('random_rotation', 0) > 0:           tf_list.append(transforms.RandomRotation(aug['random_rotation']))
    if aug.get('brightness_jitter', 0)>0 or aug.get('contrast_jitter',0)>0:
        tf_list.append(transforms.ColorJitter(
            brightness=aug.get('brightness_jitter',0),
            contrast=aug.get('contrast_jitter',0)
        ))
    tf_list += [
        transforms.ToTensor(),
        transforms.Normalize(cfg['data']['mean'], cfg['data']['std'])
    ]
    tf_train = transforms.Compose(tf_list)
    tf_val   = transforms.Compose([
        transforms.Resize(cfg['data']['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['data']['mean'], cfg['data']['std'])
    ])

    # Datasets & Loaders
    ds_train = XRayDataset(cfg['data']['train_dir'], tf_train)
    ds_val   = XRayDataset(cfg['data']['val_dir'],   tf_val)

    loader_train = DataLoader(ds_train, batch_size=cfg['training']['batch_size'],
                              shuffle=True, num_workers=cfg['training']['num_workers'])
    loader_val   = DataLoader(ds_val,   batch_size=cfg['training']['batch_size'],
                              shuffle=False, num_workers=cfg['training']['num_workers'])

    num_classes = len(ds_train.classes)

    counts = [sum(1 for _, l in ds_train.samples if l == i) for i in range(num_classes)]
    total = sum(counts)
    weights = [total / c for c in counts]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Model
    if cfg['model']['type'].lower() == 'simple':
        model = SimpleCNN(num_classes)
    else:
        model = get_transfer_model(cfg['model']['type'], num_classes)
    model.to(device)

    # Optimizer, OneCycleLR & GradScaler
    optimizer = Adam(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training'].get('weight_decay',1e-4))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg['training']['lr'],
        steps_per_epoch=len(loader_train),
        epochs=cfg['training']['epochs'],
        pct_start=cfg['training'].get('pct_start',0.1),
        anneal_strategy=cfg['training'].get('anneal_strategy','cos')
    )
    scaler = GradScaler()

    best_acc = 0.0
    for epoch in range(1, cfg['training']['epochs'] + 1):
        train_loss = train_one_epoch(model, loader_train, criterion, optimizer, scheduler, scaler, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, loader_val, criterion, device)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss:   {val_loss:.4f} | "
              f"Val Acc:    {val_acc:.4f} | "
              f"Val F1:     {val_f1:.4f}")

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
