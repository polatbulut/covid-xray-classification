import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import XRayDataset
from model import SimpleCNN, get_transfer_model
from utils import save_checkpoint, compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = val_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(all_preds, all_labels)
    return avg_loss, acc, prec, rec, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transforms
    train_tf = transforms.Compose([
        transforms.Resize(cfg['data']['img_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(cfg['data']['mean'], cfg['data']['std']),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(cfg['data']['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['data']['mean'], cfg['data']['std']),
    ])

    # Datasets & loaders
    train_ds = XRayDataset(cfg['data']['train_dir'], transform=train_tf)
    val_ds   = XRayDataset(cfg['data']['val_dir'],   transform=val_tf)
    train_loader = DataLoader(train_ds,
                              batch_size=cfg['training']['batch_size'],
                              shuffle=True,
                              num_workers=cfg['training']['num_workers'])
    val_loader   = DataLoader(val_ds,
                              batch_size=cfg['training']['batch_size'],
                              shuffle=False,
                              num_workers=cfg['training']['num_workers'])

    # Model
    num_classes = len(train_ds.classes)
    if cfg['model']['type'].lower() == 'simple':
        model = SimpleCNN(num_classes)
    else:
        model = get_transfer_model(cfg['model']['type'], num_classes)
    model.to(device)

    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg['training']['lr'])
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  patience=cfg['training']['patience'],
                                  factor=0.5)

    best_acc = 0.0
    for epoch in range(1, cfg['training']['epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                {'epoch': epoch,
                 'model_state': model.state_dict(),
                 'optim_state' : optimizer.state_dict()},
                filename=cfg['training']['checkpoint_path']
            )


if __name__ == '__main__':
    main()
