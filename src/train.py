import argparse
import yaml
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
    model.train(); total_loss = 0
    for imgs, lbls in tqdm(loader, desc='Train', leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval(); v_loss=0; preds=[]; actual=[]
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc='Val', leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            v_loss += criterion(out, lbls).item() * imgs.size(0)
            pred = out.argmax(1)
            preds.extend(pred.cpu().tolist()); actual.extend(lbls.cpu().tolist())
    metrics = compute_metrics(preds, actual)
    return v_loss/len(loader.dataset), *metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tf_train = transforms.Compose([
        transforms.Resize(cfg['data']['img_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cfg['data']['mean'], cfg['data']['std'])
    ])
    tf_val = transforms.Compose([
        transforms.Resize(cfg['data']['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['data']['mean'], cfg['data']['std'])
    ])

    ds_train = XRayDataset(cfg['data']['train_dir'], tf_train)
    ds_val   = XRayDataset(cfg['data']['val_dir'],   tf_val)
    loader_train = DataLoader(ds_train, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=cfg['training']['num_workers'])
    loader_val   = DataLoader(ds_val,   batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=cfg['training']['num_workers'])

    nc = len(ds_train.classes)
    if cfg['model']['type']=='simple': model=SimpleCNN(nc)
    else: model=get_transfer_model(cfg['model']['type'], nc)
    model.to(device)

    crit = nn.CrossEntropyLoss()
    opt  = Adam(model.parameters(), lr=cfg['training']['lr'])
    sched=ReduceLROnPlateau(opt, mode='min', patience=cfg['training']['patience'])

    best_acc=0
    for e in range(1, cfg['training']['epochs']+1):
        tr_loss = train_one_epoch(model, loader_train, crit, opt, device)
        val_loss, acc, prec, rec, f1 = validate(model, loader_val, crit, device)
        sched.step(val_loss)
        print(f"Epoch {e}: TrainLoss={tr_loss:.4f} ValLoss={val_loss:.4f} Acc={acc:.4f} F1={f1:.4f}")
        if acc>best_acc:
            best_acc=acc
            save_checkpoint({'model_state':model.state_dict(),'optim_state':opt.state_dict()}, cfg['training']['checkpoint_path'])

if __name__=='__main__':
    main()

