import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image
from PIL import Image

from src.dataset import XRayDataset
from src.model import SimpleCNN, get_transfer_model
from src.utils import load_checkpoint, compute_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test transforms
    tf_test = transforms.Compose([
        transforms.Resize(cfg['data']['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['data']['mean'], cfg['data']['std'])
    ])

    ds_test = XRayDataset(cfg['data']['test_dir'], tf_test)
    loader  = DataLoader(ds_test, batch_size=cfg['training']['batch_size'],
                         shuffle=False, num_workers=cfg['training']['num_workers'])

    num_classes = len(ds_test.classes)
    if cfg['model']['type'].lower() == 'simple':
        model = SimpleCNN(num_classes)
    else:
        model = get_transfer_model(cfg['model']['type'], num_classes)
    model.to(device)
    load_checkpoint(args.weights, model)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            with autocast():
                out = model(imgs)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(lbls.cpu().tolist())

    acc, prec, rec, f1 = compute_metrics(all_preds, all_labels)
    print(f"Test  Acc: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")


    cm = confusion_matrix(all_labels, all_preds)
    classes = ds_test.classes

    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Annotate counts
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j],
                    ha='center', va='center',
                    color='white' if cm[i,j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()

    mis = [(p, t, pr) for (p,_), t, pr in zip(ds_test.samples, all_labels, all_preds) if t!=pr]
    examples = mis[:9]  # first 9

    fig, axes = plt.subplots(3, 3, figsize=(9,9))
    for ax, (path, true_lbl, pred_lbl) in zip(axes.flatten(), examples):
        img = Image.open(path).convert('RGB')
        img = img.resize(tuple(cfg['data']['img_size']))
        ax.imshow(img)
        ax.set_title(f"T:{ds_test.classes[true_lbl]}\nP:{ds_test.classes[pred_lbl]}", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('results/misclassified_grid.png')
    plt.close()

if __name__ == '__main__':
    main()
