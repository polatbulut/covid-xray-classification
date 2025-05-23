# COVID-XRay Classification

A PyTorch-based project to classify chest X-ray images into COVID-19, viral pneumonia, or normal categories using CNNs and transfer learning.



## Quickstart
```bash
# 1. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 2. Download & split data
bash scripts/download_data.sh
python scripts/split_dataset.py

# 3. Train model
python src/train.py --config config.yaml

# 4. Evaluate on test set
python src/evaluate.py --config config.yaml --weights models/best_model.pth
```
