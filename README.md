# COVID-XRay Classification

A PyTorch-based project to classify chest X-ray images into COVID-19, viral pneumonia, or normal categories using CNNs and transfer learning.

## Project Structure
```
covid-xray-classification/
├── .gitignore
├── README.md
├── requirements.txt
├── config.yaml
│
├── data/
│   ├── raw/            # untouched downloads
│   └── processed/      # train/val/test splits
│
├── scripts/
│   ├── download_data.sh
│   └── split_dataset.py
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── utils.py
│   ├── train.py
│   └── evaluate.py
│
├── models/             # saved checkpoints (ignored)
└── results/            # figures & metrics (ignored)
```
```
## Quickstart
```bash
# 1. Clone & setup venv
git clone <your-repo-url>
cd covid-xray-classification
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download & split data
bash scripts/download_data.sh
python scripts/split_dataset.py

# 4. Train model
python src/train.py --config config.yaml

# 5. Evaluate on test set
python src/evaluate.py --config config.yaml --weights models/best_model.pth
```
