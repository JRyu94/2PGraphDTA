# 2PGraphDTA: Drug–Target Binding Affinity Prediction using GAT & GCN

This repository contains code for predicting drug–target binding affinity using Graph Neural Networks (GAT for proteins, GCN for drugs) on Davis and KIBA datasets.

---

# Directory Structure

```
.
├── data/
│   ├── davis/
│   │   ├── Davis.csv  ← download required
│   │   ├── drug_graphs_bin_davis.zip
│   │   └── protein_graphs_bin_davis.zip
│   └── kiba/
│       ├── KIBA.csv   ← download required
│       ├── drug_graphs_bin_kiba.zip
│       └── protein_graphs_bin_kiba.zip
├── data_loader.py
├── emetrics.py
├── models.py
├── test.py
├── train_5fold.py
├── requirements.txt
```

---

# Step 0: Download CSV Files

Please download the required `.csv` files and place them in the appropriate folders:

| Dataset | File         | URL                                                                 |
|---------|--------------|----------------------------------------------------------------------|
| Davis   | Davis.csv    | [Davis.csv](https://github.com/JK-Liu7/AttentionMGT-DTA/blob/main/data/Davis/Davis.csv) *(right-click → Save As)* |
| KIBA    | KIBA.csv     | [KIBA.csv](https://github.com/JK-Liu7/AttentionMGT-DTA/blob/main/data/KIBA/KIBA.csv) *(right-click → Save As)* |

Save to:
- `data/davis/Davis.csv`
- `data/kiba/KIBA.csv`

---

# Step 1: Unzip Graph Files

You have two options to prepare graph files for training:

Option 1: Use data_loader.py to generate .bin graphs

This will automatically create the following folders:

- `data/davis/drug_graphs_bin_davis/`

- `data/davis/protein_graphs_bin_davis/`

- `data/kiba/drug_graphs_bin_kiba/`

- `data/kiba/protein_graphs_bin_kiba/`

Make sure Davis.csv and KIBA.csv are correctly placed.

---

Option 2: Use pre-generated zipped graph files (faster)

Unzip the `.zip` graph files **before training**:

```bash
unzip data/davis/drug_graphs_bin_davis.zip -d data/davis/
unzip data/davis/protein_graphs_bin_davis.zip -d data/davis/
unzip data/kiba/drug_graphs_bin_kiba.zip -d data/kiba/
unzip data/kiba/protein_graphs_bin_kiba.zip -d data/kiba/
```

---

## Step 2: Train (5-Fold Cross Validation)

```bash
python train_5fold.py --dataset davis --epochs 1000
```

- `--dataset`: `kiba` or `davis`
- `--epochs`: Number of epochs (default=2000)

Trained models will be saved to:
```
model/best_model_<dataset>_fold<k>.pth
```

> If `model/` folder doesn’t exist, it will be created automatically.

---

## Step 3: Evaluate a Fold

```bash
python test.py --dataset davis --fold 1
```

- `--fold`: Fold number (1–5)
- Results: MSE, CI, R² ± std

---

## Model Architecture

- **Protein Encoder**: GATConv (2-layer by default)
- **Drug Encoder**: GCNConv (2-layer by default)
- **MLP**: Fully-connected layers for regression

---

## File Descriptions

| File             | Description                                      |
|------------------|--------------------------------------------------|
| `train_5fold.py` | 5-fold cross-validation training script          |
| `test.py`        | Evaluation on a selected fold                    |
| `models.py`      | GNN-based model definition (Protein GAT, Drug GCN, MLP) |
| `data_loader.py` | Graph and CSV loader for Davis/KIBA              |
| `emetrics.py`    | Evaluation metrics (MSE, CI, R²)                 |

---

## Requirements

Install dependencies (PyTorch, PyG, NumPy, etc.):

```bash
pip install -r requirements.txt
```
