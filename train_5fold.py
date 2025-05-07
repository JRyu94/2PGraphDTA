import os
import torch
import numpy as np
import random
from torch_geometric.data import Batch
from sklearn.model_selection import KFold
from data_loader import load_dataset
from models import BindingAffinityModel
from torch.nn import MSELoss
from torch.optim import Adam
from emetrics import concordance_index, r2_score, mean_std

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['kiba', 'davis'], required=True, help='Choose dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # ✅ Set Seed
    set_seed(args.seed)

    dataset, _ = load_dataset(name=args.dataset, batch_size=args.batch_size)
    dataset_size = len(dataset)

    # ✅ 5-fold Split with shuffle and fixed random_state
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n===== Fold {fold+1}/5 =====")

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        from data_loader import collate_fn
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        sample_protein, sample_drug, _ = next(iter(train_loader))
        protein_feat_dim = sample_protein[0].x.shape[1]
        drug_feat_dim = sample_drug[0].x.shape[1]

        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        model = BindingAffinityModel(
            num_features_prot=protein_feat_dim,
            num_features_mol=drug_feat_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            dropout=args.dropout
        ).to(device)
        optimizer = Adam(model.parameters(), lr=0.0001)
        criterion = MSELoss()

        best_val_loss = float("inf")

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for protein_batch, drug_batch, labels in train_loader:
                optimizer.zero_grad()

                protein_batch = Batch.from_data_list(protein_batch).to(device)
                drug_batch = Batch.from_data_list(drug_batch).to(device)
                labels = labels.to(device)

                predictions = model(protein_batch, drug_batch)
                loss = criterion(predictions.squeeze(), labels)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for protein_batch, drug_batch, labels in val_loader:
                    protein_batch = Batch.from_data_list(protein_batch).to(device)
                    drug_batch = Batch.from_data_list(drug_batch).to(device)
                    labels = labels.to(device)

                    predictions = model(protein_batch, drug_batch)
                    loss = criterion(predictions.squeeze(), labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Fold {fold+1} | Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs("model", exist_ok=True)
                model_path = f"model/best_model_{args.dataset}_fold{fold+1}.pth"
                torch.save(model.state_dict(), model_path)
                print(f"✅ Saved best model for Fold {fold+1}: {model_path}")
