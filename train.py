import os
import torch
import numpy as np
import random
from torch_geometric.data import Batch
from data_loader import load_dataset, collate_fn
from models import BindingAffinityModel
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import random_split

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
    parser.add_argument('--dataset', type=str, choices=['kiba', 'davis'], required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    dataset, _ = load_dataset(name=args.dataset, batch_size=args.batch_size)

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size  # test는 안씀

    train_dataset, val_dataset, _ = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    sample_protein, sample_drug, _ = next(iter(train_loader))
    protein_feat_dim = sample_protein[0].x.shape[1]
    drug_feat_dim = sample_drug[0].x.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("model", exist_ok=True)
            model_path = f"model/best_model_{args.d
