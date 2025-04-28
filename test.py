import torch
import numpy as np
from torch_geometric.data import Batch
from data_loader import load_dataset, collate_fn
from models import BindingAffinityModel
from emetrics import concordance_index, r2_score, mean_std
from sklearn.utils import resample

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['kiba', 'davis'], required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    args = parser.parse_args()

    # ✅ 데이터셋 로드
    dataset, _ = load_dataset(name=args.dataset, batch_size=args.batch_size)

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # ✅ 8:1:1 split에서 test set만
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Train 때랑 동일 seed!
    )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    sample_protein, sample_drug, _ = next(iter(test_loader))
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

    model_path = f"model/best_model_{args.dataset}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for protein_batch, drug_batch, labels in test_loader:
            protein_batch = Batch.from_data_list(protein_batch).to(device)
            drug_batch = Batch.from_data_list(drug_batch).to(device)
            labels = labels.to(device)

            predictions = model(protein_batch, drug_batch)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy().flatten().tolist())

    y_true = np.array(true_labels)
    y_pred = np.array(predicted_labels)

    n_bootstrap = 100
    mse_list, ci_list, r2_list = [], [], []

    for _ in range(n_bootstrap):
        y_t, y_p = resample(y_true, y_pred, replace=True)
        mse_list.append(np.mean((y_t - y_p)**2))
        ci_list.append(concordance_index(y_t, y_p))
        r2_list.append(r2_score(y_t, y_p))

    mse_mean, mse_std = mean_std(mse_list)
    ci_mean, ci_std = mean_std(ci_list)
    r2_mean, r2_std = mean_std(r2_list)

    print(f"\n===== Final Test Result ({args.dataset}) =====")
    print(f"MSE: {mse_mean:.4f} ± {mse_std:.4f}")
    print(f"CI : {ci_mean:.4f} ± {ci_std:.4f}")
    print(f"R² : {r2_mean:.4f} ± {r2_std:.4f}")
