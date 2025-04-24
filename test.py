import torch
import numpy as np
from torch_geometric.data import Batch
from data_loader import load_dataset, collate_fn
from models import BindingAffinityModel
from emetrics import concordance_index, r2_score, mean_std

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['kiba', 'davis'], required=True, help='Choose dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--fold', type=int, required=True, help='Which fold to test (1~5)')
    args = parser.parse_args()

    # Load full dataset and create test split
    dataset, _ = load_dataset(name=args.dataset, batch_size=args.batch_size)
    dataset_size = len(dataset)
    fold_size = dataset_size // 5

    test_start = (args.fold - 1) * fold_size
    test_end = args.fold * fold_size if args.fold != 5 else dataset_size
    test_indices = list(range(test_start, test_end))
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
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
        dropout=args.dropout,
    ).to(device)

    model_path = f"model/best_model_{args.dataset}_fold{args.fold}.pth"
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

    from sklearn.utils import resample
    n_bootstrap = 100
    mse_list, ci_list, r2_list = [], [], []

    for _ in range(n_bootstrap):
        y_t, y_p = resample(y_true, y_pred, replace=True)
        mse_list.append(np.mean((y_t - y_p) ** 2))
        ci_list.append(concordance_index(y_t, y_p))
        r2_list.append(r2_score(y_t, y_p))

    mse_mean, mse_std = mean_std(mse_list)
    ci_mean, ci_std = mean_std(ci_list)
    r2_mean, r2_std = mean_std(r2_list)

    print(f"\n===== Evaluation Result for Fold {args.fold} ({args.dataset}) =====")
    print(f"MSE: {mse_mean:.4f} ± {mse_std:.4f}")
    print(f"CI : {ci_mean:.4f} ± {ci_std:.4f}")
    print(f"R² : {r2_mean:.4f} ± {r2_std:.4f}")
