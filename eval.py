import torch
import numpy as np
from torch_geometric.data import Batch
from sklearn.utils import resample
from models import BindingAffinityModel
from data_loader import load_dataset

# 평가 지표 함수들
def concordance_index(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    num_pairs, num_correct = 0, 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:
                num_pairs += 1
                if (y_pred[i] - y_pred[j]) * (y_true[i] - y_true[j]) > 0:
                    num_correct += 1
                elif (y_pred[i] - y_pred[j]) == 0:
                    num_correct += 0.5
    return num_correct / num_pairs if num_pairs > 0 else 0.0

def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


def evaluate(model_path, dataset_name="kiba", hidden_dim=128, dropout=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, loader = load_dataset(name=dataset_name, batch_size=32)

    # 특성 차원 자동 설정
    sample_protein, sample_drug, _ = next(iter(loader))
    protein_feat_dim = sample_protein[0].x.shape[1]
    drug_feat_dim = sample_drug[0].x.shape[1]

    # 모델 로드
    model = BindingAffinityModel(
        num_features_prot=protein_feat_dim,
        num_features_mol=drug_feat_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        dropout=dropout
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for protein_batch, drug_batch, labels in loader:
            protein_batch = Batch.from_data_list(protein_batch).to(device)
            drug_batch = Batch.from_data_list(drug_batch).to(device)
            labels = labels.to(device)

            predictions = model(protein_batch, drug_batch)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy().flatten().tolist())

    y_true = np.array(true_labels)
    y_pred = np.array(predicted_labels)

    # Bootstrap 평가
    n_bootstrap = 100
    mse_list, ci_list, r2_list = [], [], []

    for _ in range(n_bootstrap):
        y_t, y_p = resample(y_true, y_pred, replace=True)
        mse_list.append(np.mean((y_t - y_p) ** 2))
        ci_list.append(concordance_index(y_t, y_p))
        r2_list.append(r2_score(y_t, y_p))

    def mean_std(arr):
        return np.mean(arr), np.std(arr)

    mse_mean, mse_std = mean_std(mse_list)
    ci_mean, ci_std = mean_std(ci_list)
    r2_mean, r2_std = mean_std(r2_list)

    print(f"Evaluation Results on {dataset_name.upper()} Dataset")
    print(f"MSE: {mse_mean:.4f} ± {mse_std:.4f}")
    print(f"CI : {ci_mean:.4f} ± {ci_std:.4f}")
    print(f"R² : {r2_mean:.4f} ± {r2_std:.4f}")


if __name__ == "__main__":
    evaluate("model/best_model_kiba.pth", dataset_name="kiba")  # 또는 "davis"
