import torch
import numpy as np
from torch_geometric.data import Batch
from sklearn.utils import resample
from model import BindingAffinityModel
from data_loader import load_dataset
from train import concordance_index, r2_score

# 테스트 프로그램

def test(model_path="model/best_model_kiba.pth", dataset_name="kiba"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 로드
    dataset, _ = load_dataset(name=dataset_name)
    test_size = int(0.1 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - test_size - val_size
    _, _, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=lambda batch: zip(*batch))

    # 특징 체험 프로필
    sample_protein, sample_drug, _ = next(iter(test_loader))
    protein_feat_dim = sample_protein[0].x.shape[1]
    drug_feat_dim = sample_drug[0].x.shape[1]

    # 모델 설정
    model = BindingAffinityModel(
        num_features_prot=protein_feat_dim,
        num_features_mol=drug_feat_dim,
        hidden_dim=128,
        output_dim=1
    ).to(device)
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

    # 성능 평가
    y_true = np.array(true_labels)
    y_pred = np.array(predicted_labels)

    n_bootstrap = 100
    mse_list, ci_list, r2_list = [], [], []

    for _ in range(n_bootstrap):
        y_t, y_p = resample(y_true, y_pred, replace=True)
        mse_list.append(np.mean((y_t - y_p)**2))
        ci_list.append(concordance_index(y_t, y_p))
        r2_list.append(r2_score(y_t, y_p))

    def mean_std(arr):
        return np.mean(arr), np.std(arr)

    mse_mean, mse_std = mean_std(mse_list)
    ci_mean, ci_std = mean_std(ci_list)
    r2_mean, r2_std = mean_std(r2_list)

    print(f" MSE: {mse_mean:.4f} ± {mse_std:.4f}")
    print(f" CI : {ci_mean:.4f} ± {ci_std:.4f}")
    print(f" R² : {r2_mean:.4f} ± {r2_std:.4f}")

if __name__ == "__main__":
    test()
