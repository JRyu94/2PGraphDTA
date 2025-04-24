import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

# 🔹 공통 그래프 데이터 경로
DRUG_GRAPH_DIR = {
    "davis": "drug_graphs_bin",
    "kiba": "drug_graphs_bin_kiba"
}
PROTEIN_GRAPH_DIR = {
    "davis": "gnn_graphs_bin",
    "kiba": "kiba_graphs_bin"
}

CSV_FILE = {
    "davis": "Davis.csv",
    "kiba": "KIBA.csv"
}

# Binding Dataset 정의
class BindingDataset(Dataset):
    def __init__(self, binding_data):
        self.binding_data = list(binding_data.items())

    def __len__(self):
        return len(self.binding_data)

    def __getitem__(self, idx):
        (protein_id, drug_id), (protein_graph_path, drug_graph_path, affinity) = self.binding_data[idx]

        protein_graph = torch.load(protein_graph_path, weights_only=False)
        drug_graph = torch.load(drug_graph_path, weights_only=False)

        affinity_tensor = torch.tensor([affinity], dtype=torch.float)
        return protein_graph, drug_graph, affinity_tensor

# PyTorch DataLoader 정의

def collate_fn(batch):
    protein_batch = [item[0] for item in batch]
    drug_batch = [item[1] for item in batch]
    labels = torch.cat([item[2] for item in batch])
    return protein_batch, drug_batch, labels

# 데이터 로드 함수 정의
def load_dataset(name="davis", batch_size=32):
    assert name in ["davis", "kiba"], "Dataset name must be either 'davis' or 'kiba'"

    csv_path = CSV_FILE[name]
    protein_dir = PROTEIN_GRAPH_DIR[name]
    drug_dir = DRUG_GRAPH_DIR[name]

    df = pd.read_csv(csv_path)
    binding_data = {}
    missing_protein = 0
    missing_drug = 0

    for _, row in df.iterrows():
        protein_id = row["PROTEIN_ID"]
        drug_id = row["COMPOUND_ID"]
        affinity = row["REG_LABEL"]

        protein_graph_file = os.path.join(protein_dir, f"{protein_id}_pocket.bin")
        drug_graph_file = os.path.join(drug_dir, f"{drug_id}.bin")

        protein_exists = os.path.exists(protein_graph_file)
        drug_exists = os.path.exists(drug_graph_file)

        if not protein_exists:
            missing_protein += 1
        if not drug_exists:
            missing_drug += 1

        if protein_exists and drug_exists:
            binding_data[(protein_id, drug_id)] = (protein_graph_file, drug_graph_file, affinity)

    print(f"✅ Total valid samples: {len(binding_data)}")
    print(f"❌ Missing protein graphs: {missing_protein}")
    print(f"❌ Missing drug graphs: {missing_drug}")

    dataset = BindingDataset(binding_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return dataset, loader
