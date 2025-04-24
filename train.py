import torch
from torch.optim import Adam
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from models import BindingAffinityModel
from data_loader import load_dataset
import os

# 🔹 학습 설정
EPOCHS = 2000
BATCH_SIZE = 32
HIDDEN_DIM = 128
OUTPUT_DIM = 1
DROPOUT = 0.3
LR = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 데이터 로드 ("davis" 또는 "kiba")
dataset_name = "kiba"  # 또는 "davis"
dataset, _ = load_dataset(name=dataset_name, batch_size=BATCH_SIZE)

# 데이터셋 분할
indices = list(range(len(dataset)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

from data_loader import collate_fn

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# 특징 차원 추출
sample_protein, sample_drug, _ = next(iter(train_loader))
protein_feat_dim = sample_protein[0].x.shape[1]
drug_feat_dim = sample_drug[0].x.shape[1]

print(f"Detected Protein Feature Dim: {protein_feat_dim}, Drug Feature Dim: {drug_feat_dim}")

# 모델 정의
model = BindingAffinityModel(
    num_features_prot=protein_feat_dim,
    num_features_mol=drug_feat_dim,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    dropout=DROPOUT
).to(device)

optimizer = Adam(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()

# 학습 함수 정의
def train(model, train_loader, val_loader, epochs=EPOCHS):
    best_val_loss = float("inf")

    for epoch in range(epochs):
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

        # 🔹 Validation
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
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Best 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("model", exist_ok=True)
            model_path = f"model/best_model_{dataset_name}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"✅ Best model saved to {model_path}")

# 학습 실행
train(model, train_loader, val_loader, epochs=EPOCHS)
