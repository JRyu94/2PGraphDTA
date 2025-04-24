import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

# Protein GAT encoder (2-layer)
class ProteinGAT(nn.Module):
    def __init__(self, num_features_prot, hidden_dim, output_dim, heads=2, dropout=0.3):
        super(ProteinGAT, self).__init__()
        self.pro_conv1 = GATConv(num_features_prot, hidden_dim, heads=heads, concat=True)
        self.pro_conv2 = GATConv(hidden_dim * heads, hidden_dim * 2, heads=1, concat=False)

        self.pro_fc_g1 = nn.Linear(hidden_dim * 2, 1024)
        self.pro_fc_g2 = nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.pro_conv1(x, edge_index))
        x = self.relu(self.pro_conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.relu(self.pro_fc_g1(x))
        x = self.dropout(x)
        x = self.pro_fc_g2(x)
        x = self.dropout(x)
        return x


# Drug GCN encoder (2-layer)
class DrugGCN(nn.Module):
    def __init__(self, num_features_mol, hidden_dim, output_dim, dropout=0.3):
        super(DrugGCN, self).__init__()
        self.mol_conv1 = GCNConv(num_features_mol, hidden_dim)
        self.mol_conv2 = GCNConv(hidden_dim, hidden_dim * 2)

        self.mol_fc_g1 = nn.Linear(hidden_dim * 2, 1024)
        self.mol_fc_g2 = nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.mol_conv1(x, edge_index))
        x = self.relu(self.mol_conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)
        return x


#Binding Affinity Prediction (MLP)
class BindingAffinityModel(nn.Module):
    def __init__(self, num_features_prot, num_features_mol, hidden_dim=128, output_dim=1, dropout=0.3):
        super(BindingAffinityModel, self).__init__()
        self.protein_encoder = ProteinGAT(num_features_prot, hidden_dim, hidden_dim, dropout=dropout)
        self.drug_encoder = DrugGCN(num_features_mol, hidden_dim, hidden_dim, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim)
        )

    def forward(self, protein, drug):
        protein_feat = self.protein_encoder(protein)
        drug_feat = self.drug_encoder(drug)
        xc = torch.cat((protein_feat, drug_feat), dim=1)
        return self.mlp(xc)