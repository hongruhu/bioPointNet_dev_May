# (sciLaMA_graph) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/sciLaMA_graph$

path = '/group/gquongrp/workspaces/hongruhu/sciLaMA_graph/test/'

import scanpy as sc
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns

from sciLaMA import *
# path = '/group/gquongrp/workspaces/hongruhu/sciLaMA/trajectory/SHAREseq_5CT/'
# output_path = path + 'output/'
# obj = sc.read_h5ad(path + 'rna_5ct_intersection.h5ad')
# obj.layers['raw'].max()
# # 880.0
# obj.X = obj.layers['raw'].copy()
# cell_embedding = pd.read_csv(path + 'cell_embeddings/direct_joint_cellVAE_embedding_esm2_mouse.csv', index_col=0)
# gene_embedding = pd.read_csv(path + 'gene_embeddings/direct_joint_geneVAE_embedding_esm2_mouse.csv', index_col=0)
# cell_metadata = obj.obs.copy()
# gene_metadata = pd.read_csv(output_path + 'gene_metadata.csv', index_col=0)
# gene_metadata.highly_variable.value_counts()
# # highly_variable
# # False    6006
# # True     1000
# saving_path = '/group/gquongrp/workspaces/hongruhu/sciLaMA_graph/test/'

# cell_embedding.to_csv(saving_path + 'cell_embedding.csv')
# gene_embedding.to_csv(saving_path + 'gene_embedding.csv')
# cell_metadata.to_csv(saving_path + 'cell_metadata.csv')
# gene_metadata.to_csv(saving_path + 'gene_metadata.csv')
# obj.write(saving_path + 'obj.h5ad')

adata = sc.read(path + 'obj.h5ad') 
gene_emb = pd.read_csv(path + 'gene_embedding.csv', index_col=0)
cell_emb = pd.read_csv(path + 'cell_embedding.csv', index_col=0)
gene_meta = pd.read_csv(path + 'gene_metadata.csv', index_col=0)
cell_meta = pd.read_csv(path + 'cell_metadata.csv', index_col=0)
sc.pp.normalize_total(adata, target_sum=1e4)
expr = pd.DataFrame(adata.X.todense(), index=cell_emb.index, columns=gene_emb.index)
hvg = gene_meta[gene_meta.highly_variable].index.tolist()
expr = expr[hvg]
gene_emb = gene_emb.loc[hvg]


print(gene_emb.shape, cell_emb.shape, expr.shape)
# (1000, 50) (6436, 50) (6436, 1000)



import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec
import umap



# 1. Build the graph with gene and cell nodes
G = nx.Graph()

# Add genes and cells as nodes with their original embeddings as attributes
for gene, emb in zip(gene_emb.index, gene_emb.values):
    G.add_node(f"gene_{gene}", type='gene', embedding=emb)

for cell, emb in zip(cell_emb.index, cell_emb.values):
    G.add_node(f"cell_{cell}", type='cell', embedding=emb)

# 2. Add edges (3 types)
# Type 1: Gene-gene edges based on gene embeddings
gene_sim = cosine_similarity(gene_emb.values)
threshold_gene = np.percentile(gene_sim.flatten(), 95)  # Keep top 5% of similarities

for i, gene_i in enumerate(gene_emb.index):
    for j, gene_j in enumerate(gene_emb.index):
        if i < j and gene_sim[i, j] > threshold_gene:  # Only connect highly similar genes
            G.add_edge(f"gene_{gene_i}", f"gene_{gene_j}", 
                       weight=gene_sim[i, j], 
                       edge_type='gene_gene')

# Type 2: Cell-cell edges based on cell embeddings
cell_sim = cosine_similarity(cell_emb.values)
threshold_cell = np.percentile(cell_sim.flatten(), 95)  # Keep top 5% of similarities

for i, cell_i in enumerate(cell_emb.index):
    for j, cell_j in enumerate(cell_emb.index):
        if i < j and cell_sim[i, j] > threshold_cell:  # Only connect highly similar cells
            G.add_edge(f"cell_{cell_i}", f"cell_{cell_j}", 
                       weight=cell_sim[i, j], 
                       edge_type='cell_cell')

# Type 3: Cell-gene edges based on expression matrix
# Normalize expression values to [0,1] for edge weights
expr_normalized = (expr - expr.min().min()) / (expr.max().max() - expr.min().min())
threshold_expr = np.percentile(expr_normalized.values.flatten(), 90)  # Keep top 10% of expressions

for cell in cell_emb.index:
    for gene in gene_emb.index:
        if expr_normalized.loc[cell, gene] > threshold_expr:
            G.add_edge(f"cell_{cell}", f"gene_{gene}", 
                       weight=float(expr_normalized.loc[cell, gene]), 
                       edge_type='cell_gene')

# 3. Project the graph into 2D using node2vec
node2vec = Node2Vec(G, dimensions=64, walk_length=20, num_walks=100, 
                   p=1, q=1, workers=4)
model = node2vec.fit(window=10, min_count=1)

# Extract embeddings
node_embeddings = {}
for node in G.nodes():
    node_embeddings[node] = model.wv[node]

# Convert to DataFrame
emb_df = pd.DataFrame.from_dict(node_embeddings, orient='index')

# 4. Use UMAP to project the 64D embeddings to 2D for visualization
reducer = umap.UMAP(n_components=2, random_state=42)
emb_2d = reducer.fit_transform(emb_df.values)

# Create a DataFrame with the 2D embeddings
emb_2d_df = pd.DataFrame(emb_2d, index=emb_df.index, columns=['UMAP1', 'UMAP2'])

# 5. Visualize the co-embedding
fig = plt.figure(figsize=(6,6))
# Plot cells
cell_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'cell']
cell_indices = [list(emb_2d_df.index).index(node) for node in cell_nodes]
sns.scatterplot(x=emb_2d_df.iloc[cell_indices, 0], y=emb_2d_df.iloc[cell_indices, 1], 
                hue=cell_meta.celltype.tolist(), s=50)
# Plot genes
gene_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'gene']
gene_indices = [list(emb_2d_df.index).index(node) for node in gene_nodes]
sns.scatterplot(x=emb_2d_df.iloc[gene_indices, 0], y=emb_2d_df.iloc[gene_indices, 1], 
                c='blue', alpha=0.7, label='Genes', s=20)
fig.savefig('G_node2vec.png')





# Save the co-embedding results
emb_2d_df.to_csv('gene_cell_co_embedding.csv')






















import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear as Lin, ReLU
from torch_geometric.nn import HeteroConv, NNConv, SAGEConv
from torch_geometric.data import HeteroData
import umap
import numpy as np
import matplotlib.pyplot as plt

# === Step 1: Load your real embeddings and expression matrix ===
# Replace these with your actual data
gene_feat = torch.tensor(gene_emb.values, dtype=torch.float)
cell_feat = torch.tensor(cell_emb.values, dtype=torch.float)
expr_vals = expr.values[np.nonzero(expr.values)]
expr_vals = torch.tensor(expr_vals, dtype=torch.float).unsqueeze(1)


cell_gene_edges = np.array(np.nonzero(expr.values)).T
cell_gene_edge_index = torch.tensor(cell_gene_edges).T

# Build edge index for cell-cell and gene-gene using kNN (as you did before)
# Use your precomputed: gene_edge_index, cell_edge_index

# === Step 2: Build HeteroData ===
data = HeteroData()
data['gene'].x = gene_feat
data['cell'].x = cell_feat
data['gene', 'interacts', 'gene'].edge_index = gene_edge_index
data['cell', 'interacts', 'cell'].edge_index = cell_edge_index
data['cell', 'expresses', 'gene'].edge_index = cell_gene_edge_index
data['gene', 'expressed_in', 'cell'].edge_index = cell_gene_edge_index.flip(0)
data['cell', 'expresses', 'gene'].edge_attr = expr_vals
data['gene', 'expressed_in', 'cell'].edge_attr = expr_vals


# === Step 3: Define NNConv-based encoder model ===
class HeteroEncoder(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=64):
        super().__init__()
        edge_nn = Sequential(Lin(1, 32), ReLU(), Lin(32, hidden_channels * 3))
        self.convs = HeteroConv({
            ('gene', 'interacts', 'gene'): SAGEConv((3, 3), hidden_channels),
            ('cell', 'interacts', 'cell'): SAGEConv((3, 3), hidden_channels),
            ('cell', 'expresses', 'gene'): NNConv(3, hidden_channels, edge_nn, aggr='mean'),
            ('gene', 'expressed_in', 'cell'): NNConv(3, hidden_channels, edge_nn, aggr='mean'),
        }, aggr='mean')
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        return self.convs(x_dict, edge_index_dict, edge_attr_dict)


class DGI_Hetero(torch.nn.Module):
    def __init__(self, encoder, metadata, hidden_channels):
        super().__init__()
        self.encoder = encoder
        self.readout = torch.nn.ModuleDict({
            'gene': torch.nn.Linear(hidden_channels, hidden_channels),
            'cell': torch.nn.Linear(hidden_channels, hidden_channels)
        })
    def corruption(self, x_dict):
        # Shuffle features within each node type
        return {
            node_type: x[torch.randperm(x.size(0))]
            for node_type, x in x_dict.items()
        }
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Positive pass
        pos_z = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        # Corruption (negative)
        corrupted_x = self.corruption(x_dict)
        neg_z = self.encoder(corrupted_x, edge_index_dict, edge_attr_dict)
        # Global summary (mean of node embeddings)
        summary = {
            k: self.readout[k](v.mean(dim=0, keepdim=True))  # [1, hidden]
            for k, v in pos_z.items()
        }
        loss = 0
        for node_type in pos_z:
            pos_score = (pos_z[node_type] * summary[node_type]).sum(dim=1)
            neg_score = (neg_z[node_type] * summary[node_type]).sum(dim=1)
            pos_loss = -F.logsigmoid(pos_score).mean()
            neg_loss = -F.logsigmoid(-neg_score).mean()
            loss += pos_loss + neg_loss
        return loss, pos_z



device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = HeteroEncoder(data.metadata(), hidden_channels=64).to(device)
model = DGI_Hetero(encoder, data.metadata(), hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
data = data.to(device)

for epoch in range(1, 301):
    model.train()
    loss, _ = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 == 0:
        print(f"[Epoch {epoch}] DGI Loss: {loss.item():.4f}")


# === Step 4: Run model (no training needed for visualization) ===
model.eval()
with torch.no_grad():
    _, embeddings = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

# embeddings['gene'], embeddings['cell'] → final learned representations


# === Step 5: UMAP visualization ===
gene_emb_ = embeddings['gene'].cpu().numpy()
cell_emb_ = embeddings['cell'].cpu().numpy()


emb_all = np.concatenate([gene_emb_, cell_emb_], axis=0)
labels = ['gene'] * len(gene_emb_) + ['cell'] * len(cell_emb_)

import umap

reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)
emb_2d = reducer.fit_transform(emb_all)

gene_2d = emb_2d[:len(gene_emb_)]
cell_2d = emb_2d[len(gene_emb_):]

import matplotlib.pyplot as plt

plt.figure(figsize=(9, 9))
plt.scatter(gene_2d[:, 0], gene_2d[:, 1], s=6, c='tab:blue', alpha=0.6, label='gene')
plt.scatter(cell_2d[:, 0], cell_2d[:, 1], s=6, c='tab:orange', alpha=0.6, label='cell')
plt.legend()
plt.title("UMAP of DGI-Trained Embeddings")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig("dgi_umap.png")
plt.show()























import torch
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Convert to torch tensors
gene_feat = torch.tensor(gene_emb.values, dtype=torch.float)
cell_feat = torch.tensor(cell_emb.values, dtype=torch.float)


from sklearn.neighbors import NearestNeighbors

k = 10  # number of neighbors
gene_knn = NearestNeighbors(n_neighbors=k + 1).fit(gene_emb.values)
_, gene_neighbors = gene_knn.kneighbors(gene_emb.values)
gene_edge_index = []
for i in range(gene_emb.shape[0]):
    for j in gene_neighbors[i][1:]:  # skip self-loop
        gene_edge_index.append([i, j])


gene_edge_index = torch.tensor(gene_edge_index).T  # shape [2, num_edges]


cell_knn = NearestNeighbors(n_neighbors=k + 1).fit(cell_emb.values)
_, cell_neighbors = cell_knn.kneighbors(cell_emb.values)
cell_edge_index = []
for i in range(cell_emb.shape[0]):
    for j in cell_neighbors[i][1:]:
        cell_edge_index.append([i, j])


cell_edge_index = torch.tensor(cell_edge_index).T

# Get non-zero expression edges
cell_gene_edges = np.array(np.nonzero(expr.values)).T
# shape = [num_edges, 2] with (cell_idx, gene_idx)
cell_gene_edge_index = torch.tensor(cell_gene_edges).T  # shape [2, num_edges]



data = HeteroData()

# Add node types
data['gene'].x = gene_feat
data['cell'].x = cell_feat

# Add gene-gene edges
data['gene', 'interacts', 'gene'].edge_index = gene_edge_index

# Add cell-cell edges
data['cell', 'interacts', 'cell'].edge_index = cell_edge_index

# Add cell-gene edges (can be bidirectional if needed)
data['cell', 'expresses', 'gene'].edge_index = cell_gene_edge_index
data['gene', 'expressed_in', 'cell'].edge_index = cell_gene_edge_index.flip(0)


expr_vals = expr.values[cell_gene_edges[:, 0], cell_gene_edges[:, 1]]
data['cell', 'expresses', 'gene'].edge_attr = torch.tensor(expr_vals, dtype=torch.float).unsqueeze(1)
data['gene', 'expressed_in', 'cell'].edge_attr = torch.tensor(expr_vals, dtype=torch.float).unsqueeze(1)
# HeteroData(
#   gene={ x=[2000, 3] },
#   cell={ x=[500, 3] },
#   (gene, interacts, gene)={ edge_index=[2, 20000] },
#   (cell, interacts, cell)={ edge_index=[2, 5000] },
#   (cell, expresses, gene)={
#     edge_index=[2, 999999],
#     edge_attr=[999999, 1],
#   },
#   (gene, expressed_in, cell)={
#     edge_index=[2, 999999],
#     edge_attr=[999999, 1],
#   }
# )




from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, Linear
import torch.nn.functional as F
import torch

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=64, out_channels=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            HeteroConv({
                ('gene', 'interacts', 'gene'): SAGEConv((3, 3), hidden_channels),
                ('cell', 'interacts', 'cell'): SAGEConv((3, 3), hidden_channels),
                ('cell', 'expresses', 'gene'): SAGEConv((3, 3), hidden_channels),
                ('gene', 'expressed_in', 'cell'): SAGEConv((3, 3), hidden_channels),
            }, aggr='mean')
        )
        self.lin = torch.nn.ModuleDict({
            'gene': Linear(hidden_channels, out_channels),
            'cell': Linear(hidden_channels, out_channels)
        })
    def forward(self, x_dict, edge_index_dict):
        x_dict = self.convs[0](x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        out_dict = {k: self.lin[k](v) for k, v in x_dict.items()}
        return out_dict



device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HeteroGNN(data.metadata(), hidden_channels=64, out_channels=2).to(device)


# Assuming you have `cell_labels` for classification
# Example: cell_labels = torch.tensor([...])  # shape: [500]
# And cell train indices: train_mask

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HeteroGNN(data.metadata()).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = CrossEntropyLoss()

data = data.to(device)
cell_labels = cell_labels.to(device)  # [num_cells]
train_idx = torch.where(train_mask)[0]  # Indices of cells to train on

for epoch in range(1, 201):
    model.train()
    out = model(data.x_dict, data.edge_index_dict)
    out_cell = out['cell']  # shape [500, num_classes]

    loss = criterion(out_cell[train_idx], cell_labels[train_idx])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")




# Send data to device
data = data.to(device)
model.eval()
with torch.no_grad():
    out = model(data.x_dict, data.edge_index_dict)




from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Extract embeddings
gene_emb_ = out['gene'].cpu().numpy()
cell_emb_ = out['cell'].cpu().numpy()

# Combine gene and cell embeddings
emb_all = np.concatenate([gene_emb_, cell_emb_], axis=0)

# Create type labels
labels = ['gene'] * gene_emb_.shape[0] + ['cell'] * cell_emb_.shape[0]

# Reduce to 2D
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
emb_2d = tsne.fit_transform(emb_all)

# Separate for plotting
emb_gene_2d = emb_2d[:gene_emb_.shape[0]]
emb_cell_2d = emb_2d[gene_emb_.shape[0]:]

# Plot
fig = plt.figure(figsize=(9, 9))
plt.scatter(emb_gene_2d[:, 0], emb_gene_2d[:, 1], s=5, alpha=0.7, label='gene', c='tab:blue')
plt.scatter(emb_cell_2d[:, 0], emb_cell_2d[:, 1], s=5, alpha=0.7, label='cell', c='tab:orange')
plt.legend()
plt.title("2D Graph Embeddings (TSNE)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.tight_layout()
plt.savefig("graph.png")






class HeteroEncoder(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=64):
        super().__init__()
        self.conv1 = HeteroConv({
            ('gene', 'interacts', 'gene'): SAGEConv((3, 3), hidden_channels),
            ('cell', 'interacts', 'cell'): SAGEConv((3, 3), hidden_channels),
            ('cell', 'expresses', 'gene'): SAGEConv((3, 3), hidden_channels),
            ('gene', 'expressed_in', 'cell'): SAGEConv((3, 3), hidden_channels),
        }, aggr='mean')
    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return x_dict



class DGI_Hetero(torch.nn.Module):
    def __init__(self, encoder, metadata, hidden_channels):
        super().__init__()
        self.encoder = encoder
        self.readout = torch.nn.ModuleDict({
            'gene': Linear(hidden_channels, hidden_channels),
            'cell': Linear(hidden_channels, hidden_channels)
        })
        self.sigmoid = torch.nn.Sigmoid()
    def corruption(self, x_dict):
        # Shuffle features within each type
        return {
            node_type: x[torch.randperm(x.size(0))]
            for node_type, x in x_dict.items()
        }
    def forward(self, x_dict, edge_index_dict):
        # Positive pass
        pos_z = self.encoder(x_dict, edge_index_dict)
        # Corrupted (negative) pass
        neg_x = self.corruption(x_dict)
        neg_z = self.encoder(neg_x, edge_index_dict)
        # Summary vector: global mean over node types
        summary = {
            k: self.readout[k](v.mean(dim=0, keepdim=True))  # shape [1, hidden]
            for k, v in pos_z.items()
        }
        loss = 0
        for node_type in pos_z:
            # Dot product between each node embedding and global summary
            pos_score = (pos_z[node_type] * summary[node_type]).sum(dim=1)
            neg_score = (neg_z[node_type] * summary[node_type]).sum(dim=1)
            pos_loss = -F.logsigmoid(pos_score).mean()
            neg_loss = -F.logsigmoid(-neg_score).mean()
            loss += pos_loss + neg_loss
        return loss, pos_z



encoder = HeteroEncoder(data.metadata(), hidden_channels=64).to(device)
model = DGI_Hetero(encoder, data.metadata(), hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

data = data.to(device)

for epoch in range(1, 201):
    model.train()
    loss, _ = model(data.x_dict, data.edge_index_dict)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")



model.eval()
with torch.no_grad():
    _, embeddings = model(data.x_dict, data.edge_index_dict)



import umap
import matplotlib.pyplot as plt
import numpy as np

# Get learned embeddings from your model
gene_emb_ = embeddings['gene'].cpu().numpy()
cell_emb_ = embeddings['cell'].cpu().numpy()

# Combine
emb_all = np.concatenate([gene_emb_, cell_emb_], axis=0)
labels = ['gene'] * len(gene_emb_) + ['cell'] * len(cell_emb_)

# Run UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
emb_2d = reducer.fit_transform(emb_all)

# Split back for plotting
gene_2d = emb_2d[:len(gene_emb_)]
cell_2d = emb_2d[len(gene_emb_):]


fig = plt.figure(figsize=(9, 9))
plt.scatter(gene_2d[:, 0], gene_2d[:, 1], s=5, alpha=0.7, label='gene', c='tab:blue')
sns.scatterplot(x=cell_2d[:, 0], y=cell_2d[:, 1], s=5, alpha=0.7, hue=adata.obs['Cell.Labels'].tolist())
plt.legend()
plt.title("2D Graph Embeddings via UMAP")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig("graph_umap.png")














import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear as Lin, ReLU
from torch_geometric.nn import HeteroConv, NNConv, SAGEConv
from torch_geometric.data import HeteroData
import umap
import numpy as np
import matplotlib.pyplot as plt

# === Step 1: Load your real embeddings and expression matrix ===
# Replace these with your actual data
gene_feat = torch.tensor(gene_emb.values, dtype=torch.float)
cell_feat = torch.tensor(cell_emb.values, dtype=torch.float)
expr_vals = expr.values[np.nonzero(expr.values)]
expr_vals = torch.tensor(expr_vals, dtype=torch.float).unsqueeze(1)





cell_gene_edges = np.array(np.nonzero(expr.values)).T
cell_gene_edge_index = torch.tensor(cell_gene_edges).T

# Build edge index for cell-cell and gene-gene using kNN (as you did before)
# Use your precomputed: gene_edge_index, cell_edge_index

# === Step 2: Build HeteroData ===
data = HeteroData()
data['gene'].x = gene_feat
data['cell'].x = cell_feat
data['gene', 'interacts', 'gene'].edge_index = gene_edge_index
data['cell', 'interacts', 'cell'].edge_index = cell_edge_index
data['cell', 'expresses', 'gene'].edge_index = cell_gene_edge_index
data['gene', 'expressed_in', 'cell'].edge_index = cell_gene_edge_index.flip(0)
data['cell', 'expresses', 'gene'].edge_attr = expr_vals
data['gene', 'expressed_in', 'cell'].edge_attr = expr_vals


# === Step 3: Define NNConv-based encoder model ===
class HeteroEncoder(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=64):
        super().__init__()
        edge_nn = Sequential(Lin(1, 32), ReLU(), Lin(32, hidden_channels * 3))
        self.convs = HeteroConv({
            ('gene', 'interacts', 'gene'): SAGEConv((3, 3), hidden_channels),
            ('cell', 'interacts', 'cell'): SAGEConv((3, 3), hidden_channels),
            ('cell', 'expresses', 'gene'): NNConv(3, hidden_channels, edge_nn, aggr='mean'),
            ('gene', 'expressed_in', 'cell'): NNConv(3, hidden_channels, edge_nn, aggr='mean'),
        }, aggr='mean')
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        return self.convs(x_dict, edge_index_dict, edge_attr_dict)


class DGI_Hetero(torch.nn.Module):
    def __init__(self, encoder, metadata, hidden_channels):
        super().__init__()
        self.encoder = encoder
        self.readout = torch.nn.ModuleDict({
            'gene': torch.nn.Linear(hidden_channels, hidden_channels),
            'cell': torch.nn.Linear(hidden_channels, hidden_channels)
        })
    def corruption(self, x_dict):
        # Shuffle features within each node type
        return {
            node_type: x[torch.randperm(x.size(0))]
            for node_type, x in x_dict.items()
        }
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Positive pass
        pos_z = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        # Corruption (negative)
        corrupted_x = self.corruption(x_dict)
        neg_z = self.encoder(corrupted_x, edge_index_dict, edge_attr_dict)
        # Global summary (mean of node embeddings)
        summary = {
            k: self.readout[k](v.mean(dim=0, keepdim=True))  # [1, hidden]
            for k, v in pos_z.items()
        }
        loss = 0
        for node_type in pos_z:
            pos_score = (pos_z[node_type] * summary[node_type]).sum(dim=1)
            neg_score = (neg_z[node_type] * summary[node_type]).sum(dim=1)
            pos_loss = -F.logsigmoid(pos_score).mean()
            neg_loss = -F.logsigmoid(-neg_score).mean()
            loss += pos_loss + neg_loss
        return loss, pos_z



device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = HeteroEncoder(data.metadata(), hidden_channels=64).to(device)
model = DGI_Hetero(encoder, data.metadata(), hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
data = data.to(device)

for epoch in range(1, 301):
    model.train()
    loss, _ = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 == 0:
        print(f"[Epoch {epoch}] DGI Loss: {loss.item():.4f}")


# === Step 4: Run model (no training needed for visualization) ===
model.eval()
with torch.no_grad():
    _, embeddings = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

# embeddings['gene'], embeddings['cell'] → final learned representations


# === Step 5: UMAP visualization ===
gene_emb_ = embeddings['gene'].cpu().numpy()
cell_emb_ = embeddings['cell'].cpu().numpy()


emb_all = np.concatenate([gene_emb_, cell_emb_], axis=0)
labels = ['gene'] * len(gene_emb_) + ['cell'] * len(cell_emb_)

import umap

reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)
emb_2d = reducer.fit_transform(emb_all)

gene_2d = emb_2d[:len(gene_emb_)]
cell_2d = emb_2d[len(gene_emb_):]

import matplotlib.pyplot as plt

plt.figure(figsize=(9, 9))
plt.scatter(gene_2d[:, 0], gene_2d[:, 1], s=6, c='tab:blue', alpha=0.6, label='gene')
plt.scatter(cell_2d[:, 0], cell_2d[:, 1], s=6, c='tab:orange', alpha=0.6, label='cell')
plt.legend()
plt.title("UMAP of DGI-Trained Embeddings")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig("dgi_umap.png")
plt.show()

















import simba as si


si.tools.discretize # Discretize continous values









gene_emb = pd.read_csv(saving_path + '3_direct_sciLaMA_GENE_embedding_si.csv', index_col=0)
cell_emb = pd.read_csv(saving_path + '3_direct_sciLaMA_CELL_embedding_si.csv', index_col=0)
expr = pd.DataFrame(adata.X - adata.X.min(), index=cell_emb.index, columns=gene_emb.index)
print(gene_emb.shape, cell_emb.shape, expr.shape)
# (2000, 3) (500, 3) (500, 2000)
