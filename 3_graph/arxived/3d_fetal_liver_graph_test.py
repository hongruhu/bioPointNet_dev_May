# (sciLaMA_graph) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/sciLaMA_graph$

path = '/group/gquongrp/workspaces/hongruhu/sciLaMA_graph/test/featal_liver/'

import scanpy as sc
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns



adata = sc.read(path + 'obj.h5ad') 

cell_emb = pd.read_csv(path + '3_direct_sciLaMA_CELL_embedding_si.csv', index_col=0)
gene_emb = pd.read_csv(path + '3_direct_sciLaMA_GENE_embedding_si.csv', index_col=0)

C = cell_emb.values
G = gene_emb.values

cell_meta = adata.obs.copy()


adata_C = sc.AnnData(X=C, obs=cell_meta)
adata_G = sc.AnnData(X=G)



import simba as si
adata_all = si.tl.embed(adata_ref=adata_C,list_adata_query=[adata_G])
palette_celltype={'TAC-1':'#1f77b4',
                  'TAC-2':'#ff7f0e',
                  'IRS':'#279e68',
                  'Medulla':"#aa40fc",
                  'Hair Shaft-cuticle.cortex':'#d62728'}
## add annotations of cells and genes
adata_all.obs['entity_anno'] = ""
adata_all.obs.loc[adata_G.obs_names, 'entity_anno'] = 'gene'
adata_all.obs.loc[adata_C.obs_names, 'entity_anno'] = adata_all.obs.loc[adata_C.obs_names, 'Cell.Labels']
adata_all.obs.head()

palette_entity_anno = palette_celltype.copy()
palette_entity_anno['gene'] = "#607e95"
si.tl.umap(adata_all,n_neighbors=15,n_components=2)
si.pl.umap(adata_all,color=['id_dataset','entity_anno'],
        #    dict_palette={'entity_anno': palette_entity_anno},
           drawing_order='original',
           fig_size=(6,5))
plt.savefig('cell_gene_umap_fetal_liver.png')






# define nodes
import numpy as np
import networkx as nx

n_cells, k = C.shape
n_genes = G.shape[0]
G_total = nx.Graph()

# Add nodes with type attribute
for i in range(n_cells):
    G_total.add_node(f"cell_{i}", node_type="cell", vec=C[i])

for j in range(n_genes):
    G_total.add_node(f"gene_{j}", node_type="gene", vec=G[j])


# define edges
from sklearn.metrics.pairwise import euclidean_distances

# Cell–cell edges (C–C)
cell_dist = euclidean_distances(C)
cc_thresh = 1.0  # or use kNN
for i in range(n_cells):
    for j in range(i + 1, n_cells):
        if cell_dist[i, j] < cc_thresh:
            G_total.add_edge(f"cell_{i}", f"cell_{j}", edge_type="cell-cell", weight=1 / (1 + cell_dist[i, j]))


# Gene–gene edges (G–G)
gene_dist = euclidean_distances(G)
gg_thresh = 1.0

for i in range(n_genes):
    for j in range(i + 1, n_genes):
        if gene_dist[i, j] < gg_thresh:
            G_total.add_edge(f"gene_{i}", f"gene_{j}", edge_type="gene-gene", weight=1 / (1 + gene_dist[i, j]))


# Cell–gene edges (C–G)
cg_dist = euclidean_distances(C, G)
cg_thresh = 1.0

for i in range(n_cells):
    for j in range(n_genes):
        if cg_dist[i, j] < cg_thresh:
            G_total.add_edge(f"cell_{i}", f"gene_{j}", edge_type="cell-gene", weight=1 / (1 + cg_dist[i, j]))



# Cell–Cell	C[i], C[j]	Euclidean (C)	Similar cells
# Gene–Gene	G[i], G[j]	Euclidean (G)	Related genes


# Co-embedding Visualization
all_embeddings = np.vstack([C, G])
labels = ["cell"] * n_cells + ["gene"] * n_genes

# UMAP or t-SNE
from umap import UMAP
coords = UMAP(n_components=2).fit_transform(all_embeddings)

plt.figure(figsize=(10,10))
plt.scatter(coords[n_cells:, 0], coords[n_cells:, 1], c='grey', label='Genes', s=10)
sns.scatterplot(x=coords[:n_cells, 0], y=coords[:n_cells, 1], hue=adata.obs['Cell.Labels'], s=50)
plt.legend()
plt.title("Cell + Gene Co-Embedding (UMAP)")
plt.show()
plt.savefig(path + 'co_embeding.png')
