# (sciLaMA_graph) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/sciLaMA_graph$

path = '/group/gquongrp/workspaces/hongruhu/sciLaMA_graph/test/'

import scanpy as sc
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns



adata = sc.read('/group/gquongrp/workspaces/hongruhu/bioPointNet/result/AD/seaad_MTG_microglia_sciLaMA/direct_sciLaMA_seaad_MTG.h5ad') 

cell_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/result/AD/seaad_MTG_microglia_sciLaMA/direct_sciLaMA_CELL_embedding.csv', index_col=0)
gene_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/result/AD/seaad_MTG_microglia_sciLaMA/direct_sciLaMA_GENE_embedding.csv', index_col=0)

C = cell_emb.values
G = gene_emb.values

cell_meta = adata.obs.copy()


adata_C = sc.AnnData(X=C, obs=cell_meta)
adata_G = sc.AnnData(X=G)



# import simba as si
# adata_all = si.tl.embed(adata_ref=adata_C,list_adata_query=[adata_G])
# palette_celltype={'TAC-1':'#1f77b4',
#                   'TAC-2':'#ff7f0e',
#                   'IRS':'#279e68',
#                   'Medulla':"#aa40fc",
#                   'Hair Shaft-cuticle.cortex':'#d62728'}
# ## add annotations of cells and genes
# adata_all.obs['entity_anno'] = ""
# adata_all.obs.loc[adata_G.obs_names, 'entity_anno'] = 'gene'
# adata_all.obs.loc[adata_C.obs_names, 'entity_anno'] = adata_all.obs.loc[adata_C.obs_names, 'cell_type']
# adata_all.obs.head()

# palette_entity_anno = palette_celltype.copy()
# palette_entity_anno['gene'] = "#607e95"
# si.tl.umap(adata_all,n_neighbors=15,n_components=2)
# si.pl.umap(adata_all,color=['id_dataset','entity_anno'],
#         #    dict_palette={'entity_anno': palette_entity_anno},
#            drawing_order='original',
#            fig_size=(6,5))
# plt.savefig('cell_gene_umap_microglia.png')



import os
from sklearn.metrics.pairwise import cosine_similarity

def gen_graph_cosine(
        adata, cell_emb, gene_emb,
        prefix_C='C',
        prefix_G='G',
        dirname='graph_emb_cosine',
        pbg_params=None,
        sim_threshold=0.2  # cosine similarity threshold
):
    filepath = './' + dirname + '/'
    os.makedirs(filepath + "input/edge", exist_ok=True)
    os.makedirs(filepath + "input/entity", exist_ok=True)
    # Get embeddings
    C = cell_emb # Cell embeddings (n_cells x k)
    G = gene_emb # Gene embeddings (n_genes x k)
    # Compute cosine similarity
    sim = cosine_similarity(C, G)  # shape: (n_cells x n_genes)
    # Cell and gene alias mappings
    ids_cells = adata.obs.index
    ids_genes = adata.var.index
    df_cells = pd.DataFrame(index=ids_cells, columns=['alias'],
                            data=[f'{prefix_C}.{i}' for i in range(len(ids_cells))])
    df_genes = pd.DataFrame(index=ids_genes, columns=['alias'],
                            data=[f'{prefix_G}.{j}' for j in range(len(ids_genes))])
    entity_alias = pd.concat([df_cells, df_genes])
    # Create edge list from cosine similarity
    edges = []
    for i in range(C.shape[0]):
        for j in range(G.shape[0]):
            if sim[i, j] > sim_threshold:
                edges.append((df_cells.iloc[i, 0], "sciLaMA_cos_sim", df_genes.iloc[j, 0]))
    df_edges = pd.DataFrame(edges, columns=["source", "relation", "destination"])
    print(f"Total edges (cosine sim > {sim_threshold}): {df_edges.shape[0]}")
    # Save files for PBG
    df_edges.to_csv(os.path.join(filepath, "pbg_graph.txt"),
                    sep='\t', header=False, index=False)
    entity_alias.to_csv(os.path.join(filepath, "entity_alias.txt"),
                        sep='\t', header=True, index=True)
    # PBG params update
    pbg_params['entity_path'] = os.path.join(filepath, "input/entity")
    pbg_params['edge_paths'] = [os.path.join(filepath, "input/edge")]
    pbg_params['checkpoint_path'] = os.path.join(filepath, "model")
    pbg_params['entities'] = {
        prefix_C: {'num_partitions': 1},
        prefix_G: {'num_partitions': 1}
    }
    pbg_params['relations'] = [{
        'name': 'sciLaMA_cos_sim',
        'lhs': prefix_C,
        'rhs': prefix_G,
        'operator': 'none',
        'weight': 1.0  # or use sim[i,j] if you want to make edge weights variable
    }]
    return df_edges, entity_alias, {
        'n_edges': df_edges.shape[0],
        'source': prefix_C,
        'destination': prefix_G,
    }


pbg_params = dict(
    entity_path="",
    edge_paths=[""],
    checkpoint_path="",
    entities={},
    relations=[],
    dynamic_relations=False,
    dimension=50,
    global_emb=False,
    comparator='dot',
    num_epochs=10,
    workers=4,
    num_batch_negs=50,
    num_uniform_negs=50,
    loss_fn='softmax',
    lr=0.1,
    early_stopping=False,
    regularization_coef=0.0,
    wd=0.0,
    wd_interval=50,
    eval_fraction=0.05,
    eval_num_batch_negs=50,
    eval_num_uniform_negs=50,
    checkpoint_preservation_interval=None,
)

df_edges, entity_alias, graph_stats = gen_graph_cosine(adata, 
                                                       cell_emb, gene_emb, 
                                                       dirname='graph_emb_cosine', 
                                                       pbg_params=pbg_params,
                                                       sim_threshold=0.2)


from coembed import *

pbg_train(
    dirname='graph_emb_cosine',
    output='model',
    auto_wd=True,
    save_wd=True,
    graph_stats=graph_stats,
    pbg_params=pbg_params
)



path_emb = pbg_params['checkpoint_path']
path_entity = pbg_params['entity_path']
num_epochs = pbg_params["num_epochs"]
prefix = []
path_entity_alias = Path(path_emb).parent.as_posix()
df_entity_alias = pd.read_csv(
    os.path.join(path_entity_alias, 'entity_alias.txt'),
    header=0,
    index_col=0,
    sep='\t')



df_entity_alias['id'] = df_entity_alias.index
df_entity_alias.index = df_entity_alias['alias'].values
convert_alias = True
dict_adata = dict()
for x in os.listdir(path_emb):
    if x.startswith('embeddings'):
        entity_type = x.split('_')[1]
        if (len(prefix) == 0) or (entity_type in prefix):
            adata = \
                read_hdf(os.path.join(path_emb,
                                        f'embeddings_{entity_type}_0.'
                                        f'v{num_epochs}.h5'),
                            key="embeddings")
            with open(
                os.path.join(path_entity,
                                f'entity_names_{entity_type}_0.json'), "rt")\
                    as tf:
                names_entity = json.load(tf)
            if convert_alias:
                names_entity = \
                    df_entity_alias.loc[names_entity, 'id'].tolist()
            adata.obs.index = names_entity
            dict_adata[entity_type] = adata






















# (sciLaMA_graph) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/sciLaMA_graph$

path = '/group/gquongrp/workspaces/hongruhu/sciLaMA_graph/test/'

import scanpy as sc
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns



adata = sc.read('/group/gquongrp/workspaces/hongruhu/bioPointNet/result/AD/seaad_MTG_microglia_sciLaMA/direct_sciLaMA_seaad_MTG.h5ad') 

cell_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/result/AD/seaad_MTG_microglia_sciLaMA/direct_sciLaMA_CELL_embedding.csv', index_col=0)
gene_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/result/AD/seaad_MTG_microglia_sciLaMA/direct_sciLaMA_GENE_embedding.csv', index_col=0)

C = cell_emb.values
G = gene_emb.values

cell_meta = adata.obs.copy()


adata_C = sc.AnnData(X=C, obs=cell_meta)
adata_G = sc.AnnData(X=G)



from coembed import *
from sklearn.metrics.pairwise import cosine_similarity

def gen_graph_cgc_cosine(cell_emb, gene_emb, cell_ids, gene_ids,
                         dirname='graph_CGC',
                         sim_threshold=0.2,
                         cc_threshold=0.3,
                         gg_threshold=0.3,
                         include_cell_cell=True,
                         include_gene_gene=True,
                         pbg_params=None,
                         prefix_C='C', prefix_G='G'):
    filepath = './' + dirname + '/'
    os.makedirs(filepath + "input/edge", exist_ok=True)
    os.makedirs(filepath + "input/entity", exist_ok=True)
    # Alias tables
    df_cells = pd.DataFrame(index=cell_ids, columns=['alias'],
                            data=[f'{prefix_C}.{i}' for i in range(len(cell_ids))])
    df_genes = pd.DataFrame(index=gene_ids, columns=['alias'],
                            data=[f'{prefix_G}.{j}' for j in range(len(gene_ids))])
    entity_alias = pd.concat([df_cells, df_genes])
    # Initialize edge list
    edges = []
    relations = []
    # --- 1. Cell ↔ Gene (C-G)
    sim_cg = cosine_similarity(cell_emb, gene_emb)
    for i in range(cell_emb.shape[0]):
        for j in range(gene_emb.shape[0]):
            if sim_cg[i, j] > sim_threshold:
                edges.append((df_cells.iloc[i, 0], "cell-gene", df_genes.iloc[j, 0]))
    relations.append({'name': 'cell-gene', 'lhs': prefix_C, 'rhs': prefix_G, 'operator': 'none', 'weight': 1.0})
    # --- 2. Cell ↔ Cell (C-C)
    if include_cell_cell:
        sim_cc = cosine_similarity(cell_emb)
        for i in range(cell_emb.shape[0]):
            for j in range(i + 1, cell_emb.shape[0]):
                if sim_cc[i, j] > cc_threshold:
                    edges.append((df_cells.iloc[i, 0], "cell-cell", df_cells.iloc[j, 0]))
        relations.append({'name': 'cell-cell', 'lhs': prefix_C, 'rhs': prefix_C, 'operator': 'none', 'weight': 1.0})
    # --- 3. Gene ↔ Gene (G-G)
    if include_gene_gene:
        sim_gg = cosine_similarity(gene_emb)
        for i in range(gene_emb.shape[0]):
            for j in range(i + 1, gene_emb.shape[0]):
                if sim_gg[i, j] > gg_threshold:
                    edges.append((df_genes.iloc[i, 0], "gene-gene", df_genes.iloc[j, 0]))
        relations.append({'name': 'gene-gene', 'lhs': prefix_G, 'rhs': prefix_G, 'operator': 'none', 'weight': 1.0})
    # Save edge list and entity aliases
    df_edges = pd.DataFrame(edges, columns=["source", "relation", "destination"])
    df_edges.to_csv(os.path.join(filepath, "input/edge/pbg_graph.txt"),
                    sep='\t', header=False, index=False)
    entity_alias.to_csv(os.path.join(filepath, "input/entity/entity_alias.txt"),
                        sep='\t', header=True, index=True)
    # Update PBG config
    pbg_params['entity_path'] = os.path.join(filepath, "input/entity")
    pbg_params['edge_paths'] = [os.path.join(filepath, "input/edge")]
    pbg_params['checkpoint_path'] = os.path.join(filepath, "model")
    pbg_params['entities'] = {
        prefix_C: {'num_partitions': 1},
        prefix_G: {'num_partitions': 1}
    }
    pbg_params['relations'] = relations
    # Print edge counts
    edge_counts = pd.Series(df_edges['relation']).value_counts().to_dict()
    print(f"Total edges: {df_edges.shape[0]} | Breakdown: {edge_counts}")
    return df_edges, entity_alias, {'n_edges': df_edges.shape[0]}



pbg_params = dict(
    # These paths are filled dynamically by `gen_graph_cgc_cosine`
    entity_path="",
    edge_paths=[""],
    checkpoint_path="",
    # Graph structure (populated by the graph generator)
    entities={},        # ← gets filled with {'C': ..., 'G': ...}
    relations=[],       # ← gets filled based on enabled edge types
    dynamic_relations=False,
    # Embedding model
    dimension=50,           # Size of embedding space
    global_emb=False,       # No global embeddings
    comparator='dot',       # Can also use 'cosine' or 'l2' if needed
    # Training configuration
    num_epochs=10,
    workers=4,              # Number of CPU threads
    num_batch_negs=50,      # Negative samples per positive
    num_uniform_negs=50,    # Global uniform negatives
    loss_fn='softmax',
    lr=0.1,
    regularization_coef=0.0,
    wd=0.0,                 # Weight decay
    wd_interval=50,
    # Evaluation during training
    eval_fraction=0.05,
    eval_num_batch_negs=50,
    eval_num_uniform_negs=50,
    checkpoint_preservation_interval=None,
)



df_edges, entity_alias, graph_stats = gen_graph_cgc_cosine(
    cell_emb=cell_emb.values,
    gene_emb=gene_emb.values,
    cell_ids=cell_emb.index,
    gene_ids=gene_emb.index,
    dirname='graph_CG_GG_only',  # or 'graph_CGC' if you include all
    sim_threshold=0.2,
    cc_threshold=0.3,
    gg_threshold=0.3,
    include_cell_cell=False, # <--- easily disable C-C here
    include_gene_gene=True,  # <--- easily disable G-G here
    pbg_params=pbg_params
)
# Total edges: 6096828 | Breakdown: {'gene-gene': 3795496, 'cell-gene': 2301332}

pbg_train(
    dirname='graph_CG_GG_only',
    output='model',
    auto_wd=True,
    save_wd=True,
    graph_stats=graph_stats,
    pbg_params=pbg_params
)






path_emb = pbg_params['checkpoint_path']
path_entity = pbg_params['entity_path']
num_epochs = pbg_params["num_epochs"]
prefix = []
path_entity_alias = Path(path_emb).parent.as_posix()
df_entity_alias = pd.read_csv(
    os.path.join(path_entity_alias, 'entity_alias.txt'),
    header=0,
    index_col=0,
    sep='\t')



df_entity_alias['id'] = df_entity_alias.index
df_entity_alias.index = df_entity_alias['alias'].values
convert_alias = True
dict_adata = dict()
for x in os.listdir(path_emb):
    if x.startswith('embeddings'):
        entity_type = x.split('_')[1]
        if (len(prefix) == 0) or (entity_type in prefix):
            adata = \
                read_hdf(os.path.join(path_emb,
                                        f'embeddings_{entity_type}_0.'
                                        f'v{num_epochs}.h5'),
                            key="embeddings")
            with open(
                os.path.join(path_entity,
                                f'entity_names_{entity_type}_0.json'), "rt")\
                    as tf:
                names_entity = json.load(tf)
            if convert_alias:
                names_entity = \
                    df_entity_alias.loc[names_entity, 'id'].tolist()
            adata.obs.index = names_entity
            dict_adata[entity_type] = adata









adata_C = dict_adata['C']  # embeddings of cells
adata_G = dict_adata['G']  # embeddings of genes



c_entity_alias = entity_alias[entity_alias.alias.isin(adata_C.obs.index)]
c_entity_alias['cell_id'] = c_entity_alias.index.tolist()
c_entity_alias.index = c_entity_alias.alias.tolist()
c_entity_alias = c_entity_alias.loc[adata_C.obs.index]

adata_C.obs['Supertype'] = adata[c_entity_alias.cell_id].obs.Supertype.tolist()
adata_C.obs['Supertype'].value_counts()
# Micro-PVM_2_2-SEAAD    453
# Micro-PVM_1_1-SEAAD    166
# Micro-PVM_2_1-SEAAD    121
# Micro-PVM_4-SEAAD       33
# Micro-PVM_2             20
# Micro-PVM_3-SEAAD        2
# Micro-PVM_1              2


import simba as si
si.tl.umap(adata_C,n_neighbors=15,n_components=2)
si.pl.umap(adata_C,color=['Supertype'],
            fig_size=(6,4),
            drawing_order='random')
plt.savefig('cell_umap_cossima.png')


# embed cells and genes into the same space
adata_all = si.tl.embed(adata_ref=adata_C,list_adata_query=[adata_G])
adata_all.obs.head()


## add annotations of cells and genes
adata_all.obs['entity_anno'] = ""
adata_all.obs.loc[adata_C.obs_names, 'entity_anno'] = adata_all.obs.loc[adata_C.obs_names, 'Supertype']
adata_all.obs.loc[adata_G.obs_names, 'entity_anno'] = 'gene'
adata_all.obs.head()


si.tl.umap(adata_all,n_neighbors=15,n_components=2)
si.pl.umap(adata_all,color=['id_dataset','entity_anno'],
           drawing_order='original',
           fig_size=(6,5))
plt.savefig('cell_gene_umap_s.png')










from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def gen_graph_cgc_cosine(cell_emb, gene_emb, cell_ids, gene_ids,
                         dirname='graph_CGC',
                         sim_threshold=0.2,
                         cc_threshold=0.3,
                         gg_threshold=0.3,
                         cc_mode='threshold',  # 'knn' or 'threshold'
                         gg_mode='threshold',
                         k_cc=10,
                         k_gg=10,
                         include_cell_cell=True,
                         include_gene_gene=True,
                         pbg_params=None,
                         prefix_C='C', prefix_G='G'):
    filepath = './' + dirname + '/'
    os.makedirs(filepath + "input/edge", exist_ok=True)
    os.makedirs(filepath + "input/entity", exist_ok=True)
    # Alias tables
    df_cells = pd.DataFrame(index=cell_ids, columns=['alias'],
                            data=[f'{prefix_C}.{i}' for i in range(len(cell_ids))])
    df_genes = pd.DataFrame(index=gene_ids, columns=['alias'],
                            data=[f'{prefix_G}.{j}' for j in range(len(gene_ids))])
    entity_alias = pd.concat([df_cells, df_genes])
    edges = []
    relations = []
    # --- 1. Cell ↔ Gene (C-G)
    sim_cg = cosine_similarity(cell_emb, gene_emb)
    for i in range(cell_emb.shape[0]):
        for j in range(gene_emb.shape[0]):
            if sim_cg[i, j] > sim_threshold:
                edges.append((df_cells.iloc[i, 0], "cell-gene", df_genes.iloc[j, 0]))
    relations.append({'name': 'cell-gene', 'lhs': prefix_C, 'rhs': prefix_G, 'operator': 'none', 'weight': 1.0})
    # --- 2. Cell ↔ Cell (C-C)
    if include_cell_cell:
        if cc_mode == 'threshold':
            sim_cc = cosine_similarity(cell_emb)
            for i in range(cell_emb.shape[0]):
                for j in range(i + 1, cell_emb.shape[0]):
                    if sim_cc[i, j] > cc_threshold:
                        edges.append((df_cells.iloc[i, 0], "cell-cell", df_cells.iloc[j, 0]))
        elif cc_mode == 'knn':
            nn = NearestNeighbors(n_neighbors=k_cc+1, metric='cosine').fit(cell_emb)
            dist, idx = nn.kneighbors(cell_emb)
            for i in range(cell_emb.shape[0]):
                for j in idx[i][1:]:  # skip self
                    edges.append((df_cells.iloc[i, 0], "cell-cell", df_cells.iloc[j, 0]))
        relations.append({'name': 'cell-cell', 'lhs': prefix_C, 'rhs': prefix_C, 'operator': 'none', 'weight': 1.0})
    # --- 3. Gene ↔ Gene (G-G)
    if include_gene_gene:
        if gg_mode == 'threshold':
            sim_gg = cosine_similarity(gene_emb)
            for i in range(gene_emb.shape[0]):
                for j in range(i + 1, gene_emb.shape[0]):
                    if sim_gg[i, j] > gg_threshold:
                        edges.append((df_genes.iloc[i, 0], "gene-gene", df_genes.iloc[j, 0]))
        elif gg_mode == 'knn':
            nn = NearestNeighbors(n_neighbors=k_gg+1, metric='cosine').fit(gene_emb)
            dist, idx = nn.kneighbors(gene_emb)
            for i in range(gene_emb.shape[0]):
                for j in idx[i][1:]:  # skip self
                    edges.append((df_genes.iloc[i, 0], "gene-gene", df_genes.iloc[j, 0]))
        relations.append({'name': 'gene-gene', 'lhs': prefix_G, 'rhs': prefix_G, 'operator': 'none', 'weight': 1.0})
    # Save edges and aliases
    df_edges = pd.DataFrame(edges, columns=["source", "relation", "destination"])
    df_edges.to_csv(os.path.join(filepath, "input/edge/pbg_graph.txt"),
                    sep='\t', header=False, index=False)
    entity_alias.to_csv(os.path.join(filepath, "input/entity/entity_alias.txt"),
                        sep='\t', header=True, index=True)
    # Update PBG config
    pbg_params['entity_path'] = os.path.join(filepath, "input/entity")
    pbg_params['edge_paths'] = [os.path.join(filepath, "input/edge")]
    pbg_params['checkpoint_path'] = os.path.join(filepath, "model")
    pbg_params['entities'] = {
        prefix_C: {'num_partitions': 1},
        prefix_G: {'num_partitions': 1}
    }
    pbg_params['relations'] = relations
    print(f"Edge counts by type:\n{pd.Series([e[1] for e in edges]).value_counts()}")
    return df_edges, entity_alias, {'n_edges': df_edges.shape[0]}


















## Add annotation of celltypes (optional)
adata_C.obs['celltype'] = adata[adata_C.obs_names,:].obs['celltype'].copy()
adata_C




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
