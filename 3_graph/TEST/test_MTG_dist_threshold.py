# (sciLaMA_graph) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/sciLaMA_graph$

path = '/group/gquongrp/workspaces/hongruhu/sciLaMA_graph/test/'

import scanpy as sc
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns



adata_all = sc.read('/group/gquongrp/workspaces/hongruhu/bioPointNet/result/AD/seaad_MTG_microglia_sciLaMA/direct_sciLaMA_seaad_MTG.h5ad') 

cell_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/result/AD/seaad_MTG_microglia_sciLaMA/direct_sciLaMA_CELL_embedding.csv', index_col=0)
gene_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/result/AD/seaad_MTG_microglia_sciLaMA/direct_sciLaMA_GENE_embedding.csv', index_col=0)

C = cell_emb.values
G = gene_emb.values

cell_meta = adata_all.obs.copy()


adata_C = sc.AnnData(X=C, obs=cell_meta)
adata_G = sc.AnnData(X=G)


from coembed import *
# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.neighbors import NearestNeighbors

# def gen_graph_euclidean_knn(cell_emb, gene_emb, cell_ids, gene_ids,
#                              dirname='graph_CGC_euclidean',
#                              dist_threshold=2.0,
#                              cc_threshold=2.0,
#                              gg_threshold=2.0,
#                              cc_mode='threshold',  # 'knn' or 'threshold'
#                              gg_mode='threshold',
#                              k_cc=10,
#                              k_gg=10,
#                              include_cell_cell=True,
#                              include_gene_gene=True,
#                              pbg_params=None,
#                              prefix_C='C', prefix_G='G'):
#     filepath = './' + dirname + '/'
#     os.makedirs(filepath + "input/edge", exist_ok=True)
#     os.makedirs(filepath + "input/entity", exist_ok=True)
#     df_cells = pd.DataFrame(index=cell_ids, columns=['alias'],
#                             data=[f'{prefix_C}.{i}' for i in range(len(cell_ids))])
#     df_genes = pd.DataFrame(index=gene_ids, columns=['alias'],
#                             data=[f'{prefix_G}.{j}' for j in range(len(gene_ids))])
#     entity_alias = pd.concat([df_cells, df_genes])
#     edges = []
#     relations = []
#     # --- 1. Cell - Gene (C-G)
#     dist_cg = euclidean_distances(cell_emb, gene_emb)
#     for i in range(cell_emb.shape[0]):
#         for j in range(gene_emb.shape[0]):
#             if dist_cg[i, j] < dist_threshold:
#                 edges.append((df_cells.iloc[i, 0], "cell-gene", df_genes.iloc[j, 0]))
#     relations.append({'name': 'cell-gene', 'lhs': prefix_C, 'rhs': prefix_G, 'operator': 'none', 'weight': 1.0})
#     print()
#     # --- 2. Cell - Cell (C-C)
#     if include_cell_cell:
#         if cc_mode == 'threshold':
#             dist_cc = euclidean_distances(cell_emb)
#             for i in range(cell_emb.shape[0]):
#                 for j in range(i + 1, cell_emb.shape[0]):
#                     if dist_cc[i, j] < cc_threshold:
#                         edges.append((df_cells.iloc[i, 0], "cell-cell", df_cells.iloc[j, 0]))
#         elif cc_mode == 'knn':
#             nn = NearestNeighbors(n_neighbors=k_cc + 1, metric='euclidean').fit(cell_emb)
#             dist, idx = nn.kneighbors(cell_emb)
#             for i in range(cell_emb.shape[0]):
#                 for j in idx[i][1:]:  # skip self
#                     edges.append((df_cells.iloc[i, 0], "cell-cell", df_cells.iloc[j, 0]))
#         relations.append({'name': 'cell-cell', 'lhs': prefix_C, 'rhs': prefix_C, 'operator': 'none', 'weight': 1.0})
#     # --- 3. Gene - Gene (G-G)
#     if include_gene_gene:
#         if gg_mode == 'threshold':
#             dist_gg = euclidean_distances(gene_emb)
#             for i in range(gene_emb.shape[0]):
#                 for j in range(i + 1, gene_emb.shape[0]):
#                     if dist_gg[i, j] < gg_threshold:
#                         edges.append((df_genes.iloc[i, 0], "gene-gene", df_genes.iloc[j, 0]))
#         elif gg_mode == 'knn':
#             nn = NearestNeighbors(n_neighbors=k_gg + 1, metric='euclidean').fit(gene_emb)
#             dist, idx = nn.kneighbors(gene_emb)
#             for i in range(gene_emb.shape[0]):
#                 for j in idx[i][1:]:  # skip self
#                     edges.append((df_genes.iloc[i, 0], "gene-gene", df_genes.iloc[j, 0]))
#         relations.append({'name': 'gene-gene', 'lhs': prefix_G, 'rhs': prefix_G, 'operator': 'none', 'weight': 1.0})
#     # Save edges and aliases
#     df_edges = pd.DataFrame(edges, columns=["source", "relation", "destination"])
#     df_edges.to_csv(os.path.join(filepath, "pbg_graph.txt"),
#                     sep='\t', header=False, index=False)
#     entity_alias.to_csv(os.path.join(filepath, "entity_alias.txt"),
#                         sep='\t', header=True, index=True)
#     # Update PBG config
#     pbg_params['entity_path'] = os.path.join(filepath, "input/entity")
#     pbg_params['edge_paths'] = [os.path.join(filepath, "input/edge")]
#     pbg_params['checkpoint_path'] = os.path.join(filepath, "model")
#     pbg_params['entities'] = {
#         prefix_C: {'num_partitions': 1},
#         prefix_G: {'num_partitions': 1}
#     }
#     pbg_params['relations'] = relations
#     print(f"Edge counts by type:\n{pd.Series([e[1] for e in edges]).value_counts()}")
#     return df_edges, entity_alias, {'n_edges': df_edges.shape[0]}


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
    dimension=30,           # Size of embedding space
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

dirname='graph_CG_GG_only'


df_edges, entity_alias, graph_stats = gen_graph_euclidean_knn(
    cell_emb=cell_emb.values,
    gene_emb=gene_emb.values,
    cell_ids=cell_emb.index,
    gene_ids=gene_emb.index,
    dirname=dirname,
    dist_threshold=1.8,
    # cc_mode='knn',
    # k_cc=10,
    gg_mode='threshold',
    gg_threshold=1.8,
    include_cell_cell=False,
    include_gene_gene=True,
    pbg_params=pbg_params
)


pbg_train(
    dirname=dirname,
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
adata_C.obs['Supertype'] = adata_all[adata_C.obs.index].obs.Supertype.tolist()
adata_C.obs['Supertype'].value_counts()


import simba as si
si.tl.umap(adata_C,n_neighbors=15,n_components=2)
si.pl.umap(adata_C,color=['Supertype'],
            fig_size=(6,4),
            drawing_order='random')
plt.savefig('cell_umap_ed_KNN.png')


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
plt.savefig('cell_gene_umap_ed_KNN.png')
