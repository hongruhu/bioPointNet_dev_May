# (sciLaMA_graph) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/sciLaMA_graph$

path = '/group/gquongrp/workspaces/hongruhu/sciLaMA_graph/test/'

import scanpy as sc
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from coembed import *



adata_all_ = sc.read('/group/gquongrp/workspaces/hongruhu/bioPointNet/result/AD/seaad_MTG_microglia_sciLaMA/direct_sciLaMA_seaad_MTG.h5ad') 

cell_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/result/AD/seaad_MTG_microglia_sciLaMA/direct_sciLaMA_CELL_embedding.csv', index_col=0)
gene_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/result/AD/seaad_MTG_microglia_sciLaMA/direct_sciLaMA_GENE_embedding.csv', index_col=0)

C = cell_emb.values
G = gene_emb.values

cell_meta = adata_all_.obs.copy()
gene_meta = adata_all_.var.copy()

adata_C = sc.AnnData(X=C, obs=cell_meta)
adata_G = sc.AnnData(X=G)



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

dirname='graph_CG_GG_CC_KNN_eucdist'


df_edges, entity_alias, graph_stats = gen_graph_euclidean_knn(
    cell_emb=cell_emb.values,
    gene_emb=gene_emb.values,
    cell_ids=cell_emb.index,
    gene_ids=gene_emb.index,
    dirname=dirname,
    dist_threshold=1.8,
    cc_mode='knn',
    k_cc=25,
    gg_mode='knn',
    k_gg=25,
    include_cell_cell=True,
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
adata_C.obs['Supertype'] = adata_all_[adata_C.obs.index].obs.Supertype.tolist()
adata_C.obs['Supertype'].value_counts()





marker_list = ['IL1B', 'CSF1R', 'STAB1', 'NINJ1', 'JAK3',
'IRF1', 'IRF7', 'IFI16',
'FCGR1A', 'FCGR1B', 'FCGR2A', 'FCGR3B',
'CD74', 'HLA-DRB5',
'C1QA', 'C1QB',
'CSF1R', 'CTSC', 'C1QA', 'C1QB', 'LY86', 'FCGR3A'
'CTSD', 'CTSS',
'LYZ',
'APOE',
'RUNX1', 'IKZF1', 'NFATC2', 'MAF']
len(marker_list)
# 29




import simba as si
si.tl.umap(adata_C,n_neighbors=15,n_components=2)
si.pl.umap(adata_C,color=['Supertype'],
            fig_size=(6,4),
            drawing_order='random')
plt.savefig(dirname + '/cell_umap_KNN.png')


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
plt.savefig(dirname + '/cell_gene_umap_KNN.png')


adata_all_umap = pd.DataFrame(adata_all.obsm['X_umap'], index=adata_all.obs.index)
si.pl.umap(adata_all[adata_all.obs.id_dataset=='ref'], color=['entity_anno'],
        drawing_order='original',
        fig_size=(6,5))
plt.savefig(dirname + '/cell_gene_umap_KNN_cell.png')


adata_gene = adata_all[adata_all.obs.id_dataset!='ref']
adata_gene.obs['feature_name'] = gene_meta.loc[adata_gene.obs.index]['feature_name']
adata_gene.obs['marker'] = 'No'
mask = adata_gene.obs['feature_name'].isin(marker_list)
adata_gene.obs.loc[mask, 'marker'] = 'Yes'
si.pl.umap(adata_gene, color=['marker'],
        drawing_order='original',
        fig_size=(6,5))
plt.savefig(dirname + '/cell_gene_umap_KNN_gene.png')



from adjustText import adjust_text
umap = adata_gene.obsm['X_umap']
marker_mask = adata_gene.obs['marker'] == 'Yes'
fig, ax = plt.subplots(figsize=(8, 6))
# Plot all points in light grey
ax.scatter(umap[:, 0], umap[:, 1], c='lightgrey', s=5, alpha=0.3)
# Plot marker=='Yes' points in red
ax.scatter(umap[marker_mask, 0], umap[marker_mask, 1], c='red', s=30, label='Markers')
# Prepare text annotations
texts = []
for i in adata_gene.obs[marker_mask].index:
    x, y = umap[adata_gene.obs.index.get_loc(i)]
    gene = adata_gene.obs.loc[i, 'feature_name']
    texts.append(ax.text(x, y, gene, fontsize=8, color='black'))


# Repel overlapping text
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
ax.set_title('UMAP with Marker Genes Highlighted and Labeled')
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig(dirname + '/cell_gene_umap_cossim_gene_marker.png')




sc.pp.neighbors(adata_gene, n_neighbors=15, use_rep='X')
sc.tl.leiden(adata_gene, resolution=0.5)  # You can tune resolution
si.pl.umap(adata_gene, color=['leiden'],
        drawing_order='original',
        fig_size=(6,5))
plt.savefig(dirname + '/cell_gene_umap_KNN_gene_cluster.png')





# UMAP coordinates are usually stored in .obsm['X_umap']
umap = adata_gene.obsm['X_umap']
# Add UMAP1 and UMAP2 as columns for easier filtering
adata_gene.obs['UMAP1'] = umap[:, 0]
adata_gene.obs['UMAP2'] = umap[:, 1]



# Apply filtering
subset_mask = (
    (adata_gene.obs['leiden'] == '5') #&
    # (adata_gene.obs['UMAP1'] >= 5) & (adata_gene.obs['UMAP1'] <= 10) &
    # (adata_gene.obs['UMAP2'] >= 7.5) & (adata_gene.obs['UMAP2'] <= 10)
)
adata_subset = adata_gene[subset_mask].copy()
gene_names = adata_subset.obs.feature_name.tolist()











from gprofiler import GProfiler
genes_cleaned = [g.split('_')[0].split('.')[0] for g in gene_names]

# Run enrichment
gp = GProfiler(return_dataframe=True)
results = gp.profile(organism='hsapiens', query=genes_cleaned)

# Show top results
print(results[['native', 'name', 'p_value', 'source']].head())
df = results[['native', 'name', 'p_value', 'source']]
# source
# GO:BP    51
# GO:CC    16
# GO:MF     3
# TF        1
# KEGG      1

df[df.source=='GO:MF']


# regulation of inflammatory response

# Run enrichment
gp = GProfiler(return_dataframe=True)
results = gp.profile(organism='hsapiens', query=marker_list)

# Show top results
print(results[['native', 'name', 'p_value', 'source']].head())

df = results[['native', 'name', 'p_value', 'source']]
# source
# GO:BP    51
# GO:CC    16
# GO:MF     3
# TF        1
# KEGG      1

df[df.source=='GO:BP']
