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

import simba as si
adata_all = si.tl.embed(adata_ref=adata_C,list_adata_query=[adata_G])
## add annotations of cells and genes
adata_all.obs['entity_anno'] = ""
adata_all.obs.loc[adata_G.obs_names, 'entity_anno'] = 'gene'
adata_all.obs.loc[adata_C.obs_names, 'entity_anno'] = adata_all.obs.loc[adata_C.obs_names, 'Supertype']
adata_all.obs.head()
si.tl.umap(adata_all,n_neighbors=15,n_components=2)
si.pl.umap(adata_all,color=['id_dataset','entity_anno'],
           drawing_order='original',
           fig_size=(6,5))
plt.savefig('cell_gene_umap_microglia.png')
