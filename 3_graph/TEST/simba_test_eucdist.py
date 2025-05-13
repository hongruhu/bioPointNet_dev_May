# (sciLaMA_graph) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/sciLaMA_graph$

path = '/group/gquongrp/workspaces/hongruhu/sciLaMA_graph/test/featal_liver/'

import scanpy as sc
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns

from sciLaMA import *


adata_CG = sc.read(path + 'obj.h5ad') 
adata_CG.obs['celltype'] = adata_CG.obs['Cell.Labels']


import os
import simba as si
si.__version__
workdir = 'result_simba_euclidean_distances'
si.settings.set_workdir(workdir)
cell_emb = pd.read_csv(path + 'cell_embedding.csv', index_col=0)
gene_emb = pd.read_csv(path + 'gene_embedding.csv', index_col=0)
C = cell_emb.values
G = gene_emb.values
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity

euclidean_distances(C,G).shape

adata_CG.X = euclidean_distances(C,G)
cutoff = 2
adata_CG.X[adata_CG.X > cutoff] = 0
adata_CG.X = scipy.sparse.csr_matrix(adata_CG.X)


si.tl.discretize(adata_CG,n_bins=5)
si.pl.discretize(adata_CG,kde=False)
plt.savefig('discretize_euclidean_distances.png')
si.tl.gen_graph(list_CG=[adata_CG],
                layer='simba',
                use_highly_variable=False,
                dirname='graph0')
si.tl.pbg_train(auto_wd=True, save_wd=True, output='model')


# read in entity embeddings obtained from pbg training.
dict_adata = si.read_embedding()
dict_adata
adata_C = dict_adata['C']  # embeddings of cells
adata_G = dict_adata['G']  # embeddings of genes
## Add annotation of celltypes (optional)
adata_C.obs['celltype'] = adata_CG[adata_C.obs_names,:].obs['celltype'].copy()
adata_C
adata_CG.obs.celltype.value_counts()
# palette_celltype={'TAC-1':'#1f77b4',
#                   'TAC-2':'#ff7f0e',
#                   'IRS':'#279e68',
#                   'Medulla':"#aa40fc",
#                   'Hair Shaft-cuticle.cortex':'#d62728'}
si.tl.umap(adata_C,n_neighbors=15,n_components=2)
si.pl.umap(adata_C,color=['celltype'],
            # dict_palette={'celltype': palette_celltype},
            fig_size=(6,4),
            drawing_order='random')
plt.savefig('cell_umap_euclidean_distances.png')
# embed cells and genes into the same space
adata_all = si.tl.embed(adata_ref=adata_C,list_adata_query=[adata_G])
adata_all.obs.head()
## add annotations of cells and genes
adata_all.obs['entity_anno'] = ""
adata_all.obs.loc[adata_C.obs_names, 'entity_anno'] = adata_all.obs.loc[adata_C.obs_names, 'celltype']
adata_all.obs.loc[adata_G.obs_names, 'entity_anno'] = 'gene'
adata_all.obs.head()
# palette_entity_anno = palette_celltype.copy()
# palette_entity_anno['gene'] = "#607e95"
si.tl.umap(adata_all,n_neighbors=15,n_components=2)
si.pl.umap(adata_all,color=['id_dataset','entity_anno'],
        #    dict_palette={'entity_anno': palette_entity_anno},
           drawing_order='original',
           fig_size=(6,5))
plt.savefig('cell_gene_umap_euclidean_distances.png')
