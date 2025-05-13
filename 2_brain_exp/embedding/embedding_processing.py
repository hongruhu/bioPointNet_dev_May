# (mil) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet$ python
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns

MTG = sc.read_h5ad('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/MTG_glial.h5ad')
DLPFC = sc.read_h5ad('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/DLPFC_glial.h5ad')
ROSMAP = sc.read_h5ad('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/ROSMAP_glia_processed_.h5ad')


MTG = MTG[MTG.obs.assay=="10x 3' v3"]
DLPFC = DLPFC[DLPFC.obs.assay=="10x 3' v3"]
MTG = MTG[MTG.obs.Subclass.isin(['Astrocyte', 'OPC', 'Oligodendrocyte', 'Microglia-PVM'])]
DLPFC = DLPFC[DLPFC.obs.Subclass.isin(['Astrocyte', 'OPC', 'Oligodendrocyte', 'Microglia-PVM'])]

MTG.obs = MTG.obs.drop('observation_joinid', axis=1)
DLPFC.obs = DLPFC.obs.drop('observation_joinid', axis=1)

sc.pp.filter_genes(MTG, min_cells=5)
sc.pp.filter_genes(DLPFC, min_cells=5)

print(MTG.X.max(), DLPFC.X.max(), ROSMAP.X.max())
# 8.181981 8.417528 4069
print(MTG.shape, DLPFC.shape, ROSMAP.shape)
# (226669, 33190) (287307, 33363) (930739, 29268)

gene_list = list(set(MTG.var.index).intersection(set(DLPFC.var.index)))
len(gene_list)
# 33051

MTG = MTG[:,MTG.var.feature_type=='protein_coding']
DLPFC = DLPFC[:,DLPFC.var.feature_type=='protein_coding']
print(MTG.shape, DLPFC.shape)
# (226669, 18226) (287307, 18270)
gene_list = list(set(MTG.var.index).intersection(set(DLPFC.var.index)))
len(gene_list)
# 18187

orthologs = pd.read_csv('/group/gquongrp/workspaces/hongruhu/sciLaMA/STATIC_GENE_EMBEDDINGS/GRCH38_human_mouse_orthologs_Biomart_autosomal.txt',
                        sep = '\t')
orthologs.columns
# Index(['Transcript stable ID', 'Mouse gene name', 'Mouse gene stable ID',
#        'Mouse homology type', 'Mouse orthology confidence [0 low, 1 high]',
#        'Gene name', 'Gene stable ID', 'Chromosome/scaffold name'],
#       dtype='object')
orthologs_human = orthologs[orthologs['Gene stable ID'].isin(gene_list)]
orthologs_human.index = orthologs_human['Gene stable ID']
# [150959 rows x 8 columns]
# remove duplicated genes
orthologs_human = orthologs_human.loc[~orthologs_human.index.duplicated(keep='first')]
orthologs_human = orthologs_human[['Gene name', 'Gene stable ID']]
# [16158 rows x 2 columns]
intersection_genes = list(set(gene_list) & set(orthologs_human.index))
orthologs_human = orthologs_human.loc[intersection_genes]
orthologs_human.to_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/gene_list.csv')
# 16158

human_genes = orthologs_human['Gene name'].tolist()
human_ids = orthologs_human['Gene stable ID'].tolist()
# CHECK GENES
##########################
# Natural Language Model #
##########################
import pickle
genept_path = '/group/gquongrp/workspaces/hongruhu/sciLaMA/STATIC_GENE_EMBEDDINGS/natural_embedding/GenePT_gene_embeddings_35k.pickle'
with open(genept_path, 'rb') as f:
    genept = pickle.load(f) # dict
    genept = pd.DataFrame(genept).T


print(genept.shape) # (33985, 1536)
genept.loc[list(set(human_genes)&set(genept.index))].shape # (15941, 1536)
##############################
# Amino Acid Languange Model #
##############################
import torch
esm2_human_path = '/group/gquongrp/workspaces/hongruhu/sciLaMA/STATIC_GENE_EMBEDDINGS/protein_embedding/SATURN_aka_UCE_20k/ESM2_SATURN/human_embedding.torch'
esm2_human = pd.DataFrame(torch.load(esm2_human_path)).T
print(esm2_human.shape) # (19790, 5120)
esm2_human.loc[list(set(human_genes)&set(esm2_human.index))].shape # (16087, 5120)
###############################
# Single Cell Languange Model #
###############################
scGPT_path = '/group/gquongrp/workspaces/hongruhu/sciLaMA/STATIC_GENE_EMBEDDINGS/gene_embedding_SC/gene_embeddings_scGPT_whole.csv'
scgpt = pd.read_csv(scGPT_path, index_col=0)
print(scgpt.shape)     # (60697, 512)
scgpt.loc[list(set(human_genes)&set(scgpt.index))].shape    # (16104, 512)
###########################
# Find Intersecting Genes #
###########################
genept_ = genept.loc[list(set(human_genes)&set(genept.index))]                      
esm2_human_ = esm2_human.loc[list(set(human_genes)&set(esm2_human.index))]          
scgpt_ = scgpt.loc[list(set(human_genes)&set(scgpt.index))]  
intersection_genes = list(set(genept_.index) & set(esm2_human_.index) & set(scgpt_.index))
len(intersection_genes) # 15934
orthologs_human = orthologs_human[orthologs_human['Gene name'].isin(intersection_genes)]
orthologs_human.to_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/gene_list_intersection.csv')
genept_ = genept_.loc[orthologs_human['Gene name']]
esm2_human_ = esm2_human_.loc[orthologs_human['Gene name']]
scgpt_ = scgpt_.loc[orthologs_human['Gene name']]
MTG = MTG[:,orthologs_human.index.tolist()]
DLPFC = DLPFC[:,orthologs_human.index.tolist()]



np.sum(genept_.index == MTG.var.feature_name)
np.sum(esm2_human_.index == MTG.var.feature_name)
np.sum(scgpt_.index == MTG.var.feature_name)

intersection_genes = set(genept_.index).intersection(set(MTG.var.feature_name))
len(intersection_genes)
# 15909

MTG = MTG[:, MTG.var.feature_name.isin(intersection_genes)]
DLPFC = DLPFC[:, DLPFC.var.feature_name.isin(intersection_genes)]

genept_ = genept_.loc[MTG.var.feature_name]
esm2_human_ = esm2_human_.loc[MTG.var.feature_name]
scgpt_ = scgpt_.loc[MTG.var.feature_name]

np.sum(genept_.index == MTG.var.feature_name)
np.sum(esm2_human_.index == MTG.var.feature_name)
np.sum(scgpt_.index == MTG.var.feature_name)

np.sum(genept_.index == DLPFC.var.feature_name)
np.sum(esm2_human_.index == DLPFC.var.feature_name)
np.sum(scgpt_.index == DLPFC.var.feature_name)

genept_.to_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/genept_embedding.csv')
esm2_human_.to_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/esm2_embedding.csv')
scgpt_.to_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/scgpt_embedding.csv')

MTG.write_h5ad('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/MTG_glia_intersection_norm.h5ad', compression='gzip')
DLPFC.write_h5ad('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/DLPFC_glia_intersection_norm.h5ad', compression='gzip')





ROSMAP = ROSMAP[ROSMAP.obs.cell_type.isin(['Ast', 'Mic', 'OPC', 'Oli'])]
# View of AnnData object with n_obs × n_vars = 926374 × 29268
sc.pp.filter_genes(ROSMAP, min_cells=1)
# AnnData object with n_obs × n_vars = 926374 × 29267
gene_list = list(ROSMAP.var.index)
len(gene_list)
# 29267

orthologs = pd.read_csv('/group/gquongrp/workspaces/hongruhu/sciLaMA/STATIC_GENE_EMBEDDINGS/GRCH38_human_mouse_orthologs_Biomart_autosomal.txt',
                        sep = '\t')
orthologs.columns
# Index(['Transcript stable ID', 'Mouse gene name', 'Mouse gene stable ID',
#        'Mouse homology type', 'Mouse orthology confidence [0 low, 1 high]',
#        'Gene name', 'Gene stable ID', 'Chromosome/scaffold name'],
#       dtype='object')
orthologs_human = orthologs[orthologs['Gene name'].isin(gene_list)]
orthologs_human.index = orthologs_human['Gene name']
# [146643 rows x 8 columns]
# remove duplicated genes
orthologs_human = orthologs_human.loc[~orthologs_human.index.duplicated(keep='first')]
orthologs_human = orthologs_human[['Gene name', 'Gene stable ID']]
# [15832 rows x 2 columns]
intersection_genes = list(set(gene_list) & set(orthologs_human.index))
orthologs_human.to_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/gene_list.csv')


human_genes = orthologs_human['Gene name'].tolist()
human_ids = orthologs_human['Gene stable ID'].tolist()
# CHECK GENES
##########################
# Natural Language Model #
##########################
import pickle
genept_path = '/group/gquongrp/workspaces/hongruhu/sciLaMA/STATIC_GENE_EMBEDDINGS/natural_embedding/GenePT_gene_embeddings_35k.pickle'
with open(genept_path, 'rb') as f:
    genept = pickle.load(f) # dict
    genept = pd.DataFrame(genept).T


print(genept.shape) # (33985, 1536)
genept.loc[list(set(human_genes)&set(genept.index))].shape # (15957, 1536)
##############################
# Amino Acid Languange Model #
##############################
import torch
esm2_human_path = '/group/gquongrp/workspaces/hongruhu/sciLaMA/STATIC_GENE_EMBEDDINGS/protein_embedding/SATURN_aka_UCE_20k/ESM2_SATURN/human_embedding.torch'
esm2_human = pd.DataFrame(torch.load(esm2_human_path)).T
print(esm2_human.shape) # (19790, 5120)
esm2_human.loc[list(set(human_genes)&set(esm2_human.index))].shape # (16119, 5120)
###############################
# Single Cell Languange Model #
###############################
scGPT_path = '/group/gquongrp/workspaces/hongruhu/sciLaMA/STATIC_GENE_EMBEDDINGS/gene_embedding_SC/gene_embeddings_scGPT_whole.csv'
scgpt = pd.read_csv(scGPT_path, index_col=0)
print(scgpt.shape)     # (60697, 512)
scgpt.loc[list(set(human_genes)&set(scgpt.index))].shape    # (16145, 512)
###########################
# Find Intersecting Genes #
###########################
genept_ = genept.loc[list(set(human_genes)&set(genept.index))]                      
esm2_human_ = esm2_human.loc[list(set(human_genes)&set(esm2_human.index))]          
scgpt_ = scgpt.loc[list(set(human_genes)&set(scgpt.index))]  
intersection_genes = list(set(genept_.index) & set(esm2_human_.index) & set(scgpt_.index))
len(intersection_genes) # 15384
orthologs_human = orthologs_human[orthologs_human['Gene name'].isin(intersection_genes)]
orthologs_human.to_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/gene_list_intersection.csv')
genept_ = genept_.loc[orthologs_human['Gene name']]
esm2_human_ = esm2_human_.loc[orthologs_human['Gene name']]
scgpt_ = scgpt_.loc[orthologs_human['Gene name']]
genept_.to_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/genept_embedding.csv')
esm2_human_.to_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/esm2_embedding.csv')
scgpt_.to_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/scgpt_embedding.csv')


sc.pp.normalize_total(ROSMAP, 1e4)
ROSMAP.X.max() # 3889.5859473023843
ROSMAP = ROSMAP[:,orthologs_human['Gene name']]
ROSMAP.write_h5ad('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/ROSMAP_glia_intersection_libnorm.h5ad', compression='gzip')









# (mil) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet$ python
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns


ROSMAP = sc.read_h5ad('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/ROSMAP_glia_intersection_libnorm.h5ad')
MTG = sc.read_h5ad('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/MTG_glia_intersection_norm.h5ad')
DLPFC = sc.read_h5ad('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/DLPFC_glia_intersection_norm.h5ad')

ROSMAP_meta = ROSMAP.obs.copy()
MTG_meta = MTG.obs.copy()
DLPFC_meta = DLPFC.obs.copy()

print(ROSMAP_meta.shape, MTG_meta.shape, DLPFC_meta.shape)
# (926374, 8) (226669, 41) (287307, 40)

ROSMAP_meta.to_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/ROSMAP_glia_metadata.csv')
MTG_meta.to_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/MTG_glia_metadata.csv')
DLPFC_meta.to_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/DLPFC_glia_metadata.csv')