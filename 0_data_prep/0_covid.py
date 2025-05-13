# (mil) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet$ python
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns

import scvi
import multimil as mil

from sciLaMA import *
from bioPointNet_Apr2025 import *


# ProtoCell4P [https://doi.org/10.1093/bioinformatics/btad493]
# https://doi.org/10.1016/j.cell.2021.07.023
# Ziegler et al. Jose Ordovas-Montanes, Cell 2021 [*]


path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/covid/'

counts = pd.read_csv(path + 'scMILD/Ziegler/expression/20210220_NasalSwab_RawCounts.txt', sep='\t')
genes = counts.index.tolist()
barcodes = counts.columns.tolist()
sparse_counts = counts.values.T
sparse_counts = sparse_counts.astype(np.float32)
from scipy import sparse
sparse_counts = sparse.csr_matrix(sparse_counts)

metadata = pd.read_csv(path + 'scMILD/Ziegler/metadata/20210701_NasalSwab_MetaData.txt', sep='\t', index_col=0)
metadata.drop(index=['TYPE'], inplace=True)
metadata.index = metadata.index.tolist()
np.sum(metadata.index == barcodes) # 32588

umap = pd.read_csv(path + 'scMILD/Ziegler/cluster/20210220_NasalSwab_UMAP.txt', sep='\t', index_col=0)
umap.drop(index=['TYPE'], inplace=True)
umap.index = umap.index.tolist()
np.sum(umap.index == barcodes) # 32588


adata = AnnData(X=sparse_counts, obs=metadata, var=pd.DataFrame(index=genes, columns=['gene_ids']))
adata.var_names = adata.var.index.tolist()
adata.obs['celltype'] = umap['Category'].tolist()
adata.obsm['X_umap'] = umap.iloc[:,:-1].values


adata
# AnnData object with n_obs × n_vars = 32588 × 32871
#     obs: 'donor_id', 'Peak_Respiratory_Support_WHO_Score', 'Bloody_Swab', 'Percent_Mitochondrial', 
#           'SARSCoV2_PCR_Status', 'SARSCoV2_PCR_Status_and_WHO_Score', 'Cohort_Disease_WHO_Score', 
#           'biosample_id', 'SingleCell_SARSCoV2_RNA_Status', 
#           'SARSCoV2_Unspliced_TRS_Total_Corrected', 'SARSCoV2_Spliced_TRS_Total_Corrected', 'SARSCoV2_NegativeStrand_Total_Corrected', 'SARSCoV2_PositiveStrand_Total_Corrected', 'SARSCoV2_Total_Corrected', 
#           'species', 'species__ontology_label', 'sex', 
#           'disease', 'disease__ontology_label', 'organ', 'organ__ontology_label', 'library_preparation_protocol', 'library_preparation_protocol__ontology_label', 
#           'age', 'Coarse_Cell_Annotations', 'Detailed_Cell_Annotations', 'celltype'
#     var: 'gene_ids'
#     obsm: 'X_umap'


adata.obs.disease.value_counts()
# disease
# MONDO_0100096    18073
# PATO_0000461      8874
# MONDO_0021113     3335
# MONDO_0100233     2306

adata.obs.disease__ontology_label.value_counts()
# disease__ontology_label
# COVID-19               18073
# normal                  8874
# respiratory failure     3335
# long COVID-19           2306



adata.obs.donor_id.value_counts()       # 58
adata.obs.biosample_id.value_counts()   # 58



adata.obs[['disease__ontology_label','donor_id']].value_counts().sort_index()




adata.X.max()
# 4259.0
adata.X.min()
# 0.0

adata 
# AnnData object with n_obs × n_vars = 32588 × 32871
sc.pp.filter_genes(adata, min_cells=5)
adata
# AnnData object with n_obs × n_vars = 32588 × 28696
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat_v3')

sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)


adata = adata[:, adata.var.highly_variable]
adata.var['gene_ids'] = adata.var.index.tolist()
adata.write_h5ad(path + 'covid_Ziegler_processed.h5ad')