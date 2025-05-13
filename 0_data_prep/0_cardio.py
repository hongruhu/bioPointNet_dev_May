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
# Cardio (Chaffin et al. 2022) https://www.nature.com/articles/s41586-022-04817-8
# https://singlecell.broadinstitute.org/single_cell/study/SCP1303/
# - DCM_HCM_Expression_Matrix_raw_counts_V1.mtx
# - DCM_HCM_Expression_Matrix_genes_V1.tsv
# - DCM_HCM_Expression_Matrix_barcodes_V1.tsv
# - DCM_HCM_MetaData_V1.txt

path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/cardio/'
adata = sc.read_h5ad(path + 'anndata/human_dcm_hcm_scportal_03.17.2022.h5ad')
# AnnData object with n_obs × n_vars = 592689 × 36601
#     obs: 'biosample_id', 'donor_id', 'disease', 'sex', 'age', 'lvef', 'cell_type_leiden0.6', 'SubCluster', 'cellbender_ncount', 'cellbender_ngenes', 'cellranger_percent_mito', 'exon_prop', 'cellbender_entropy', 'cellranger_doublet_scores'
#     var: 'gene_ids', 'feature_types', 'genome'
#     obsm: 'X_umap'
#     layers: 'cellranger_raw'

adata.obs.disease.value_counts()
# disease
# HCM    235252
# NF     185441
# DCM    171996

adata.obs.biosample_id.value_counts() # 80
adata.obs.donor_id.value_counts()     # 42

adata.obs[['disease','donor_id']].value_counts().sort_index()
# DCM      P1290        8357
#          P1300       17318
#          P1304       19123
#          P1358       23984
#          P1371       16233
#          P1430       11131
#          P1437       21695
#          P1472       19513
#          P1504        8133
#          P1606        8523
#          P1617       17986
# HCM      P1422       23315
#          P1425       15378
#          P1447       11151
#          P1462       21715
#          P1479       19124
#          P1508       20536
#          P1510       10744
#          P1602       16580
#          P1630       17633
#          P1631       13686
#          P1685       10043
#          P1707        9517
#          P1722       21432
#          P1726       12389
#          P1735       12009
# NF       P1515       14502
#          P1516        9361
#          P1539       11076
#          P1540       11638
#          P1547        8253
#          P1549       11709
#          P1558       10469
#          P1561       10016
#          P1582       18855
#          P1600       14882
#          P1603       10638
#          P1610       13919
#          P1622        7210
#          P1678       10085
#          P1702       13550
#          P1718        9278
# DCM 11 | HCM 15 | NF 16





adata.X.max()
# 14643.0
adata.X.min()
# 0.0

adata 
# AnnData object with n_obs × n_vars = 592689 × 36601
sc.pp.filter_genes(adata, min_cells=5)
adata
# AnnData object with n_obs × n_vars = 592689 × 32151
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat_v3')

sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)


adata = adata[:, adata.var.highly_variable]
adata.write_h5ad(path + 'cardio_processed.h5ad')