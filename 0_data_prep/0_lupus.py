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


# 142 patients consisting of 566,453 cells
# disease: 22 healthy controls, 120 patients with lupus / SLE
# race: 80 European, 62 Asian 

# Lupus (disease) (Mandric et al. 2020) 
# https://www.nature.com/articles/s41467-020-19365-w
# https://ucsf.app.box.com/s/tds2gotok3lyeanlrt13prj40am5w720
# CLUESImmVar_nonorm.V6.h5ad


path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/lupus/'
adata = sc.read_h5ad(path + 'CLUESImmVar_nonorm.V6.h5ad')
# AnnData object with n_obs × n_vars = 834096 × 32738
#     obs: 'disease_cov', 'ct_cov', 'pop_cov', 'ind_cov', 'well', 'batch_cov', 'batch'
#     var: 'gene_ids-0-0-0-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0', 'gene_ids-1-0-0-0-0', 'gene_ids-1-0-0-0', 'gene_ids-1-0-0', 'gene_ids-1-0', 'gene_ids-1'

adata.obs.disease_cov.value_counts()
# disease_cov
# sle        557630
# healthy    276466
adata.obs.pop_cov.value_counts()
# pop_cov
# WHITE    533178
# ASIAN    300918

adata.obs.ind_cov.value_counts()    # 169


adata.obs[['disease_cov','ind_cov']].value_counts().sort_index()
# disease_cov  ind_cov
# healthy      IGTB141                 7308
#              IGTB143                10229
#              IGTB195                 9727
#              IGTB256                 5645
#              IGTB469                 9874
#                                     ...
# sle          904405200_904405200     3658
#              904425200_904425200     3861
#              904463200_904463200     5595
#              904464200_904464200     6805
#              904477200_904477200     4130


adata.obs.well.value_counts()    # 54
adata.obs.batch_cov.value_counts()    # 14



adata.X.max()
# 7425.0
adata.X.min()
# 0.0

adata 
# AnnData object with n_obs × n_vars = 834096 × 32738
sc.pp.filter_genes(adata, min_cells=5)
adata
# AnnData object with n_obs × n_vars = 834096 × 24205
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat_v3')

sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)


adata = adata[:, adata.var.highly_variable]
adata.write_h5ad(path + 'lupus_processed.h5ad')