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


path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/cardio/'
adata = sc.read_h5ad(path + 'cardio_processed.h5ad')
# AnnData object with n_obs × n_vars = 592689 × 2000
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


df = pd.DataFrame(adata.obs[['disease','donor_id']].value_counts().sort_index())
df['disease'] = df.index.get_level_values(0)
df.index = df.index.get_level_values(1)
df.index = df.index.tolist()
df.drop(columns='count', inplace=True)
df['samples'] = df.index.tolist()
df['temp_idx'] = range(len(df))

# 5 folds
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df['fold'] = 0

for fold, (_, val_idx) in enumerate(skf.split(X=df['temp_idx'], y=df.iloc[:,0]), 1):
    df.iloc[val_idx, df.columns.get_loc('fold')] = fold

# Remove the temporary column
df = df.drop('temp_idx', axis=1)
df.to_csv(path + 'cardio_5folds.csv', index=False)





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


path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/cardio/'
adata = sc.read_h5ad(path + 'cardio_processed.h5ad')
# AnnData object with n_obs × n_vars = 592689 × 2000
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

adata.obs[['disease','biosample_id']].value_counts().sort_index()


df = pd.DataFrame(adata.obs[['disease','biosample_id']].value_counts().sort_index())
df['disease'] = df.index.get_level_values(0)
df.index = df.index.get_level_values(1)
df.index = df.index.tolist()
df.drop(columns='count', inplace=True)
df['samples'] = df.index.tolist()
df['temp_idx'] = range(len(df))

# 5 folds
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df['fold'] = 0

for fold, (_, val_idx) in enumerate(skf.split(X=df['temp_idx'], y=df.iloc[:,0]), 1):
    df.iloc[val_idx, df.columns.get_loc('fold')] = fold

# Remove the temporary column
df = df.drop('temp_idx', axis=1)
df.to_csv(path + 'cardio_biosample_id_5folds.csv', index=False)









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



path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/covid/'
adata = sc.read_h5ad(path + 'covid_Ziegler_processed.h5ad')


adata.obs.disease__ontology_label.value_counts()
# disease__ontology_label
# COVID-19               18073
# normal                  8874
# respiratory failure     3335
# long COVID-19           2306

adata.obs.donor_id.value_counts()    # 

adata.obs[['disease__ontology_label','donor_id']].value_counts().sort_index()


adata.obs['disease_id'] = 'control'

# COVID-19  and long COVID-19 to 'case'
adata.obs.loc[adata.obs['disease__ontology_label'] == 'COVID-19', 'disease_id'] = 'case'
adata.obs.loc[adata.obs['disease__ontology_label'] == 'long COVID-19', 'disease_id'] = 'case'
adata.obs.disease_id.value_counts()


adata.obs[['disease_id','donor_id']].value_counts().sort_index()


df = pd.DataFrame(adata.obs[['disease_id','donor_id']].value_counts().sort_index())
df['disease_id'] = df.index.get_level_values(0)
df.index = df.index.get_level_values(1)
df.index = df.index.tolist()
df.drop(columns='count', inplace=True)
df['samples'] = df.index.tolist()
df['temp_idx'] = range(len(df))

# 5 folds
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df['fold'] = 0

for fold, (_, val_idx) in enumerate(skf.split(X=df['temp_idx'], y=df.iloc[:,0]), 1):
    df.iloc[val_idx, df.columns.get_loc('fold')] = fold

# Remove the temporary column
df = df.drop('temp_idx', axis=1)
df.to_csv(path + 'covid_5folds.csv', index=False)







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

path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/lupus/'
adata = sc.read_h5ad(path + 'lupus_processed.h5ad')
# AnnData object with n_obs × n_vars = 834096 × 2000
#     obs: 'disease_cov', 'ct_cov', 'pop_cov', 'ind_cov', 'well', 'batch_cov', 'batch'
#     var: 'gene_ids-0-0-0-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0', 'gene_ids-1-0-0-0-0', 'gene_ids-1-0-0-0', 'gene_ids-1-0-0', 'gene_ids-1-0', 'gene_ids-1', 'n_cells', 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm'
#     uns: 'hvg', 'log1p'
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




adata.obs[['disease_cov','ind_cov']].value_counts().sort_index()


df = pd.DataFrame(adata.obs[['disease_cov','ind_cov']].value_counts().sort_index())
df['disease_cov'] = df.index.get_level_values(0)
df.index = df.index.get_level_values(1)
df.index = df.index.tolist()
df.drop(columns='count', inplace=True)
df['samples'] = df.index.tolist()
df['temp_idx'] = range(len(df))

# 5 folds
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df['fold'] = 0

for fold, (_, val_idx) in enumerate(skf.split(X=df['temp_idx'], y=df.iloc[:,0]), 1):
    df.iloc[val_idx, df.columns.get_loc('fold')] = fold

# Remove the temporary column
df = df.drop('temp_idx', axis=1)
df.to_csv(path + 'lupus_5folds.csv', index=False)








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

path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/lupus/'
adata = sc.read_h5ad(path + 'lupus_processed.h5ad')
# AnnData object with n_obs × n_vars = 834096 × 2000
#     obs: 'disease_cov', 'ct_cov', 'pop_cov', 'ind_cov', 'well', 'batch_cov', 'batch'
#     var: 'gene_ids-0-0-0-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0', 'gene_ids-1-0-0-0-0', 'gene_ids-1-0-0-0', 'gene_ids-1-0-0', 'gene_ids-1-0', 'gene_ids-1', 'n_cells', 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm'
#     uns: 'hvg', 'log1p'
adata.obs.disease_cov.value_counts()
# disease_cov
# sle        557630
# healthy    276466
adata.obs.pop_cov.value_counts()
# pop_cov
# WHITE    533178
# ASIAN    300918

adata.obs.pop_cov.value_counts()    # 169
# pop_cov
# WHITE    533178
# ASIAN    300918




adata.obs[['pop_cov','ind_cov']].value_counts().sort_index()


df = pd.DataFrame(adata.obs[['pop_cov','ind_cov']].value_counts().sort_index())
df['pop_cov'] = df.index.get_level_values(0)
df.index = df.index.get_level_values(1)
df.index = df.index.tolist()
df.drop(columns='count', inplace=True)
df['samples'] = df.index.tolist()
df['temp_idx'] = range(len(df))

# 5 folds
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df['fold'] = 0

for fold, (_, val_idx) in enumerate(skf.split(X=df['temp_idx'], y=df.iloc[:,0]), 1):
    df.iloc[val_idx, df.columns.get_loc('fold')] = fold

# Remove the temporary column
df = df.drop('temp_idx', axis=1)
df.to_csv(path + 'race_5folds.csv', index=False)