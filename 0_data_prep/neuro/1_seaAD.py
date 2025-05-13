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


path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/'
adata = sc.read_h5ad(path + 'DLPFC.h5ad')

adata.obs.cell_type.value_counts()
# cell_type
# L2/3-6 intratelencephalic projecting glutamatergic neuron         589217
# oligodendrocyte                                                   145995
# pvalb GABAergic cortical interneuron                              116142
# VIP GABAergic cortical interneuron                                100215
# astrocyte of the cerebral cortex                                   87444
# lamp5 GABAergic cortical interneuron                               78859
# sst GABAergic cortical interneuron                                 77583
# microglial cell                                                    42486
# corticothalamic-projecting glutamatergic cortical neuron           29343
# oligodendrocyte precursor cell                                     29041
# sncg GABAergic cortical interneuron                                24792
# near-projecting glutamatergic cortical neuron                      19191
# L6b glutamatergic cortical neuron                                  18903
# chandelier pvalb GABAergic cortical interneuron                    15412
# caudal ganglionic eminence derived interneuron                      9446
# vascular leptomeningeal cell                                        4860
# L5 extratelencephalic projecting glutamatergic cortical neuron      4097
# cerebral cortex endothelial cell                                    2575

adata.obs.Class.value_counts()
# Class
# Neuronal: Glutamatergic        660751
# Neuronal: GABAergic            422449
# Non-neuronal and Non-neural    312401

glial = adata[adata.obs.Class == 'Non-neuronal and Non-neural']
glial.obs['donor_id'].value_counts() # 83 samples





sc.pp.filter_genes(glial, min_cells=10)
glial
# AnnData object with n_obs × n_vars = 312401 × 33390
sc.pp.highly_variable_genes(glial, n_top_genes=3000, flavor='seurat_v3')


glial.obs.cell_type.value_counts()
# cell_type
# oligodendrocyte                     145995
# astrocyte of the cerebral cortex     87444
# microglial cell                      42486
# oligodendrocyte precursor cell       29041
# vascular leptomeningeal cell          4860
# cerebral cortex endothelial cell      2575


glial.write_h5ad(path + 'DLPFC_glial.h5ad')
























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


path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/'
adata = sc.read_h5ad(path + 'MTG.h5ad')

adata.obs.cell_type.value_counts()
# cell_type
# L2/3-6 intratelencephalic projecting glutamatergic neuron         589217
# oligodendrocyte                                                   145995
# pvalb GABAergic cortical interneuron                              116142
# VIP GABAergic cortical interneuron                                100215
# astrocyte of the cerebral cortex                                   87444
# lamp5 GABAergic cortical interneuron                               78859
# sst GABAergic cortical interneuron                                 77583
# microglial cell                                                    42486
# corticothalamic-projecting glutamatergic cortical neuron           29343
# oligodendrocyte precursor cell                                     29041
# sncg GABAergic cortical interneuron                                24792
# near-projecting glutamatergic cortical neuron                      19191
# L6b glutamatergic cortical neuron                                  18903
# chandelier pvalb GABAergic cortical interneuron                    15412
# caudal ganglionic eminence derived interneuron                      9446
# vascular leptomeningeal cell                                        4860
# L5 extratelencephalic projecting glutamatergic cortical neuron      4097
# cerebral cortex endothelial cell                                    2575

# cell_type
# L2/3-6 intratelencephalic projecting glutamatergic neuron         698416
# oligodendrocyte                                                   111194
# VIP GABAergic cortical interneuron                                104514
# pvalb GABAergic cortical interneuron                               90804
# astrocyte of the cerebral cortex                                   70009
# lamp5 GABAergic cortical interneuron                               64364
# sst GABAergic cortical interneuron                                 59761
# microglial cell                                                    40000
# oligodendrocyte precursor cell                                     32493
# sncg GABAergic cortical interneuron                                22168
# near-projecting glutamatergic cortical neuron                      20741
# corticothalamic-projecting glutamatergic cortical neuron           18402
# L6b glutamatergic cortical neuron                                  16227
# chandelier pvalb GABAergic cortical interneuron                    10928
# caudal ganglionic eminence derived interneuron                      9203
# vascular leptomeningeal cell                                        4328
# L5 extratelencephalic projecting glutamatergic cortical neuron      2590
# cerebral cortex endothelial cell                                    2069

adata.obs.Class.value_counts()
# Class
# Neuronal: Glutamatergic        660751
# Neuronal: GABAergic            422449
# Non-neuronal and Non-neural    312401

# Class
# Neuronal: Glutamatergic        756376
# Neuronal: GABAergic            361742
# Non-neuronal and Non-neural    260093

glial = adata[adata.obs.Class == 'Non-neuronal and Non-neural']
glial.obs['donor_id'].value_counts() # 89 samples

glial = glial[glial.obs.donor_id != 'H18.30.001']
glial = glial[glial.obs.donor_id != 'H200.1023']
glial.obs['donor_id'].value_counts() # 87 samples




sc.pp.filter_genes(glial, min_cells=10)
glial
# AnnData object with n_obs × n_vars = 312401 × 34016
# AnnData object with n_obs × n_vars = 260066 × 33222
sc.pp.highly_variable_genes(glial, n_top_genes=3000, flavor='seurat_v3')


glial.obs.cell_type.value_counts()
# cell_type
# oligodendrocyte                     145995
# astrocyte of the cerebral cortex     87444
# microglial cell                      42486
# oligodendrocyte precursor cell       29041
# vascular leptomeningeal cell          4860
# cerebral cortex endothelial cell      2575

# cell_type
# oligodendrocyte                     111187
# astrocyte of the cerebral cortex     70008
# microglial cell                      39998
# oligodendrocyte precursor cell       32491
# vascular leptomeningeal cell          4328
# cerebral cortex endothelial cell      2054


glial.write_h5ad(path + 'MTG_glial.h5ad')