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


# MultiMIL [*]
# Stephenson et al [Haniffa data] https://www.nature.com/articles/s41591-021-01329-2


# scRAT
# Haniffa data [https://www.nature.com/articles/s41591-021-01329-2] [*]
# COMBAT data [https://doi.org/10.1016/j.cell.2022.01.012]          [*]
# SC4 data Ren et al. Zemin Zhang [https://doi.org/10.1016/j.cell.2021.01.053]
# https://figshare.com/projects/ScRAT_Early_Phenotype_Prediction_From_Single-cell_RNA-seq_Data_using_Attention-Based_Neural_Networks/151659


# scMILD
# https://doi.org/10.1016/j.cell.2021.07.023
# Ziegler et al. Jose Ordovas-Montanes, Cell 2021

# https://doi.org/10.1016/j.cell.2020.10.037
# Su et al. Jame Heath, Cell 2020
# 258 healthy. 110 mild. 102 moderate. 52 severe
# E-MTAB-9357


# Ziegler, Haniffa, COMBAT 