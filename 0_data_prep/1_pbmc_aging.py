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



# ref: Terekhova 2023 Cell: https://doi.org/10.1016/j.immuni.2023.10.013
# 317 samples
# syn49637038 | https://www.synapse.org/Synapse:syn50542388
# query: Zhu 2023 SciAdv: https://doi.org/10.1126/sciadv.abq7599
# Processed gene expression data are deposited in Gene Expression Omnibus (accession no. GSE213516).