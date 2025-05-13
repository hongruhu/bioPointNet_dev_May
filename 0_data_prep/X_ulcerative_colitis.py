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


# Smillie et al, 2019 Cell 
# https://doi.org/10.1016/j.cell.2019.06.029
# 24 healthy controls, 18 patients with ulcerative colitis
