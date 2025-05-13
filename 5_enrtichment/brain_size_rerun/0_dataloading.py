# (mil) hongruhu@gpu-5-50:/group/gquongrp/workspaces/hongruhu/bioPointNet$ python

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns
from bioPointNet_Apr2025 import *


obj = torch.load('/group/gquongrp/collaborations/viral_evo/longevity/embeddings.pt')
len(list(obj.keys())) # 266

brain_size = pd.read_csv('/group/gquongrp/collaborations/viral_evo/longevity/brainsize_labels.csv', index_col=0)
# 132
# longevity = pd.read_csv('/group/gquongrp/collaborations/viral_evo/longevity/longevity_labels.csv', index_col=0)
# 234


all_samples = list(obj.keys())
brain_size_samples = brain_size.Proteome_ID.tolist()

intersection_samples = list(set(all_samples) & set(brain_size_samples))
len(intersection_samples) # 131


brain_size.index = brain_size.Proteome_ID

obj_brainsize = {k: obj[k] for k in intersection_samples}
df_brainsize = brain_size.loc[intersection_samples]

# plot the distribution of "log_brain_mass"
fig = plt.figure(figsize=(10, 5))
sns.histplot(df_brainsize['log_brain_mass'], bins=20)
plt.xlabel('log_brain_mass')
plt.ylabel('Frequency')
plt.title('Distribution of log_brain_mass')
plt.show()
fig.savefig('log_brain_mass_distribution.png')


# split the data into 10 folds randomly
np.random.seed(0)
df_brainsize['fold'] = np.random.randint(0, 10, size=len(df_brainsize))
df_brainsize.to_csv('log_brain_mass_labels.csv')

torch.save(obj_brainsize, 'embeddings_brainsize.pt')


import json
with open('/group/gquongrp/collaborations/viral_evo/longevity/all_proteins.json', 'r') as file:
    data = json.load(file)

data = {x.split('.')[0]: v for x, v in data.items()}
data.keys()

torch.save(data, 'species_protein_name.pt')



metadata = pd.read_csv('/group/gquongrp/collaborations/viral_evo/longevity/metadata.csv', index_col=0)

metadata.index = metadata['Proteome_ID_x'].tolist()

len(set(metadata.index)) == metadata.shape[0]

metadata.to_csv('species_id.csv')