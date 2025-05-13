# (mil) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet$ python

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns


adata = sc.read("/group/gquongrp/workspaces/hongruhu/MIL/CT/top4_disease/HNOCA_disease_attn.h5ad")







adata.obs['high_attention_score'] = 'no'
adata.obs.loc[adata.obs['attention_score_norm_cellnum'] > 20, 'high_attention_score'] = 'yes'




umap_df = pd.DataFrame(adata.obsm['X_umap'], index=adata.obs.index.tolist(), columns=['umap_0','umap_1'])

df = umap_df.join(adata.obs)


df['size'] = df['attention_score_norm_cellnum_clip'].apply(lambda x: 20 if x > 5 else 5)

fig = plt.figure(figsize=(8, 7))
sns.scatterplot(
    data=df,
    x='umap_0', y='umap_1',
    hue='attention_score_norm_cellnum_clip',
    size='size',
    sizes=(10, 20),
    palette='viridis',
    edgecolor=None,
    legend=None,
    alpha = 0.5
)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
sns.despine()
plt.tight_layout()
fig.savefig('HNOCA_disease_attn_clip.png')




df['color'] = df['high_attention_score'].map({'yes': 'crimson', 'no': 'grey'})
df['alpha'] = df['high_attention_score'].map({'yes': 1.0, 'no': 0.5})
df['size_binary'] = df['high_attention_score'].map({'yes': 20, 'no': 10})

fig = plt.figure(figsize=(8, 7))
for label in ['no', 'yes']:
    sub_df = df[df['high_attention_score'] == label]
    plt.scatter(
        sub_df['umap_0'], sub_df['umap_1'],
        c=sub_df['color'],
        s=sub_df['size_binary'],
        alpha=sub_df['alpha'].iloc[0],
        edgecolors='none'
    )

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
sns.despine()
plt.tight_layout()
fig.savefig('HNOCA_disease_attn_binary.png')
