import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns
from bioPointNet_Apr2025 import *

adata = sc.read_h5ad("/group/gquongrp/workspaces/hongruhu/MIL/neural_organoid/scPN_modeling/hdbca_intersected_union1000hvg.h5ad")
# AnnData object with n_obs × n_vars = 1665937 × 1766



umap = np.load('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/hdbca_intersected_union1000hvg/hdbca_umap.npy')
umap.shape # (1665937, 2)
adata.obsm['X_umap'] = umap.copy()

metadata = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/hnoca_intersected_union1000hvg/metadata_tr_HNOCA_model_HDBCA.csv', index_col=0)
attn = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/hnoca_intersected_union1000hvg/attention_mtx_HNOCA_model_HDBCA.csv', index_col=0)

adata.obs.index = adata.obs.index.tolist()




umap_df = pd.DataFrame(adata.obsm['X_umap'], index=adata.obs.index.tolist(), columns=['umap_0','umap_1'])

umap_df = umap_df.loc[attn.index]

metadata['attention_score_norm_cellnum'] = attn.attention_score_norm_cellnum
metadata['attention_score_norm_cellnum_clip'] = attn.attention_score_norm_cellnum_clip

metadata['high_attention_score'] = 'no'
metadata.loc[metadata['attention_score_norm_cellnum'] > 15, 'high_attention_score'] = 'yes'





df = umap_df.join(metadata)


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
fig.savefig('HDBCA_attn_clip.png')




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
fig.savefig('HDBCA_attn_binary.png')






# Define colors and sizes for CellClass
df['color'] = 'grey'
df['alpha'] = 0.5
df['size'] = 5

df.loc[df['CellClass'] == 'Radial glia', 'color'] = 'steelblue'
df.loc[df['CellClass'] == 'Glioblast', 'color'] = 'crimson'
df.loc[df['CellClass'].isin(['Radial glia', 'Glioblast']), 'alpha'] = 0.5
df.loc[df['CellClass'].isin(['Radial glia', 'Glioblast']), 'size'] = 5

# Plot with Radial glia and Glioblast on top
fig = plt.figure(figsize=(8, 7))
plt.scatter(
    df['umap_0'], df['umap_1'],
    c=df['color'],
    s=df['size'],
    alpha=df['alpha'],
    edgecolors='none'
)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
sns.despine()
plt.tight_layout()
fig.savefig('HDBCA_CellClass.png')


adata.obs['color'] = df['color'] 

sc.pl.umap(adata, color=["color", 'CellClass'], ncols=1, save="_HDBCA_CellClass.png")



