# (mil) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet$ python
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns

from bioPointNet_Apr2025 import *


data_source = 'DLPFC'

sample_key = 'donor_id'
task_key = 'disease'
cov_key = 'sex'
ct_key = 'cell_type'
ct_key_fine = 'Supertype'

class_num = 2

path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/SEAAD_Glia/'




SEED = 2

adata_tr = sc.read(path + data_source + '/' + data_source + '_cls/ALL_sex_cov/' + "Adata_Tr_" + data_source + '_' + str(SEED) + ".h5ad")






umap_df = pd.DataFrame(adata_tr.obsm['X_umap'], index=adata_tr.obs.index.tolist(), columns=['umap_0','umap_1'])
metadata = adata_tr.obs.copy()


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
fig.savefig('DLPFC_all_sexcov_SEED2_attn_clip.png')




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
fig.savefig('DLPFC_all_sexcov_SEED2_attn_binary.png')




adata_tr_0 = adata_tr[adata_tr.obs[task_key] != metadata[task_key][0]] # AD
adata_tr_1 = adata_tr[adata_tr.obs[task_key] == metadata[task_key][0]] # normal



umap_df = pd.DataFrame(adata_tr_1.obsm['X_umap'], index=adata_tr_1.obs.index.tolist(), columns=['umap_0','umap_1'])
metadata = adata_tr_1.obs.copy()
metadata['high_attention_score'] = 'no'
metadata.loc[metadata['attention_score_norm_cellnum'] > 8, 'high_attention_score'] = 'yes'
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
fig.savefig('DLPFC_all_sexcov_SEED2_attn_clip_adata_tr_1Healthy.png')
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
fig.savefig('DLPFC_all_sexcov_SEED2_attn_binary_adata_tr_1Healthy.png')









umap_df = pd.DataFrame(adata_tr_0.obsm['X_umap'], index=adata_tr_0.obs.index.tolist(), columns=['umap_0','umap_1'])
metadata = adata_tr_0.obs.copy()
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
fig.savefig('DLPFC_all_sexcov_SEED2_attn_clip_adata_tr_0AD.png')
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
fig.savefig('DLPFC_all_sexcov_SEED2_attn_binary_adata_tr_0AD.png')




donors = adata_tr_0.obs[sample_key].unique()
# remove specific donors
donors = [donor for donor in donors if donor not in ['H21.33.027', 'H21.33.016', 'H21.33.043', 'H20.33.041']]



n = len(donors)
threshold = 100
proportion_dict = {}
for i in donors:
    print(i)
    adata_donor = adata_tr_0[adata_tr_0.obs[sample_key] == i] # AD
    umap_df = pd.DataFrame(adata_donor.obsm['X_umap'], index=adata_donor.obs.index.tolist(), columns=['umap_0','umap_1'])
    metadata = adata_donor.obs.copy()
    metadata['high_attention_score'] = 'no'
    top_indices = metadata['attention_score_norm_cellnum'].nlargest(threshold).index
    metadata.loc[top_indices, 'high_attention_score'] = 'yes'
    df = umap_df.join(metadata)
    df['size'] = df['attention_score_norm_cellnum_clip'].apply(lambda x: 20 if x > 5 else 5)
    # fig = plt.figure(figsize=(8, 7))
    # sns.scatterplot(
    #     data=df,
    #     x='umap_0', y='umap_1',
    #     hue='attention_score_norm_cellnum_clip',
    #     size='size',
    #     sizes=(10, 20),
    #     palette='viridis',
    #     edgecolor=None,
    #     legend=None,
    #     alpha = 0.5
    # )
    # plt.xlabel("UMAP 1")
    # plt.ylabel("UMAP 2")
    # sns.despine()
    # plt.tight_layout()
    # fig.savefig('DLPFC_all_sexcov_SEED2_attn_clip_' + i+ '.png')
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
    fig.savefig('DLPFC_all_sexcov_SEED2_attn_binary_' + i+ '.png')
    proportion_dict[i]=metadata.loc[top_indices]['Subclass'].value_counts(normalize=True)



proportion_dict_df = pd.DataFrame.from_dict(proportion_dict, orient='index')

# Clustermap on rows
sns.clustermap(proportion_dict_df, cmap="viridis", figsize=(10, 10), row_cluster=True, col_cluster=False)
plt.savefig('proportion_clustermap.png')












