
# (mil) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet$
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from bioPointNet_Apr2025 import *

import umap
from adjustText import adjust_text


loading_path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/SEAAD_Glia/MTG/MTG_cls/correct_sample/split_from_all/Split_sex_cov_ALT/'
celltype_key = 'Microglia-PVM'
sample_key = 'donor_id'
task_key = 'disease'
cov_key = 'sex'
class_num = 2 


data_source = 'MTG'

sample_key = 'donor_id'
task_key = 'disease'
cov_key = 'sex'
ct_key = 'cell_type'
ct_key_fine = 'Supertype'

class_num = 2

path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/SEAAD_Glia/'



split_id = 0


adata_all = sc.read(loading_path + celltype_key + '/' + 'Adata_Tr_' + celltype_key + '_' + str(split_id)+ '.h5ad')
pred_label = pd.read_csv(loading_path + celltype_key + '/' + 'pred_label_tr_' + celltype_key + '_' + str(split_id)+ '_.csv', index_col=0)
true_label = pd.read_csv(loading_path + celltype_key + '/' + 'true_label_tr_' + celltype_key + '_' + str(split_id)+ '.csv', index_col=0)
wrong_preds = pred_label.idxmax(axis=1)[pred_label.idxmax(axis=1) != true_label.idxmax(axis=1)].index.tolist()
correct_preds = pred_label.idxmax(axis=1)[pred_label.idxmax(axis=1) == true_label.idxmax(axis=1)].index.tolist()
adata_all = adata_all[~adata_all.obs[sample_key].isin(wrong_preds)]
true_label = true_label.loc[correct_preds]
pred_label = pd.read_csv(loading_path + celltype_key + '/' + 'pred_label_tr_' + celltype_key + '_' + str(split_id)+ '.csv', index_col=0)
pred_label = pred_label.loc[correct_preds]
sample = correct_preds[0]
label = true_label.loc[sample]
label = label[label==1].index
perturb_sample = correct_preds[1]
perturb_label = true_label.loc[perturb_sample]
perturb_label = perturb_label[perturb_label==1].index
adata_sample = adata_all[adata_all.obs[sample_key]==sample]
adata_sample_high_attn_cells = adata_sample[adata_sample.obs.attention_score_norm_cellnum.sort_values()[-10:].index]
adata_sample_high_attn_cells.obs[sample_key] = perturb_sample


adata_tr_0 = adata_all[adata_all.obs[task_key] != adata_all.obs[task_key][0]] # 'dementia'
adata_tr_1 = adata_all[adata_all.obs[task_key] == adata_all.obs[task_key][0]]
sc.pl.umap(adata_tr_0, color=['attention_score_norm_cellnum_clip'], 
        title = adata_tr_0.obs[task_key][0],
            ncols=1, wspace=0.5, save='_normed_atten_' + adata_tr_0.obs[task_key][0] + '.pdf')
sc.pl.umap(adata_tr_1, color=['attention_score_norm_cellnum_clip'], 
        title = adata_tr_1.obs[task_key][0],
            ncols=1, wspace=0.5, save='_normed_atten_' + adata_tr_1.obs[task_key][0] + '.pdf')









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
fig.savefig('MTG_mic_sexcov_SEED0_attn_clip_adata_tr_1Healthy.png')
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
fig.savefig('MTG_mic_sexcov_SEED0_attn_binary_adata_tr_1Healthy.png')




umap_df = pd.DataFrame(adata_tr_0.obsm['X_umap'], index=adata_tr_0.obs.index.tolist(), columns=['umap_0','umap_1'])
metadata = adata_tr_0.obs.copy()
metadata['high_attention_score'] = 'no'
metadata.loc[metadata['attention_score_norm_cellnum'] > 7, 'high_attention_score'] = 'yes'
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
fig.savefig('MTG_mic_sexcov_SEED0_attn_clip_adata_tr_0AD.png')
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
fig.savefig('MTG_mic_sexcov_SEED0_attn_binary_adata_tr_0AD.png')
















umap_df = pd.DataFrame(adata_all.obsm['X_umap'], index=adata_all.obs.index.tolist(), columns=['umap_0','umap_1'])
metadata = adata_all.obs.copy()
metadata['is_3'] = 'no'
metadata.loc[metadata['Supertype'] == 'Micro-PVM_3-SEAAD', 'is_3'] = 'yes'
df = umap_df.join(metadata)



df['color'] = df['is_3'].map({'yes': 'crimson', 'no': 'grey'})
df['alpha'] = df['is_3'].map({'yes': 1.0, 'no': 0.5})
df['size_binary'] = df['is_3'].map({'yes': 20, 'no': 10})
fig = plt.figure(figsize=(8, 7))
for label in ['no', 'yes']:
    sub_df = df[df['is_3'] == label]
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
fig.savefig('MTG_mic_sexcov_SEED0_adata_all_ct_is_3.png')






umap_df = pd.DataFrame(adata_tr_1.obsm['X_umap'], index=adata_tr_1.obs.index.tolist(), columns=['umap_0','umap_1'])
metadata = adata_tr_1.obs.copy()
metadata['is_3'] = 'no'
metadata.loc[metadata['Supertype'] == 'Micro-PVM_3-SEAAD', 'is_3'] = 'yes'
df = umap_df.join(metadata)



df['color'] = df['is_3'].map({'yes': 'crimson', 'no': 'grey'})
df['alpha'] = df['is_3'].map({'yes': 1.0, 'no': 0.5})
df['size_binary'] = df['is_3'].map({'yes': 20, 'no': 10})
fig = plt.figure(figsize=(8, 7))
for label in ['no', 'yes']:
    sub_df = df[df['is_3'] == label]
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
fig.savefig('MTG_mic_sexcov_SEED0_adata_tr_1_ct_is_3.png')



adata_tr_0 = adata_all[adata_all.obs[task_key] != adata_all.obs[task_key][0]] # 'dementia'
adata_tr_1 = adata_all[adata_all.obs[task_key] == adata_all.obs[task_key][0]] # dementia
