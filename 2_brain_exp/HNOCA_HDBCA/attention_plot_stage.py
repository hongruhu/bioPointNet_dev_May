import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns
from bioPointNet_Apr2025 import *



# reference HNOCA
ref = sc.read_h5ad("/group/gquongrp/workspaces/hongruhu/MIL/neural_organoid/scPN_modeling/hnoca_intersected_union1000hvg.h5ad")
# umap_ref = np.load('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/hnoca_intersected_union1000hvg/hnoca_umap.npy')
# obj = sc.read_h5ad("/group/gquongrp/workspaces/hongruhu/MIL/neural_organoid/HNOCA_healthy_embedding.h5ad")
# umap_df_ref = pd.DataFrame(umap_ref, index=obj.obs.index.tolist(), columns=['umap_0','umap_1'])
# umap_df_ref = umap_df_ref.loc[ref.obs.index.tolist()]
# umap_df_ref.to_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/hnoca_intersected_union1000hvg/umap_HNOCA_df.csv')

metadata_ref = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/hnoca_intersected_union1000hvg/metadata_tr_HNOCA.csv', index_col=0)
attn_ref = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/hnoca_intersected_union1000hvg/attention_mtx_HNOCA.csv', index_col=0)

metadata_ref['attention_score_norm_cellnum'] = attn_ref.attention_score_norm_cellnum
metadata_ref['attention_score_norm_cellnum_clip'] = attn_ref.attention_score_norm_cellnum_clip
metadata_ref['high_attention_score'] = 'no'
metadata_ref.loc[metadata_ref['attention_score_norm_cellnum'] > 15, 'high_attention_score'] = 'yes'
metadata_ref.index = metadata_ref.index.tolist()

umap_df_ref = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/hnoca_intersected_union1000hvg/umap_HNOCA_df.csv', index_col=0)
umap_df_ref = umap_df_ref.loc[metadata_ref.original_cell_id.tolist()]
umap_df_ref.index = metadata_ref.index.tolist()






# query HDBCA
query = sc.read_h5ad("/group/gquongrp/workspaces/hongruhu/MIL/neural_organoid/scPN_modeling/hdbca_intersected_union1000hvg.h5ad")
query.obs.index = query.obs.index.tolist()

metadata_query = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/hnoca_intersected_union1000hvg/metadata_tr_HNOCA_model_HDBCA.csv', index_col=0)
attn_query = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/hnoca_intersected_union1000hvg/attention_mtx_HNOCA_model_HDBCA.csv', index_col=0)

metadata_query['attention_score_norm_cellnum'] = attn_query.attention_score_norm_cellnum
metadata_query['attention_score_norm_cellnum_clip'] = attn_query.attention_score_norm_cellnum_clip
metadata_query['high_attention_score'] = 'no'
metadata_query.loc[metadata_query['attention_score_norm_cellnum'] > 15, 'high_attention_score'] = 'yes'
metadata_query.index = metadata_query.index.tolist()

# umap_query = np.load('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/hdbca_intersected_union1000hvg/hdbca_umap.npy')
# umap_df_query = pd.DataFrame(query.obsm['X_umap'], index=query.obs.index.tolist(), columns=['umap_0','umap_1'])
# umap_df_query = umap_df_query.loc[attn_query.index]
# umap_df_query.index = umap_df_query.index.tolist()
# umap_df_query.to_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/hdbca_intersected_union1000hvg/umap_HDBCA_df.csv')
umap_df_query = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/hdbca_intersected_union1000hvg/umap_HDBCA_df.csv', index_col=0)









# Split UMAP by different "metadata_ref.organoid_age_days"
unique_stages = metadata_ref['organoid_age_days'].unique()
unique_stages.sort()
for stage in unique_stages:
    stage_df = umap_df_ref.join(metadata_ref[metadata_ref['organoid_age_days'] == stage])
    stage_df['size'] = stage_df['attention_score_norm_cellnum_clip'].apply(lambda x: 20 if x > 5 else 5)
    fig = plt.figure(figsize=(8, 7))
    sns.scatterplot(
        data=stage_df,
        x='umap_0', y='umap_1',
        hue='attention_score_norm_cellnum_clip',
        size='size',
        sizes=(10, 20),
        palette='viridis',
        edgecolor=None,
        legend=None,
        alpha=0.5
    )
    plt.title(f"Development Stage: {stage}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    sns.despine()
    plt.tight_layout()
    fig.savefig(f'HNOCA_attn_clip_{stage}.png')
    plt.close(fig)



unique_stages = metadata_ref['organoid_age_days'].unique()
unique_stages.sort()
for stage in unique_stages:
    stage_metadata = metadata_ref[metadata_ref['organoid_age_days'] == stage]
    stage_df = umap_df_ref.join(stage_metadata)
    threshold = stage_df['attention_score_norm_cellnum'].quantile(0.95)  # Top 5% threshold
    stage_df['high_attention_score'] = 'no'
    stage_df.loc[stage_df['attention_score_norm_cellnum'] > threshold, 'high_attention_score'] = 'yes'
    stage_df['color'] = stage_df['high_attention_score'].map({'yes': 'crimson', 'no': 'grey'})
    stage_df['alpha'] = stage_df['high_attention_score'].map({'yes': 1.0, 'no': 0.5})
    stage_df['size_binary'] = stage_df['high_attention_score'].map({'yes': 20, 'no': 10})
    fig = plt.figure(figsize=(8, 7))
    for label in ['no', 'yes']:
        sub_df = stage_df[stage_df['high_attention_score'] == label]
        plt.scatter(
            sub_df['umap_0'], sub_df['umap_1'],
            c=sub_df['color'],
            s=sub_df['size_binary'],
            alpha=sub_df['alpha'].iloc[0],
            edgecolors='none'
        )
    plt.title(f"Binary Attention - Development Stage: {stage}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    sns.despine()
    plt.tight_layout()
    fig.savefig(f'HNOCA_attn_binary_{stage}.png')
    plt.close(fig)









# Split UMAP by different "metadata_query.development_stage"
unique_stages = metadata_query['development_stage'].unique()
unique_stages.sort()

for stage in unique_stages:
    stage_df = umap_df_query.join(metadata_query[metadata_query['development_stage'] == stage])
    stage_df['size'] = stage_df['attention_score_norm_cellnum_clip'].apply(lambda x: 20 if x > 5 else 5)
    fig = plt.figure(figsize=(8, 7))
    sns.scatterplot(
        data=stage_df,
        x='umap_0', y='umap_1',
        hue='attention_score_norm_cellnum_clip',
        size='size',
        sizes=(10, 20),
        palette='viridis',
        edgecolor=None,
        legend=None,
        alpha=0.5
    )
    plt.title(f"Development Stage: {stage}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    sns.despine()
    plt.tight_layout()
    fig.savefig(f'HDBCA_attn_clip_{stage}.png')
    plt.close(fig)




unique_stages = metadata_query['development_stage'].unique()
unique_stages.sort()
for stage in unique_stages:
    stage_metadata = metadata_query[metadata_query['development_stage'] == stage]
    stage_df = umap_df_query.join(stage_metadata)
    threshold = stage_df['attention_score_norm_cellnum'].quantile(0.95)  # Top 5% threshold
    stage_df['high_attention_score'] = 'no'
    stage_df.loc[stage_df['attention_score_norm_cellnum'] > threshold, 'high_attention_score'] = 'yes'
    stage_df['color'] = stage_df['high_attention_score'].map({'yes': 'crimson', 'no': 'grey'})
    stage_df['alpha'] = stage_df['high_attention_score'].map({'yes': 1.0, 'no': 0.5})
    stage_df['size_binary'] = stage_df['high_attention_score'].map({'yes': 20, 'no': 10})
    fig = plt.figure(figsize=(8, 7))
    for label in ['no', 'yes']:
        sub_df = stage_df[stage_df['high_attention_score'] == label]
        plt.scatter(
            sub_df['umap_0'], sub_df['umap_1'],
            c=sub_df['color'],
            s=sub_df['size_binary'],
            alpha=sub_df['alpha'].iloc[0],
            edgecolors='none'
        )
    plt.title(f"Binary Attention - Development Stage: {stage}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    sns.despine()
    plt.tight_layout()
    fig.savefig(f'HDBCA_attn_binary_{stage}.png')
    plt.close(fig)


