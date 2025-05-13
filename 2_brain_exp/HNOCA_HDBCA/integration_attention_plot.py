# (mil) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet


import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns


hnoca = sc.read_h5ad("/group/gquongrp/workspaces/hongruhu/MIL/neural_organoid/scPN_modeling/hnoca_intersected_union1000hvg.h5ad")
hdbca = sc.read_h5ad("/group/gquongrp/workspaces/hongruhu/MIL/neural_organoid/scPN_modeling/hdbca_intersected_union1000hvg.h5ad")



umap_hnoca_df = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/integration/umap_hnoca.csv',
                            index_col=0)
umap_hdbca_df = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/integration/umap_hdbca.csv',
                            index_col=0)
cell_id_df_hnoca = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/integration/cell_hnoca.csv',
                                index_col=0)
cell_id_df_hdbca = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/integration/cell_hdbca.csv',
                                index_col=0)
# Add UMAP results to AnnData objects
hnoca.obsm['X_umap'] = umap_hnoca_df.loc[hnoca.obs.index.tolist()].to_numpy()
hdbca.obsm['X_umap'] = umap_hdbca_df.loc[hdbca.obs.index.tolist()].to_numpy()
# Plot color by cell_type and save
sc.pl.umap(hnoca, color='annot_level_2', save='_hnoca_cell_type_project_HNOCA_.png')
sc.pl.umap(hdbca, color='CellClass', save='_hdbca_cell_type_project_HNOCA_.png')

sc.pl.umap(hnoca, color='annot_level_2', save='_hnoca_cell_type_project_HNOCA_legend.png', legend_loc='on data')
sc.pl.umap(hdbca, color='CellClass', save='_hdbca_cell_type_project_HNOCA_legend.png', legend_loc='on data')



cell_id_df_hnoca['stage'] = hnoca.obs.loc[cell_id_df_hnoca.index]['organoid_age_days']
unique_stages = cell_id_df_hnoca['stage'].unique()
unique_stages.sort()
for stage in unique_stages:
    stage_metadata = cell_id_df_hnoca[cell_id_df_hnoca['stage'] == stage]
    stage_df = umap_hnoca_df.join(stage_metadata)
    threshold = stage_df['attn_norm'].quantile(0.95)  # Top 5% threshold
    stage_df['high_attention_score'] = 'no'
    stage_df.loc[stage_df['attn_norm'] > threshold, 'high_attention_score'] = 'yes'
    stage_df['color'] = stage_df['high_attention_score'].map({'yes': 'crimson', 'no': 'grey'})
    stage_df['alpha'] = stage_df['high_attention_score'].map({'yes': 1.0, 'no': 0.5})
    stage_df['size_binary'] = stage_df['high_attention_score'].map({'yes': 20, 'no': 10})
    fig = plt.figure(figsize=(8, 7))
    for label in ['no', 'yes']:
        sub_df = stage_df[stage_df['high_attention_score'] == label]
        plt.scatter(
            sub_df['UMAP1'], sub_df['UMAP2'],
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
    fig.savefig(f'HNOCA_integreated_attn_binary_{stage}.png')
    plt.close(fig)





cell_id_df_hdbca['stage'] = hdbca.obs.loc[cell_id_df_hdbca.index]['development_stage']
unique_stages = cell_id_df_hdbca['stage'].astype(str).unique()
unique_stages.sort()
for stage in unique_stages:
    stage_metadata = cell_id_df_hdbca[cell_id_df_hdbca['stage'] == stage]
    stage_df = umap_hdbca_df.join(stage_metadata)
    threshold = stage_df['attn_norm'].quantile(0.95)  # Top 5% threshold
    stage_df['high_attention_score'] = 'no'
    stage_df.loc[stage_df['attn_norm'] > threshold, 'high_attention_score'] = 'yes'
    stage_df['color'] = stage_df['high_attention_score'].map({'yes': 'crimson', 'no': 'grey'})
    stage_df['alpha'] = stage_df['high_attention_score'].map({'yes': 1.0, 'no': 0.5})
    stage_df['size_binary'] = stage_df['high_attention_score'].map({'yes': 20, 'no': 10})
    fig = plt.figure(figsize=(8, 7))
    for label in ['no', 'yes']:
        sub_df = stage_df[stage_df['high_attention_score'] == label]
        plt.scatter(
            sub_df['UMAP1'], sub_df['UMAP2'],
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
    fig.savefig(f'HDBCA_integreated_attn_binary_{stage}.png')
    plt.close(fig)





# Plot split on the color-based annotation for hnoca
unique_annotations_hnoca = hnoca.obs['annot_level_2'].unique()
for annotation in unique_annotations_hnoca:
    fig = plt.figure(figsize=(8, 7))
    annotation_df = umap_hnoca_df.copy()
    annotation_df['highlight'] = hnoca.obs['annot_level_2'] == annotation
    annotation_df['color'] = annotation_df['highlight'].map({True: 'crimson', False: 'grey'})
    annotation_df['alpha'] = annotation_df['highlight'].map({True: 1.0, False: 0.5})
    annotation_df['size'] = annotation_df['highlight'].map({True: 20, False: 10})
    # Plot non-highlighted cells first (grey)
    sub_df = annotation_df[~annotation_df['highlight']]
    plt.scatter(
        sub_df['UMAP1'], sub_df['UMAP2'],
        c=sub_df['color'],
        s=sub_df['size'],
        alpha=sub_df['alpha'].iloc[0],
        edgecolors='none'
    )
    # Plot highlighted cells on top (crimson)
    sub_df = annotation_df[annotation_df['highlight']]
    plt.scatter(
        sub_df['UMAP1'], sub_df['UMAP2'],
        c=sub_df['color'],
        s=sub_df['size'],
        alpha=sub_df['alpha'].iloc[0],
        edgecolors='none'
    )
    plt.title(f"HNOCA - Highlighted: {annotation}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    sns.despine()
    plt.tight_layout()
    fig.savefig(f'HNOCA_highlight_{annotation}.png')
    plt.close(fig)


# Plot split on the color-based annotation for hdbca
unique_annotations_hdbca = hdbca.obs['CellClass'].unique()
for annotation in unique_annotations_hdbca:
    fig = plt.figure(figsize=(8, 7))
    annotation_df = umap_hdbca_df.copy()
    annotation_df['highlight'] = hdbca.obs['CellClass'] == annotation
    annotation_df['color'] = annotation_df['highlight'].map({True: 'crimson', False: 'grey'})
    annotation_df['alpha'] = annotation_df['highlight'].map({True: 1.0, False: 0.5})
    annotation_df['size'] = annotation_df['highlight'].map({True: 20, False: 10})
    # Plot non-highlighted cells first (grey)
    sub_df = annotation_df[~annotation_df['highlight']]
    plt.scatter(
        sub_df['UMAP1'], sub_df['UMAP2'],
        c=sub_df['color'],
        s=sub_df['size'],
        alpha=sub_df['alpha'].iloc[0],
        edgecolors='none'
    )
    # Plot highlighted cells on top (crimson)
    sub_df = annotation_df[annotation_df['highlight']]
    plt.scatter(
        sub_df['UMAP1'], sub_df['UMAP2'],
        c=sub_df['color'],
        s=sub_df['size'],
        alpha=sub_df['alpha'].iloc[0],
        edgecolors='none'
    )
    plt.title(f"HDBCA - Highlighted: {annotation}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    sns.despine()
    plt.tight_layout()
    fig.savefig(f'HDBCA_highlight_{annotation}.png')
    plt.close(fig)
