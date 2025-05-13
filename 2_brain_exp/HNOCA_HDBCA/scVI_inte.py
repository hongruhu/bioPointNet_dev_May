# (mil) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns
from scvi.model import SCVI
import scvi


hnoca = sc.read_h5ad("/group/gquongrp/workspaces/hongruhu/MIL/neural_organoid/scPN_modeling/hnoca_intersected_union1000hvg.h5ad")
hdbca = sc.read_h5ad("/group/gquongrp/workspaces/hongruhu/MIL/neural_organoid/scPN_modeling/hdbca_intersected_union1000hvg.h5ad")

hnoca.obs['sample_batch'] = ['hnoca_' + i for i in hnoca.obs.donor_id.tolist()]
hdbca.obs['sample_batch'] = ['hdbca_' + i for i in hdbca.obs.donor_id.tolist()]

hnoca.obs.index = hnoca.obs.index.tolist()
hdbca.obs.index = hdbca.obs.index.tolist()


# Combine the two datasets
import anndata

hnoca.obs.rename(columns={'dissection': 'dissection_hnoca'}, inplace=True)
hdbca.obs.rename(columns={'dissection': 'dissection_hdbca'}, inplace=True)
hnoca.obs = hnoca.obs.loc[:, ~hnoca.obs.columns.duplicated()]
hdbca.obs = hdbca.obs.loc[:, ~hdbca.obs.columns.duplicated()]


adata = anndata.concat([hnoca, hdbca], label="sample_batch", keys=["hnoca", "hdbca"], index_unique=None)



# Set up scVI model
adata.layers["counts"] = adata.X.copy()  # Preserve raw counts
SCVI.setup_anndata(adata, batch_key="sample_batch", layer="counts")

# Train the scVI model

model = SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb", max_epochs=20)
model.train()

# Get the latent representation
adata.obsm["X_scVI"] = model.get_latent_representation()


# Save the scVI embeddings to a CSV file
scVI_emb = pd.DataFrame(adata.obsm["X_scVI"], index=adata.obs.index.tolist())
scVI_emb.to_csv("scVI_embeddings_20epochs.csv")

# Perform UMAP on the latent representation
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.umap(adata)

# Plot the UMAP
# Plot the UMAP and save the figure
sc.pl.umap(adata, color=["sample_batch", "cell_type"], save="_sample_batch_cell_type_20epochs.png")


sc.pl.umap(adata, color=["_scvi_batch"], save="__scvi_batch_20epochs.png")

# Save the scVI embeddings to a CSV file
scVI_emb_umap = pd.DataFrame(adata.obsm["X_umap"], index=adata.obs.index.tolist())
scVI_emb_umap .to_csv("scVI_umap_20epochs.csv")




cell_id_df_hnoca = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/integration/cell_hnoca.csv',
                                index_col=0)
cell_id_df_hdbca = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/integration/cell_hdbca.csv',
                                index_col=0)


umap_hnoca_df = scVI_emb_umap.loc[hnoca.obs.index.tolist()]
umap_hnoca_df.columns = ['UMAP1','UMAP2']
hnoca.obsm['X_umap'] = umap_hnoca_df.values
sc.pl.umap(hnoca, color=["annot_level_2"], save="_annot_level_2_20epochs_HNOCA.png")
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





sc.pl.umap(hnoca, color=["annot_level_1"], save="_annot_level_1_20epochs_HNOCA.png")


umap_hdbca_df = scVI_emb_umap.loc[hdbca.obs.index.tolist()]
umap_hdbca_df.columns = ['UMAP1','UMAP2']
hdbca.obsm['X_umap'] = umap_hdbca_df.values
sc.pl.umap(hdbca, color=["CellClass"], save="_CellClass_20epochs_HDBCA.png")
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





cell_id_df_hnoca['stage'] = hnoca.obs.loc[cell_id_df_hnoca.index]['organoid_age_days']
cell_id_df_hnoca = cell_id_df_hnoca.loc[umap_hnoca_df.index.tolist()]
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
cell_id_df_hdbca = cell_id_df_hdbca.loc[umap_hdbca_df.index.tolist()]
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
