

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import sparse
from scipy.io import mmread
from scipy.spatial.distance import pdist, squareform  # Ensure pdist is imported
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import zscore




path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/'
adata = sc.read_h5ad(path + 'MTG_glia_intersection_norm.h5ad')


adata_all = adata.copy()

adata = adata_all.copy()



adata = adata_all.copy()
# Calculate cell type proportions for each subject
# Calculate cell type proportions for each subject
for disease_status in ['normal', 'dementia']:
    adata_subset = adata[adata.obs.disease == disease_status]
    # Subset the first 15 samples per condition
    first_15_subjects = adata_subset.obs['donor_id'].unique()[:15]
    adata_subset = adata_subset[adata_subset.obs['donor_id'].isin(first_15_subjects)]
    cell_type_proportions = (
        adata_subset.obs.groupby(['donor_id', 'Subclass'])
        .size()
        .unstack(fill_value=0)
        .apply(lambda x: x / x.sum(), axis=1)
    )
    cell_type_proportions = cell_type_proportions.sort_values(by='Oligodendrocyte', ascending=False)
    # Plot the sorted stacked barplot as a horizontal bar chart
    plt.figure(figsize=(10, 10))
    cell_type_proportions.plot(
        kind='barh',
        stacked=True,
        colormap='tab20',
        figsize=(10, 10),
        legend=False,
        width=0.8  # Adjust the width to create white space between bars
    )
    plt.legend(
        cell_type_proportions.columns,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        title='Cell Types'
    )
    plt.ylabel('donor_id')
    plt.xlabel('Proportion')
    plt.title(f'Cell Type Proportions per Subject (Sorted by Oli Proportion) - AD: {disease_status}')
    output_path = f'cell_type_proportions_per_subject_sorted_horizontal_SEAAD_disease_{disease_status}_first15.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()




sc.pp.highly_variable_genes(adata, flavor='cell_ranger')
hvg = adata.var[adata.var['highly_variable']].index
adata = adata[:, hvg].copy()

# Normalize the data
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)




# Calculate cell type proportions for each subject
for disease_status in ['yes', 'no']:
    adata_subset = adata[adata.obs.AD == disease_status]
    # Subset the first 15 samples per condition
    first_15_subjects = adata_subset.obs['subject'].unique()[:15]
    adata_subset = adata_subset[adata_subset.obs['subject'].isin(first_15_subjects)]
    subject_gene_matrix = (
        adata_subset.to_df().groupby(adata_subset.obs['subject']).mean()
    )
    subject_gene_matrix_zscore = subject_gene_matrix.apply(zscore, axis=0)
    sns.heatmap(
        subject_gene_matrix_zscore,
        cmap='plasma',
        cbar=True,
        xticklabels=True,
        yticklabels=True,
    )  
    heatmap_output_path = f'subject_gene_heatmap_minmax_disease_{disease_status}.png'
    plt.title(f'Subject-Gene Heatmap (Min-Max Normalized) - AD: {disease_status}')
    plt.tight_layout()
    plt.savefig(heatmap_output_path, dpi=150)
    plt.close()


# Calculate cell type proportions for each subject
for disease_status in ['yes', 'no']:
    adata_subset = adata[adata.obs.AD == disease_status]
    # Subset the first 15 samples per condition
    first_15_subjects = adata_subset.obs['subject'].unique()[:15]
    adata_subset = adata_subset[adata_subset.obs['subject'].isin(first_15_subjects)]
    cell_type_proportions = (
        adata_subset.obs.groupby(['subject', 'cell_type'])
        .size()
        .unstack(fill_value=0)
        .apply(lambda x: x / x.sum(), axis=1)
    )
    # Sort subjects based on the proportion of a specific cell type (e.g., 'Oli')
    cell_type_proportions = cell_type_proportions.sort_values(by='Oli', ascending=False)
    # Plot the sorted stacked barplot as a horizontal bar chart
    plt.figure(figsize=(10, 10))
    cell_type_proportions.plot(
        kind='barh',
        stacked=True,
        colormap='tab20',
        figsize=(10, 10),
        legend=False,
        width=0.8  # Adjust the width to create white space between bars
    )
    plt.legend(
        cell_type_proportions.columns,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        title='Cell Types'
    )
    plt.ylabel('Subject')
    plt.xlabel('Proportion')
    plt.title(f'Cell Type Proportions per Subject (Sorted by Oli Proportion) - AD: {disease_status}')
    output_path = f'cell_type_proportions_per_subject_sorted_horizontal_disease_{disease_status}_first15.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
