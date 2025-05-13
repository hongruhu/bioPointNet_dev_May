# (mil) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet$ python
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

path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/'


counts = mmread(path + "ROSMAP_glia/raw_count.mtx")
genes = pd.read_table(path + "ROSMAP_glia/genes.txt", header=None)
cells = pd.read_table(path + "ROSMAP_glia/barcodes.txt", header=None)
metadata = pd.read_csv(path + "ROSMAP_glia/ROSMAP_meta.csv", index_col=0, sep=",")


adata = sc.AnnData(X=counts, obs=metadata)
adata.var = pd.DataFrame(index = genes.iloc[:,0].tolist())
adata.var['gene_id'] = genes.iloc[:,0].tolist()
adata
# AnnData object with n_obs × n_vars = 969091 × 33538
#     obs: 'cell_type_high_resolution', 'subject'
#     var: 'gene_id'


adata.obs
#                        cell_type_high_resolution       subject
# CACATTTCAAGTCTAC-1-0                    Ast GRM3  ROSMAP-52226
# GAAGCAGCAGCTCCGA-1-0                  Ast CHI3L1  ROSMAP-52226
# GCGCGATCAGCCTGTG-1-0                  Ast CHI3L1  ROSMAP-52226
# TCACGAAGTGCAGGTA-1-0                    Ast GRM3  ROSMAP-52226
# TTTCCTCCAATGACCT-1-0                  Ast CHI3L1  ROSMAP-52226
# ...                                          ...           ...
# TTACCATTCGAACTCA-16-14                       OPC  ROSMAP-37074
# TTAGGGTCACACAGAG-16-14                       OPC  ROSMAP-37074
# TTCCTAAAGAGGCGGA-16-14                       OPC  ROSMAP-37074
# TTCTTGAGTCGTTGGC-16-14                       OPC  ROSMAP-37074
# TTGTGTTTCCTCAGAA-16-14                       OPC  ROSMAP-37074


adata.obs.cell_type_high_resolution.value_counts()
# cell_type_high_resolution
# Oli           645142
# Ast GRM3       92527
# OPC            90502
# Mic P2RY12     73061
# Ast DPP10      32779
# Ast CHI3L1     24252
# Mic TPT1        5261
# T cells         2534
# CAMs            2167
# Mic MKI67        866

adata.obs['cell_type'] = [i[0] for i in adata.obs.cell_type_high_resolution.str.split(' ')]
adata.obs.cell_type.value_counts()
# cell_type
# Oli     645142
# Ast     149558
# OPC      90502
# Mic      79188
# T         2534
# CAMs      2167


adata.obs.subject.value_counts()
# Name: count, Length: 427, dtype: int64


# individual meta
ind_meta = pd.read_csv(path + 'individual_metadata_deidentified.tsv', sep='\t', index_col=0)
ind_meta
#               msex age_death   pmi  race Pathologic_diagnosis_of_AD
# subject
# ROSMAP-10132     0       90+   7.0     1                        yes
# ROSMAP-10643     0       90+  13.0     1                        yes
# ROSMAP-10859     1   (80,85]  14.0     2                        yes
# ROSMAP-12078     1   (85,90]   3.0     1                        yes
# ROSMAP-12256     1   (85,90]   4.0     1                        yes
# ...            ...       ...   ...   ...                        ...
# ROSMAP-98582     1   (70,75]   8.0     1                         no
# ROSMAP-98683     0   (85,90]   7.0     1                        yes
# ROSMAP-99419     0   (75,80]   8.0     1                         no
# ROSMAP-99585     1       90+   2.0     1                        yes
# ROSMAP-99981     1       90+   5.0     1                        yes

# Ensure subject is the index in ind_meta, or reset it if it's the index
if ind_meta.index.name == 'subject':
    ind_meta = ind_meta.reset_index()

# Merge metadata into adata.obs based on subject ID
adata.obs = adata.obs.merge(
    ind_meta[['subject', 'msex', 'age_death', 'pmi', 'race', 'Pathologic_diagnosis_of_AD']],
    on='subject',
    how='left'
)

# Optionally rename the column for clarity
adata.obs.rename(columns={'Pathologic_diagnosis_of_AD': 'AD'}, inplace=True)




adata.obs.AD.value_counts()
# AD
# yes    525491
# no     443600


adata.obs.subject.value_counts()
# subject
# ROSMAP-41333    10252
# ROSMAP-87836     8190
# ROSMAP-82353     6918
# ROSMAP-53808     6658
# ROSMAP-74690     6623
#                 ...
# ROSMAP-72912      148
# ROSMAP-37415      134
# ROSMAP-15389      134
# ROSMAP-52226      115
# ROSMAP-50976       61



adata.X = adata.X.tocsr()

# Count number of cells per subject
cell_counts = adata.obs['subject'].value_counts()
# Get subjects with at least 1000 cells
valid_subjects = cell_counts[cell_counts >= 1000].index
# Subset the AnnData object
adata = adata[adata.obs['subject'].isin(valid_subjects)].copy()

adata.obs.subject.value_counts()
# Name: count, Length: 349, dtype: int64

adata.obs.AD.value_counts()
# AD
# yes    505578
# no     425161

adata.obs.msex.value_counts()
# msex
# 0    490461
# 1    440278

adata.obs.pmi.value_counts()
# pmi
# 6.0     158803
# 5.0     155112
# 4.0     140625
# 7.0     112749
# 8.0     103132
# 3.0      47564
# 9.0      38947
# 10.0     29722
# 11.0     18281
# 12.0     17840
# 2.0      16636
# 14.0     12912
# 20.0     11534
# 16.0     10238
# 13.0      9673
# 17.0      9038
# 23.0      6560
# 15.0      5261
# 1.0       5110
# 18.0      5107
# 21.0      4897
# 22.0      4120
# 30.0      2485
# 41.0      1685
# 19.0      1332

adata.obs.age_death.value_counts()
# age_death
# 90+        332833
# (85,90]    302136
# (80,85]    181798
# (75,80]     82397
# (70,75]     31575

adata.obs.race.value_counts()
# race
# 1    928789
# 2      1950


adata
# AnnData object with n_obs × n_vars = 930739 × 33538
#     obs: 'cell_type_high_resolution', 'subject', 'cell_type', 'msex', 'age_death', 'pmi', 'race', 'AD'
#     var: 'gene_id'




sc.pp.filter_genes(adata, min_cells=10)
adata
# AnnData object with n_obs × n_vars = 930739 × 33538 --> 
# AnnData object with n_obs × n_vars = 930739 × 29268

sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat_v3')

adata.write_h5ad(path + 'ROSMAP_glia_processed.h5ad')

adata.write_h5ad(path + 'ROSMAP_glia_processed_.h5ad', compression='gzip')









# Load the processed AnnData object
adata = sc.read_h5ad('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/ROSMAP_glia_processed_.h5ad')

# Calculate cell type proportions for each subject
cell_type_proportions = (
    adata.obs.groupby(['subject', 'cell_type_high_resolution'])
    .size()
    .unstack(fill_value=0)
    .apply(lambda x: x / x.sum(), axis=1)
)

# Plot the stacked barplot
plt.figure(figsize=(20, 10))
cell_type_proportions.plot(
    kind='bar',
    stacked=True,
    colormap='tab20',
    figsize=(20, 10),
    legend=False
)

# Add legend outside the plot
plt.legend(
    cell_type_proportions.columns,
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    title='Cell Types'
)

# Add labels and title
plt.xlabel('Subject')
plt.ylabel('Proportion')
plt.title('Cell Type Proportions per Subject')

# Save the plot
output_path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/cell_type_proportions_per_subject.png'
plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.close()









# Sort subjects based on similarity in cell type proportions

# Compute pairwise distances between subjects based on cell type proportions
distance_matrix = pdist(cell_type_proportions, metric='euclidean')

# Perform hierarchical clustering
linkage_matrix = linkage(distance_matrix, method='ward')

# Get the order of subjects based on the clustering
ordered_subjects = cell_type_proportions.index[leaves_list(linkage_matrix)]

# Reorder the cell type proportions dataframe
cell_type_proportions = cell_type_proportions.loc[ordered_subjects]
# Plot the reordered stacked barplot as a horizontal bar chart
plt.figure(figsize=(10, 20))
cell_type_proportions.plot(
    kind='barh',
    stacked=True,
    colormap='tab20',
    figsize=(10, 20),
    legend=False,
    width=1.0  # Set the bar width to 1.0 to minimize gaps
)

# Add legend outside the plot
plt.legend(
    cell_type_proportions.columns,
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    title='Cell Types'
)

# Add labels and title
plt.ylabel('Subject')
plt.xlabel('Proportion')
plt.title('Cell Type Proportions per Subject (Ordered by Similarity)')

# Save the plot
output_path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/cell_type_proportions_per_subject_ordered_horizontal.png'
plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.close()















hvg = adata.var[adata.var['highly_variable']].index
adata = adata[:, hvg].copy()

# Normalize the data
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Pseudobulk by averaging normalized gene profiles for each subject
subject_gene_matrix = (
    adata.to_df()
    .groupby(adata.obs['subject'])
    .mean()
)

# Save the resulting matrix to a CSV file
output_path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/subject_x_gene_matrix.csv'
# Save the resulting matrix to a CSV file
subject_gene_matrix.to_csv(output_path)



subject_gene_matrix = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/subject_x_gene_matrix.csv', index_col=0)

# Normalize the subject_gene_matrix using min-max scaling
subject_gene_matrix_minmax = (subject_gene_matrix - subject_gene_matrix.min(axis=0)) / (subject_gene_matrix.max(axis=0) - subject_gene_matrix.min(axis=0))

# Plot a clustermap using Ward's method (ward2 linkage) with min-max scaled data
sns.clustermap(
    subject_gene_matrix_minmax,
    method='ward',
    metric='euclidean',
    cmap='plasma',
    figsize=(20, 15)
)

# Save the clustermap
clustermap_output_path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/subject_gene_clustermap_minmax.png'
plt.savefig(clustermap_output_path, dpi=150)
plt.close()








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

path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/'




# Load the processed AnnData object
adata = sc.read_h5ad('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/ROSMAP_glia_processed_.h5ad')

adata_all = adata.copy()

adata = adata_all[adata_all.obs.AD=='yes']
# Calculate cell type proportions for each subject
cell_type_proportions = (
    adata.obs.groupby(['subject', 'cell_type_high_resolution'])
    .size()
    .unstack(fill_value=0)
    .apply(lambda x: x / x.sum(), axis=1)
)
# Sort subjects based on similarity in cell type proportions
# Compute pairwise distances between subjects based on cell type proportions
distance_matrix = pdist(cell_type_proportions, metric='euclidean')
# Perform hierarchical clustering
linkage_matrix = linkage(distance_matrix, method='ward')
# Get the order of subjects based on the clustering
ordered_subjects = cell_type_proportions.index[leaves_list(linkage_matrix)]
# Reorder the cell type proportions dataframe
cell_type_proportions = cell_type_proportions.loc[ordered_subjects]
# Plot the reordered stacked barplot as a horizontal bar chart
plt.figure(figsize=(10, 20))
cell_type_proportions.plot(
    kind='barh',
    stacked=True,
    colormap='tab20',
    figsize=(10, 20),
    legend=False,
    width=1.0  # Set the bar width to 1.0 to minimize gaps
)
# Add legend outside the plot
plt.legend(
    cell_type_proportions.columns,
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    title='Cell Types'
)
# Add labels and title
plt.ylabel('Subject')
plt.xlabel('Proportion')
plt.title('Cell Type Proportions per Subject (Ordered by Similarity)')
# Save the plot
output_path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/cell_type_proportions_per_subject_ordered_horizontal_AD.png'
plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.close()




adata = adata_all[adata_all.obs.AD=='no']
# Calculate cell type proportions for each subject
cell_type_proportions = (
    adata.obs.groupby(['subject', 'cell_type_high_resolution'])
    .size()
    .unstack(fill_value=0)
    .apply(lambda x: x / x.sum(), axis=1)
)
# Sort subjects based on similarity in cell type proportions
# Compute pairwise distances between subjects based on cell type proportions
distance_matrix = pdist(cell_type_proportions, metric='euclidean')
# Perform hierarchical clustering
linkage_matrix = linkage(distance_matrix, method='ward')
# Get the order of subjects based on the clustering
ordered_subjects = cell_type_proportions.index[leaves_list(linkage_matrix)]
# Reorder the cell type proportions dataframe
cell_type_proportions = cell_type_proportions.loc[ordered_subjects]
# Plot the reordered stacked barplot as a horizontal bar chart
plt.figure(figsize=(10, 20))
cell_type_proportions.plot(
    kind='barh',
    stacked=True,
    colormap='tab20',
    figsize=(10, 20),
    legend=False,
    width=1.0  # Set the bar width to 1.0 to minimize gaps
)
# Add legend outside the plot
plt.legend(
    cell_type_proportions.columns,
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    title='Cell Types'
)
# Add labels and title
plt.ylabel('Subject')
plt.xlabel('Proportion')
plt.title('Cell Type Proportions per Subject (Ordered by Similarity)')
# Save the plot
output_path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/cell_type_proportions_per_subject_ordered_horizontal_CTRL.png'
plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.close()








adata = adata_all.copy()
# Calculate cell type proportions for each subject
for disease_status in ['yes', 'no']:
    adata_subset = adata[adata.obs.AD == disease_status]
    cell_type_proportions = (
        adata_subset.obs.groupby(['subject', 'cell_type_high_resolution'])
        .size()
        .unstack(fill_value=0)
        .apply(lambda x: x / x.sum(), axis=1)
    )
    # Sort subjects based on the proportion of a specific cell type (e.g., 'Oli')
    cell_type_proportions = cell_type_proportions.sort_values(by='Oli', ascending=False)
    # Plot the sorted stacked barplot as a horizontal bar chart
    plt.figure(figsize=(10, 20))
    cell_type_proportions.plot(
        kind='barh',
        stacked=True,
        colormap='tab20',
        figsize=(10, 20),
        legend=False,
        width=1.0
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
    output_path = f'/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/cell_type_proportions_per_subject_sorted_horizontal_AD_{disease_status}.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
