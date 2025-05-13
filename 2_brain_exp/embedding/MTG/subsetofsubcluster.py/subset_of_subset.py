# (mil) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet$ python
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
from adjustText import adjust_text
import seaborn as sns

from bioPointNet_Apr2025 import *



# Load the data
adata = sc.read_h5ad('/group/gquongrp/workspaces/hongruhu/bioPointNet/Micro3/MTG_sciMultiLaMA.h5ad')
# AnnData object with n_obs × n_vars = 4531 × 14248
adata_raw = sc.read_h5ad('/group/gquongrp/workspaces/hongruhu/bioPointNet/Micro3/MTG_sciMultiLaMA_norm.h5ad')
# AnnData object with n_obs × n_vars = 4531 × 14248



adata.X = adata_raw.X.copy()
# Set the embedding as the basis for clustering
adata.obsm['X_emb'] = adata.obsm['X_sciMultiLaMA']

# Perform Leiden clustering
sc.pp.neighbors(adata, use_rep='X_emb')  # Use the embedding for neighborhood graph construction
sc.tl.leiden(adata)  # Perform Leiden clustering

# Compute UMAP for visualization
sc.tl.umap(adata)

# Plot UMAP with Leiden clusters
sc.pl.umap(adata, color='leiden', title='Leiden Clusters on UMAP', save='_leiden')


adata_tr = sc.read_h5ad('/group/gquongrp/workspaces/hongruhu/bioPointNet/Micro3/emb_pred_sex_cov/Adata_Tr_MTG_4.h5ad')
# AnnData object with n_obs × n_vars = 3495 × 50


adata = adata[adata_tr.obs.index]
adata.obs[['attention_score_norm_cellnum', 'attention_score_norm_cellnum_clip']] = adata_tr.obs[['attention_score_norm_cellnum','attention_score_norm_cellnum_clip']]
adata.var.index = adata.var.feature_name.tolist()


sc.pl.umap(adata, color=['leiden', 'disease','attention_score_norm_cellnum','attention_score_norm_cellnum_clip'], 
           save='_leiden_atten.pdf')



marker_list = ['IL1B', 'CSF1R', 'STAB1', 'NINJ1', 'JAK3',
                'IRF1', 'IRF7', 'IFI16',
                'FCGR1A', 'FCGR1B', 'FCGR2A', 'FCGR3B', # | FCGR1B
                'CD74', 'HLA-DRB5',
                'C1QA', 'C1QB',
                'CSF1R', 'CTSC', 'C1QA', 'C1QB', 'LY86', 'FCGR3A',
                'CTSD', 'CTSS',
                'LYZ',
                'APOE',
                'RUNX1', 'IKZF1', 'NFATC2', 'MAF']
len(marker_list)
# 30

len(set(marker_list)) # 27

intersection_list = list(set(adata.var.index) & set(marker_list))
len(intersection_list)
# 26

# Check if all genes in intersection_list are present in adata.var.index
missing_genes = set(marker_list) -  set(intersection_list)
if missing_genes:
    print(f"Warning: The following genes are missing in adata.var.index: {missing_genes}")





adata.raw = adata  

sc.pl.umap(adata, color=intersection_list, 
           save='_intersection_marker_list.pdf')





task_key = 'disease'


adata_tr_0 = adata[adata.obs[task_key] != adata.obs[task_key][0]] # 'dementia'
adata_tr_1 = adata[adata.obs[task_key] == adata.obs[task_key][0]]
sc.pl.umap(adata_tr_0, color=['attention_score_norm_cellnum_clip'], 
        title = adata_tr_0.obs[task_key][0],
            ncols=1, wspace=0.5, save='_normed_atten_' + adata_tr_0.obs[task_key][0] + '.pdf')
sc.pl.umap(adata_tr_1, color=['attention_score_norm_cellnum_clip'], 
        title = adata_tr_1.obs[task_key][0],
            ncols=1, wspace=0.5, save='_normed_atten_' + adata_tr_1.obs[task_key][0] + '.pdf')



adata_tr_0_top = adata_tr_0[adata_tr_0.obs.sort_values(by='attention_score_norm_cellnum').iloc[-100:].index]
adata_tr_1_top = adata_tr_1[adata_tr_1.obs.sort_values(by='attention_score_norm_cellnum').iloc[-100:].index]




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Same as before - function to compute percentages
def get_supertype_percentage(adata):
    counts = adata.obs['leiden'].value_counts()
    percentages = counts / counts.sum() * 100
    return percentages

# Step 2: Get percentages for each adata
percent_all = get_supertype_percentage(adata)
percent_tr0 = get_supertype_percentage(adata_tr_0)
percent_tr1 = get_supertype_percentage(adata_tr_1)
percent_tr0_top = get_supertype_percentage(adata_tr_0_top)
percent_tr1_top = get_supertype_percentage(adata_tr_1_top)

# Step 3: Combine into a DataFrame
supertype_union = sorted(
    set(percent_all.index) |
    set(percent_tr0.index) |
    set(percent_tr1.index) |
    set(percent_tr0_top.index) |
    set(percent_tr1_top.index)
)

# Align everything
data_combined = pd.DataFrame({
    'All': percent_all.reindex(supertype_union).fillna(0),
    'Train_0': percent_tr0.reindex(supertype_union).fillna(0),
    'Train_1': percent_tr1.reindex(supertype_union).fillna(0),
    'Train_0_Top': percent_tr0_top.reindex(supertype_union).fillna(0),
    'Train_1_Top': percent_tr1_top.reindex(supertype_union).fillna(0),
}).T  # <--- transpose so datasets are rows

# Step 4: Use 'leiden_colors' for the color palette
leiden_colors = adata.uns['leiden_colors']  # Retrieve the leiden colors
color_dict = dict(zip(supertype_union, leiden_colors[:len(supertype_union)]))

# Step 5: Plot stacked bar plot
fig, ax = plt.subplots(figsize=(12, 8))

bottom = pd.Series([0] * len(data_combined), index=data_combined.index)

for supertype in supertype_union:
    ax.bar(
        data_combined.index,
        data_combined[supertype],
        bottom=bottom,
        label=supertype,
        color=color_dict.get(supertype, '#d3d3d3')  # Default to gray if supertype not in color_dict
    )
    bottom += data_combined[supertype]

ax.set_ylabel('Percentage of Cells (%)')
ax.set_xlabel('Dataset')
ax.set_title('Stacked Bar Plot of Cell Type Proportions')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
sns.despine()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig('stacked_celltype_proportions.pdf')








# Find markers for each Leiden cluster in adata
sc.tl.rank_genes_groups(adata, groupby='leiden', method='wilcoxon')
# Plot the top markers for each cluster
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False, save='_leiden_markers.pdf')
# Save the marker genes to a CSV file
marker_genes = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(30)
marker_genes.to_csv('leiden_cluster_markers.csv', index=False)


# Check overlapping between marker_genes and intersection_list
overlapping_genes = set(marker_genes.values.flatten()) & set(intersection_list)
# Overlapping genes: {'APOE', 'CD74', 'C1QA', 'C1QB'}
print(f"Number of overlapping genes: {len(overlapping_genes)}")
print(f"Overlapping genes: {overlapping_genes}")





from gprofiler import GProfiler
from adjustText import adjust_text
# Run enrichment
gp = GProfiler(return_dataframe=True)
results_3 = gp.profile(organism='hsapiens', query=marker_genes['3'].tolist())
results_4 = gp.profile(organism='hsapiens', query=marker_genes['4'].tolist())
results_5 = gp.profile(organism='hsapiens', query=marker_genes['5'].tolist())

# Show top results
print(results_3[['native', 'name', 'p_value', 'source']].head())
print(results_4[['native', 'name', 'p_value', 'source']].head())
print(results_5[['native', 'name', 'p_value', 'source']].head())

results_3.to_csv('result_leiden3.csv')
results_4.to_csv('result_leiden4.csv')
results_5.to_csv('result_leiden5.csv')










adata.obs['high_attention_score'] = 'no'
# adata.obs.loc[adata.obs['attention_score_norm_cellnum'] > 10, 'high_attention_score'] = 'yes'

adata.obs.loc[adata_tr_0_top.obs.index, 'high_attention_score']= 'yes'




# Set custom colors for 'high_attention_score'
adata.uns['high_attention_score_colors'] = ['#808080', '#DC143C']  # Grey for 'no', Crimson for 'yes'

sc.pl.umap(adata, color=['leiden', 'disease', 'attention_score_norm_cellnum', 'attention_score_norm_cellnum_clip',
                         'high_attention_score'], 
           save='_leiden_atten_binary.pdf')

sc.tl.rank_genes_groups(adata, groupby='high_attention_score', method='wilcoxon')
# Plot the top markers for each cluster
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False, save='_high_attention_score_markers.pdf')
# Save the marker genes to a CSV file
marker_genes = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(30)
marker_genes.to_csv('atten_cluster_markers.csv', index=False)





# Draw a volcano plot for 'yes' vs 'no'
import matplotlib.pyplot as plt
from adjustText import adjust_text

ranked_genes = pd.DataFrame({
    'names': adata.uns['rank_genes_groups']['names']['yes'],
    'scores': adata.uns['rank_genes_groups']['scores']['yes'],
    'logfoldchanges': adata.uns['rank_genes_groups']['logfoldchanges']['yes'],
    'pvals': adata.uns['rank_genes_groups']['pvals']['yes']
})

# Plot
plt.figure(figsize=(6, 6))
upregulated = ranked_genes[ranked_genes['logfoldchanges'] > 0]
downregulated = ranked_genes[ranked_genes['logfoldchanges'] <= 0]

plt.scatter(upregulated['logfoldchanges'], -np.log10(upregulated['pvals']), alpha=0.7, color='red', label='Upregulated')
plt.scatter(downregulated['logfoldchanges'], -np.log10(downregulated['pvals']), alpha=0.7, color='blue', label='Downregulated')

plt.xlabel('Log Fold Change')
plt.ylabel('-Log10(P-value)')
plt.title('Volcano Plot for High Attention Score (Right side)')
plt.axhline(-np.log10(0.05), color='red', linestyle='--', label='P-value = 0.05')
plt.xlim(-5, 5)

# Annotate top 30 genes with repelled text
top_30_genes = ranked_genes.sort_values(by='scores', ascending=False).head(50)
texts = []
for _, row in top_30_genes.iterrows():
    texts.append(plt.text(row['logfoldchanges'], -np.log10(row['pvals']), row['names'], fontsize=8))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

plt.legend()
sns.despine()
plt.tight_layout()
plt.savefig('volcano_plot_high_attention_score_with_labels_top30DEG.pdf')

# Save the top 30 genes to a CSV file
print(top_30_genes)
top_30_genes.to_csv('top_30_genes_high_attention_score.csv', index=False)


top_30_genes.names
# [      'FTL', 'RPL28', 'RPS19', 'APOE', 'RPL7A', 'RPL15', 'RPS28',
    #    'EEF1A1', 'DDX5', 'RPL13', 'RPL13A', 'ASAH1', 'B2M', 'SERF2',
    #    'LIPA', 'RPL5', 'RPLP1', 'C1QC', 'PSAP', 'C1QB', 'CST3', 'TRAM1',
    #    'SPP1', 'ADAM9', 'RPS9', 'EIF1', 'NINJ1', 'RPL32', 'GRN', 'RPL11',
    #    'CD63', 'ATP6V1B2', 'HNRNPK', 'MYL6', 'ANKUB1', 'S100A11', 'STOM',
    #    'EEF2', 'KCTD12', 'RNF13', 'RPL8', 'RPS18', 'CTSD', 'TYROBP',
    #    'PIK3IP1', 'ALOX5AP', 'ACTG1', 'NPC2', 'FCER1G', 'GPNMB'],
top_30_genes = top_30_genes[top_30_genes.names.isin(['APOE', 'C1QB', 'C1QC', 'SPP1','SPP1', 'TYROBP', 'CD63',
                                                     'FTL', 'ADAM9', 'PSAP', 'ATP6V1B2', 'FCER1G', 'GPNMB','CST3', 'LIPA', 'TRAM1'])]

# Plot
plt.figure(figsize=(6, 6))
upregulated = ranked_genes[ranked_genes['logfoldchanges'] > 0]
downregulated = ranked_genes[ranked_genes['logfoldchanges'] <= 0]

plt.scatter(upregulated['logfoldchanges'], -np.log10(upregulated['pvals']), alpha=0.7, color='red', label='Upregulated')
plt.scatter(downregulated['logfoldchanges'], -np.log10(downregulated['pvals']), alpha=0.7, color='blue', label='Downregulated')

plt.xlabel('Log Fold Change')
plt.ylabel('-Log10(P-value)')
plt.title('Volcano Plot for High Attention Score (Yes vs No)')
plt.axhline(-np.log10(0.05), color='red', linestyle='--', label='P-value = 0.05')
plt.xlim(-5, 5)

texts = []
for _, row in top_30_genes.iterrows():
    texts.append(plt.text(row['logfoldchanges'], -np.log10(row['pvals']), row['names'], fontsize=8))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

plt.legend()
sns.despine()
plt.tight_layout()
plt.savefig('volcano_plot_high_attention_score_with_labels_.pdf')




results_high = gp.profile(organism='hsapiens', query=marker_genes['yes'].tolist())
print(results_high[['native', 'name', 'p_value', 'source']].head())
results_high
results_high.to_csv('results_high_attn.csv')

overlapping_genes = set(marker_genes.values.flatten()) & set(marker_list)
# {'APOE', 'C1QB'}

# ['FTL', 'RPS19', 'RPLP1', 'RPS28', 'ADAM9', 'EEF1A1', 'PSAP', 'RPL28', 'RAB11A', 'RPL5', 'EIF1', 'RPL13', 'DDX5', 'RPL18', 'RPSA', 'RPL15', 'ANKUB1', 'APOE', 'RPL7A', 'C1QB', 'RPL23', 'GATAD1', 'RPL13A', 'LIPA', 'RPL11', 'TYROBP', 'OAZ1', 'HLA-DRB1', 'CEBPD', 'CTSH']

adata.write('micro3_see4.h5ad')






gene_embedding_df = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/Micro3/MTG_sciMultiLaMA_GENE_embedding.csv', index_col=0)
gene_embedding_df.index = adata.var.index

marker_genes
# ITGAX
sc.pp.highly_variable_genes(adata, flavor='cell_ranger')
hvgs = adata.var[adata.var.highly_variable].index
# 3788
genes = list(set(hvgs).union(set(marker_genes.yes)))
# 3797
gene_embedding_df = gene_embedding_df.loc[genes]

import umap
U = umap.UMAP(random_state=0)
umap_coords = U.fit_transform(gene_embedding_df)
umap_coords = pd.DataFrame(umap_coords, index=gene_embedding_df.index, columns=['UMAP1','UMAP2'])

fig, ax = plt.subplots(figsize=(10, 8))
# Plot all genes (faint)
ax.scatter(umap_coords['UMAP1'], umap_coords['UMAP2'], c='lightgrey', s=10, label='All genes', alpha=0.5)
# Plot marker genes (larger, red)
marker_mask = umap_coords.index.isin(list(set(marker_genes.yes)))
ax.scatter(umap_coords.loc[marker_mask, 'UMAP1'],
           umap_coords.loc[marker_mask, 'UMAP2'],
           c='red', s=50, label='Marker genes')
# Step 3: Add text with repel
texts = []
for gene in list(set(marker_genes.yes)) + ['LPL', 'ITGAX']:
    if gene in umap_coords.index:
        x, y = umap_coords.loc[gene, ['UMAP1', 'UMAP2']]
        texts.append(ax.text(x, y, gene, fontsize=8, color='black'))


from adjustText import adjust_text

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
# Styling
ax.set_title('UMAP of Gene Embeddings with Highlighted Marker Genes')
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig('gene_embedding.pdf')



sc.pl.umap(adata, color=['ITGAX', 'LPL'], 
           save='_intersection_marker_list_.pdf')

# all high


gene_sum = pd.DataFrame(adata.X.sum(0).tolist(), index=['sum_expr'], columns=adata.var_names.tolist())
gene_sum = gene_sum.T

gene_sum = gene_sum.loc[genes]
gene_sum = gene_sum.sort_values('sum_expr')
