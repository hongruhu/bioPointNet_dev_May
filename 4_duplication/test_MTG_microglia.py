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
split_list = [0,2,3]
celltype_key = 'Microglia-PVM'
sample_key = 'donor_id'
task_key = 'disease'
cov_key = 'sex'
class_num = 2 

cell_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/SEAAD_Glia/MTG/Microglia-PVM_MTG_sciMultiLaMA_CELL_embedding.csv', index_col=0)
gene_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/SEAAD_Glia/MTG/Microglia-PVM_MTG_sciMultiLaMA_GENE_embedding.csv', index_col=0)
# [35887 rows x 50 columns] [15507 rows x 50 columns]


cell_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/SEAAD_Glia/MTG/MTG_sciMultiLaMA_CELL_embedding.csv', index_col=0)
gene_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/SEAAD_Glia/MTG/MTG_sciMultiLaMA_GENE_embedding.csv', index_col=0)
# [226669 rows x 50 columns] [15909 rows x 50 columns]



# Run UMAP on gene embeddings
reducer = umap.UMAP(random_state=42)
gene_umap = reducer.fit_transform(gene_emb.values)
umap_df = pd.DataFrame(gene_umap, index=gene_emb.index, columns=["UMAP1", "UMAP2"])




obj_all = sc.read_h5ad('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/MTG_glia_intersection_norm.h5ad')



marker_list = ['IL1B', 'CSF1R', 'STAB1', 'NINJ1', 'JAK3',
'IRF1', 'IRF7', 'IFI16',
'FCGR1A', 'FCGR1B', 'FCGR2A', 'FCGR3B',
'CD74', 'HLA-DRB5',
'C1QA', 'C1QB',
'CSF1R', 'CTSC', 'C1QA', 'C1QB', 'LY86', 'FCGR3A'
'CTSD', 'CTSS',
'LYZ',
'APOE',
'RUNX1', 'IKZF1', 'NFATC2', 'MAF']
len(marker_list)
# 29



marker_list = [ 'IL1B', 'CSF1R', 'STAB1', 'NINJ1', 'JAK3',  # inflammatory process
                'IRF1', 'IRF7', 'IFI16',                    # interferon response
                'FCGR1A', 'FCGR2A', 'FCGR3B',     # Fc receptors | FCGR1B is not in the data
                'CD74', 'HLA-DRB5',                         # MHC
                'C1QA', 'C1QB']                             # complement components
len(marker_list) # 16 - 1 = 15

marker_df = obj_all.var[obj_all.var.feature_name.isin(marker_list)]
marker_id_list = marker_df.index.tolist()
marker_name_list = marker_df.feature_name.tolist()
gene_emb_dist = pd.DataFrame(np.sqrt(np.sum(gene_emb**2, 1)), index=gene_emb.index, columns=['dist'])
gene_emb_dist.sort_values('dist')
gene_emb_dist.mean() # dist    1.327988
gene_emb_dist.loc[marker_id_list].sort_values('dist')


# Step 2: Plot
fig, ax = plt.subplots(figsize=(10, 8))
# Plot all genes (faint)
ax.scatter(umap_df['UMAP1'], umap_df['UMAP2'], c='lightgrey', s=10, label='All genes', alpha=0.5)
# Plot marker genes (larger, red)
marker_mask = umap_df.index.isin(marker_id_list)
ax.scatter(umap_df.loc[marker_mask, 'UMAP1'],
           umap_df.loc[marker_mask, 'UMAP2'],
           c='red', s=50, label='Marker genes')
# Step 3: Add text with repel
texts = []
for gene in marker_id_list:
    if gene in umap_df.index:
        x, y = umap_df.loc[gene, ['UMAP1', 'UMAP2']]
        texts.append(ax.text(x, y, marker_df.loc[gene].feature_name, fontsize=8, color='black'))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
# Styling
ax.set_title('UMAP of Gene Embeddings with Highlighted Marker Genes')
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig('gene_embedding.png')





from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
cell_emb_ct = cell_emb[obj_all.obs.Subclass=='Microglia-PVM'].copy()
gene_emb_marker = gene_emb.loc[marker_id_list].copy()
cg_eucdis = pd.DataFrame(euclidean_distances(cell_emb_ct, gene_emb_marker), index=cell_emb_ct.index, columns=marker_id_list)
cg_cossim = pd.DataFrame(cosine_similarity(cell_emb_ct, gene_emb_marker), index=cell_emb_ct.index, columns=marker_id_list)

cell_list_eucdis = []
for id in marker_id_list:
    cell_list_eucdis.append(cg_eucdis.sort_values(id)[id][0:10].index.tolist())


cell_set_eucdis = list(set([cell for sublist in cell_list_eucdis for cell in sublist]))
len(cell_set_eucdis) 

cell_list_cossim = []
for id in marker_id_list:
    cell_list_cossim.append(cg_cossim.sort_values(id)[id][-10:].index.tolist())


cell_set_cossim = list(set([cell for sublist in cell_list_cossim for cell in sublist]))
len(cell_set_cossim) 

len(set(cell_set_eucdis).intersection(set(cell_set_cossim))) # 3

obj_all[cell_set_eucdis].obs.Supertype.value_counts()
# Supertype
# Micro-PVM_2            31
# Micro-PVM_2_3-SEAAD    12
# Micro-PVM_4-SEAAD       9
# Micro-PVM_3-SEAAD       9
# Micro-PVM_1             4
# Micro-PVM_2_2-SEAAD     1

obj_all[cell_set_cossim].obs.Supertype.value_counts()
# Supertype
# Micro-PVM_2            30
# Micro-PVM_2_3-SEAAD    28
# Micro-PVM_3-SEAAD      19
# Micro-PVM_2_2-SEAAD    18
# Micro-PVM_1            13
# Micro-PVM_1_1-SEAAD     3
# Micro-PVM_2_1-SEAAD     2
# Micro-PVM_4-SEAAD       2

obj_all[cell_set_eucdis].obs.Supertype.value_counts()/obj_all[cell_set_eucdis].shape[0]
# Supertype
# Micro-PVM_2            0.603774
# Micro-PVM_2_3-SEAAD    0.202830
# Micro-PVM_3-SEAAD      0.165094
# Micro-PVM_1            0.018868
# Micro-PVM_4-SEAAD      0.004717
# Micro-PVM_2_2-SEAAD    0.004717
obj_all[cell_set_cossim].obs.Supertype.value_counts()/obj_all[cell_set_cossim].shape[0]
# Supertype
# Micro-PVM_2            0.537037
# Micro-PVM_2_3-SEAAD    0.208333
# Micro-PVM_3-SEAAD      0.129630
# Micro-PVM_1            0.078704
# Micro-PVM_2_2-SEAAD    0.046296

obj_all[cell_emb_ct.index].obs.Supertype.value_counts()
# Supertype
# Micro-PVM_2            20118
# Micro-PVM_2_3-SEAAD     8200
# Micro-PVM_3-SEAAD       4531
# Micro-PVM_1             1436
# Micro-PVM_2_2-SEAAD      757
# Micro-PVM_4-SEAAD        353
# Micro-PVM_2_1-SEAAD      259
# Micro-PVM_1_1-SEAAD      233
obj_all[cell_emb_ct.index].obs.Supertype.value_counts()/obj_all[cell_emb_ct.index].shape[0]
# Supertype
# Micro-PVM_2            0.560593
# Micro-PVM_2_3-SEAAD    0.228495
# Micro-PVM_3-SEAAD      0.126257
# Micro-PVM_1            0.040014
# Micro-PVM_2_2-SEAAD    0.021094
# Micro-PVM_4-SEAAD      0.009836
# Micro-PVM_2_1-SEAAD    0.007217
# Micro-PVM_1_1-SEAAD    0.006493






for split_id in split_list:
    print(split_id)




split_id = 0



enc = torch.load(loading_path + celltype_key + '/' + 'ENCODER_' + celltype_key + '_' + str(split_id)+ '.pt')
cls = torch.load(loading_path + celltype_key + '/' + 'CLASSIFIER_' + celltype_key + '_' + str(split_id)+ '.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformation_function = nn.Softmax(dim=1)



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




sc.pl.umap(adata_all, color=['attention_score_norm_cellnum_clip'], 
            ncols=1, wspace=0.5, save='_normed_atten_.png')


sc.pl.umap(adata_all, color=['Supertype'], 
            ncols=1, wspace=0.5, save='_supertype.pdf')

adata_tr_0 = adata_all[adata_all.obs[task_key] != adata_all.obs[task_key][0]] # 'dementia'
adata_tr_1 = adata_all[adata_all.obs[task_key] == adata_all.obs[task_key][0]]
sc.pl.umap(adata_tr_0, color=['attention_score_norm_cellnum_clip'], 
        title = adata_tr_0.obs[task_key][0],
            ncols=1, wspace=0.5, save='_normed_atten_' + adata_tr_0.obs[task_key][0] + '.pdf')
sc.pl.umap(adata_tr_1, color=['attention_score_norm_cellnum_clip'], 
        title = adata_tr_1.obs[task_key][0],
            ncols=1, wspace=0.5, save='_normed_atten_' + adata_tr_1.obs[task_key][0] + '.pdf')


# 
adata_tr_0.obs.sort_values(by='attention_score_norm_cellnum').iloc[-30:].Supertype
# GTGCGTGGTCGATTAC-L8TX_210325_01_A09-1153814172      Micro-PVM_3-SEAAD
# GGAGGTAAGGTAAGTT-L8TX_210318_01_H04-1142430390    Micro-PVM_2_3-SEAAD
# AGATCGTGTGCCTAAT-L8TX_210722_01_F08-1153814334      Micro-PVM_3-SEAAD
# TTGGATGGTACATACC-L8TX_210429_01_C03-1142430411    Micro-PVM_2_3-SEAAD
# GATGGAGAGGTTGGTG-L8TX_210513_01_B10-1142430437            Micro-PVM_1
# AAATGGACATGTGGCC-L8TX_210325_01_H08-1142430397            Micro-PVM_2
# ACACTGAAGTAGAATC-L8TX_210325_01_A09-1153814172      Micro-PVM_3-SEAAD
# GGTGTTAAGAGTTCGG-L8TX_210513_01_A10-1153814239      Micro-PVM_3-SEAAD
# AGTGTTGTCATGGATC-L8TX_210701_01_H06-1153814265    Micro-PVM_2_3-SEAAD
# GAAGCCCCAGGGCTTC-L8TX_210708_01_C09-1153814271      Micro-PVM_3-SEAAD
# AGCTACATCCGTGTGG-L8TX_210722_01_C07-1142430451            Micro-PVM_1
# AAGCATCTCATGCCAA-L8TX_210513_01_B11-1142430447      Micro-PVM_3-SEAAD
# TGTGGCGAGTTCCGTA-L8TX_210513_01_B10-1142430437    Micro-PVM_2_3-SEAAD
# AGGTCATAGCGTCTCG-L8TX_210325_01_E07-1153814170            Micro-PVM_2
# TCATCATAGAGATGCC-L8TX_210722_01_E08-1153814332      Micro-PVM_3-SEAAD
# AGAGCCCTCGGAGCAA-L8TX_210722_01_E08-1153814332      Micro-PVM_3-SEAAD
# AGGCTGCAGAGACAAG-L8TX_210701_01_H06-1153814265    Micro-PVM_2_3-SEAAD
# CTATCTATCGCCTTTG-L8TX_210701_01_E07-1153814263    Micro-PVM_2_3-SEAAD
# TCCTAATGTCAGTCCG-L8TX_210812_01_D10-1153814354      Micro-PVM_3-SEAAD
# GGGCTCAAGCATTGTC-L8TX_210513_01_B10-1142430437    Micro-PVM_2_3-SEAAD
# TGAGCGCTCGCTCTAC-L8TX_210513_01_B10-1142430437      Micro-PVM_3-SEAAD
# CGCATGGCAAATGGAT-L8TX_210430_01_A05-1153814213            Micro-PVM_2
# CATTGAGAGGGACTGT-L8TX_210429_01_C03-1142430411      Micro-PVM_3-SEAAD
# GTTGTCCGTCTCAAGT-L8TX_210513_01_E10-1153814251    Micro-PVM_2_3-SEAAD
# GTCAGCGAGGATCATA-L8TX_210701_01_H06-1153814265    Micro-PVM_2_3-SEAAD
# AGGATAACACACCAGC-L8TX_210305_01_G02-1153814153      Micro-PVM_3-SEAAD
# GAGCTGCTCCGTTGAA-L8TX_210513_01_B10-1142430437            Micro-PVM_1
# GAGGCAACACGTTGGC-L8TX_210701_01_E07-1153814263    Micro-PVM_2_3-SEAAD
# AAATGGAGTACCTGTA-L8TX_210701_01_H06-1153814265            Micro-PVM_1
# TACTTGTCACGTAGTT-L8TX_210513_01_B10-1142430437      Micro-PVM_3-SEAAD
# AD
adata_tr_1.obs.sort_values(by='attention_score_norm_cellnum').iloc[-30:].Supertype # 16/30


adata_tr_0_top = adata_tr_0[adata_tr_0.obs.sort_values(by='attention_score_norm_cellnum').iloc[-100:].index]
adata_tr_1_top = adata_tr_1[adata_tr_1.obs.sort_values(by='attention_score_norm_cellnum').iloc[-100:].index]

import matplotlib.pyplot as plt

# Step 1: Get percentages for each adata (same as before)
def get_supertype_percentage(adata, target_supertype):
    counts = adata.obs['Supertype'].value_counts(normalize=True) * 100
    return counts.get(target_supertype, 0)  # return 0 if not found

# Calculate
target_supertype = 'Micro-PVM_3-SEAAD'

percentages = {
    'All': get_supertype_percentage(adata_all, target_supertype),
    'Train_0': get_supertype_percentage(adata_tr_0, target_supertype),
    'Train_1': get_supertype_percentage(adata_tr_1, target_supertype),
    'Train_0_Top': get_supertype_percentage(adata_tr_0_top, target_supertype),
    'Train_1_Top': get_supertype_percentage(adata_tr_1_top, target_supertype),
}

# Step 2: Plot
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(percentages.keys(), percentages.values(), color='skyblue')

# Add percentage text on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,  # x position: center of the bar
        height + 0.5,  # y position: slightly above the bar
        f'{height:.1f}%',  # formatted percentage
        ha='center', va='bottom', fontsize=9
    )

ax.set_ylabel('Percentage of Cells (%)')
ax.set_xlabel('Dataset')
ax.set_title(f'Proportion of {target_supertype}')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, max(percentages.values()) * 1.2)  # Add a little headroom
plt.tight_layout()

plt.savefig('Micro_PVM3_SEAAD_percentage_annotated.pdf')
plt.show()

print("Saved annotated plot as 'Micro_PVM3_SEAAD_percentage_annotated.pdf'.")



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Extract proportions
def get_supertype_percentage(adata):
    counts = adata.obs['Supertype'].value_counts()
    percentages = counts / counts.sum() * 100
    return percentages

# Get percentages
percent_all = get_supertype_percentage(adata_all)
percent_tr0 = get_supertype_percentage(adata_tr_0)
percent_tr1 = get_supertype_percentage(adata_tr_1)
percent_tr0_top = get_supertype_percentage(adata_tr_0_top)
percent_tr1_top = get_supertype_percentage(adata_tr_1_top)

# Step 2: Union of all supertypes
supertype_union = sorted(
    set(percent_all.index) |
    set(percent_tr0.index) |
    set(percent_tr1.index) |
    set(percent_tr0_top.index) |
    set(percent_tr1_top.index)
)

# Step 3: Combine into a DataFrame
data_combined = pd.DataFrame({
    'All': percent_all.reindex(supertype_union).fillna(0),
    'Train_0': percent_tr0.reindex(supertype_union).fillna(0),
    'Train_1': percent_tr1.reindex(supertype_union).fillna(0),
    'Train_0_Top': percent_tr0_top.reindex(supertype_union).fillna(0),
    'Train_1_Top': percent_tr1_top.reindex(supertype_union).fillna(0),
})

# Step 4: Assign consistent colors
palette = sns.color_palette("tab20", len(supertype_union))
color_dict = dict(zip(supertype_union, palette))

# Step 5: Plot
fig, axes = plt.subplots(1, 5, figsize=(25, 6), sharey=True)

datasets = ['All', 'Train_0', 'Train_1', 'Train_0_Top', 'Train_1_Top']
titles = ['All Cells', 'Training Set 0', 'Training Set 1', 'Top Train 0', 'Top Train 1']

for i, (dataset, title) in enumerate(zip(datasets, titles)):
    subset = data_combined[dataset]
    axes[i].bar(subset.index, subset.values, color=[color_dict[stype] for stype in subset.index])
    axes[i].set_title(title)
    axes[i].set_xticklabels(subset.index, rotation=90)
    axes[i].set_ylabel('Percentage of Cells (%)')
    axes[i].grid(axis='y')

plt.tight_layout()
plt.savefig('cell_type_proportions_comparison_5sets.pdf')
plt.show()

print("Saved figure as 'cell_type_proportions_comparison_5sets.pdf'.")




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Same as before - function to compute percentages
def get_supertype_percentage(adata):
    counts = adata.obs['Supertype'].value_counts()
    percentages = counts / counts.sum() * 100
    return percentages

# Step 2: Get percentages for each adata
percent_all = get_supertype_percentage(adata_all)
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

# Step 4: Set colors
palette = sns.color_palette("tab20", len(supertype_union))
color_dict = dict(zip(supertype_union, palette))

# Step 5: Plot stacked bar plot
fig, ax = plt.subplots(figsize=(12, 8))

bottom = pd.Series([0] * len(data_combined), index=data_combined.index)

for supertype in supertype_union:
    ax.bar(
        data_combined.index,
        data_combined[supertype],
        bottom=bottom,
        label=supertype,
        color=color_dict[supertype]
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
plt.show()

print("Saved stacked bar plot as 'stacked_celltype_proportions.pdf'.")











# perturb_sample is 'H20.33.034'
selected_patient = adata_all[adata_all.obs[sample_key] == 'H20.33.034']
sc.pl.umap(selected_patient, color=['attention_score_norm_cellnum_clip'], 
            ncols=1, wspace=0.5, save='_normed_atten_sample.png')
sc.pl.umap(selected_patient, color=['Supertype'], 
            ncols=1, wspace=0.5, save='_supertype_sample.pdf')



selected_patient = adata_all[adata_all.obs[sample_key] == sample]
sc.pl.umap(selected_patient, color=['attention_score_norm_cellnum'], 
            ncols=1, wspace=0.5, save='_normed_atten_sample.pdf')
sc.pl.umap(selected_patient, color=['Supertype'], 
            ncols=1, wspace=0.5, save='_supertype_sample.pdf')












Ns = adata_all.X.copy()
Ns_df = pd.DataFrame(Ns, index=adata_all.obs.index.tolist())
Ys_df = pd.get_dummies(adata_all.obs[task_key].copy()).astype(int)
Cs_df = pd.get_dummies(adata_all.obs[cov_key].copy()).astype(int)
Ns_df = pd.concat([Ns_df, Cs_df], axis=1)
Xs, Ys, ins, meta_ids = Create_MIL_Dataset(ins_df=Ns_df, label_df=Ys_df, metadata=adata_all.obs, bag_column=sample_key)
print(f"Number of bags: {len(Xs)}", f"Number of labels: {Ys.shape}", f"Number of max instances: {max(ins)}")
mil_dataset = MILDataset(Xs, Ys, ins, meta_ids)
train_loader = DataLoader(mil_dataset, batch_size=len(mil_dataset), shuffle=False, collate_fn=MIL_Collate_fn)
for tr_padded_bags, tr_labels, tr_lengths, tr_id in train_loader:
                print(len(tr_lengths))


pred_label_tr_list = []
true_label_tr_list = []
attention_mtx_list = []
instance_level_list = []
embedding_tr_list = []
cell_id_list = []
loss_tr = 0
for tr_idx in range(len(tr_lengths)):
        tr_length = tr_lengths[tr_idx]
        if tr_length == 1:
            continue
        elif tr_length > 1:
            input_tr = tr_padded_bags[tr_idx, :tr_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_tr, embedding_tr, _, attn_weights, _ = cls(enc(input_tr.squeeze(0).transpose(0,1), None).transpose(0,1).unsqueeze(0))
            pred_label_tr = transformation_function(cls_tr)
            true_label_tr = tr_labels[tr_idx]
            pred_label_tr_list.append(pred_label_tr.detach().cpu().numpy())
            true_label_tr_list.append(true_label_tr.detach().cpu().numpy())
            instance_level_list.append(input_tr.squeeze(0).permute(1,0).detach().cpu().numpy())
            attention_mtx_list.append(attn_weights.squeeze(0,1).detach().cpu().numpy())
            cell_id_list.append(tr_id[tr_idx])
            embedding_tr_list.append(embedding_tr.squeeze(0).detach().cpu().numpy())



tr_sample_ids = [adata_all[adata_all.obs.index==x[0]].obs[sample_key][0] for x in cell_id_list]
pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=Ys_df.columns, index=tr_sample_ids)
pred_label_tr_df.loc[perturb_sample]
pred_label.loc[perturb_sample]
# dementia    0.012624
# normal      0.987375
cell_id_list = [i for sublist in cell_id_list for i in sublist]
cell_id_df = pd.DataFrame([i for i in cell_id_list], columns=["cell_id"])
metadata_tr = adata_all.obs.loc[cell_id_df.cell_id]
sample_meta = [i for i in tr_id]
sample_meta_tr = metadata_tr.loc[[i[0] for i in sample_meta]]
embedding_tr_df = pd.DataFrame(np.vstack(embedding_tr_list))
sample_meta_tr
# purtub 'H20.33.034'
perturb_sample_cell_rep_id = sample_meta_tr.iloc[tr_sample_ids.index(perturb_sample)].index
# umap embedding
import umap
reducer = umap.UMAP(n_components=2, random_state=42)
embedding_tr_umap = reducer.fit_transform(embedding_tr_df)
embedding_tr_umap_df = pd.DataFrame(embedding_tr_umap, index=sample_meta_tr.index)
fig = plt.figure(figsize=(8, 7))
sns.scatterplot(x=embedding_tr_umap[:, 0], y=embedding_tr_umap[:, 1], s=300, linewidth=0.1,
                hue=sample_meta_tr.disease)
plt.scatter(embedding_tr_umap_df.iloc[60, 0], embedding_tr_umap_df.iloc[60, 1], s=1000, c='crimson', alpha=0.56)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend([],[], frameon=False)
sns.despine()
plt.xlim(-1,5.5)
fig.savefig("UMAP_sample_level_embedding_AD.pdf", dpi=150)






adata_merged = adata_all.concatenate(adata_sample_high_attn_cells, join='outer', batch_key='source', batch_categories=['all', 'sample'])

Ns = adata_merged.X.copy()
Ns_df = pd.DataFrame(Ns, index=adata_merged.obs.index.tolist())
Ys_df = pd.get_dummies(adata_merged.obs[task_key].copy()).astype(int)
Cs_df = pd.get_dummies(adata_merged.obs[cov_key].copy()).astype(int)
Ns_df = pd.concat([Ns_df, Cs_df], axis=1)
Xs, Ys, ins, meta_ids = Create_MIL_Dataset(ins_df=Ns_df, label_df=Ys_df, metadata=adata_merged.obs, bag_column=sample_key)
print(f"Number of bags: {len(Xs)}", f"Number of labels: {Ys.shape}", f"Number of max instances: {max(ins)}")
mil_dataset = MILDataset(Xs, Ys, ins, meta_ids)
train_loader = DataLoader(mil_dataset, batch_size=len(mil_dataset), shuffle=False, collate_fn=MIL_Collate_fn)
for tr_padded_bags, tr_labels, tr_lengths, tr_id in train_loader:
                print(len(tr_lengths))



pred_label_tr_list = []
true_label_tr_list = []
attention_mtx_list = []
instance_level_list = []
embedding_tr_list = []
cell_id_list = []
loss_tr = 0
for tr_idx in range(len(tr_lengths)):
        tr_length = tr_lengths[tr_idx]
        if tr_length == 1:
            continue
        elif tr_length > 1:
            input_tr = tr_padded_bags[tr_idx, :tr_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_tr, embedding_tr, _, attn_weights, _ = cls(enc(input_tr.squeeze(0).transpose(0,1), None).transpose(0,1).unsqueeze(0))
            pred_label_tr = transformation_function(cls_tr)
            true_label_tr = tr_labels[tr_idx]
            pred_label_tr_list.append(pred_label_tr.detach().cpu().numpy())
            true_label_tr_list.append(true_label_tr.detach().cpu().numpy())
            instance_level_list.append(input_tr.squeeze(0).permute(1,0).detach().cpu().numpy())
            attention_mtx_list.append(attn_weights.squeeze(0,1).detach().cpu().numpy())
            cell_id_list.append(tr_id[tr_idx])
            embedding_tr_list.append(embedding_tr.squeeze(0).detach().cpu().numpy())



tr_sample_ids = [adata_merged[adata_merged.obs.index==x[0]].obs[sample_key][0] for x in cell_id_list]
pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=Ys_df.columns, index=tr_sample_ids)
pred_label_tr_df.loc[perturb_sample]
pred_label.loc[perturb_sample]



cell_id_list = [i for sublist in cell_id_list for i in sublist]
cell_id_df = pd.DataFrame([i for i in cell_id_list], columns=["cell_id"])
metadata_tr = adata_merged.obs.loc[cell_id_df.cell_id]
sample_meta = [i for i in tr_id]
sample_meta_tr = metadata_tr.loc[[i[0] for i in sample_meta]]
embedding_tr_df = pd.DataFrame(np.vstack(embedding_tr_list))
sample_meta_tr
# purtub 'H20.33.034'
perturb_sample_cell_rep_id = sample_meta_tr.iloc[tr_sample_ids.index(perturb_sample)].index
embedding_tr_umap = reducer.transform(embedding_tr_df)

fig = plt.figure(figsize=(8, 7))
sns.scatterplot(x=embedding_tr_umap_df.iloc[:, 0].tolist(), y=embedding_tr_umap_df.iloc[:, 1].tolist(), s=300, linewidth=0.1,
                hue=sample_meta_tr.disease)
plt.scatter(embedding_tr_umap[60, 0], embedding_tr_umap[60, 1], s=1000, c='crimson', alpha=0.56)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.xlim(-1,5.5)
plt.legend([],[], frameon=False)
sns.despine()
fig.savefig("UMAP_sample_level_embedding_AD_merged.pdf", dpi=150)






adata_merged = adata_merged.concatenate(adata_sample_high_attn_cells, join='outer', batch_key='source', batch_categories=['all', 'sample'])

Ns = adata_merged.X.copy()
Ns_df = pd.DataFrame(Ns, index=adata_merged.obs.index.tolist())
Ys_df = pd.get_dummies(adata_merged.obs[task_key].copy()).astype(int)
Cs_df = pd.get_dummies(adata_merged.obs[cov_key].copy()).astype(int)
Ns_df = pd.concat([Ns_df, Cs_df], axis=1)
Xs, Ys, ins, meta_ids = Create_MIL_Dataset(ins_df=Ns_df, label_df=Ys_df, metadata=adata_merged.obs, bag_column=sample_key)
print(f"Number of bags: {len(Xs)}", f"Number of labels: {Ys.shape}", f"Number of max instances: {max(ins)}")
mil_dataset = MILDataset(Xs, Ys, ins, meta_ids)
train_loader = DataLoader(mil_dataset, batch_size=len(mil_dataset), shuffle=False, collate_fn=MIL_Collate_fn)
for tr_padded_bags, tr_labels, tr_lengths, tr_id in train_loader:
                print(len(tr_lengths))



pred_label_tr_list = []
true_label_tr_list = []
attention_mtx_list = []
instance_level_list = []
embedding_tr_list = []
cell_id_list = []
loss_tr = 0
for tr_idx in range(len(tr_lengths)):
        tr_length = tr_lengths[tr_idx]
        if tr_length == 1:
            continue
        elif tr_length > 1:
            input_tr = tr_padded_bags[tr_idx, :tr_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_tr, embedding_tr, _, attn_weights, _ = cls(enc(input_tr.squeeze(0).transpose(0,1), None).transpose(0,1).unsqueeze(0))
            pred_label_tr = transformation_function(cls_tr)
            true_label_tr = tr_labels[tr_idx]
            pred_label_tr_list.append(pred_label_tr.detach().cpu().numpy())
            true_label_tr_list.append(true_label_tr.detach().cpu().numpy())
            instance_level_list.append(input_tr.squeeze(0).permute(1,0).detach().cpu().numpy())
            attention_mtx_list.append(attn_weights.squeeze(0,1).detach().cpu().numpy())
            cell_id_list.append(tr_id[tr_idx])
            embedding_tr_list.append(embedding_tr.squeeze(0).detach().cpu().numpy())



tr_sample_ids = [adata_merged[adata_merged.obs.index==x[0]].obs[sample_key][0] for x in cell_id_list]
pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=Ys_df.columns, index=tr_sample_ids)
pred_label_tr_df.loc[perturb_sample]
pred_label.loc[perturb_sample]



cell_id_list = [i for sublist in cell_id_list for i in sublist]
cell_id_df = pd.DataFrame([i for i in cell_id_list], columns=["cell_id"])
metadata_tr = adata_merged.obs.loc[cell_id_df.cell_id]
sample_meta = [i for i in tr_id]
sample_meta_tr = metadata_tr.loc[[i[0] for i in sample_meta]]
embedding_tr_df = pd.DataFrame(np.vstack(embedding_tr_list))
sample_meta_tr
# purtub 'H20.33.034'
perturb_sample_cell_rep_id = sample_meta_tr.iloc[tr_sample_ids.index(perturb_sample)].index
embedding_tr_umap = reducer.transform(embedding_tr_df)

fig = plt.figure(figsize=(8, 7))
sns.scatterplot(x=embedding_tr_umap_df.iloc[:, 0].tolist(), y=embedding_tr_umap_df.iloc[:, 1].tolist(), s=300, linewidth=0.1,
                hue=sample_meta_tr.disease)
plt.scatter(embedding_tr_umap[60, 0], embedding_tr_umap[60, 1], s=1000, c='crimson', alpha=0.56)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.xlim(-1,5.5)
plt.legend([],[], frameon=False)
sns.despine()
fig.savefig("UMAP_sample_level_embedding_AD_merged_7.pdf", dpi=150)








tr_sample_ids = [adata_merged[adata_merged.obs.index==x[0]].obs[sample_key][0] for x in cell_id_list]
pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=Ys_df.columns, index=tr_sample_ids)
pred_label_tr_df.loc[perturb_sample]
pred_label.loc[perturb_sample]
# dementia    0.091777
# normal      0.908223

adata_all.shape[0]
# 23853
adata_merged.shape[0]
# 23863



adata_merged_ = adata_merged.concatenate(adata_sample_high_attn_cells, join='outer', batch_key='source', batch_categories=['all', 'sample'])

Ns = adata_merged_.X.copy()
Ns_df = pd.DataFrame(Ns, index=adata_merged_.obs.index.tolist())
Ys_df = pd.get_dummies(adata_merged_.obs[task_key].copy()).astype(int)
Cs_df = pd.get_dummies(adata_merged_.obs[cov_key].copy()).astype(int)
Ns_df = pd.concat([Ns_df, Cs_df], axis=1)
Xs, Ys, ins, meta_ids = Create_MIL_Dataset(ins_df=Ns_df, label_df=Ys_df, metadata=adata_merged_.obs, bag_column=sample_key)
print(f"Number of bags: {len(Xs)}", f"Number of labels: {Ys.shape}", f"Number of max instances: {max(ins)}")
mil_dataset = MILDataset(Xs, Ys, ins, meta_ids)
train_loader = DataLoader(mil_dataset, batch_size=len(mil_dataset), shuffle=False, collate_fn=MIL_Collate_fn)
for tr_padded_bags, tr_labels, tr_lengths, tr_id in train_loader:
                print(len(tr_lengths))


pred_label_tr_list = []
true_label_tr_list = []
attention_mtx_list = []
instance_level_list = []
cell_id_list = []
loss_tr = 0
for tr_idx in range(len(tr_lengths)):
        tr_length = tr_lengths[tr_idx]
        if tr_length == 1:
            continue
        elif tr_length > 1:
            input_tr = tr_padded_bags[tr_idx, :tr_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_tr, _, _, attn_weights, _ = cls(enc(input_tr.squeeze(0).transpose(0,1), None).transpose(0,1).unsqueeze(0))
            pred_label_tr = transformation_function(cls_tr)
            true_label_tr = tr_labels[tr_idx]
            pred_label_tr_list.append(pred_label_tr.detach().cpu().numpy())
            true_label_tr_list.append(true_label_tr.detach().cpu().numpy())
            instance_level_list.append(input_tr.squeeze(0).permute(1,0).detach().cpu().numpy())
            attention_mtx_list.append(attn_weights.squeeze(0,1).detach().cpu().numpy())
            cell_id_list.append(tr_id[tr_idx])


tr_sample_ids = [adata_merged_[adata_merged_.obs.index==x[0]].obs[sample_key][0] for x in cell_id_list]
pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=Ys_df.columns, index=tr_sample_ids)
pred_label_tr_df.loc[perturb_sample]
pred_label.loc[perturb_sample]
# dementia    0.296515
# normal      0.703485



adata_merged_ = adata_merged_.concatenate(adata_sample_high_attn_cells, join='outer', batch_key='source', batch_categories=['all', 'sample'])

Ns = adata_merged_.X.copy()
Ns_df = pd.DataFrame(Ns, index=adata_merged_.obs.index.tolist())
Ys_df = pd.get_dummies(adata_merged_.obs[task_key].copy()).astype(int)
Cs_df = pd.get_dummies(adata_merged_.obs[cov_key].copy()).astype(int)
Ns_df = pd.concat([Ns_df, Cs_df], axis=1)
Xs, Ys, ins, meta_ids = Create_MIL_Dataset(ins_df=Ns_df, label_df=Ys_df, metadata=adata_merged_.obs, bag_column=sample_key)
print(f"Number of bags: {len(Xs)}", f"Number of labels: {Ys.shape}", f"Number of max instances: {max(ins)}")
mil_dataset = MILDataset(Xs, Ys, ins, meta_ids)
train_loader = DataLoader(mil_dataset, batch_size=len(mil_dataset), shuffle=False, collate_fn=MIL_Collate_fn)
for tr_padded_bags, tr_labels, tr_lengths, tr_id in train_loader:
                print(len(tr_lengths))


pred_label_tr_list = []
true_label_tr_list = []
attention_mtx_list = []
instance_level_list = []
cell_id_list = []
loss_tr = 0
for tr_idx in range(len(tr_lengths)):
        tr_length = tr_lengths[tr_idx]
        if tr_length == 1:
            continue
        elif tr_length > 1:
            input_tr = tr_padded_bags[tr_idx, :tr_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_tr, _, _, attn_weights, _ = cls(enc(input_tr.squeeze(0).transpose(0,1), None).transpose(0,1).unsqueeze(0))
            pred_label_tr = transformation_function(cls_tr)
            true_label_tr = tr_labels[tr_idx]
            pred_label_tr_list.append(pred_label_tr.detach().cpu().numpy())
            true_label_tr_list.append(true_label_tr.detach().cpu().numpy())
            instance_level_list.append(input_tr.squeeze(0).permute(1,0).detach().cpu().numpy())
            attention_mtx_list.append(attn_weights.squeeze(0,1).detach().cpu().numpy())
            cell_id_list.append(tr_id[tr_idx])


tr_sample_ids = [adata_merged_[adata_merged_.obs.index==x[0]].obs[sample_key][0] for x in cell_id_list]
pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=Ys_df.columns, index=tr_sample_ids)
pred_label_tr_df.loc[perturb_sample]
pred_label.loc[perturb_sample]

# dementia    0.733312
# normal      0.266688

# dementia    0.957838
# normal      0.042162

# dementia    0.990135
# normal      0.009865





adata_sample_high_attn_cells.obs.Supertype


cell_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/SEAAD_Glia/MTG/MTG_sciMultiLaMA_CELL_embedding.csv', index_col=0)
gene_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/SEAAD_Glia/MTG/MTG_sciMultiLaMA_GENE_embedding.csv', index_col=0)
#