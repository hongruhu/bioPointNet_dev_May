# (mil) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet


import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns
from bioPointNet_Apr2025 import *






pn_cls_checkpoint = torch.load("/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/hnoca_intersected_union1000hvg/best_model_HNOCA_from_Gene.pt")
# Number of bags: 340 Number of labels: (340, 1) Number of max instances: 11323
device = "cpu"
def MIL_Collate_fn(batch):
    """
    Custom collate function for MIL datasets: pads bags to the same length.
    Args:
        batch (list): List of tuples (bag, label) from the dataset.
    Returns:
        padded_bags (torch.Tensor): Padded bags of shape (batch_size, max_instances, num_features).
        labels (torch.Tensor): Labels of shape (batch_size, num_classes).
        lengths (torch.Tensor): Lengths of each bag in the batch, shape (batch_size,).
    """
    bags, labels, ins, metadata_idx= zip(*batch)
    lengths = ins
    max_length = max(ins)                            # Maximum number of instances in the batch
    # Pad bags to the same length
    padded_bags = torch.zeros((len(bags), max_length, bags[0].shape[1]), dtype=torch.float32)
    for i, bag in enumerate(bags):
        padded_bags[i, :len(bag)] = bag
    labels = torch.stack(labels)        # Stack labels into a single tensor
    return padded_bags.to(device), labels.to(device), lengths, metadata_idx






hnoca = sc.read_h5ad("/group/gquongrp/workspaces/hongruhu/MIL/neural_organoid/scPN_modeling/hnoca_intersected_union1000hvg.h5ad")
sample_key_ref = "bio_sample"
metadata_ref = hnoca.obs.copy()
Ns = hnoca.X.copy() # sparse matrix
Ns_df = pd.DataFrame(Ns.todense(), index=metadata_ref.index.tolist(), columns=hnoca.var.index.tolist())
metadata_ref['organoid_age_days_rank'] = metadata_ref['organoid_age_days'].rank(method='dense') - 1
metadata_ref.organoid_age_days_rank.value_counts().sort_index()
Ys_df = metadata_ref[['organoid_age_days_rank']].copy()
Xs, Ys, ins, meta_ids = Create_MIL_Dataset(ins_df=Ns_df, label_df=Ys_df, metadata=metadata_ref, bag_column=sample_key_ref)
print(f"Number of bags: {len(Xs)}", f"Number of labels: {Ys.shape}", f"Number of max instances: {max(ins)}")
# Number of bags: 282 Number of labels: (282, 1) Number of max instances: 34610
# Create the dataset
mil_dataset = MILDataset(Xs, Ys, ins, meta_ids)
train_size = len(mil_dataset)
train_loader_ = DataLoader(mil_dataset, batch_size=train_size, shuffle=False, collate_fn=MIL_Collate_fn)
for tr_padded_bags, tr_labels, tr_lengths, tr_id in train_loader_:
        print(len(tr_lengths))


attention_mtx_list = []
ins_emb_list = []
cell_id_list = []
pn_cls_checkpoint.to(device)
for tr_idx in range(len(tr_lengths)):
        tr_length = tr_lengths[tr_idx]
        x = tr_padded_bags[tr_idx, :tr_length,:].unsqueeze(0).permute(0, 2, 1)
        cls_tr, embedding_tr, global_feature_tr, attn_weights_tr, _ = pn_cls_checkpoint(x)
        attention_mtx_list.append(attn_weights_tr.squeeze(0,1).detach().cpu().numpy())
        A_input = pn_cls_checkpoint.backbone.tnet1(x)
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)
        x = pn_cls_checkpoint.backbone.shared_mlp1(x)
        A_feat = pn_cls_checkpoint.backbone.tnet2(x)
        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)
        x = pn_cls_checkpoint.backbone.shared_mlp2(x)
        ins_level_emb = x.squeeze(0).T.cpu().detach().numpy()
        cell_id_list.append(tr_id[tr_idx])
        ins_emb_list.append(ins_level_emb)



ins_emb_hnoca = np.vstack(ins_emb_list) # (1306759, 1766)
cell_id_df_hnoca = pd.DataFrame([i for i in [i for sublist in cell_id_list for i in sublist]], columns=["cell_id"])
attention_mtx_raw_hnoca = np.concatenate(attention_mtx_list, axis=0) # (53193, )
attention_mtx_list_norm_cellnum = [i*len(i) for i in attention_mtx_list]
attention_mtx_norm_cellnum = np.concatenate(attention_mtx_list_norm_cellnum, axis=0) # (53193, )
cell_id_df_hnoca['attn_raw'] = attention_mtx_raw_hnoca
cell_id_df_hnoca['attn_norm'] = attention_mtx_norm_cellnum
cell_id_df_hnoca.index = cell_id_df_hnoca.cell_id.tolist()
np.save('ins_emb_hnoca.npy', ins_emb_hnoca)
cell_id_df_hnoca.to_csv('cell_hnoca.csv')



hdbca = sc.read_h5ad("/group/gquongrp/workspaces/hongruhu/MIL/neural_organoid/scPN_modeling/hdbca_intersected_union1000hvg.h5ad")
sample_key_query = "sample_id"
metadata_query = hdbca.obs.copy()
metadata_query['original_cell_id'] = metadata_query.index.tolist()
Ns = hdbca.X.copy() # sparse matrix
Ns_df = pd.DataFrame(Ns.todense(), index=metadata_query.index.tolist(), columns=hdbca.var.index.tolist())
Ys_df = metadata_query[['Age']].copy()
Xs, Ys, ins, meta_ids = Create_MIL_Dataset(ins_df=Ns_df, label_df=Ys_df, metadata=metadata_query, bag_column=sample_key_query)
print(f"Number of bags: {len(Xs)}", f"Number of labels: {Ys.shape}", f"Number of max instances: {max(ins)}")

# Create the dataset
mil_dataset = MILDataset(Xs, Ys, ins, meta_ids)
train_size = len(mil_dataset)
train_loader_ = DataLoader(mil_dataset, batch_size=train_size, shuffle=False, collate_fn=MIL_Collate_fn)
for tr_padded_bags, tr_labels, tr_lengths, tr_id in train_loader_:
        print(len(tr_lengths))


attention_mtx_list = []
ins_emb_list = []
cell_id_list = []
pn_cls_checkpoint.to(device)
for tr_idx in range(len(tr_lengths)):
        tr_length = tr_lengths[tr_idx]
        x = tr_padded_bags[tr_idx, :tr_length,:].unsqueeze(0).permute(0, 2, 1)
        cls_tr, embedding_tr, global_feature_tr, attn_weights_tr, _ = pn_cls_checkpoint(x)
        attention_mtx_list.append(attn_weights_tr.squeeze(0,1).detach().cpu().numpy())
        A_input = pn_cls_checkpoint.backbone.tnet1(x)
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)
        x = pn_cls_checkpoint.backbone.shared_mlp1(x)
        A_feat = pn_cls_checkpoint.backbone.tnet2(x)
        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)
        x = pn_cls_checkpoint.backbone.shared_mlp2(x)
        ins_level_emb = x.squeeze(0).T.cpu().detach().numpy()
        cell_id_list.append(tr_id[tr_idx])
        ins_emb_list.append(ins_level_emb)



ins_emb_hdbca = np.vstack(ins_emb_list) # (1306759, 1766)
cell_id_df_hdbca = pd.DataFrame([i for i in [i for sublist in cell_id_list for i in sublist]], columns=["cell_id"])
attention_mtx_raw_hdbca = np.concatenate(attention_mtx_list, axis=0) # (53193, )
attention_mtx_list_norm_cellnum = [i*len(i) for i in attention_mtx_list]
attention_mtx_norm_cellnum = np.concatenate(attention_mtx_list_norm_cellnum, axis=0) # (53193, )
cell_id_df_hdbca['attn_raw'] = attention_mtx_raw_hdbca
cell_id_df_hdbca['attn_norm'] = attention_mtx_norm_cellnum
cell_id_df_hdbca.index = cell_id_df_hdbca.cell_id.tolist()
np.save('ins_emb_hdbca.npy', ins_emb_hdbca)
cell_id_df_hdbca.to_csv('cell_hdbca.csv')








import umap
# Fit UMAP on ins_emb_hnoca
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
umap_hnoca = umap_model.fit_transform(ins_emb_hnoca)
# Apply the learned UMAP transformation to ins_emb_hdbca
umap_hdbca = umap_model.transform(ins_emb_hdbca)

# Save the UMAP results
umap_hnoca_df = pd.DataFrame(umap_hnoca, columns=['UMAP1', 'UMAP2'], index=cell_id_df_hnoca.index)
umap_hdbca_df = pd.DataFrame(umap_hdbca, columns=['UMAP1', 'UMAP2'], index=cell_id_df_hdbca.index)



umap_hnoca_df.to_csv('umap_hnoca.csv')
umap_hdbca_df.to_csv('umap_hdbca.csv')



hnoca.obsm['X_umap'] = umap_hnoca_df.loc[hnoca.obs.index.tolist()].to_numpy()
hdbca.obsm['X_umap'] = umap_hdbca_df.loc[hdbca.obs.index.tolist()].to_numpy()

# plot color by cell_type and save
sc.pl.umap(hnoca, color='annot_level_1', save='_hnoca_cell_type.png')
sc.pl.umap(hdbca, color='CellClass', save='_hdbca_cell_type.png')







import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns
from bioPointNet_Apr2025 import *

import umap

ins_emb_hnoca = np.load('ins_emb_hnoca.npy')
ins_emb_hdbca = np.load('ins_emb_hdbca.npy')
cell_hnoca = pd.read_csv('cell_hnoca.csv', index_col=0)
cell_hdbca = pd.read_csv('cell_hdbca.csv', index_col=0)



hnoca = sc.read_h5ad("/group/gquongrp/workspaces/hongruhu/MIL/neural_organoid/scPN_modeling/hnoca_intersected_union1000hvg.h5ad")
hdbca = sc.read_h5ad("/group/gquongrp/workspaces/hongruhu/MIL/neural_organoid/scPN_modeling/hdbca_intersected_union1000hvg.h5ad")

# Concatenate embeddings from both datasets
ins_emb_combined = np.vstack([ins_emb_hnoca, ins_emb_hdbca])

# Fit UMAP on the combined embeddings
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
umap_combined = umap_model.fit_transform(ins_emb_combined)

# Split the UMAP results back into hnoca and hdbca
umap_hnoca = umap_combined[:len(ins_emb_hnoca)]
umap_hdbca = umap_combined[len(ins_emb_hnoca):]

# Save the UMAP results
umap_hnoca_df = pd.DataFrame(umap_hnoca, columns=['UMAP1', 'UMAP2'], index=cell_hnoca.index)
umap_hdbca_df = pd.DataFrame(umap_hdbca, columns=['UMAP1', 'UMAP2'], index=cell_hdbca.index)

umap_hnoca_df.to_csv('umap_hnoca_integrated.csv')
umap_hdbca_df.to_csv('umap_hdbca_integrated.csv')

# Add UMAP results to AnnData objects
hnoca.obsm['X_umap'] = umap_hnoca_df.loc[hnoca.obs.index.tolist()].to_numpy()
hdbca.obsm['X_umap'] = umap_hdbca_df.loc[hdbca.obs.index.tolist()].to_numpy()

# Plot color by cell_type and save
sc.pl.umap(hnoca, color='annot_level_1', save='_hnoca_cell_type_integrated.png')
sc.pl.umap(hdbca, color='CellClass', save='_hdbca_cell_type_integrated.png')









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
