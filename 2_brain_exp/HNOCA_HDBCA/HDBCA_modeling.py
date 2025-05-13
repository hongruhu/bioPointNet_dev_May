# (mil) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet


import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns
from bioPointNet_Apr2025 import *



adata = sc.read_h5ad("/group/gquongrp/workspaces/hongruhu/MIL/neural_organoid/scPN_modeling/hdbca_intersected_union1000hvg.h5ad")
# AnnData object with n_obs × n_vars = 1641349 × 1766 -> AnnData object with n_obs × n_vars = 1665937 × 1766
adata.obs.sample_id.value_counts()
sample_key = "sample_id"
adata.obs[sample_key].unique() # 340 samples

metadata = adata.obs.copy()
metadata.development_stage.value_counts().sort_index()
# development_stage
# Carnegie stage 15 (33-36d/ 5 weeks)  to   15th week post-fertilization stage

metadata.Age.value_counts().sort_index()


np.sum(metadata.index == adata.obs.index) # 1665937
metadata['original_cell_id'] = metadata.index.tolist()

Ns = adata.X.copy() # sparse matrix
Ns_df = pd.DataFrame(Ns.todense(), index=metadata.index.tolist(), columns=adata.var.index.tolist())
metadata.Age.value_counts().sort_index()
# 18  time points
Ys_df = metadata[['Age']].copy()
Ys_df



Xs, Ys, ins, meta_ids = Create_MIL_Dataset(ins_df=Ns_df, label_df=Ys_df, metadata=metadata, bag_column=sample_key)
print(f"Number of bags: {len(Xs)}", f"Number of labels: {Ys.shape}", f"Number of max instances: {max(ins)}")
# Number of bags: 340 Number of labels: (340, 1) Number of max instances: 11323


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the dataset
mil_dataset = MILDataset(Xs, Ys, ins, meta_ids)
# Define the proportions for training and validation sets
train_size = int(0.8 * len(mil_dataset))
val_size = len(mil_dataset) - train_size
train_dataset, val_dataset = random_split(mil_dataset, [train_size, val_size])
len(train_dataset.indices)
# 225
len(val_dataset.indices)
# 57
# Print the sizes of the datasets
print(f"Training set size: {len(train_dataset)}", f"Validation set size: {len(val_dataset)}")
# Training set size: 225 Validation set size: 57

batch_size = 15
# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=MIL_Collate_fn)
val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False, collate_fn=MIL_Collate_fn)


SEED = 15
set_seed(SEED)
pn_cls = PointNetClassHead(input_dim=Ns.shape[1], k=1, global_features=256, attention_dim=64, agg_method="gated_attention")
pn_cls.apply(init_weights)
pn_cls.to(device)

learning_rate = 1e-4
optimizer = torch.optim.Adam(pn_cls.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()
criterion = OrdinalRegressionLoss(num_class=Ys_df.value_counts().size, train_cutpoints=True)
transformation_function = nn.Softmax(dim=1)

best_val_loss = float('inf')
patience_counter = 0
patience = 20
best_model_state = None
epochs = 200
for val_padded_bags, val_labels, val_lengths, val_id in val_loader:
        print(len(val_lengths))





import copy
for epoch in range(epochs):
    for padded_bags, labels, lengths, _ in train_loader:
        # train_model(pn_cls)
        optimizer.zero_grad()
        loss_tr = 0
        for idx in range(len(lengths)):
            length = lengths[idx]
            input_tr = padded_bags[idx, :length,:].unsqueeze(0).permute(0, 2, 1)
            res_tr = pn_cls(input_tr)[0]
            loss_per_sample = criterion(res_tr.to('cpu'), labels[idx].view(-1, 1).to('cpu')) # equivalent to the following
            loss_tr += loss_per_sample.to(device)
        (loss_tr/len(lengths)).backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss_tr:.4f}")
    loss_val = 0
    # eval_model(pn_cls)
    for val_idx in range(len(val_lengths)):
        val_length = val_lengths[val_idx]
        input_val = val_padded_bags[val_idx, :val_length,:].unsqueeze(0).permute(0, 2, 1)
        res_val = pn_cls(input_val)[0]
        val_loss_per_sample = criterion(res_val.to('cpu'), val_labels[val_idx].view(-1, 1).to('cpu'))
        loss_val += val_loss_per_sample
    loss_val_avg = loss_val/len(val_lengths)
    print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {loss_val_avg:.4f}")
    if loss_val_avg < best_val_loss:
        best_val_loss = loss_val_avg
        torch.save(pn_cls, "best_model_HDBCA_from_Gene.pt")
        patience_counter = 0
        print(f"Saving the best model with validation loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


pn_cls_checkpoint = torch.load("best_model_HDBCA_from_Gene.pt")
# pn_cls_checkpoint.eval()
pred_label_val_list = []
true_label_val_list = []
loss_val = 0
for val_idx in range(len(val_lengths)):
        val_length = val_lengths[val_idx]
        input_val = val_padded_bags[val_idx, :val_length,:].unsqueeze(0).permute(0, 2, 1)
        cls_val, embedding_val, global_feature_val, attention_weights_val, _ = pn_cls_checkpoint(input_val)
        pred_label_val_list.append(cls_val.detach().cpu().numpy())
        true_label_val_list.append(val_labels[val_idx].detach().cpu().numpy())
        loss_per_sample = criterion(cls_val.cpu(), val_labels[val_idx].view(-1, 1).cpu())
        loss_val += loss_per_sample


loss_val/len(val_lengths) # 
true_label_val_df = pd.DataFrame(np.vstack(true_label_val_list), columns=Ys_df.columns)
pred_label_val_df = pd.DataFrame(np.vstack(pred_label_val_list), columns=Ys_df.columns)
# correlation between true and predicted labels
true_label_val_df.corrwith(pred_label_val_df, method="spearman")
# organoid_age_days_rank    0.96052 hnoca   | hdbca 0.957239
true_label_val_df.corrwith(pred_label_val_df, method="pearson")
# organoid_age_days_rank    0.970783 hnoca  | hdbca 0.998474
# correlation between true and predicted labels
spearman_corr = true_label_val_df.corrwith(pred_label_val_df, method="spearman").values[0]
pearson_corr = true_label_val_df.corrwith(pred_label_val_df, method="pearson").values[0]
# scatterplot, x-axis: true label, y-axis: predicted label
fig = plt.figure(figsize=(8, 7))
sns.scatterplot(x=true_label_val_df.values.flatten(), y=pred_label_val_df.values.flatten(), s=150, linewidth=0.01)
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.title("True vs Predicted Label")
sns.despine()
# Add text of correlations to the plot
plt.text(0.05, 0.95, f'Spearman: {spearman_corr:.2f}', transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')
plt.text(0.05, 0.90, f'Pearson: {pearson_corr:.2f}', transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')
sns.despine()
plt.show()
fig.savefig("true_vs_pred_label_ValSet.pdf", dpi=150)


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


train_loader_ = DataLoader(train_dataset, batch_size=train_size, shuffle=False, collate_fn=MIL_Collate_fn)
for tr_padded_bags, tr_labels, tr_lengths, tr_id in train_loader_:
        print(len(tr_lengths))


pred_label_tr_list = []
true_label_tr_list = []
attention_mtx_list = []
instance_level_list = []
cell_id_list = []
embedding_tr_list = []
global_feature_tr_list = []
loss_tr = 0
pn_cls_checkpoint.to(device)
for tr_idx in range(len(tr_lengths)):
        tr_length = tr_lengths[tr_idx]
        input_tr = tr_padded_bags[tr_idx, :tr_length,:].unsqueeze(0).permute(0, 2, 1)
        cls_tr, embedding_tr, global_feature_tr, attn_weights_tr, _ = pn_cls_checkpoint(input_tr)
        embedding_tr_list.append(embedding_tr.squeeze(0).detach().cpu().numpy())
        global_feature_tr_list.append(global_feature_tr.squeeze(0).detach().cpu().numpy())
        pred_label_tr = cls_tr
        true_label_tr = tr_labels[tr_idx]
        pred_label_tr_list.append(pred_label_tr.detach().cpu().numpy())
        true_label_tr_list.append(true_label_tr.detach().cpu().numpy())
        instance_level_list.append(input_tr.squeeze(0).permute(1,0).detach().cpu().numpy())
        attention_mtx_list.append(attn_weights_tr.squeeze(0,1).detach().cpu().numpy())
        loss_per_sample = criterion(pred_label_tr, true_label_tr.view(-1, 1))
        loss_tr += loss_per_sample
        cell_id_list.append(tr_id[tr_idx])


loss_tr/len(tr_lengths) #
true_label_tr_df = pd.DataFrame(np.vstack(true_label_tr_list), columns=Ys_df.columns)
pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=Ys_df.columns)
# correlation between true and predicted labels
true_label_tr_df.corrwith(pred_label_tr_df, method="spearman")
# organoid_age_days_rank    0.995861 hnoca | hdbca 0.967258
true_label_tr_df.corrwith(pred_label_tr_df, method="pearson")
# organoid_age_days_rank    0.998635 hnoca | hdaca 0.999918

# correlation between true and predicted labels
spearman_corr = true_label_tr_df.corrwith(pred_label_tr_df, method="spearman").values[0]
pearson_corr = true_label_tr_df.corrwith(pred_label_tr_df, method="pearson").values[0]
# scatterplot, x-axis: true label, y-axis: predicted label
fig = plt.figure(figsize=(8, 7))
sns.scatterplot(x=true_label_tr_df.values.flatten(), y=pred_label_tr_df.values.flatten(), s=150, linewidth=0.01)
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.title("True vs Predicted Label")
sns.despine()
# Add text of correlations to the plot
plt.text(0.05, 0.95, f'Spearman: {spearman_corr:.2f}', transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')
plt.text(0.05, 0.90, f'Pearson: {pearson_corr:.2f}', transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')
sns.despine()
plt.show()
fig.savefig("true_vs_pred_label_TrSet.pdf", dpi=150)

len(attention_mtx_list)     # 107 | 272
len(instance_level_list)    # 107 | 272
# instance_mtx = np.vstack(instance_level_list) # (1306759, 1766)
# unlist the cell_id_list
cell_id_list = [i for sublist in cell_id_list for i in sublist]
cell_id_df = pd.DataFrame([i for i in cell_id_list], columns=["cell_id"])
metadata_tr = metadata.loc[cell_id_df.cell_id]


attention_mtx_raw = np.concatenate(attention_mtx_list, axis=0) # (53193, )
attention_mtx_raw_df = pd.DataFrame(attention_mtx_raw, columns=["attention_score_raw"], index=cell_id_df.cell_id)
from scipy.stats import zscore
attention_mtx_list_norm = [zscore(i) for i in attention_mtx_list]
attention_mtx = np.concatenate(attention_mtx_list_norm, axis=0) # (53193, )
attention_mtx_list_norm_cellnum = [i*len(i) for i in attention_mtx_list]
attention_mtx_norm_cellnum = np.concatenate(attention_mtx_list_norm_cellnum, axis=0) # (53193, )
attention_mtx_raw_df['attention_score_zscore'] = attention_mtx.tolist()
attention_mtx_raw_df['attention_score_norm_cellnum'] = attention_mtx_norm_cellnum.tolist()
attention_mtx_raw_df['attention_score_zscore_clip'] = attention_mtx_raw_df['attention_score_zscore'].apply(lambda x: min(x, 10))
attention_mtx_raw_df['attention_score_norm_cellnum_clip'] = attention_mtx_raw_df['attention_score_norm_cellnum'].apply(lambda x: min(x, 10))

metadata['original_cell_id'] = adata.obs.index.tolist()
metadata_tr = metadata.loc[attention_mtx_raw_df.index]
metadata_tr.to_csv("metadata_tr_HDBCA.csv")
attention_mtx_raw_df.to_csv("attention_mtx_HDBCA.csv")






hnoca = sc.read_h5ad("/group/gquongrp/workspaces/hongruhu/MIL/neural_organoid/scPN_modeling/hnoca_intersected_union1000hvg.h5ad")
# AnnData object with n_obs × n_vars = 1641349 × 1766
hnoca.obs.bio_sample.value_counts()

sample_key = "bio_sample"
hnoca.obs[sample_key].unique() # 282 samples

metadata_query = hnoca.obs.copy()
metadata_query.organoid_age_days.value_counts().sort_index()
# 7d - 192d, 50 time points.




metadata_query.index = ['cell_' + str(i) for i in range(metadata_query.shape[0])]
Ns = hnoca.X.copy() # sparse matrix
Ns_df = pd.DataFrame(Ns.todense(), index=metadata_query.index.tolist(), columns=hnoca.var.index.tolist())
# rank the organoid_age_days from 0 by order
metadata_query['organoid_age_days_rank'] = metadata_query['organoid_age_days'].rank(method='dense') - 1
metadata_query.organoid_age_days_rank.value_counts().sort_index()


Ys_df = metadata_query[['organoid_age_days_rank']].copy()
Ys_df

Xs, Ys, ins, meta_ids = Create_MIL_Dataset(ins_df=Ns_df, label_df=Ys_df, metadata=metadata_query, bag_column=sample_key)
print(f"Number of bags: {len(Xs)}", f"Number of labels: {Ys.shape}", f"Number of max instances: {max(ins)}")
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


# Create the dataset
mil_dataset = MILDataset(Xs, Ys, ins, meta_ids)
train_size = len(mil_dataset)

train_loader_ = DataLoader(mil_dataset, batch_size=train_size, shuffle=False, collate_fn=MIL_Collate_fn)
for tr_padded_bags, tr_labels, tr_lengths, tr_id in train_loader_:
        print(len(tr_lengths))


# pn_cls_checkpoint = torch.load("/group/gquongrp/workspaces/hongruhu/bioPointNet/HNOCA_HDBCA/hdbca_intersected_union1000hvg/best_model_HDBCA_from_Gene.pt")
criterion = OrdinalRegressionLoss(num_class=Ys_df.value_counts().size, train_cutpoints=True)
pred_label_tr_list = []
true_label_tr_list = []
attention_mtx_list = []
instance_level_list = []
cell_id_list = []
embedding_tr_list = []
global_feature_tr_list = []
loss_tr = 0
pn_cls_checkpoint.to(device)
for tr_idx in range(len(tr_lengths)):
        tr_length = tr_lengths[tr_idx]
        input_tr = tr_padded_bags[tr_idx, :tr_length,:].unsqueeze(0).permute(0, 2, 1)
        cls_tr, embedding_tr, global_feature_tr, attn_weights_tr, _ = pn_cls_checkpoint(input_tr)
        embedding_tr_list.append(embedding_tr.squeeze(0).detach().cpu().numpy())
        global_feature_tr_list.append(global_feature_tr.squeeze(0).detach().cpu().numpy())
        pred_label_tr = cls_tr
        true_label_tr = tr_labels[tr_idx]
        pred_label_tr_list.append(pred_label_tr.detach().cpu().numpy())
        true_label_tr_list.append(true_label_tr.detach().cpu().numpy())
        instance_level_list.append(input_tr.squeeze(0).permute(1,0).detach().cpu().numpy())
        attention_mtx_list.append(attn_weights_tr.squeeze(0,1).detach().cpu().numpy())
        loss_per_sample = criterion(pred_label_tr, true_label_tr.view(-1, 1))
        loss_tr += loss_per_sample
        cell_id_list.append(tr_id[tr_idx])


loss_tr/len(tr_lengths) #
true_label_tr_df = pd.DataFrame(np.vstack(true_label_tr_list), columns=Ys_df.columns)
pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=Ys_df.columns)
# correlation between true and predicted labels
true_label_tr_df.corrwith(pred_label_tr_df, method="spearman")
# organoid_age_days_rank    0.876859 hnoca -> hdbca | hdbca -> hnoca 0.8534
true_label_tr_df.corrwith(pred_label_tr_df, method="pearson")
# organoid_age_days_rank    0.875731 hnoca -> hdbca | hdbca -> hnoca 0.847528
# correlation between true and predicted labels
spearman_corr = true_label_tr_df.corrwith(pred_label_tr_df, method="spearman").values[0]
pearson_corr = true_label_tr_df.corrwith(pred_label_tr_df, method="pearson").values[0]
# scatterplot, x-axis: true label, y-axis: predicted label
fig = plt.figure(figsize=(8, 7))
sns.scatterplot(x=true_label_tr_df.values.flatten(), y=pred_label_tr_df.values.flatten(), s=150, linewidth=0.01)
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.title("True vs Predicted Label")
sns.despine()
# Add text of correlations to the plot
plt.text(0.05, 0.95, f'Spearman: {spearman_corr:.2f}', transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')
plt.text(0.05, 0.90, f'Pearson: {pearson_corr:.2f}', transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')
sns.despine()
plt.show()
fig.savefig("true_vs_pred_label_HDBCA_to_HNOCA.pdf", dpi=150)

len(attention_mtx_list)     # 340
len(instance_level_list)    # 340
# instance_mtx = np.vstack(instance_level_list) # (1306759, 1766)
# unlist the cell_id_list
cell_id_list = [i for sublist in cell_id_list for i in sublist]
cell_id_list == Ns_df.index.tolist() # False
cell_id_df = pd.DataFrame([i for i in cell_id_list], columns=["cell_id"])
metadata_tr = metadata_query.loc[cell_id_df.cell_id]

attention_mtx_raw = np.concatenate(attention_mtx_list, axis=0) # (53193, )
attention_mtx_raw_df = pd.DataFrame(attention_mtx_raw, columns=["attention_score_raw"], index=cell_id_df.cell_id)
from scipy.stats import zscore
attention_mtx_list_norm = [zscore(i) for i in attention_mtx_list]
attention_mtx = np.concatenate(attention_mtx_list_norm, axis=0) # (53193, )
attention_mtx_list_norm_cellnum = [i*len(i) for i in attention_mtx_list]
attention_mtx_norm_cellnum = np.concatenate(attention_mtx_list_norm_cellnum, axis=0) # (53193, )
attention_mtx_raw_df['attention_score_zscore'] = attention_mtx.tolist()
attention_mtx_raw_df['attention_score_norm_cellnum'] = attention_mtx_norm_cellnum.tolist()
attention_mtx_raw_df['attention_score_zscore_clip'] = attention_mtx_raw_df['attention_score_zscore'].apply(lambda x: min(x, 10))
attention_mtx_raw_df['attention_score_norm_cellnum_clip'] = attention_mtx_raw_df['attention_score_norm_cellnum'].apply(lambda x: min(x, 10))


hnoca.obs.index == metadata_query.index
hnoca.obs.index == metadata_tr.index   # false

attention_mtx_raw_df = attention_mtx_raw_df.loc[metadata_query.index]
attention_mtx_raw_df.to_csv("attention_mtx_HDBCA_model_HNOCA.csv")
metadata_tr = metadata_query.loc[attention_mtx_raw_df.index]
metadata_tr.to_csv("metadata_tr_HDBCA_model_HNOCA.csv")







