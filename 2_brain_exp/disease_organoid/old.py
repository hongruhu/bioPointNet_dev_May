# cd /group/gquongrp/workspaces/hongruhu/MIL/
# (scpair) hongruhu@gpu-5-50:/group/gquongrp/workspaces/hongruhu/MIL/$ python

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns
from scPointNet_Attn_Feb23 import *


obj = sc.read_h5ad('/group/gquongrp/workspaces/hongruhu/HNOCA/disease_atlas.h5ad')
sample_key = "sample"
obj.obs[sample_key].unique() # 89 samples

classification_keys = ["disease"]
obj.obs[classification_keys].value_counts()
# disease
# control                 140306
# ALS/FTD                  67840
# glioblastoma             59829
# ASD                      48133
# corrected                23195
# fragile_x                19052
# pitt_hopkins             14566
# schizophrenia            11829
# myotonic_dystrophy        9597
# microcephaly              5971
# neuronal_heterotopia      4544
# AD                        4415

# only keep control, ALS/FTD, glioblastoma, ASD
obj = obj[obj.obs.disease.isin(["control", "ALS/FTD", "glioblastoma", "ASD"])].copy()

categorical_covariate_keys = classification_keys + [sample_key]
# ['disease', 'sample']
sample_key = "sample"
obj.obs[sample_key].unique() # 64 samples



obj.obsm['X_scpoli_HNOCA'].shape  # 10
obj.obsm['X_pca'].shape           # 20
obj.obsm['X_HNOCA_Braun'].shape   # 30


adata = sc.AnnData(obj.obsm['X_HNOCA_Braun'], obs=obj.obs)

adata.obsm['X_umap'] = obj.obsm['X_umap']
adata.obsm['X_umap_raw'] = obj.obsm['X_umap_raw']
adata.obsm['X_umap_scpoli_HNOCA'] = obj.obsm['X_umap_scpoli_HNOCA']


metadata = adata.obs.copy()
metadata[classification_keys].value_counts().sort_index()
# disease
# ALS/FTD          67840
# ASD              48133
# control         140306
# glioblastoma     59829

Ns = adata.X.copy() # sparse matrix
Ns_df = pd.DataFrame(Ns, index=metadata.index.tolist(), columns=adata.var.index.tolist())
# rank the organoid_age_days from 0 by order
Ys_df = pd.get_dummies(metadata[classification_keys]).astype(int)
Xs, Ys, ins, meta_ids = Create_MIL_Dataset(ins_df=Ns_df, label_df=Ys_df, metadata=metadata, bag_column=sample_key)
print(f"Number of bags: {len(Xs)}", f"Number of labels: {Ys.shape}", f"Number of max instances: {max(ins)}")
# Number of bags: 64 Number of labels: (64, 4) Number of max instances: 26744


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create the dataset
mil_dataset = MILDataset(Xs, Ys, ins, meta_ids)
# Define the proportions for training and validation sets
train_size = int(0.8 * len(mil_dataset))
val_size = len(mil_dataset) - train_size
# set the seed
SEED = 15
# Split the dataset into training and validation sets
torch.manual_seed(SEED)
set_seed(SEED)
train_dataset, val_dataset = random_split(mil_dataset, [train_size, val_size])
# Print the sizes of the datasets
print(f"Training set size: {len(train_dataset)}", f"Validation set size: {len(val_dataset)}")
# Training set size: 51 Validation set size: 13





batch_size = 15
# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=MIL_Collate_fn)
val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False, collate_fn=MIL_Collate_fn)
set_seed(SEED)
pn_cls = PointNetClassHead(input_dim=Ns.shape[1], k=Ys.shape[1], global_features=256, attention_dim=64, agg_method="gated_attention")
pn_cls.apply(init_weights)
pn_cls.to(device)
learning_rate = 1e-4
optimizer = torch.optim.Adam(pn_cls.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
transformation_function = nn.Softmax(dim=1)

# criterion = OrdinalRegressionLoss(num_class=Ys_df.value_counts().size, train_cutpoints=True)
best_val_loss = float('inf')
patience_counter = 0
patience = 20
best_model_state = None
epochs = 200
for val_padded_bags, val_labels, val_lengths, val_id in val_loader:
        print(len(val_lengths))


def train_model(model):
    model.train()


def eval_model(model):
    model.eval()


for epoch in range(epochs):
    for padded_bags, labels, lengths, _ in train_loader:
        # train_model(pn_cls)
        optimizer.zero_grad()
        loss_tr = 0
        for idx in range(len(lengths)):
            length = lengths[idx]
            if length <= 1:
                continue
            else:
                input_tr = padded_bags[idx, :length,:].unsqueeze(0).permute(0, 2, 1)
                res_tr = pn_cls(input_tr)[0]
                pred_label = transformation_function(res_tr)
                true_label = labels[idx]
                loss_per_sample = criterion(pred_label, torch.argmax(true_label).view(-1))
                loss_tr += loss_per_sample.to(device)
        (loss_tr/len(lengths)).backward() if loss_tr > 1 else None
        optimizer.step() if loss_tr > 1 else None
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss_tr:.4f}")
    loss_val = 0
    # eval_model(pn_cls)
    for val_idx in range(len(val_lengths)):
        val_length = val_lengths[val_idx]
        if val_length <= 1:
            continue
        else:
            input_val = val_padded_bags[val_idx, :val_length,:].unsqueeze(0).permute(0, 2, 1)
            res_val = pn_cls(input_val)[0]
            pred_label_val = transformation_function(res_val)
            true_label_val = val_labels[val_idx]
            val_loss_per_sample = criterion(pred_label_val, torch.argmax(true_label_val).view(-1))
            loss_val += val_loss_per_sample
    loss_val_avg = loss_val/len(val_lengths)
    print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {loss_val_avg:.4f}")
    if loss_val_avg < best_val_loss:
        best_val_loss = loss_val_avg
        torch.save(pn_cls, "/group/gquongrp/workspaces/hongruhu/MIL/CT/disease_best_model_direct.pt")
        patience_counter = 0
        print(f"Saving the best model with validation loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break





pn_cls_checkpoint = torch.load("/group/gquongrp/workspaces/hongruhu/MIL/CT/disease_best_model_direct.pt")


# pn_cls_checkpoint.eval()
pred_label_val_list = []
true_label_val_list = []
for val_idx in range(len(val_lengths)):
        if val_lengths[val_idx] <= 1:
            continue
        else:
            val_length = val_lengths[val_idx]
            input_val = val_padded_bags[val_idx, :val_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_val, embedding_val, global_feature_val, attention_weights_val, _ = pn_cls_checkpoint(input_val)
            pred_label_val_list.append(transformation_function(cls_val.detach().cpu()).numpy())
            true_label_val_list.append(val_labels[val_idx].detach().cpu().numpy())


true_label_val_df = pd.DataFrame(np.vstack(true_label_val_list), columns=Ys_df.columns)
pred_label_val_df = pd.DataFrame(np.vstack(pred_label_val_list), columns=Ys_df.columns)


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
pn_cls_checkpoint.to(device)
for tr_idx in range(len(tr_lengths)):
        tr_length = tr_lengths[tr_idx]
        if tr_length <= 1:
            continue
        else:
            input_tr = tr_padded_bags[tr_idx, :tr_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_tr, embedding_tr, global_feature_tr, attn_weights_tr, _ = pn_cls_checkpoint(input_tr)
            embedding_tr_list.append(embedding_tr.squeeze(0).detach().cpu().numpy())
            global_feature_tr_list.append(global_feature_tr.squeeze(0).detach().cpu().numpy())
            pred_label_tr = cls_tr
            true_label_tr = tr_labels[tr_idx]
            pred_label_tr_list.append(transformation_function(pred_label_tr.detach().cpu()).numpy())
            true_label_tr_list.append(true_label_tr.detach().cpu().numpy())
            instance_level_list.append(input_tr.squeeze(0).permute(1,0).detach().cpu().numpy())
            attention_mtx_list.append(attn_weights_tr.squeeze(0,1).detach().cpu().numpy())
            cell_id_list.append(tr_id[tr_idx])


true_label_tr_df = pd.DataFrame(np.vstack(true_label_tr_list), columns=Ys_df.columns)
pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=Ys_df.columns)



fig = plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.heatmap(true_label_val_df, cmap="coolwarm", cbar=True)
plt.subplot(1, 2, 2)
sns.heatmap(pred_label_val_df, cmap="coolwarm", cbar=True)
plt.show()
fig.savefig("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_pred_label_val.png")

fig = plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.heatmap(true_label_tr_df, cmap="coolwarm", cbar=True)
plt.subplot(1, 2, 2)
sns.heatmap(pred_label_tr_df, cmap="coolwarm", cbar=True)
plt.show()
fig.savefig("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_pred_label_tr.png")



sorted_true_label_val_df = true_label_val_df.sort_values(by=true_label_val_df.columns.tolist(), axis=0)
sorted_pred_label_val_df = pred_label_val_df.loc[sorted_true_label_val_df.index]
sorted_true_label_tr_df = true_label_tr_df.sort_values(by=true_label_tr_df.columns.tolist(), axis=0)
sorted_pred_label_tr_df = pred_label_tr_df.loc[sorted_true_label_tr_df.index]

fig = plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.heatmap(sorted_true_label_val_df, cmap="viridis", cbar=True, vmin=0, vmax=1)
plt.subplot(1, 2, 2)
sns.heatmap(sorted_pred_label_val_df, cmap="viridis", cbar=True, vmin=0, vmax=1)
plt.show()
fig.savefig("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_pred_label_val_sorted.pdf")

fig = plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.heatmap(sorted_true_label_tr_df, cmap="viridis", cbar=True, vmin=0, vmax=1)
plt.subplot(1, 2, 2)
sns.heatmap(sorted_pred_label_tr_df, cmap="viridis", cbar=True, vmin=0, vmax=1)
plt.show()
fig.savefig("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_pred_label_tr_sorted.pdf")


# umap visualization
sc.pl.umap(adata, color=["disease", 'annot_level_1_plus', 'annot_level_2_plus', 'annot_region_rev_plus', 'annot_region_rev2_plus',], ncols=1, save="HNOCA_disease.png")



sorted_true_label_val_df
sorted_pred_label_val_df

# F1 score
from sklearn.metrics import f1_score
true_label_val_df = sorted_true_label_val_df.idxmax(axis=1)
pred_label_val_df = sorted_pred_label_val_df.idxmax(axis=1)
f1_score(true_label_val_df, pred_label_val_df, average='weighted') # 0.6080246913580247
# accuracy
from sklearn.metrics import accuracy_score
accuracy_score(true_label_val_df, pred_label_val_df) # 0.6666666666666666

# precision, recall, F1 score
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(true_label_val_df, pred_label_val_df, average='weighted')
# (0.5851851851851851, 0.6666666666666666, 0.6080246913580247, None)

# F1 score and other multiclassification metrics
from sklearn.metrics import classification_report
true_label_val_df = sorted_true_label_val_df.idxmax(axis=1)
pred_label_val_df = sorted_pred_label_val_df.idxmax(axis=1)

true_label_val_df = np.argmax(sorted_true_label_val_df.values,1)
pred_label_val_df = np.argmax(sorted_pred_label_val_df.values,1)
print(classification_report(true_label_val_df, pred_label_val_df, target_names=Ys_df.columns.tolist()))
#                       precision    recall  f1-score   support

#      disease_ALS/FTD       1.00      1.00      1.00         1
#          disease_ASD       1.00      1.00      1.00         6
#      disease_control       1.00      1.00      1.00         5
# disease_glioblastoma       1.00      1.00      1.00         1

#             accuracy                           1.00        13
#            macro avg       1.00      1.00      1.00        13
#         weighted avg       1.00      1.00      1.00        13

sorted_true_label_val_df.to_csv("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_true_label_val.csv")
sorted_pred_label_val_df.to_csv("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_pred_label_val.csv")
sorted_true_label_tr_df.to_csv("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_true_label_tr.csv")
sorted_pred_label_tr_df.to_csv("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_pred_label_tr.csv")




instance_mtx = np.vstack(instance_level_list) # (1306759, 1766)
sample_meta = [i[0] for i in cell_id_list]
cell_id_list_ = [i for sublist in cell_id_list for i in sublist]
cell_id_list_ == Ns_df.index.tolist() # False
cell_id_df = pd.DataFrame([i for i in cell_id_list_], columns=["cell_id"])
metadata_tr = metadata.loc[cell_id_df.cell_id]
sample_meta_tr = metadata_tr.loc[sample_meta]
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

adata.obs.index == metadata.index

adata = adata[metadata_tr.index]
adata.obs = metadata_tr.copy()
adata.obs = adata.obs.join(attention_mtx_raw_df)

sc.pl.umap(adata, color=["disease", 'annot_level_2_plus', 'attention_score_norm_cellnum_clip',], ncols=1, save="HNOCA_disease_attn.png")
adata.write("/group/gquongrp/workspaces/hongruhu/MIL/CT/HNOCA_disease_attn.h5ad")



embedding_tr_df = pd.DataFrame(np.vstack(embedding_tr_list))
embedding_tr_meta = pd.DataFrame(tr_labels)
embedding_tr_df.index = sample_meta_tr.index




# umap embedding
import umap
reducer = umap.UMAP(n_components=2, random_state=42)
embedding_tr_umap = reducer.fit_transform(embedding_tr_df)

fig = plt.figure(figsize=(8, 7))
sns.scatterplot(x=embedding_tr_umap[:, 0], y=embedding_tr_umap[:, 1], hue=sample_meta_tr.disease,
                s=78, alpha=0.78)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP for Sample-level Embedding")
# plt.legend([],[], frameon=False)
# legend on left side outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
sns.despine()
plt.show()
fig.savefig("/group/gquongrp/workspaces/hongruhu/MIL/CT/UMAP_Sample_level_embedding_Disease.pdf", dpi=150)


embedding_tr_df.to_csv("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_embedding_tr.csv")
sample_meta_tr.to_csv("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_embedding_tr_meta.csv")









train_loader_ = DataLoader(mil_dataset, batch_size=len(mil_dataset), shuffle=False, collate_fn=MIL_Collate_fn)
for tr_padded_bags, tr_labels, tr_lengths, tr_id in train_loader_:
        print(len(tr_lengths))



pred_label_tr_list = []
true_label_tr_list = []
attention_mtx_list = []
instance_level_list = []
cell_id_list = []
embedding_tr_list = []
global_feature_tr_list = []
pn_cls_checkpoint.to(device)
for tr_idx in range(len(tr_lengths)):
        tr_length = tr_lengths[tr_idx]
        if tr_length <= 1:
            continue
        else:
            input_tr = tr_padded_bags[tr_idx, :tr_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_tr, embedding_tr, global_feature_tr, attn_weights_tr, _ = pn_cls_checkpoint(input_tr)
            embedding_tr_list.append(embedding_tr.squeeze(0).detach().cpu().numpy())
            global_feature_tr_list.append(global_feature_tr.squeeze(0).detach().cpu().numpy())
            pred_label_tr = cls_tr
            true_label_tr = tr_labels[tr_idx]
            pred_label_tr_list.append(transformation_function(pred_label_tr.detach().cpu()).numpy())
            true_label_tr_list.append(true_label_tr.detach().cpu().numpy())
            instance_level_list.append(input_tr.squeeze(0).permute(1,0).detach().cpu().numpy())
            attention_mtx_list.append(attn_weights_tr.squeeze(0,1).detach().cpu().numpy())
            cell_id_list.append(tr_id[tr_idx])


true_label_tr_df = pd.DataFrame(np.vstack(true_label_tr_list), columns=Ys_df.columns)
pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=Ys_df.columns)



fig = plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.heatmap(true_label_val_df, cmap="coolwarm", cbar=True)
plt.subplot(1, 2, 2)
sns.heatmap(pred_label_val_df, cmap="coolwarm", cbar=True)
plt.show()
fig.savefig("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_pred_label_val.png")

fig = plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.heatmap(true_label_tr_df, cmap="coolwarm", cbar=True)
plt.subplot(1, 2, 2)
sns.heatmap(pred_label_tr_df, cmap="coolwarm", cbar=True)
plt.show()
fig.savefig("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_pred_label_tr.png")



sorted_true_label_val_df = true_label_val_df.sort_values(by=true_label_val_df.columns.tolist(), axis=0)
sorted_pred_label_val_df = pred_label_val_df.loc[sorted_true_label_val_df.index]
sorted_true_label_tr_df = true_label_tr_df.sort_values(by=true_label_tr_df.columns.tolist(), axis=0)
sorted_pred_label_tr_df = pred_label_tr_df.loc[sorted_true_label_tr_df.index]

fig = plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.heatmap(sorted_true_label_val_df, cmap="viridis", cbar=True, vmin=0, vmax=1)
plt.subplot(1, 2, 2)
sns.heatmap(sorted_pred_label_val_df, cmap="viridis", cbar=True, vmin=0, vmax=1)
plt.show()
fig.savefig("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_pred_label_val_sorted.pdf")

fig = plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.heatmap(sorted_true_label_tr_df, cmap="viridis", cbar=True, vmin=0, vmax=1)
plt.subplot(1, 2, 2)
sns.heatmap(sorted_pred_label_tr_df, cmap="viridis", cbar=True, vmin=0, vmax=1)
plt.show()
fig.savefig("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_pred_label_tr_sorted.pdf")


# umap visualization
sc.pl.umap(adata, color=["disease", 'annot_level_1_plus', 'annot_level_2_plus', 'annot_region_rev_plus', 'annot_region_rev2_plus',], ncols=1, save="HNOCA_disease.png")



sorted_true_label_val_df
sorted_pred_label_val_df

# F1 score
from sklearn.metrics import f1_score
true_label_val_df = sorted_true_label_val_df.idxmax(axis=1)
pred_label_val_df = sorted_pred_label_val_df.idxmax(axis=1)
f1_score(true_label_val_df, pred_label_val_df, average='weighted') # 0.6080246913580247
# accuracy
from sklearn.metrics import accuracy_score
accuracy_score(true_label_val_df, pred_label_val_df) # 0.6666666666666666

# precision, recall, F1 score
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(true_label_val_df, pred_label_val_df, average='weighted')
# (0.5851851851851851, 0.6666666666666666, 0.6080246913580247, None)

# F1 score and other multiclassification metrics
from sklearn.metrics import classification_report
true_label_val_df = sorted_true_label_val_df.idxmax(axis=1)
pred_label_val_df = sorted_pred_label_val_df.idxmax(axis=1)

true_label_val_df = np.argmax(sorted_true_label_val_df.values,1)
pred_label_val_df = np.argmax(sorted_pred_label_val_df.values,1)
print(classification_report(true_label_val_df, pred_label_val_df, target_names=Ys_df.columns.tolist()))
#                       precision    recall  f1-score   support

#      disease_ALS/FTD       1.00      1.00      1.00         1
#          disease_ASD       1.00      1.00      1.00         6
#      disease_control       1.00      1.00      1.00         5
# disease_glioblastoma       1.00      1.00      1.00         1

#             accuracy                           1.00        13
#            macro avg       1.00      1.00      1.00        13
#         weighted avg       1.00      1.00      1.00        13

sorted_true_label_val_df.to_csv("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_true_label_val.csv")
sorted_pred_label_val_df.to_csv("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_pred_label_val.csv")
sorted_true_label_tr_df.to_csv("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_true_label_tr.csv")
sorted_pred_label_tr_df.to_csv("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_pred_label_tr.csv")

sorted_true_label_val_df = pd.read_csv("/group/gquongrp/workspaces/hongruhu/MIL/CT/all_disease/Disease_true_label_val.csv", index_col=0)
sorted_pred_label_val_df = pd.read_csv("/group/gquongrp/workspaces/hongruhu/MIL/CT/all_disease/Disease_pred_label_val.csv", index_col=0)
sorted_true_label_tr_df = pd.read_csv("/group/gquongrp/workspaces/hongruhu/MIL/CT/all_disease/Disease_true_label_tr.csv", index_col=0)
sorted_pred_label_tr_df = pd.read_csv("/group/gquongrp/workspaces/hongruhu/MIL/CT/all_disease/Disease_pred_label_tr.csv", index_col=0)


# F1 score
from sklearn.metrics import f1_score
f1_score(sorted_true_label_val_df.idxmax(axis=1), sorted_pred_label_val_df.idxmax(axis=1), average='weighted') # 0.6080246913580247

f1_score(sorted_true_label_val_df, sorted_pred_label_val_df, average=None) # 0.6080246913580247

from sklearn.metrics import precision_recall_fscore_support

true_label_all_df = pd.concat([sorted_true_label_val_df, sorted_true_label_tr_df], axis=0)
pred_label_all_df = pd.concat([sorted_pred_label_val_df, sorted_pred_label_tr_df], axis=0)

precision_recall_fscore_support(true_label_all_df.idxmax(axis=1), pred_label_all_df.idxmax(axis=1), average='weighted')




instance_mtx = np.vstack(instance_level_list) # (1306759, 1766)
sample_meta = [i[0] for i in cell_id_list]
cell_id_list_ = [i for sublist in cell_id_list for i in sublist]
cell_id_list_ == Ns_df.index.tolist() # False
cell_id_df = pd.DataFrame([i for i in cell_id_list_], columns=["cell_id"])
metadata_tr = metadata.loc[cell_id_df.cell_id]
sample_meta_tr = metadata_tr.loc[sample_meta]
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

adata.obs.index == metadata.index

adata = adata[metadata_tr.index]
adata.obs = metadata_tr.copy()
adata.obs = adata.obs.join(attention_mtx_raw_df)

sc.pl.umap(adata, color=["disease", 'annot_level_2_plus', 'attention_score_norm_cellnum_clip',], ncols=1, save="HNOCA_disease_attn.png")
adata.write("/group/gquongrp/workspaces/hongruhu/MIL/CT/HNOCA_disease_attn.h5ad")



adata = sc.read("/group/gquongrp/workspaces/hongruhu/MIL/CT/top4_disease/HNOCA_disease_attn.h5ad")

sc.pl.umap(adata, color=["disease", 'annot_level_2_plus', 'attention_score_norm_cellnum_clip',], 
           ncols=1, save="HNOCA_disease_attn.png")


adata_control = adata[adata.obs.disease=='control']
adata_asd = adata[adata.obs.disease=='ASD']
adata_als = adata[adata.obs.disease=='ALS/FTD']
adata_g = adata[adata.obs.disease=='glioblastoma']

sc.pl.umap(adata_control, color=["disease", 'annot_level_2_plus', 'attention_score_norm_cellnum_clip',], 
           ncols=1, save="HNOCA_disease_attn_adata_control.png")


sc.pl.umap(adata_asd, color=["disease", 'annot_level_2_plus', 'attention_score_norm_cellnum_clip',], 
           ncols=1, save="HNOCA_disease_attn_adata_asd.png")

sc.pl.umap(adata_als, color=["disease", 'annot_level_2_plus', 'attention_score_norm_cellnum_clip',], 
           ncols=1, save="HNOCA_disease_attn_adata_als.png")

sc.pl.umap(adata_g, color=["disease", 'annot_level_2_plus', 'attention_score_norm_cellnum_clip',], 
           ncols=1, save="HNOCA_disease_attn_adata_g.png")




embedding_tr_df = pd.DataFrame(np.vstack(embedding_tr_list))
embedding_tr_meta = pd.DataFrame(tr_labels)
embedding_tr_df.index = sample_meta_tr.index




# umap embedding
import umap
reducer = umap.UMAP(n_components=2, random_state=42)
embedding_tr_umap = reducer.fit_transform(embedding_tr_df)

fig = plt.figure(figsize=(8, 7))
sns.scatterplot(x=embedding_tr_umap[:, 0], y=embedding_tr_umap[:, 1], hue=sample_meta_tr.disease,
                s=78, alpha=0.78)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP for Sample-level Embedding")
# plt.legend([],[], frameon=False)
# legend on left side outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
sns.despine()
plt.show()
fig.savefig("/group/gquongrp/workspaces/hongruhu/MIL/CT/UMAP_Sample_level_embedding_Disease_all.pdf", dpi=150)


embedding_tr_df.to_csv("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_embedding_all.csv")
sample_meta_tr.to_csv("/group/gquongrp/workspaces/hongruhu/MIL/CT/Disease_embedding_all_meta.csv")

# Index(['sample', 'age', 'sex', 'disease', 'condition', 'publication',
#        'full_sample', 'is_control', 'sc_method', 'batch_in_data'],
#       dtype='object')

fig = plt.figure(figsize=(8, 7))
sns.scatterplot(x=embedding_tr_umap[:, 0], y=embedding_tr_umap[:, 1], hue=sample_meta_tr.is_control,
                s=78, alpha=0.78)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP for Sample-level Embedding")
# plt.legend([],[], frameon=False)
# legend on left side outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
sns.despine()
plt.show()
fig.savefig("/group/gquongrp/workspaces/hongruhu/MIL/CT/UMAP_Sample_level_embedding_is_control.pdf", dpi=150)


fig = plt.figure(figsize=(8, 7))
sns.scatterplot(x=embedding_tr_umap[:, 0], y=embedding_tr_umap[:, 1], hue=sample_meta_tr.sc_method,
                s=78, alpha=0.78)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP for Sample-level Embedding")
# plt.legend([],[], frameon=False)
# legend on left side outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
sns.despine()
plt.show()
fig.savefig("/group/gquongrp/workspaces/hongruhu/MIL/CT/UMAP_Sample_level_embedding_sc_method.pdf", dpi=150)


fig = plt.figure(figsize=(8, 7))
sns.scatterplot(x=embedding_tr_umap[:, 0], y=embedding_tr_umap[:, 1], hue=sample_meta_tr.sex,
                s=78, alpha=0.78)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP for Sample-level Embedding")
# plt.legend([],[], frameon=False)
# legend on left side outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
sns.despine()
plt.show()
fig.savefig("/group/gquongrp/workspaces/hongruhu/MIL/CT/UMAP_Sample_level_embedding_sex.pdf", dpi=150)



fig = plt.figure(figsize=(8, 7))
sns.scatterplot(x=embedding_tr_umap[:, 0], y=embedding_tr_umap[:, 1], hue=sample_meta_tr.age,
                s=78, alpha=0.78)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP for Sample-level Embedding")
# plt.legend([],[], frameon=False)
# legend on left side outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
sns.despine()
plt.show()
fig.savefig("/group/gquongrp/workspaces/hongruhu/MIL/CT/UMAP_Sample_level_embedding_age.pdf", dpi=150)
