# (mil) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet$ python
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns

import scvi
import multimil as mil

from sciLaMA import *
from bioPointNet_Apr2025 import *



path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/cardio/'
adata = sc.read_h5ad(path + 'cardio_processed.h5ad')
# AnnData object with n_obs × n_vars = 592689 × 2000
#     obs: 'biosample_id', 'donor_id', 'disease', 'sex', 'age', 'lvef', 'cell_type_leiden0.6', 'SubCluster', 'cellbender_ncount', 'cellbender_ngenes', 'cellranger_percent_mito', 'exon_prop', 'cellbender_entropy', 'cellranger_doublet_scores'
#     var: 'gene_ids', 'feature_types', 'genome'
#     obsm: 'X_umap'
#     layers: 'cellranger_raw'

adata.obs.disease.value_counts()
# disease
# HCM    235252
# NF     185441
# DCM    171996

adata.obs.biosample_id.value_counts() # 80
adata.obs.donor_id.value_counts()     # 42

adata.obs[['disease','donor_id']].value_counts().sort_index()
# DCM      P1290        8357
#          P1300       17318
#          P1304       19123
#          P1358       23984
#          P1371       16233
#          P1430       11131
#          P1437       21695
#          P1472       19513
#          P1504        8133
#          P1606        8523
#          P1617       17986
# HCM      P1422       23315
#          P1425       15378
#          P1447       11151
#          P1462       21715
#          P1479       19124
#          P1508       20536
#          P1510       10744
#          P1602       16580
#          P1630       17633
#          P1631       13686
#          P1685       10043
#          P1707        9517
#          P1722       21432
#          P1726       12389
#          P1735       12009
# NF       P1515       14502
#          P1516        9361
#          P1539       11076
#          P1540       11638
#          P1547        8253
#          P1549       11709
#          P1558       10469
#          P1561       10016
#          P1582       18855
#          P1600       14882
#          P1603       10638
#          P1610       13919
#          P1622        7210
#          P1678       10085
#          P1702       13550
#          P1718        9278
# DCM 11 | HCM 15 | NF 16


class_num = 3

metadata = adata.obs.copy()
sample_key = 'donor_id'
sample_key = 'biosample_id'

Ns = adata.X.copy() # sparse matrix
Ns_df = pd.DataFrame(Ns.todense(), index=metadata.index.tolist(), columns=adata.var.index.tolist())
Ys_df = pd.get_dummies(adata.obs['disease'].copy()).astype(int)
Xs, Ys, ins, meta_ids = Create_MIL_Dataset(ins_df=Ns_df, label_df=Ys_df, metadata=metadata, bag_column=sample_key)
print(f"Number of bags: {len(Xs)}", f"Number of labels: {Ys.shape}", f"Number of max instances: {max(ins)}")
# Number of bags: 42 Number of labels: (42, 3) Number of max instances: 23984


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
batch_size = 5
# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=MIL_Collate_fn)
val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False, collate_fn=MIL_Collate_fn)





set_seed(SEED)

enc = CellEncoder(input_dim=Ns.shape[1], input_batch_num=0, hidden_layer=[256, 32], 
                  layernorm=True, activation=nn.ReLU(), batchnorm=False, dropout_rate=0.1, 
                  add_linear_layer=True, clip_threshold=None)
enc.apply(init_weights)
enc.to(device)
pn_cls = PointNetClassHead(input_dim=32, k=class_num, global_features=128, attention_dim=32, agg_method="gated_attention")
pn_cls.apply(init_weights)
pn_cls.to(device)


learning_rate = 1e-4
# optimizer = torch.optim.Adam(pn_cls.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(list(pn_cls.parameters())+list(enc.parameters()), lr=learning_rate)

criterion = nn.CrossEntropyLoss()
# criterion = OrdinalRegressionLoss(num_class=Ys_df.value_counts().size, train_cutpoints=True)
transformation_function = nn.Softmax(dim=1)
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



import copy
for epoch in range(epochs):
    for padded_bags, labels, lengths, _ in train_loader:
        # train_model(pn_cls)
        optimizer.zero_grad()
        loss_tr = 0
        for idx in range(len(lengths)):
            length = lengths[idx]
            if length == 1:
                continue
            elif length > 1:
                input_tr = padded_bags[idx, :length,:].unsqueeze(0).permute(0, 2, 1)
                # cls, embedding, global_features, attn_weights, crit_idxs = pn_cls(input_tr)
                cls, embedding, global_features, attn_weights, crit_idxs = pn_cls(enc(input_tr.squeeze(0).transpose(0,1), None).transpose(0,1).unsqueeze(0))
                pred_label = transformation_function(cls)
                true_label = labels[idx]
                loss_per_sample = criterion(pred_label, torch.max(true_label.reshape(-1,class_num),1)[1]) # equivalent to the following
                loss_tr += loss_per_sample
        (loss_tr/len(lengths)).backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss_tr:.4f}")
    loss_val = 0
    # eval_model(pn_cls)
    for val_idx in range(len(val_lengths)):
        val_length = val_lengths[val_idx]
        if val_length == 1:
            continue
        elif val_length > 1:
            input_val = val_padded_bags[val_idx, :val_length,:].unsqueeze(0).permute(0, 2, 1)
            # cls_val, _, _, _, _ = pn_cls(input_val)
            cls_val, _, _, _, _ = pn_cls(enc(input_val.squeeze(0).transpose(0,1), None).transpose(0,1).unsqueeze(0))
            pred_label_val = transformation_function(cls_val)
            true_label_val = val_labels[val_idx]
            val_loss_per_sample = criterion(pred_label_val, torch.max(true_label_val.reshape(-1,class_num),1)[1])
            loss_val += val_loss_per_sample
    loss_val_avg = loss_val/len(val_lengths)
    print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {loss_val_avg:.4f}")
    if loss_val_avg < best_val_loss:
        best_val_loss = loss_val_avg
        torch.save(pn_cls, "best_model_cardio_.pt")
        torch.save(enc, "best_enc_cardio_.pt")
        patience_counter = 0
        print(f"Saving the best model with validation loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break




pn_cls_checkpoint = torch.load("best_model_cardio_.pt")
enc_checkpoint = torch.load("best_enc_cardio_.pt")
# pn_cls_checkpoint.eval()
pred_label_val_list = []
true_label_val_list = []
loss_val = 0
for val_idx in range(len(val_lengths)):
        val_length = val_lengths[val_idx]
        if val_length == 1:
            continue
        if val_length > 1:
            input_val = val_padded_bags[val_idx, :val_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_val, _, _, _, _ = pn_cls_checkpoint(enc(input_val.squeeze(0).transpose(0,1), None).transpose(0,1).unsqueeze(0))
            pred_label_val = transformation_function(cls_val)
            true_label_val = val_labels[val_idx]
            pred_label_val_list.append(pred_label_val.detach().cpu().numpy())
            true_label_val_list.append(true_label_val.detach().cpu().numpy())
            val_loss_per_sample = criterion(pred_label_val, torch.max(true_label_val.reshape(-1,class_num),1)[1])
            loss_val += val_loss_per_sample


loss_val/len(val_lengths) # 0.7001
true_label_val_df = pd.DataFrame(np.vstack(true_label_val_list), columns=Ys_df.columns)
pred_label_val_df = pd.DataFrame(np.vstack(pred_label_val_list), columns=Ys_df.columns)

true_label_val_df = true_label_val_df.sort_values(by=["DCM"], ascending=False)
pred_label_val_df = pred_label_val_df.loc[true_label_val_df.index]
fig = plt.figure(figsize=(10, 10))
sns.heatmap(pred_label_val_df, cmap="coolwarm", cbar=True)
plt.show()
fig.savefig("pred_label_val_cardio_.png")
fig = plt.figure(figsize=(10, 10))
sns.heatmap(true_label_val_df, cmap="coolwarm", cbar=True)
plt.show()
fig.savefig("true_label_val_cardio_.png")



# auROC and macro F1
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
auROC_val = roc_auc_score(true_label_val_df, pred_label_val_df, average='macro', multi_class='ovr')
# 1

# macro F1
pred_label_val_df_ = pred_label_val_df.copy()
# make the largetest value as 1 and the rest as 0 for each row
pred_label_val_df_ = pred_label_val_df_.apply(lambda x: x == x.max(), axis=1).astype(int)
macroF1_val = f1_score(true_label_val_df, pred_label_val_df_, average='macro')
# 1





device = 'cpu'
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


train_loader_ = DataLoader(train_dataset, batch_size=train_size, shuffle=True, collate_fn=MIL_Collate_fn)
for tr_padded_bags, tr_labels, tr_lengths, tr_id in train_loader_:
        print(len(tr_lengths))


pn_cls_checkpoint=pn_cls_checkpoint.to(device)
enc_checkpoint=enc_checkpoint.to(device)
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
            # cls_tr, _, _, attn_weights, _ = pn_cls_checkpoint(input_tr.to(device))
            cls_tr, _, _, attn_weights, _ = pn_cls_checkpoint(enc_checkpoint(input_tr.squeeze(0).transpose(0,1), None).transpose(0,1).unsqueeze(0))
            pred_label_tr = transformation_function(cls_tr)
            true_label_tr = tr_labels[tr_idx]
            pred_label_tr_list.append(pred_label_tr.detach().cpu().numpy())
            true_label_tr_list.append(true_label_tr.detach().cpu().numpy())
            instance_level_list.append(input_tr.squeeze(0).permute(1,0).detach().cpu().numpy())
            attention_mtx_list.append(attn_weights.squeeze(0,1).detach().cpu().numpy())
            loss_per_sample = criterion(pred_label_tr, torch.max(true_label_tr.to(device).reshape(-1,class_num),1)[1])
            loss_tr += loss_per_sample
            cell_id_list.append(tr_id[tr_idx])


loss_tr/len(tr_lengths) # 0.3402

true_label_tr_df = pd.DataFrame(np.vstack(true_label_tr_list), columns=Ys_df.columns)
pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=Ys_df.columns)
true_label_tr_df = true_label_tr_df.sort_values(by=["DCM"], ascending=False)
pred_label_tr_df = pred_label_tr_df.loc[true_label_tr_df.index]

fig = plt.figure(figsize=(10, 10))
sns.heatmap(pred_label_tr_df, cmap="coolwarm", cbar=True)
plt.show()
fig.savefig("pred_label_tr_cardio_.png")
fig = plt.figure(figsize=(10, 10))
sns.heatmap(true_label_tr_df, cmap="coolwarm", cbar=True)
plt.show()
fig.savefig("true_label_tr_cardio_.png")


auROC_tr = roc_auc_score(true_label_tr_df, pred_label_tr_df, average='macro', multi_class='ovr')
# 1.0
pred_label_tr_df_ = pred_label_tr_df.copy()
pred_label_tr_df_ = pred_label_tr_df_.apply(lambda x: x == x.max(), axis=1).astype(int)
macroF1_tr = f1_score(true_label_tr_df, pred_label_tr_df_, average='macro')
# 1.0





print(sample_key, '\n',
      "auROC_tr", auROC_tr,'\n',
      "auROC_val", auROC_val,'\n',
      "macroF1_tr", macroF1_tr,'\n',
      "macroF1_val", macroF1_val)

# donor_id
#  auROC_tr 1.0
#  auROC_val 1.0
#  macroF1_tr 1.0
#  macroF1_val 1.0


# biosample_id
#  auROC_tr 1.0
#  auROC_val 1.0
#  macroF1_tr 1.0
#  macroF1_val 1.0