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



path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/'
adata = sc.read_h5ad(path + 'MTG_glial.h5ad')
# AnnData object with n_obs × n_vars = 260066 × 33222
#     obs: 'assay_ontology_term_id', 'cell_type_ontology_term_id', 'disease_ontology_term_id', 'self_reported_ethnicity_ontology_term_id', 'organism_ontology_term_id', 'sex_ontology_term_id', 'tissue_ontology_term_id', 'is_primary_data', 'Neurotypical reference', 
#           'Class', 'Subclass', 'Supertype', 'Age at death', 'Years of education', 'Cognitive status', 'ADNC', 'Braak stage', 'Thal phase', 'CERAD score', 'APOE4 status', 'Lewy body disease pathology', 'LATE-NC stage', 'Microinfarct pathology', 
#           'Specimen ID', 'donor_id', 'PMI', 'Number of UMIs', 'Genes detected', 'Fraction mitochrondrial UMIs', 'suspension_type', 'development_stage_ontology_term_id', 'Continuous Pseudo-progression Score', 'tissue_type', 
#           'cell_type', 'assay', 'disease', 'organism', 'sex', 'tissue', 'self_reported_ethnicity', 'development_stage', 'observation_joinid'
#     var: 'feature_is_filtered', 'feature_name', 'feature_reference', 'feature_biotype', 'feature_length', 'feature_type', 'n_cells', 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm'
#     uns: 'ADNC_colors', 'APOE4 status_colors', 'Age at death_colors', 'Braak stage_colors', 'CERAD score_colors', 'Cognitive status_colors', 'Great Apes Metadata', 'LATE-NC stage_colors', 'Lewy body disease pathology_colors', 'Microinfarct pathology_colors', 'PMI_colors', 'Subclass_colors', 'Supertype_colors', 'Thal phase_colors', 'UW Clinical Metadata', 'Years of education_colors', 'batch_condition', 'citation', 'default_embedding', 'hvg', 'neighbors', 'schema_reference', 'schema_version', 'title', 'umap'
#     obsm: 'X_scVI', 'X_umap'
#     obsp: 'connectivities', 'distances'


sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat_v3')
sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
adata = adata[:, adata.var.highly_variable]
# View of AnnData object with n_obs × n_vars = 260066 × 3000


adata.obs.disease.value_counts()
# disease
# normal      146067
# dementia    113999

adata.obs.donor_id.value_counts() # 87


adata.obs.sex.value_counts()
# sex
# female    147575
# male      112491

class_num = 2

metadata = adata.obs.copy()
sample_key = 'donor_id'
task_key = 'disease'
cov_key = 'sex'

Ns = adata.X.copy() # sparse matrix
Ns_df = pd.DataFrame(Ns.todense(), index=metadata.index.tolist(), columns=adata.var.index.tolist())
Ys_df = pd.get_dummies(adata.obs[task_key].copy()).astype(int)
Cs_df = pd.get_dummies(adata.obs[cov_key].copy()).astype(int)

Ns_df = pd.concat([Ns_df, Cs_df], axis=1)
Xs, Ys, ins, meta_ids = Create_MIL_Dataset(ins_df=Ns_df, label_df=Ys_df, metadata=metadata, bag_column=sample_key)
print(f"Number of bags: {len(Xs)}", f"Number of labels: {Ys.shape}", f"Number of max instances: {max(ins)}")
# Number of bags: 87 Number of labels: (87, 2) Number of max instances: 6755



# Create the dataset
mil_dataset = MILDataset(Xs, Ys, ins, meta_ids)
# Define the proportions for training and validation sets
train_size = int(0.8 * len(mil_dataset))
val_size = len(mil_dataset) - train_size
# set the seed
SEED = 0
# Split the dataset into training and validation sets
torch.manual_seed(SEED)
set_seed(SEED)
train_dataset, val_dataset = random_split(mil_dataset, [train_size, val_size])
# Print the sizes of the datasets
print(f"Training set size: {len(train_dataset)}", f"Validation set size: {len(val_dataset)}")
batch_size = 15
# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=MIL_Collate_fn)
val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False, collate_fn=MIL_Collate_fn)





set_seed(SEED)
enc = CellEncoder(input_dim=Ns_df.shape[1], input_batch_num=0, hidden_layer=[256, 32], 
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
        torch.save(pn_cls, "best_model_seaadMTG.pt")
        torch.save(enc, "best_enc_seaadMTG.pt")
        patience_counter = 0
        print(f"Saving the best model with validation loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break




pn_cls_checkpoint = torch.load("best_model_seaadMTG.pt")
enc_checkpoint = torch.load("best_enc_seaadMTG.pt")
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


loss_val/len(val_lengths) # 0.5913
true_label_val_df = pd.DataFrame(np.vstack(true_label_val_list), columns=Ys_df.columns)
pred_label_val_df = pd.DataFrame(np.vstack(pred_label_val_list), columns=Ys_df.columns)

true_label_val_df = true_label_val_df.sort_values(by=["normal"], ascending=False)
pred_label_val_df = pred_label_val_df.loc[true_label_val_df.index]
fig = plt.figure(figsize=(10, 10))
sns.heatmap(pred_label_val_df, cmap="coolwarm", cbar=True)
plt.show()
fig.savefig("pred_label_val_seaadMTG.png")
fig = plt.figure(figsize=(10, 10))
sns.heatmap(true_label_val_df, cmap="coolwarm", cbar=True)
plt.show()
fig.savefig("true_label_val_seaadMTG.png")



# auROC and macro F1
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
auROC_val = roc_auc_score(true_label_val_df, pred_label_val_df, average='macro', multi_class='ovr')
# macro F1
pred_label_val_df_ = pred_label_val_df.copy()
# make the largetest value as 1 and the rest as 0 for each row
pred_label_val_df_ = pred_label_val_df_.apply(lambda x: x == x.max(), axis=1).astype(int)
macroF1_val = f1_score(true_label_val_df, pred_label_val_df_, average='macro')


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


loss_tr/len(tr_lengths) # 

true_label_tr_df = pd.DataFrame(np.vstack(true_label_tr_list), columns=Ys_df.columns)
pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=Ys_df.columns)
true_label_tr_df = true_label_tr_df.sort_values(by=["normal"], ascending=False)
pred_label_tr_df = pred_label_tr_df.loc[true_label_tr_df.index]

fig = plt.figure(figsize=(10, 10))
sns.heatmap(pred_label_tr_df, cmap="coolwarm", cbar=True)
plt.show()
fig.savefig("pred_label_tr_seaadMTG.png")
fig = plt.figure(figsize=(10, 10))
sns.heatmap(true_label_tr_df, cmap="coolwarm", cbar=True)
plt.show()
fig.savefig("true_label_tr_seaadMTG.png")


auROC_tr = roc_auc_score(true_label_tr_df, pred_label_tr_df, average='macro', multi_class='ovr')
pred_label_tr_df_ = pred_label_tr_df.copy()
pred_label_tr_df_ = pred_label_tr_df_.apply(lambda x: x == x.max(), axis=1).astype(int)
macroF1_tr = f1_score(true_label_tr_df, pred_label_tr_df_, average='macro')



print(sample_key, '\n',
      "auROC_tr", auROC_tr,'\n',
      "auROC_val", auROC_val,'\n',
      "macroF1_tr", macroF1_tr,'\n',
      "macroF1_val", macroF1_val)


# donor_id
#  auROC_tr 0.9941077441077442
#  auROC_val 0.8271604938271605
#  macroF1_tr 0.970959595959596
#  macroF1_val 0.6989966555183946

# donor_id
#  auROC_tr 0.978956228956229
#  auROC_val 0.8395061728395061
#  macroF1_tr 0.9414261460101867
#  macroF1_val 0.775



len(attention_mtx_list)     # 279
len(instance_level_list)    # 279

instance_mtx = np.vstack(instance_level_list) # (1306759, 1766)
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

metadata_tr.index == attention_mtx_raw_df.index



metadata_tr = pd.concat([metadata_tr,attention_mtx_raw_df], axis=1)


metadata_tr.sort_values(by='attention_score_norm_cellnum',ascending=False)


metadata_tr.sort_values(by='attention_score_norm_cellnum',ascending=False)[0:1000].cell_type.value_counts()
# cell_type
# microglial cell                     585
# astrocyte of the cerebral cortex    383
# oligodendrocyte precursor cell       15
# oligodendrocyte                      10
# vascular leptomeningeal cell          7
# cerebral cortex endothelial cell      0


# cell_type
# oligodendrocyte                     954
# astrocyte of the cerebral cortex     21
# oligodendrocyte precursor cell       14
# cerebral cortex endothelial cell      8
# vascular leptomeningeal cell          3
# microglial cell                       0


metadata_tr.sort_values(by='attention_score_norm_cellnum',ascending=False)[0:1000].Supertype.value_counts()
# Supertype
# Micro-PVM_2            389
# Astro_2                150
# Micro-PVM_2_3-SEAAD    128
# Astro_3                118
# Micro-PVM_3-SEAAD       39
# Astro_5                 36
# Astro_1                 34
# Astro_6-SEAAD           32
# Micro-PVM_1             19
# Astro_4                 13
# OPC_2_2-SEAAD            9
# OPC_2                    6
# Oligo_2                  6
# Micro-PVM_2_1-SEAAD      5
# VLMC_2                   4
# Oligo_4                  4
# Micro-PVM_4-SEAAD        2
# Micro-PVM_2_2-SEAAD      2
# VLMC_1                   1
# VLMC_2_1-SEAAD           1
# VLMC_2_2-SEAAD           1
# Micro-PVM_1_1-SEAAD      1
# Endo_3                   0
# Endo_2                   0
# Oligo_3                  0
# Oligo_2_1-SEAAD          0
# Oligo_1                  0
# OPC_2_1-SEAAD            0
# OPC_1                    0
# Endo_1                   0


# Supertype
# Oligo_4                501
# Oligo_2                339
# Oligo_2_1-SEAAD         71
# Oligo_1                 34
# Astro_2                 19
# Oligo_3                  9
# Endo_2                   8
# OPC_2                    8
# OPC_2_1-SEAAD            4
# VLMC_2                   2
# OPC_1                    2
# Astro_6-SEAAD            1
# VLMC_1                   1
# Astro_1                  1
# Micro-PVM_1_1-SEAAD      0
# Micro-PVM_2              0
# Micro-PVM_2_3-SEAAD      0
# Micro-PVM_3-SEAAD        0
# Micro-PVM_4-SEAAD        0
# Micro-PVM_2_1-SEAAD      0
# Micro-PVM_1              0
# Endo_1                   0
# VLMC_2_2-SEAAD           0
# VLMC_2_1-SEAAD           0
# Endo_3                   0
# OPC_2_2-SEAAD            0
# Astro_4                  0
# Astro_3                  0
# Astro_5                  0
# Micro-PVM_2_2-SEAAD      0