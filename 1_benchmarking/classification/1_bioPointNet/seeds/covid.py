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



path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/covid/'
adata = sc.read_h5ad(path + 'covid_Ziegler_processed.h5ad')
adata.obs.disease__ontology_label.value_counts()
# disease__ontology_label
# COVID-19               18073
# normal                  8874
# respiratory failure     3335
# long COVID-19           2306
adata.obs[['disease__ontology_label','donor_id']].value_counts().sort_index()
adata.obs['disease_id'] = 'control'
# COVID-19  and long COVID-19 to 'case'
adata.obs.loc[adata.obs['disease__ontology_label'] == 'COVID-19', 'disease_id'] = 'case'
adata.obs.loc[adata.obs['disease__ontology_label'] == 'long COVID-19', 'disease_id'] = 'case'
# adata.obs.loc[adata.obs['disease__ontology_label'] == 'respiratory failure', 'disease_id'] = 'case'
adata.obs.disease_id.value_counts()
# disease_id
# case       23714
# control     8874
adata_full = adata.copy()


task_name = 'covid_cls'

sample_key = 'donor_id'
task_key = 'disease_id'
class_num = 2
folds = pd.read_csv(path + 'covid_5folds.csv', index_col=0)



res_stat_dict = {}
for SEED in range(10):
    for fold in [1,2,3,4,5]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folds_trainval = folds[folds['fold'] != fold]
        folds_test = folds[folds['fold'] == fold]
        adata = adata_full[adata_full.obs[sample_key].isin(folds_trainval['samples'])]
        adata_test = adata_full[adata_full.obs[sample_key].isin(folds_test['samples'])]
        #
        metadata = adata.obs.copy()
        Ns = adata.X.copy() # sparse matrix
        Ns_df = pd.DataFrame(Ns.todense(), index=metadata.index.tolist(), columns=adata.var.index.tolist())
        Ys_df = pd.get_dummies(adata.obs[task_key].copy()).astype(int)
        Xs, Ys, ins, meta_ids = Create_MIL_Dataset(ins_df=Ns_df, label_df=Ys_df, metadata=metadata, bag_column=sample_key)
        print(f"Number of bags: {len(Xs)}", f"Number of labels: {Ys.shape}", f"Number of max instances: {max(ins)}")
        mil_dataset = MILDataset(Xs, Ys, ins, meta_ids)
        #
        metadata_test = adata_test.obs.copy()
        Ns_test = adata_test.X.copy() # sparse matrix
        Ns_df_test = pd.DataFrame(Ns_test.todense(), index=metadata_test.index.tolist(), columns=adata_test.var.index.tolist())
        Ys_df_test = pd.get_dummies(adata_test.obs[task_key].copy()).astype(int)
        Xs_test, Ys_test, ins_test, meta_ids_test = Create_MIL_Dataset(ins_df=Ns_df_test, label_df=Ys_df_test, metadata=metadata_test, bag_column=sample_key)
        print(f"Number of bags: {len(Xs_test)}", f"Number of labels: {Ys_test.shape}", f"Number of max instances: {max(ins_test)}")
        mil_dataset_test = MILDataset(Xs_test, Ys_test, ins_test, meta_ids_test)
        # Print the sizes of the datasets
        print(f"Training set size: {len(mil_dataset)}", f"Testing set size: {len(mil_dataset_test)}", '\n')
        train_size = int(0.8 * len(mil_dataset))
        val_size = len(mil_dataset) - train_size
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
        pn_cls = PointNetClassHead(input_dim=32, k=class_num, global_features=128, attention_dim=32, 
                                   agg_method="gated_attention")
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
                torch.save(pn_cls, "best_pn_" + task_name + '_' + str(fold) + '_seed_' + str(SEED) + ".pt")
                torch.save(enc, "best_enc_" + task_name + '_' + str(fold) + '_seed_' + str(SEED) + ".pt")
                patience_counter = 0
                print(f"Saving the best model with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        pn_cls_checkpoint = torch.load("best_pn_" + task_name + '_' + str(fold) + '_seed_' + str(SEED) + ".pt")
        enc_checkpoint = torch.load("best_enc_" + task_name + '_' + str(fold) + '_seed_' + str(SEED) + ".pt")
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
        true_label_val_df = pd.DataFrame(np.vstack(true_label_val_list), columns=Ys_df.columns)
        pred_label_val_df = pd.DataFrame(np.vstack(pred_label_val_list), columns=Ys_df.columns)
        # true_label_val_df = true_label_val_df.sort_values(by=["case"], ascending=False)
        # pred_label_val_df = pred_label_val_df.loc[true_label_val_df.index]
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(pred_label_val_df, cmap="coolwarm", cbar=True)
        # plt.show()
        # fig.savefig("pred_label_val_" + task_name + '_' + str(fold) + '_seed_' + str(SEED) +  ".png")
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(true_label_val_df, cmap="coolwarm", cbar=True)
        # plt.show()
        # fig.savefig("true_label_val_" + task_name + '_' + str(fold) + '_seed_' + str(SEED) +  ".png")
        # auROC and macro F1
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import f1_score
        auROC_val = roc_auc_score(true_label_val_df, pred_label_val_df, average='macro', multi_class='ovr')
        pred_label_val_df_ = pred_label_val_df.copy()
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
        train_loader_ = DataLoader(train_dataset, batch_size=train_size, shuffle=False, collate_fn=MIL_Collate_fn)
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
        true_label_tr_df = pd.DataFrame(np.vstack(true_label_tr_list), columns=Ys_df.columns)
        pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=Ys_df.columns)
        # true_label_tr_df = true_label_tr_df.sort_values(by=["case"], ascending=False)
        # pred_label_tr_df = pred_label_tr_df.loc[true_label_tr_df.index]
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(pred_label_tr_df, cmap="coolwarm", cbar=True)
        # plt.show()
        # fig.savefig("pred_label_tr_" + task_name + '_' + str(fold) + '_seed_' + str(SEED) +  ".png")
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(true_label_tr_df, cmap="coolwarm", cbar=True)
        # plt.show()
        # fig.savefig("true_label_tr_" + task_name + '_' + str(fold) + '_seed_' + str(SEED) +  ".png")
        auROC_tr = roc_auc_score(true_label_tr_df, pred_label_tr_df, average='macro', multi_class='ovr')
        pred_label_tr_df_ = pred_label_tr_df.copy()
        pred_label_tr_df_ = pred_label_tr_df_.apply(lambda x: x == x.max(), axis=1).astype(int)
        macroF1_tr = f1_score(true_label_tr_df, pred_label_tr_df_, average='macro')
        test_loader = DataLoader(mil_dataset_test, batch_size=len(mil_dataset_test), shuffle=False, collate_fn=MIL_Collate_fn)
        for te_padded_bags, te_labels, te_lengths, te_id in test_loader:
                print(len(te_lengths))
        pred_label_test_list = []
        true_label_test_list = []
        attention_mtx_list_test = []
        instance_level_list_test = []
        cell_id_list_test = []
        loss_test = 0
        for test_idx in range(len(te_lengths)):
                    test_length = te_lengths[test_idx]
                    if test_length == 1:
                        continue
                    elif test_length > 1:
                        input_test = te_padded_bags[test_idx, :test_length,:].unsqueeze(0).permute(0, 2, 1)
                        # cls_test, _, _, attn_weights, _ = pn_cls_checkpoint(input_test.to(device))
                        cls_test, _, _, attn_weights, _ = pn_cls_checkpoint(enc_checkpoint(input_test.squeeze(0).transpose(0,1), None).transpose(0,1).unsqueeze(0))
                        pred_label_test = transformation_function(cls_test)
                        true_label_test = te_labels[test_idx]
                        pred_label_test_list.append(pred_label_test.detach().cpu().numpy())
                        true_label_test_list.append(true_label_test.detach().cpu().numpy())
                        instance_level_list_test.append(input_test.squeeze(0).permute(1,0).detach().cpu().numpy())
                        attention_mtx_list_test.append(attn_weights.squeeze(0,1).detach().cpu().numpy())
                        loss_per_sample = criterion(pred_label_test, torch.max(true_label_test.to(device).reshape(-1,class_num),1)[1])
                        loss_test += loss_per_sample
                        cell_id_list_test.append(te_id[test_idx])
        true_label_test_df = pd.DataFrame(np.vstack(true_label_test_list), columns=Ys_df.columns)
        pred_label_test_df = pd.DataFrame(np.vstack(pred_label_test_list), columns=Ys_df.columns)
        # true_label_test_df = true_label_test_df.sort_values(by=["case"], ascending=False)
        # pred_label_test_df = pred_label_test_df.loc[true_label_test_df.index]
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(pred_label_test_df, cmap="coolwarm", cbar=True)
        # plt.show()
        # fig.savefig("pred_label_test_" + task_name + '_' + str(fold) + '_seed_' + str(SEED) +  ".png")
        # fig = plt.figure(figsize=(10, 10))
        # sns.heatmap(true_label_test_df, cmap="coolwarm", cbar=True)
        # plt.show()
        # fig.savefig("true_label_test_" + task_name + '_' + str(fold) + '_seed_' + str(SEED) +  ".png")
        auROC_test = roc_auc_score(true_label_test_df, pred_label_test_df, average='macro', multi_class='ovr')
        pred_label_test_df_ = pred_label_test_df.copy()
        pred_label_test_df_ = pred_label_test_df_.apply(lambda x: x == x.max(), axis=1).astype(int)
        macroF1_test = f1_score(true_label_test_df, pred_label_test_df_, average='macro')
        print(sample_key, '\n',
            "auROC_tr", auROC_tr,'\n',
            "auROC_val", auROC_val,'\n',
            "macroF1_tr", macroF1_tr,'\n',
            "macroF1_val", macroF1_val,'\n',
            "auROC_test", auROC_test,'\n',
            "macroF1_test", macroF1_test,'\n',
            "fold", fold, '\n',
            'seed', SEED
            )
        res_df = pd.DataFrame(
            {
                'sample_key': sample_key,
                'auROC_tr': auROC_tr,
                'macroF1_tr': macroF1_tr,
                'auROC_val': auROC_val,
                'macroF1_val': macroF1_val,
                'auROC_test': auROC_test,
                'macroF1_test': macroF1_test,
                'fold': fold
            }, index=[0]
        )
        res_df = res_df.T
        res_df.columns = [task_name]
        res_df.to_csv(task_name + '_results_fold_' + str(fold) + '_seed_' + str(SEED) +  '.csv')
        res_stat_dict['fold_' + str(fold) + '_seed_' + str(SEED)] = res_df



torch.save(res_stat_dict, task_name + '_results_all_folds.pth')



res_stat_dict.keys()
dict_keys(['fold_1_seed_0', 'fold_2_seed_0', 'fold_3_seed_0', 'fold_4_seed_0', 'fold_5_seed_0', 
           'fold_1_seed_1', 'fold_2_seed_1', 'fold_3_seed_1', 'fold_4_seed_1', 'fold_5_seed_1', 
           'fold_1_seed_2', 'fold_2_seed_2', 'fold_3_seed_2', 'fold_4_seed_2', 'fold_5_seed_2', 
           'fold_1_seed_3', 'fold_2_seed_3', 'fold_3_seed_3', 'fold_4_seed_3', 'fold_5_seed_3', 
           'fold_1_seed_4', 'fold_2_seed_4', 'fold_3_seed_4', 'fold_4_seed_4', 'fold_5_seed_4', 
           'fold_1_seed_5', 'fold_2_seed_5', 'fold_3_seed_5', 'fold_4_seed_5', 'fold_5_seed_5', 
           'fold_1_seed_6', 'fold_2_seed_6', 'fold_3_seed_6', 'fold_4_seed_6', 'fold_5_seed_6', 
           'fold_1_seed_7', 'fold_2_seed_7', 'fold_3_seed_7', 'fold_4_seed_7', 'fold_5_seed_7', 
           'fold_1_seed_8', 'fold_2_seed_8', 'fold_3_seed_8', 'fold_4_seed_8', 'fold_5_seed_8', 
           'fold_1_seed_9', 'fold_2_seed_9', 'fold_3_seed_9', 'fold_4_seed_9', 'fold_5_seed_9'])


res_test_df = pd.DataFrame(index=list(res_stat_dict.keys()), 
                           columns=['auROC_test','macroF1_test'])


for i in list(res_stat_dict.keys()):
     print(i)
     res_test_df.loc[i,'auROC_test'] = res_stat_dict[i].loc['auROC_test'].values
     res_test_df.loc[i,'macroF1_test'] = res_stat_dict[i].loc['macroF1_test'].values


res_test_df.mean()
# auROC_test      0.703321
# macroF1_test    0.576133


for seed in range(10):
    print(seed)
    res_test_df[res_test_df.index.str.contains('seed_' + str(seed))].mean()
    print('\n')


# 0
# auROC_test      0.709107
# macroF1_test    0.666087
# dtype: object


# 1
# auROC_test      0.763929
# macroF1_test    0.635556
# dtype: object


# 2
# auROC_test      0.780357
# macroF1_test    0.609319
# dtype: object


# 3
# auROC_test      0.644107
# macroF1_test    0.472657
# dtype: object


# 4
# auROC_test      0.650179
# macroF1_test    0.597744
# dtype: object


# 5
# auROC_test      0.608929
# macroF1_test    0.402856
# dtype: object


# 6
# auROC_test      0.693214
# macroF1_test    0.617718
# dtype: object


# 7
# auROC_test        0.6825
# macroF1_test    0.566223
# dtype: object


# 8
# auROC_test      0.699107
# macroF1_test     0.57662
# dtype: object


# 9
# auROC_test      0.801786
# macroF1_test     0.61655
# dtype: object


for fold in [1,2,3,4,5]:
    res_test_df[res_test_df.index.str.contains('fold_' + str(fold))].mean()
    print('\n')


# auROC_test       0.71875
# macroF1_test    0.603626
# dtype: object


# auROC_test         0.825
# macroF1_test    0.720957
# dtype: object


# auROC_test      0.694286
# macroF1_test    0.519653
# dtype: object


# auROC_test         0.675
# macroF1_test    0.592527
# dtype: object


# auROC_test      0.603571
# macroF1_test    0.443902
# dtype: object