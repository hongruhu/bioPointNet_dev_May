# (mil) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet$ python
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns
import os
from bioPointNet_Apr2025 import *


data_source = 'DLPFC'

sample_key = 'donor_id'
task_key = 'disease'
cov_key = 'sex'
ct_key = 'Subclass'
ct_key_fine = 'Supertype'
class_num = 2

path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/SEAAD_Glia/'


# all glial types
metadata_all = pd.read_csv(path + data_source + '/' + data_source + '_glia_metadata.csv', index_col=0)


for ct in list(metadata_all[ct_key].unique()):
    print(ct)
    saving_path = path + data_source + '/' + data_source + '_cls/' + ct + '/'
    if os.path.isdir(saving_path):
        print("Directory exists.")
    else:
        os.mkdir(saving_path)
        print("Creating directory.")
    metadata = metadata_all[metadata_all[ct_key] == ct].copy()   
    cell_mtx = pd.read_csv(path + data_source + '/' + ct + '_' + data_source + '_sciMultiLaMA_CELL_embedding.csv', index_col=0)
    metadata = metadata.loc[cell_mtx.index.tolist()]
    adata = sc.AnnData(X=cell_mtx.values, obs=metadata.loc[cell_mtx.index.tolist()])
    Ns = adata.X.copy()
    Ns_df = pd.DataFrame(Ns, index=metadata.index.tolist())
    Ys_df = pd.get_dummies(adata.obs[task_key].copy()).astype(int)
    # Cs_df = pd.get_dummies(adata.obs[cov_key].copy()).astype(int)
    # Ns_df = pd.concat([Ns_df, Cs_df], axis=1)
    Xs, Ys, ins, meta_ids = Create_MIL_Dataset(ins_df=Ns_df, label_df=Ys_df, metadata=metadata, bag_column=sample_key)
    print(f"Number of bags: {len(Xs)}", f"Number of labels: {Ys.shape}", f"Number of max instances: {max(ins)}")
    # Number of bags: 87 Number of labels: (87, 2) Number of max instances: 6473
    # Create the dataset
    mil_dataset = MILDataset(Xs, Ys, ins, meta_ids)
    # Define the proportions for training and validation sets
    train_size = int(0.8 * len(mil_dataset))
    val_size = len(mil_dataset) - train_size
    for SEED in range(5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        def MIL_Collate_fn(batch):
            bags, labels, ins, metadata_idx= zip(*batch)
            lengths = ins
            max_length = max(ins)                            # Maximum number of instances in the batch
            # Pad bags to the same length
            padded_bags = torch.zeros((len(bags), max_length, bags[0].shape[1]), dtype=torch.float32)
            for i, bag in enumerate(bags):
                padded_bags[i, :len(bag)] = bag
            labels = torch.stack(labels)        # Stack labels into a single tensor
            return padded_bags.to(device), labels.to(device), lengths, metadata_idx
        print(SEED)
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
        enc = CellEncoder(input_dim=Ns_df.shape[1], input_batch_num=0, hidden_layer=[32], 
                        layernorm=True, activation=nn.ReLU(), batchnorm=False, dropout_rate=0, 
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
        patience = 25
        best_model_state = None
        epochs = 200
        for val_padded_bags, val_labels, val_lengths, val_id in val_loader:
                print(len(val_lengths))
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
                torch.save(pn_cls, saving_path + "CLASSIFIER_" + ct + '_' + str(SEED) + ".pt")
                torch.save(enc, saving_path + "ENCODER_" + ct + '_' + str(SEED) + ".pt")
                patience_counter = 0
                print(f"Saving the best model with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        pn_cls_checkpoint = torch.load(saving_path + "CLASSIFIER_" + ct + '_' + str(SEED) + ".pt")
        enc_checkpoint = torch.load(saving_path + "ENCODER_" + ct + '_' + str(SEED) + ".pt")
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
        true_label_val_df = pd.DataFrame(np.vstack(true_label_val_list), columns=Ys_df.columns)
        pred_label_val_df = pd.DataFrame(np.vstack(pred_label_val_list), columns=Ys_df.columns)
        true_label_val_df = true_label_val_df.sort_values(by=[metadata[task_key][0]], ascending=False)
        pred_label_val_df = pred_label_val_df.loc[true_label_val_df.index]
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(pred_label_val_df, cmap="coolwarm", cbar=True)
        plt.show()
        fig.savefig(saving_path + "pred_label_val_" + ct + '_' + str(SEED) + ".png")
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(true_label_val_df, cmap="coolwarm", cbar=True)
        plt.show()
        fig.savefig(saving_path + "true_label_val_" + ct + '_' + str(SEED) + ".png")
        # auROC and macro F1
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import f1_score
        auROC_val = roc_auc_score(true_label_val_df, pred_label_val_df)
        # macro F1
        pred_label_val_df_ = pred_label_val_df.copy()
        # make the largetest value as 1 and the rest as 0 for each row
        pred_label_val_df_ = pred_label_val_df_.apply(lambda x: x == x.max(), axis=1).astype(int)
        macroF1_val = f1_score(true_label_val_df, pred_label_val_df_, average='macro')
        device = 'cpu'
        def MIL_Collate_fn(batch):
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
        tr_sample_ids = [adata[adata.obs.index==x[0]].obs[sample_key][0] for x in cell_id_list]
        true_label_tr_df = pd.DataFrame(np.vstack(true_label_tr_list), columns=Ys_df.columns, index=tr_sample_ids)
        pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=Ys_df.columns, index=tr_sample_ids)
        true_label_tr_df = true_label_tr_df.sort_values(by=[metadata[task_key][0]], ascending=False)
        pred_label_tr_df = pred_label_tr_df.loc[true_label_tr_df.index]
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(pred_label_tr_df, cmap="coolwarm", cbar=True)
        plt.show()
        fig.savefig(saving_path + "pred_label_tr_" + ct + '_' + str(SEED) + ".png")
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(true_label_tr_df, cmap="coolwarm", cbar=True)
        plt.show()
        fig.savefig(saving_path + "true_label_tr_" + ct + '_' + str(SEED) + ".png")
        auROC_tr = roc_auc_score(true_label_tr_df, pred_label_tr_df, average='macro', multi_class='ovr')
        pred_label_tr_df_ = pred_label_tr_df.copy()
        pred_label_tr_df_ = pred_label_tr_df_.apply(lambda x: x == x.max(), axis=1).astype(int)
        macroF1_tr = f1_score(true_label_tr_df, pred_label_tr_df_, average='macro')
        print(sample_key, '\n',
            "auROC_tr", auROC_tr,'\n',
            "auROC_val", auROC_val,'\n',
            "macroF1_tr", macroF1_tr,'\n',
            "macroF1_val", macroF1_val)
        res_df = pd.DataFrame(
                    {
                        'sample_key': sample_key,
                        'auROC_tr': auROC_tr,
                        'macroF1_tr': macroF1_tr,
                        'auROC_val': auROC_val,
                        'macroF1_val': macroF1_val,
                        'SEED': SEED
                    }, index=[0]).T
        res_df.to_csv(saving_path + "RES_STAT_" + ct + '_' + str(SEED) + ".csv")
        true_label_tr_df.to_csv(saving_path + "true_label_tr_" + ct + '_' + str(SEED) + ".csv")
        pred_label_tr_df.to_csv(saving_path + "pred_label_tr_" + ct + '_' + str(SEED) + ".csv")
        pred_label_tr_df_.to_csv(saving_path + "pred_label_tr_" + ct + '_' + str(SEED) + "_.csv")
        # Find mismatches
        wrong_preds = pred_label_tr_df_.idxmax(axis=1)[pred_label_tr_df_.idxmax(axis=1) != true_label_tr_df.idxmax(axis=1)].index.tolist()
        instance_mtx = np.vstack(instance_level_list) # 
        instance_mtx = np.vstack(instance_level_list) # 
        # unlist the cell_id_list
        cell_id_list = [i for sublist in cell_id_list for i in sublist]
        cell_id_df = pd.DataFrame([i for i in cell_id_list], columns=["cell_id"])
        metadata_tr = metadata.loc[cell_id_df.cell_id]
        from scipy.stats import zscore
        attention_mtx_raw = np.concatenate(attention_mtx_list, axis=0) # 
        attention_mtx_raw_df = pd.DataFrame(attention_mtx_raw, columns=["attention_score_raw"], index=cell_id_df.cell_id)
        attention_mtx_list_norm = [zscore(i) for i in attention_mtx_list]
        attention_mtx = np.concatenate(attention_mtx_list_norm, axis=0) # 
        attention_mtx_list_norm_cellnum = [i*len(i) for i in attention_mtx_list]
        attention_mtx_norm_cellnum = np.concatenate(attention_mtx_list_norm_cellnum, axis=0) # 
        attention_mtx_raw_df['attention_score_zscore'] = attention_mtx.tolist()
        attention_mtx_raw_df['attention_score_norm_cellnum'] = attention_mtx_norm_cellnum.tolist()
        attention_mtx_raw_df['attention_score_zscore_clip'] = attention_mtx_raw_df['attention_score_zscore'].apply(lambda x: min(x, 10))
        attention_mtx_raw_df['attention_score_norm_cellnum_clip'] = attention_mtx_raw_df['attention_score_norm_cellnum'].apply(lambda x: min(x, 10))
        metadata_tr = pd.concat([metadata_tr,attention_mtx_raw_df], axis=1)
        adata_tr = adata[metadata_tr.index.tolist()]
        adata_tr.obs = metadata_tr.copy()
        adata_tr.obs.index = adata_tr.obs.index.tolist()
        adata_tr.obsm['X_emb'] = adata_tr.X.copy()
        sc.pp.neighbors(adata_tr, n_neighbors=25, use_rep='X_emb')
        sc.tl.umap(adata_tr)
        adata_tr.write_h5ad(saving_path + "Adata_Tr_" + ct + '_' + str(SEED) + ".h5ad")
        sc.pl.umap(adata_tr, color=['attention_score_norm_cellnum_clip'], 
                    ncols=1, wspace=0.5, save='_normed_atten_' + ct + '_' + str(SEED) + '.png')
        sc.pl.umap(adata_tr, color=['attention_score_zscore_clip'], 
                    ncols=1, wspace=0.5, save='_zscored_atten_' + ct + '_' + str(SEED) + '.png')
        sc.pl.umap(adata_tr, color=[sample_key, task_key, cov_key, ct_key_fine], legend_loc='on data',
                        ncols=2, wspace=0.5, save='_cov_' + ct + '_' + str(SEED) + '.png')
        adata_tr_0 = adata_tr[adata_tr.obs[task_key] != metadata[task_key][0]]
        adata_tr_1 = adata_tr[adata_tr.obs[task_key] == metadata[task_key][0]]
        sc.pl.umap(adata_tr_0, color=['attention_score_norm_cellnum_clip'], 
                title = adata_tr_0.obs[task_key][0],
                    ncols=1, wspace=0.5, save='_normed_atten_' + adata_tr_0.obs[task_key][0] + ct + '_' + str(SEED) + '.png')
        sc.pl.umap(adata_tr_1, color=['attention_score_norm_cellnum_clip'], 
                title = adata_tr_1.obs[task_key][0],
                    ncols=1, wspace=0.5, save='_normed_atten_' + adata_tr_1.obs[task_key][0] + ct + '_' + str(SEED) + '.png')
        donors = adata_tr_0.obs[sample_key].unique()
        n = len(donors)
        cols = 5
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        for ax, donor in zip(axes.flat, donors):
            pred_label = ' Right Pred'
            if donor in wrong_preds:
                    print(donor)
                    pred_label = ' Wrong Pred'
                    sc.pl.umap(adata_tr_0[adata_tr_0.obs[sample_key] == donor],
                        color=['attention_score_norm_cellnum_clip'],  # or any obs key
                        ax=ax, palette='plasma',
                        show=False,
                        title=donor + pred_label)
            else:
                sc.pl.umap(adata_tr_0[adata_tr_0.obs[sample_key] == donor],
                        color=['attention_score_norm_cellnum_clip'],  # or any obs key
                        ax=ax,
                        show=False,
                        title=donor + pred_label)
        # Remove any unused subplots
        for i in range(n, len(axes.flat)):
            fig.delaxes(axes.flat[i])
        plt.tight_layout()
        plt.savefig('figures/umap_normed_atten_' + adata_tr_0.obs[task_key][0] + ct + '_' + str(SEED) + '_all_donors.png')
        donors = adata_tr_1.obs[sample_key].unique()
        n = len(donors)
        cols = 5
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        for ax, donor in zip(axes.flat, donors):
            pred_label = ' Right Pred'
            if donor in wrong_preds:
                    print(donor)
                    pred_label = ' Wrong Pred'
                    sc.pl.umap(adata_tr_1[adata_tr_1.obs[sample_key] == donor],
                        color=['attention_score_norm_cellnum_clip'],  # or any obs key
                        ax=ax, palette='plasma',
                        show=False,
                        title=donor + pred_label)
            else:
                sc.pl.umap(adata_tr_1[adata_tr_1.obs[sample_key] == donor],
                        color=['attention_score_norm_cellnum_clip'],  # or any obs key
                        ax=ax,
                        show=False,
                        title=donor + pred_label)
        # Remove any unused subplots
        for i in range(n, len(axes.flat)):
            fig.delaxes(axes.flat[i])
        plt.tight_layout()
        plt.savefig('figures/umap_normed_atten_' + adata_tr_1.obs[task_key][0] + ct + '_' + str(SEED) + '_all_donors.png')
        adata_tr_0 = adata_tr_0[~adata_tr_0.obs[sample_key].isin(wrong_preds)]
        adata_tr_1 = adata_tr_1[~adata_tr_1.obs[sample_key].isin(wrong_preds)]
        sc.pl.umap(adata_tr_0, color=['attention_score_norm_cellnum_clip'], 
                title = adata_tr_0.obs[task_key][0],
                    ncols=1, wspace=0.5, save='_normed_atten_' + adata_tr_0.obs[task_key][0] + ct + '_' + str(SEED) + '_CLEAN.png')
        sc.pl.umap(adata_tr_1, color=['attention_score_norm_cellnum_clip'], 
                title = adata_tr_1.obs[task_key][0],
                    ncols=1, wspace=0.5, save='_normed_atten_' + adata_tr_1.obs[task_key][0] + ct + '_' + str(SEED) + '_CLEAN.png')


