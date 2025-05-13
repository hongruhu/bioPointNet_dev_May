# (scpair) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/MIL/$

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scPointNet_Attn_March10 import *


target_pheno = 'Hand-Wing.Index'

obj = torch.load('/group/gquongrp/workspaces/hongruhu/MIL/bird/embeddings_bird_'+target_pheno+'.pt')
len(obj.keys()) # 201
df_pheno= pd.read_csv('/group/gquongrp/workspaces/hongruhu/MIL/bird/bird_'+target_pheno+'_labels.csv', index_col=0)
df_pheno    # [201 rows x 2 columns]


df_pheno['rank'] = df_pheno[target_pheno].rank(method='dense') - 1
np.corrcoef(df_pheno['rank'], df_pheno[target_pheno]) 
# array([[1.        , 0.94130739],
#        [0.94130739, 1.        ]])

ins_all = [obj[k].shape[0] for k in obj.keys()]


np.corrcoef(df_pheno[target_pheno], ins_all) 
# array([[1.        , 0.02082245],
#        [0.02082245, 1.        ]])
np.corrcoef(df_pheno['rank'], ins_all) 
# array([[1.       , 0.0463431],
#        [0.0463431, 1.       ]])





list(obj.keys()) == list(df_pheno.index) # False
df_pheno.columns= ['phenotype', 'fold' ,'rank']
# Index(['phenotype', 'fold', 'rank'], dtype='object')

for fold in range(10):
    val_idx = df_pheno[df_pheno['fold'] == fold].index
    train_idx = df_pheno[df_pheno['fold'] != fold].index
    train_obj = {k: obj[k] for k in train_idx}
    val_obj = {k: obj[k] for k in val_idx}
    train_labels = df_pheno.loc[train_idx, 'rank']
    val_labels = df_pheno.loc[val_idx, 'rank']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    # turn into list
    Xs_tr = [torch.tensor(train_obj[k]) for k in train_obj.keys()]
    Xs_val = [torch.tensor(val_obj[k]) for k in val_obj.keys()]
    Ys_tr = torch.tensor(train_labels.tolist()).reshape(-1, 1)
    Ys_val = torch.tensor(val_labels.tolist()).reshape(-1, 1)
    ins_tr = [Xs_tr[i].shape[0] for i in range(len(Xs_tr))]
    ins_val = [Xs_val[i].shape[0] for i in range(len(Xs_val))]
    tr_dataset = MILDataset(Xs_tr, Ys_tr, ins_tr, train_labels.index.tolist(), task="regression")
    val_dataset = MILDataset(Xs_val, Ys_val, ins_val, val_labels.index.tolist(), task="regression")
    batch_size = 15
    # Create the dataloaders
    train_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, collate_fn=MIL_Collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=len(ins_val), shuffle=False, collate_fn=MIL_Collate_fn)
    # set the seed
    SEED = 15
    set_seed(SEED)
    pn_cls = PointNetClassHead(input_dim=Xs_tr[0].shape[1], k=1, 
                            global_features=256, attention_dim=64, agg_method="gated_attention")
    pn_cls.apply(init_weights)
    pn_cls.to(device)
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(pn_cls.parameters(), lr=learning_rate)
    # criterion = nn.CrossEntropyLoss()
    criterion = OrdinalRegressionLoss(num_class=df_pheno['rank'].value_counts().size, train_cutpoints=True)
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
                    loss_per_sample = criterion(res_tr.to('cpu'), labels[idx].long().view(-1, 1).to('cpu')) # equivalent to the following
                    loss_tr += loss_per_sample.to(device)
            (loss_tr/len(lengths)).backward() if loss_tr > 0 else None
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
                val_loss_per_sample = criterion(res_val.to('cpu'), val_labels[val_idx].long().view(-1, 1).to('cpu'))
                loss_val += val_loss_per_sample
        loss_val_avg = loss_val/len(val_lengths)
        print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {loss_val_avg:.4f}")
        if loss_val_avg < best_val_loss:
            best_val_loss = loss_val_avg
            torch.save(pn_cls, str(fold) + "_bird_" + target_pheno + ".pt")
            patience_counter = 0
            print(f"Saving the best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    pn_cls_checkpoint = torch.load(str(fold) + "_bird_" + target_pheno + ".pt")
    saving_path = "./bird"
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
                pred_label_val_list.append(cls_val.detach().cpu().numpy())
                true_label_val_list.append(val_labels[val_idx].detach().cpu().numpy())
    true_label_val_df = pd.DataFrame(np.vstack(true_label_val_list), columns=[target_pheno], index=val_id)
    pred_label_val_df = pd.DataFrame(np.vstack(pred_label_val_list), columns=[target_pheno], index=val_id)
    true_label_val_df.corrwith(pred_label_val_df, method="spearman") # 
    true_label_val_df.corrwith(pred_label_val_df, method="pearson")  # 
    spearman_corr = true_label_val_df.corrwith(pred_label_val_df, method="spearman").values[0]
    pearson_corr = true_label_val_df.corrwith(pred_label_val_df, method="pearson").values[0]
    fig = plt.figure(figsize=(8, 7))
    sns.scatterplot(x=true_label_val_df.values.flatten(), y=pred_label_val_df.values.flatten(), s=56)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.title("True vs Predicted Label")
    sns.despine()
    # Add text of correlations to the plot
    plt.text(0.05, 0.95, f'Spearman: {spearman_corr:.6f}', transform=plt.gca().transAxes, fontsize=16, verticalalignment='top')
    plt.text(0.05, 0.90, f'Pearson: {pearson_corr:.6f}', transform=plt.gca().transAxes, fontsize=16, verticalalignment='top')
    plt.show()
    fig.savefig(str(fold) + "_True_vs_pred_label_Val_" + target_pheno + ".pdf", dpi=150)
    true_label_val_df[target_pheno] = df_pheno.loc[true_label_val_df.index, 'phenotype']
    true_label_val_df.corrwith(pred_label_val_df, method="spearman") # -0.060448
    true_label_val_df.corrwith(pred_label_val_df, method="pearson")  # -0.093125
    spearman_corr = true_label_val_df.corrwith(pred_label_val_df, method="spearman").values[0]
    pearson_corr = true_label_val_df.corrwith(pred_label_val_df, method="pearson").values[0]
    fig = plt.figure(figsize=(8, 7))
    sns.scatterplot(x=true_label_val_df.values.flatten(), y=pred_label_val_df.values.flatten(), s=56)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.title("True vs Predicted Label")
    sns.despine()
    # Add text of correlations to the plot
    plt.text(0.05, 0.95, f'Spearman: {spearman_corr:.6f}', transform=plt.gca().transAxes, fontsize=16, verticalalignment='top')
    plt.text(0.05, 0.90, f'Pearson: {pearson_corr:.6f}', transform=plt.gca().transAxes, fontsize=16, verticalalignment='top')
    plt.show()
    fig.savefig(str(fold) + "_True_vs_pred_label_Val_" + target_pheno + "_.pdf", dpi=150)
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
    train_loader_ = DataLoader(tr_dataset, batch_size=len(tr_dataset), shuffle=False, collate_fn=MIL_Collate_fn)
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
                pred_label_tr_list.append(pred_label_tr.detach().cpu().numpy())
                true_label_tr_list.append(true_label_tr.detach().cpu().numpy())
                instance_level_list.append(input_tr.squeeze(0).permute(1,0).detach().cpu().numpy())
                attention_mtx_list.append(attn_weights_tr.squeeze(0,1).detach().cpu().numpy())
                cell_id_list.append(tr_id[tr_idx])
    true_label_tr_df = pd.DataFrame(np.vstack(true_label_tr_list), columns=[target_pheno], index=tr_id)
    pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=[target_pheno], index=tr_id)
    true_label_tr_df.corrwith(pred_label_tr_df, method="spearman") # 0.000000
    true_label_tr_df.corrwith(pred_label_tr_df, method="pearson")  # 0.000000
    # correlation between true and predicted labels
    spearman_corr = true_label_tr_df.corrwith(pred_label_tr_df, method="spearman").values[0] # 0.980850271704683
    pearson_corr = true_label_tr_df.corrwith(pred_label_tr_df, method="pearson").values[0]   # 0.9799932062905035
    # scatterplot, x-axis: true label, y-axis: predicted label
    fig = plt.figure(figsize=(8, 7))
    sns.scatterplot(x=true_label_tr_df.values.flatten(), y=pred_label_tr_df.values.flatten(), s=78)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.title("True vs Predicted Label")
    sns.despine()
    # Add text of correlations to the plot
    plt.text(0.05, 0.95, f'Spearman: {spearman_corr:.6f}', transform=plt.gca().transAxes, fontsize=16, verticalalignment='top')
    plt.text(0.05, 0.90, f'Pearson: {pearson_corr:.6f}', transform=plt.gca().transAxes, fontsize=16, verticalalignment='top')
    plt.show()
    fig.savefig(str(fold) + "_True_vs_pred_label_Tr_" + target_pheno + ".pdf", dpi=150)
    true_label_tr_df[target_pheno] = df_pheno.loc[true_label_tr_df.index, 'phenotype']
    true_label_tr_df.corrwith(pred_label_tr_df, method="spearman") # 0.000000
    true_label_tr_df.corrwith(pred_label_tr_df, method="pearson")  # 0.000000
    # correlation between true and predicted labels
    spearman_corr = true_label_tr_df.corrwith(pred_label_tr_df, method="spearman").values[0] # 0.980850271704683
    pearson_corr = true_label_tr_df.corrwith(pred_label_tr_df, method="pearson").values[0]   # 0.9799932062905035
    # scatterplot, x-axis: true label, y-axis: predicted label
    fig = plt.figure(figsize=(8, 7))
    sns.scatterplot(x=true_label_tr_df.values.flatten(), y=pred_label_tr_df.values.flatten(), s=78)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.title("True vs Predicted Label")
    sns.despine()
    # Add text of correlations to the plot
    plt.text(0.05, 0.95, f'Spearman: {spearman_corr:.6f}', transform=plt.gca().transAxes, fontsize=16, verticalalignment='top')
    plt.text(0.05, 0.90, f'Pearson: {pearson_corr:.6f}', transform=plt.gca().transAxes, fontsize=16, verticalalignment='top')
    plt.show()
    fig.savefig(str(fold) + "_True_vs_pred_label_Tr_" + target_pheno + "_.pdf", dpi=150)
    sample_meta = [i for i in tr_id]
    sample_meta_tr = df_pheno.loc[sample_meta]
    embedding_tr_df = pd.DataFrame(np.vstack(embedding_tr_list))
    sample_meta_tr
    # umap embedding
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_tr_umap = reducer.fit_transform(embedding_tr_df)
    fig = plt.figure(figsize=(8, 7))
    sns.scatterplot(x=embedding_tr_umap[:, 0], y=embedding_tr_umap[:, 1], 
                    hue=sample_meta_tr['rank'], palette="viridis")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP for Sample-level Embedding " + target_pheno + ' rank') 
    plt.legend([],[], frameon=False)
    sns.despine()
    plt.show()
    fig.savefig(str(fold) + "_UMAP_sample_level_embedding_" + target_pheno + "_rank.pdf", dpi=150)
    fig = plt.figure(figsize=(8, 7))
    sns.scatterplot(x=embedding_tr_umap[:, 0], y=embedding_tr_umap[:, 1], 
                    hue=sample_meta_tr['phenotype'], palette="viridis")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    sns.despine()
    plt.show()
    plt.title("UMAP for Sample-level Embedding " + target_pheno + ' value')
    fig.savefig(str(fold) + "_UMAP_sample_level_embedding_" + target_pheno + "_value.pdf", dpi=150)
