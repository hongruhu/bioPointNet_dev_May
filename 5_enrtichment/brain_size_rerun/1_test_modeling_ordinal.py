# (mil) hongruhu@gpu-5-50:/group/gquongrp/workspaces/hongruhu/bioPointNet$

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns
from bioPointNet_Apr2025 import *

obj = torch.load('/group/gquongrp/workspaces/hongruhu/bioPointNet/organism_level/species_protein_embedding.pt')
len(obj.keys()) # 131
gene_map = torch.load('/group/gquongrp/workspaces/hongruhu/bioPointNet/organism_level/species_protein_name.pt')
gene_map = {x.split('_')[0]: v for x, v in gene_map.items()}


df_brainsize = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/organism_level/log_brain_mass_labels.csv', index_col=0)
df_brainsize    # [131 rows x 4 columns]

species_meta = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/organism_level/species_id.csv', index_col=0)
df_brainsize['size_order'] = df_brainsize['log_brain_mass'].rank(method='dense') - 1
fig = plt.figure(figsize=(10, 5))
sns.histplot(df_brainsize['size_order'], bins=20)
plt.xlabel('size_order')
plt.ylabel('Frequency')
plt.title('Distribution of size_order')
plt.show()
fig.savefig('brainsize_distribution_rank.png')

ins_all = [obj[k].shape[0] for k in obj.keys()]

np.corrcoef(ins_all, df_brainsize['log_brain_mass']) # 0.000000
# array([[1.        , 0.27450154],
#        [0.27450154, 1.        ]])

np.corrcoef(ins_all, df_brainsize['size_order']) # 0.000000
# array([[1.        , 0.27006757],
#        [0.27006757, 1.        ]])


for fold in range(10):
    val_idx = df_brainsize[df_brainsize['fold'] == fold].index
    train_idx = df_brainsize[df_brainsize['fold'] != fold].index
    train_obj = {k: obj[k] for k in train_idx}
    val_obj = {k: obj[k] for k in val_idx}
    train_labels = df_brainsize.loc[train_idx, 'size_order']
    val_labels = df_brainsize.loc[val_idx, 'size_order']
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
    tr_dataset = MILDataset(Xs_tr, Ys_tr, ins_tr, train_labels.index.tolist())
    val_dataset = MILDataset(Xs_val, Ys_val, ins_val, val_labels.index.tolist())
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
    criterion = OrdinalRegressionLoss(num_class=df_brainsize.size_order.value_counts().size, train_cutpoints=True)
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
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
                if length <= 1:
                    continue
                else:
                    input_tr = padded_bags[idx, :length,:].unsqueeze(0).permute(0, 2, 1)
                    res_tr = pn_cls(input_tr)[0]
                    loss_per_sample = criterion(res_tr.to('cpu'), labels[idx].view(-1, 1).to('cpu')) # equivalent to the following
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
                val_loss_per_sample = criterion(res_val.to('cpu'), val_labels[val_idx].view(-1, 1).to('cpu'))
                loss_val += val_loss_per_sample
        loss_val_avg = loss_val/len(val_lengths)
        print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {loss_val_avg:.4f}")
        if loss_val_avg < best_val_loss:
            best_val_loss = loss_val_avg
            torch.save(pn_cls, str(fold) + "_brainsize.pt")
            patience_counter = 0
            print(f"Saving the best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    pn_cls_checkpoint = torch.load(str(fold) + "_brainsize.pt")
    saving_path = "./"
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
    true_label_val_df = pd.DataFrame(np.vstack(true_label_val_list), columns=['brainsize'], index=val_id)
    pred_label_val_df = pd.DataFrame(np.vstack(pred_label_val_list), columns=['brainsize'], index=val_id)
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
    fig.savefig(str(fold) + "_True_vs_pred_label_Val_brainsize_order.pdf", dpi=150)
    true_label_val_df['brainsize'] = df_brainsize.loc[true_label_val_df.index, 'log_brain_mass']
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
    fig.savefig(str(fold) + "_True_vs_pred_label_Val_brainsize_value.pdf", dpi=150)
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
    true_label_tr_df = pd.DataFrame(np.vstack(true_label_tr_list), columns=['brainsize'], index=tr_id)
    pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=['brainsize'], index=tr_id)
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
    fig.savefig(str(fold) + "_True_vs_pred_label_Tr_brainsize_order.pdf", dpi=150)
    true_label_tr_df['brainsize'] = df_brainsize.loc[true_label_tr_df.index, 'log_brain_mass']
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
    fig.savefig(str(fold) + "_True_vs_pred_label_Tr_brainsize_value.pdf", dpi=150)
    sample_meta = [i for i in tr_id]
    sample_meta_tr = df_brainsize.loc[sample_meta]
    embedding_tr_df = pd.DataFrame(np.vstack(embedding_tr_list))
    sample_meta_tr
    # umap embedding
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_tr_umap = reducer.fit_transform(embedding_tr_df)
    fig = plt.figure(figsize=(8, 7))
    sns.scatterplot(x=embedding_tr_umap[:, 0], y=embedding_tr_umap[:, 1], s=300, linewidth=0.1,
                    hue=sample_meta_tr.size_order, palette="viridis")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP for Sample-level Embedding (brainsize Order)")
    plt.legend([],[], frameon=False)
    sns.despine()
    fig.savefig(str(fold) + "_UMAP_sample_level_embedding_brainsize_order.pdf", dpi=150)
    fig = plt.figure(figsize=(8, 7))
    sns.scatterplot(x=embedding_tr_umap[:, 0], y=embedding_tr_umap[:, 1], s=300, linewidth=0.1,
                    hue=sample_meta_tr['log_brain_mass'], palette="viridis")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    sns.despine()
    plt.legend([],[], frameon=False)
    plt.title("UMAP for Sample-level Embedding (log brain mass)")
    fig.savefig(str(fold) + "_UMAP_sample_level_embedding_brainsize_value.pdf", dpi=150)
    len(cell_id_list)           # 118
    len(instance_level_list)    # 118
    gene_id_list = [gene_map[i] for i in cell_id_list]
    len(gene_id_list)           # 118
    mismatching_species = []
    for bag_id_tr in range(len(gene_id_list)):
        if len(gene_id_list[bag_id_tr]) != len(instance_level_list[bag_id_tr]):
            mismatching_species.append(bag_id_tr)    
            print(cell_id_list[bag_id_tr], len(gene_id_list[bag_id_tr])-len(instance_level_list[bag_id_tr]))
    filtered_gene_id_list = [item for idx, item in enumerate(gene_id_list) if idx not in mismatching_species]
    filtered_instance_level_list = [item for idx, item in enumerate(instance_level_list) if idx not in mismatching_species]
    filtered_attention_mtx_list = [item for idx, item in enumerate(attention_mtx_list) if idx not in mismatching_species]
    [len(x) for x in filtered_gene_id_list] == [len(x) for x in filtered_instance_level_list]
    instance_mtx = np.vstack(filtered_instance_level_list) # (1833547, 320)
    gene_id_df = pd.DataFrame([i for sublist in filtered_gene_id_list for i in sublist], columns=["gene_id"])
    attention_mtx_raw = np.concatenate(filtered_attention_mtx_list, axis=0) # 
    from scipy.stats import zscore
    attention_mtx_list_zscore= [zscore(i) for i in filtered_attention_mtx_list]
    attention_mtx_zscore = np.concatenate(attention_mtx_list_zscore, axis=0) # 
    attention_mtx_list_norm = [i*len(i) for i in filtered_attention_mtx_list]
    attention_mtx_norm = np.concatenate(attention_mtx_list_norm, axis=0) # 
    attention_mtx_df = pd.DataFrame(attention_mtx_raw, columns=["attention_score_raw"], index=gene_id_df.gene_id)
    attention_mtx_df['attention_score_zscore'] = attention_mtx_zscore.tolist()
    attention_mtx_df['attention_score_norm_norm'] = attention_mtx_norm.tolist()
    gene_to_species = {gene: species for species, genes in gene_map.items() for gene in genes}
    attention_mtx_df['species_id'] = attention_mtx_df.index.map(gene_to_species)
    species_info = species_meta[['Pheno Name', 'Uniprot Name']]
    species_id_to_name = species_info['Pheno Name'].to_dict()
    species_id_to_name_uniprot = species_info['Uniprot Name'].to_dict()
    attention_mtx_df['species_name'] = attention_mtx_df.species_id.map(species_id_to_name)
    attention_mtx_df['species_Uniprot_name'] = attention_mtx_df.species_id.map(species_id_to_name_uniprot)
    # UP000005640 is human
    attention_mtx_df.to_csv(str(fold) + "_attention_tr.csv")



human_attention_mtx_df = attention_mtx_df[attention_mtx_df.species_id=='UP000005640']
human_df = human_attention_mtx_df.sort_values(by='attention_score_norm_norm')[['attention_score_norm_norm', 'species_name']]


from bioservices import UniProt
u = UniProt()
result = u.mapping("UniProtKB_AC-ID", "UniProtKB", query=human_df.index.tolist())
result.keys()
# dict_keys(['results', 'failedIds'])
gene_ids_human = [result['results'][i]['to']['uniProtkbId'] for i in range(len(result['results']))]
gene_ids_human = [x.split('_HUMAN')[0] for x in gene_ids_human]
human_df['gene_name'] = gene_ids_human

from gprofiler import GProfiler
# Run enrichment
gp = GProfiler(return_dataframe=True)
gsea_results = gp.profile(organism='hsapiens', query=human_df['gene_name'][-30:].tolist())
df = gsea_results[['native', 'name', 'p_value', 'source']]
df.source.value_counts()
# source
# TF       40
# CORUM    11
# GO:MF     6
# KEGG      4
# HP        3
# GO:CC     2
# WP        2
# REAC      1

df[df.source=='GO:CC']
df[df.source=='GO:MF']

# WDR81
# PIDD1





instance_mtx.shape     # (1833547, 320)
attention_mtx_df.shape # (1833547, 6)
human_df[human_df.gene_name=='WDR81'] # Q562E7
if "Q562E7" in attention_mtx_df.index:
    row_number = attention_mtx_df.index.get_loc("Q562E7")
    print(row_number)
    # 882033

attention_mtx_df.iloc[row_number,:]
# attention_score_raw                      0.000791
# attention_score_zscore                   6.824608
# attention_score_norm_norm               15.701357
# species_id                            UP000005640
# species_name                                human
# species_Uniprot_name         homo sapiens (human)
WDR81_emb = instance_mtx[row_number].copy()
WDR81_emb = WDR81_emb.reshape(1,-1)


filtered_instance_level_list    # input instance matrix list
filtered_gene_id_list           # input instance name list
print(len(filtered_instance_level_list), len(filtered_gene_id_list))
# 115 115
# input bag list?
mismatching_species # [1, 54, 81]
true_label_tr_df # 118
pred_label_tr_df # 118
filtered_true_label = true_label_tr_df.drop(true_label_tr_df.index[mismatching_species])
filtered_pred_label = pred_label_tr_df.drop(pred_label_tr_df.index[mismatching_species])
filtered_true_label['species'] = filtered_true_label.index.map(species_id_to_name)
filtered_pred_label['species'] = filtered_pred_label.index.map(species_id_to_name)
np.sum(filtered_true_label.species == filtered_pred_label.species) # 115
species_keys = filtered_true_label.species.tolist()






eval_obj = {species_keys[k]:filtered_instance_level_list[k] for k in range(len(species_keys))}
Xs_eval = [torch.tensor(eval_obj[k]) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label 






WDR81_emb
[eval_obj[k] for k in eval_obj.keys()][0].shape
# ([13969, 320])
np.vstack([WDR81_emb,[eval_obj[k] for k in eval_obj.keys()][0]]).shape
# (13970, 320)
np.vstack([np.repeat(WDR81_emb, repeats=10, axis=0),[eval_obj[k] for k in eval_obj.keys()][0]]).shape
# (13979, 320)


# add 1
Xs_eval = [torch.tensor(np.vstack([WDR81_emb, eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_1_copy'] = pred_label_eval_df.values



# add 2
Xs_eval = [torch.tensor(np.vstack([np.repeat(WDR81_emb, repeats=2, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_2_copy'] = pred_label_eval_df.values




# add 5
Xs_eval = [torch.tensor(np.vstack([np.repeat(WDR81_emb, repeats=5, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_5_copy'] = pred_label_eval_df.values





# add 10
Xs_eval = [torch.tensor(np.vstack([np.repeat(WDR81_emb, repeats=10, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_10_copy'] = pred_label_eval_df.values





# add 20
Xs_eval = [torch.tensor(np.vstack([np.repeat(WDR81_emb, repeats=20, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_20_copy'] = pred_label_eval_df.values



# add 50
Xs_eval = [torch.tensor(np.vstack([np.repeat(WDR81_emb, repeats=50, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_50_copy'] = pred_label_eval_df.values




# add 100
Xs_eval = [torch.tensor(np.vstack([np.repeat(WDR81_emb, repeats=100, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_100_copy'] = pred_label_eval_df.values



# add 200
Xs_eval = [torch.tensor(np.vstack([np.repeat(WDR81_emb, repeats=200, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_200_copy'] = pred_label_eval_df.values


# add 500
Xs_eval = [torch.tensor(np.vstack([np.repeat(WDR81_emb, repeats=500, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_500_copy'] = pred_label_eval_df.values




# add 1000
Xs_eval = [torch.tensor(np.vstack([np.repeat(WDR81_emb, repeats=1000, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_1000_copy'] = pred_label_eval_df.values



filtered_pred_label.to_csv('seed_9_duplication_UP000005640_Human_Q562E7_WDR81.csv')
filtered_true_label.to_csv('seed_9_true.csv')

# Disease	                    Brain effect
# Cerebellar atrophy	        Shrinkage of the cerebellum (part of brain controlling movement and balance)
# Congenital hydrocephalus	    Enlarged brain ventricles due to fluid buildup
# Microcephaly-like syndromes	Smaller than normal brain size

# Loss of function mutations in WDR81 lead to severely reduced brain size, especially in the cerebellum.
# In mouse models, Wdr81 mutations cause defects in neuronal survival and brain volume loss.

import pandas as pd
import matplotlib.pyplot as plt
import os

df = filtered_pred_label.copy()
x_labels = ['original'] + [col for col in df.columns if col.startswith('add_')]
x = list(range(len(x_labels)))  # 0, 1, 2, ...
os.makedirs('plots_per_species', exist_ok=True)

# Version 1: One PDF per species
for idx, row in df.iterrows():
    species_name = row['species']
    y = [row['brainsize']] + [row[col] for col in df.columns if col.startswith('add_')]
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o')
    plt.xticks(x, x_labels, rotation=45, ha='right')
    plt.title(f'Brain Size Change - {species_name}')
    plt.xlabel('Number of Added Copies')
    plt.ylabel('Brain Size (Prediction)')
    plt.grid(True)
    safe_species_name = species_name.replace('/', '_')  # Handle weird characters
    plt.tight_layout()
    plt.savefig(f'plots_per_species/{safe_species_name}.pdf')
    plt.close()



# Version 2: All species on one figure
plt.figure(figsize=(15, 12))
for idx, row in df.iterrows():
    species_name = row['species']
    y = [row['brainsize']] + [row[col] for col in df.columns if col.startswith('add_')]
    plt.plot(x, y, marker='o', label=species_name)

plt.xticks(x, x_labels, rotation=45, ha='right')
plt.title('Brain Size Change Across Species')
plt.xlabel('Number of Added Copies')
plt.ylabel('Brain Size (Prediction)')
plt.grid(True)
plt.legend(fontsize=5, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('brain_size_all_species.pdf')
plt.close()





# control
# Q9Y6J3                    0.016481        human     SMA5O

human_df[human_df.gene_name=='SMA5O'] # Q9Y6J3
if "Q9Y6J3" in attention_mtx_df.index:
    row_number = attention_mtx_df.index.get_loc("Q9Y6J3")
    print(row_number)
    # 893747

attention_mtx_df.iloc[row_number,:]
# attention_score_raw                      0.000001
# attention_score_zscore                  -0.456565
# attention_score_norm_norm                0.016481
# species_id                            UP000005640
# species_name                                human
# species_Uniprot_name         homo sapiens (human)
SMA5O_emb = instance_mtx[row_number].copy()
SMA5O_emb = SMA5O_emb.reshape(1,-1)



SMA5O_emb
[eval_obj[k] for k in eval_obj.keys()][0].shape
# ([13969, 320])
np.vstack([SMA5O_emb,[eval_obj[k] for k in eval_obj.keys()][0]]).shape
# (13970, 320)
np.vstack([np.repeat(SMA5O_emb, repeats=10, axis=0),[eval_obj[k] for k in eval_obj.keys()][0]]).shape
# (13979, 320)


# add 1
Xs_eval = [torch.tensor(np.vstack([SMA5O_emb, eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_1_copy'] = pred_label_eval_df.values



# add 2
Xs_eval = [torch.tensor(np.vstack([np.repeat(SMA5O_emb, repeats=2, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_2_copy'] = pred_label_eval_df.values




# add 5
Xs_eval = [torch.tensor(np.vstack([np.repeat(SMA5O_emb, repeats=5, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_5_copy'] = pred_label_eval_df.values





# add 10
Xs_eval = [torch.tensor(np.vstack([np.repeat(SMA5O_emb, repeats=10, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_10_copy'] = pred_label_eval_df.values





# add 20
Xs_eval = [torch.tensor(np.vstack([np.repeat(SMA5O_emb, repeats=20, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_20_copy'] = pred_label_eval_df.values



# add 50
Xs_eval = [torch.tensor(np.vstack([np.repeat(SMA5O_emb, repeats=50, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_50_copy'] = pred_label_eval_df.values




# add 100
Xs_eval = [torch.tensor(np.vstack([np.repeat(SMA5O_emb, repeats=100, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_100_copy'] = pred_label_eval_df.values



# add 200
Xs_eval = [torch.tensor(np.vstack([np.repeat(SMA5O_emb, repeats=200, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_200_copy'] = pred_label_eval_df.values


# add 500
Xs_eval = [torch.tensor(np.vstack([np.repeat(SMA5O_emb, repeats=500, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_500_copy'] = pred_label_eval_df.values




# add 1000
Xs_eval = [torch.tensor(np.vstack([np.repeat(SMA5O_emb, repeats=1000, axis=0), eval_obj[k]])) for k in eval_obj.keys()]
Ys_eval = torch.tensor(filtered_true_label.brainsize.tolist()).reshape(-1, 1)
ins_eval = [Xs_eval[i].shape[0] for i in range(len(Xs_eval))]
eval_dataset = MILDataset(Xs_eval, Ys_eval, ins_eval, species_keys)
eval_loader = DataLoader(eval_dataset, batch_size=len(ins_eval), shuffle=False, collate_fn=MIL_Collate_fn)
for eval_padded_bags, eval_labels, eval_lengths, eval_id in eval_loader:
    print(len(eval_lengths))


pred_label_eval_list = []
pn_cls_checkpoint.to(device)
for eval_idx in range(len(eval_lengths)):
        eval_length = eval_lengths[eval_idx]
        if eval_length <= 1:
            continue
        else:
            input_eval = eval_padded_bags[eval_idx, :eval_length,:].unsqueeze(0).permute(0, 2, 1)
            cls_eval, _, _, _, _ = pn_cls_checkpoint(input_eval)
            pred_label_eval = cls_eval
            pred_label_eval_list.append(pred_label_eval.detach().cpu().numpy())


pred_label_eval_df = pd.DataFrame(np.vstack(pred_label_eval_list), columns=['brainsize'], index=eval_id)
pred_label_eval_df
filtered_pred_label['add_1000_copy'] = pred_label_eval_df.values



filtered_pred_label.to_csv('seed_9_duplication_UP000005640_Human_Q9Y6J3_SMA5O_emb_ctrl.csv')
filtered_true_label.to_csv('seed_9_true.csv')



import pandas as pd
import matplotlib.pyplot as plt
import os

df = filtered_pred_label.copy()
x_labels = ['original'] + [col for col in df.columns if col.startswith('add_')]
x = list(range(len(x_labels)))  # 0, 1, 2, ...
os.makedirs('plots_per_species', exist_ok=True)

# Version 1: One PDF per species
for idx, row in df.iterrows():
    species_name = row['species']
    y = [row['brainsize']] + [row[col] for col in df.columns if col.startswith('add_')]
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o')
    plt.xticks(x, x_labels, rotation=45, ha='right')
    plt.title(f'Brain Size Change - {species_name}')
    plt.xlabel('Number of Added Copies')
    plt.ylabel('Brain Size (Prediction)')
    plt.grid(True)
    safe_species_name = species_name.replace('/', '_')  # Handle weird characters
    plt.tight_layout()
    plt.savefig(f'plots_per_species/{safe_species_name}.pdf')
    plt.close()



# Version 2: All species on one figure
plt.figure(figsize=(15, 12))
for idx, row in df.iterrows():
    species_name = row['species']
    y = [row['brainsize']] + [row[col] for col in df.columns if col.startswith('add_')]
    plt.plot(x, y, marker='o', label=species_name)

plt.xticks(x, x_labels, rotation=45, ha='right')
plt.title('Brain Size Change Across Species')
plt.xlabel('Number of Added Copies')
plt.ylabel('Brain Size (Prediction)')
plt.grid(True)
plt.legend(fontsize=5, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('brain_size_all_species.pdf')
plt.close()







import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the file
df = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/organism_level/low_attn_ctrl/seed_9_duplication_UP000005640_Human_Q9Y6J3_SMA5O_emb_ctrl.csv', 
                 index_col=0)

# Create output folder
os.makedirs('plots_per_species', exist_ok=True)
# Prepare x-axis based on actual number of copies added
x_mapping = {'brainsize': 0}
for col in df.columns:
    if col.startswith('add_'):
        # Extract number from column name like 'add_5_copy'
        copy_num = int(col.split('_')[1])
        x_mapping[col] = copy_num


# x-axis values (sorted by increasing copy number)
x_labels = list(x_mapping.keys())
x_values = [x_mapping[label] for label in x_labels]


# Version 2: all species together
plt.figure(figsize=(15, 12))

for idx, row in df.iterrows():
    species_name = row['species']
    y = [row['brainsize']] + [row[col] for col in df.columns if col.startswith('add_')]
    plt.plot(x_values, y, marker='o', label=species_name)

plt.xticks(x_values)
plt.title('Brain Size Change Across Species')
plt.xlabel('Number of Added Copies')
plt.ylabel('Brain Size (Prediction)')
plt.grid(True)
plt.legend(fontsize=5, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('brain_size_all_species_corrected_xaxis.pdf')
plt.close()





import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the file
df = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/organism_level/high_attn/seed_9_duplication_UP000005640_Human_Q562E7_WDR81.csv', 
                 index_col=0)

# Prepare x-axis based on actual number of copies added
x_mapping = {'brainsize': 0}
for col in df.columns:
    if col.startswith('add_'):
        # Extract number from column name like 'add_5_copy'
        copy_num = int(col.split('_')[1])
        x_mapping[col] = copy_num


# x-axis values (sorted by increasing copy number)
x_labels = list(x_mapping.keys())
x_values = [x_mapping[label] for label in x_labels]


# Version 2: all species together
plt.figure(figsize=(15, 12))

for idx, row in df.iterrows():
    species_name = row['species']
    y = [row['brainsize']] + [row[col] for col in df.columns if col.startswith('add_')]
    plt.plot(x_values, y, marker='o', label=species_name)

plt.xticks(x_values)
plt.title('Brain Size Change Across Species')
plt.xlabel('Number of Added Copies')
plt.ylabel('Brain Size (Prediction)')
plt.grid(True)
plt.legend(fontsize=5, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('brain_size_all_species_corrected_xaxis_.pdf')
plt.close()

