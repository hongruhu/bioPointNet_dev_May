# (mil) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet/benchmarking/ProtoCell4P$

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns

from src import *

import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ProtoCell(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim, n_layers, n_proto, n_classes, lambdas, n_ct=None, device="cpu", d_min=1):
        super(ProtoCell, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_proto = n_proto
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_ct = n_ct
        self.d_min = d_min
        assert self.n_layers > 0
        self.lambda_1 = lambdas["lambda_1"]
        self.lambda_2 = lambdas["lambda_2"]
        self.lambda_3 = lambdas["lambda_3"]
        self.lambda_4 = lambdas["lambda_4"]
        self.lambda_5 = lambdas["lambda_5"]
        self.lambda_6 = lambdas["lambda_6"]
        self.enc_i = nn.Linear(self.input_dim, self.h_dim)
        self.enc_h = nn.Sequential(*([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.n_layers - 1)]))
        self.enc_z = nn.Linear(self.h_dim, self.z_dim)
        self.dec_z = nn.Linear(self.z_dim, self.h_dim)
        self.dec_h = nn.Sequential(*([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.n_layers - 1)]))
        self.dec_i = nn.Linear(self.h_dim, self.input_dim)
        self.imp_i = nn.Linear(self.input_dim, self.h_dim)
        self.imp_h = nn.Sequential(*([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.n_layers - 1)]))
        self.imp_p = nn.Linear(self.h_dim, self.n_proto * self.n_classes)
        self.activate = nn.LeakyReLU()
        self.prototypes = nn.parameter.Parameter(torch.empty(self.n_proto, self.z_dim), requires_grad = True)
        self.ce_ = nn.CrossEntropyLoss(reduction="mean")
        nn.init.xavier_normal_(self.enc_i.weight)
        nn.init.xavier_normal_(self.enc_z.weight)
        nn.init.xavier_normal_(self.dec_z.weight)
        nn.init.xavier_normal_(self.dec_i.weight)
        nn.init.xavier_normal_(self.imp_i.weight)
        nn.init.xavier_normal_(self.imp_p.weight)
        nn.init.xavier_normal_(self.prototypes)
        for i in range(self.n_layers - 1):
            nn.init.xavier_normal_(self.enc_h[i].weight)
            nn.init.xavier_normal_(self.dec_h[i].weight)
            nn.init.xavier_normal_(self.imp_h[i].weight)
        if self.n_ct is not None:
            self.ct_clf1 = nn.Linear(self.n_proto, self.n_ct)
            self.ct_clf2 = nn.Linear(self.n_proto * self.n_classes, self.n_ct)
            nn.init.xavier_normal_(self.ct_clf1.weight)
            nn.init.xavier_normal_(self.ct_clf2.weight)
    def forward(self, x, y, ct=None, sparse=True):
        split_idx = [0]
        for i in range(len(x)):
            split_idx.append(split_idx[-1]+x[i].shape[0])
        if sparse:
            x = torch.cat([torch.tensor(x[i].toarray()) for i in range(len(x))]).to(self.device)
        else:
            x = torch.cat([torch.tensor(x[i]) for i in range(len(x))]).to(self.device)
        y = y.to(self.device)
        z = self.encode(x)
        import_scores = self.compute_importance(x) # (n_cell, n_proto, n_class)
        c2p_dists = torch.pow(z[:, None] - self.prototypes[None, :], 2).sum(-1)
        c_logits = (1 / (c2p_dists+0.5))[:,None,:].matmul(import_scores).squeeze(1) # (n_cell, n_classes)
        logits = torch.stack([c_logits[split_idx[i]:split_idx[i+1]].mean(dim=0) for i in range(len(split_idx)-1)])
        clf_loss = self.ce_(logits, y)
        if self.n_ct is not None and ct is not None:
            ct_logits = self.ct_clf2(import_scores.reshape(-1, self.n_proto * self.n_classes))
            ct_loss = self.ce_(ct_logits, torch.tensor([j for i in ct for j in i]).to(self.device))
        else:
            ct_loss = 0
        total_loss = clf_loss + self.lambda_6 * ct_loss
        if ct is not None:
            return total_loss, logits, ct_logits    
        return total_loss, logits
    def encode(self, x):
        h_e = self.activate(self.enc_i(x))
        for i in range(self.n_layers - 1):
            h_e = self.activate(self.enc_h[i](h_e))
        z = self.activate(self.enc_z(h_e))
        return z
    def decode(self, z):
        h_d = self.activate(self.dec_z(z))
        for i in range(self.n_layers - 1):
            h_d = self.activate(self.dec_h[i](h_d))
        x_hat = torch.relu(self.dec_i(h_d))
        return x_hat
    def compute_importance(self, x):
        h_i = self.activate(self.imp_i(x))
        for i in range(self.n_layers - 1):
            h_i = self.activate(self.imp_h[i](h_i))
        import_scores = torch.sigmoid(self.imp_p(h_i)).reshape(-1, self.n_proto, self.n_classes)
        return import_scores
    def pretrain(self, x, y, ct=None, sparse=True):
        split_idx = [0]
        for i in range(len(x)):
            split_idx.append(split_idx[-1]+x[i].shape[0])
        if sparse:
            x = torch.cat([torch.tensor(x[i].toarray()) for i in range(len(x))]).to(self.device)
        else:
            x = torch.cat([torch.tensor(x[i]) for i in range(len(x))]).to(self.device)
        y = y.to(self.device)
        if ct is not None:
            ct = torch.tensor([j for i in ct for j in i]).to(self.device)
        z = self.encode(x)
        x_hat = self.decode(z)
        c2p_dists = torch.pow(z[:, None] - self.prototypes[None, :], 2).sum(-1)
        if ct is None:
            p2c_dists = torch.pow(self.prototypes[:, None] - z[None, :], 2).sum(-1)
        else:
            p2c_dists = torch.stack([torch.pow(self.prototypes[:, None] - z[ct == t][None, :], 2).sum(-1).mean(-1) for t in ct.unique().tolist()]).T # n_proto * n_ct
        p2p_dists = (torch.pow(self.prototypes[:, None] - self.prototypes[None, :], 2).sum(-1)+1e-16).sqrt()
        recon_loss = (x - x_hat).pow(2).mean()
        c2p_loss = (c2p_dists).min(dim=1)[0].mean()
        p2c_loss = (p2c_dists + (torch.ones_like(p2c_dists).uniform_() < 0.3) * 1e9).min(dim=1)[0].mean()
        p2p_loss = ((self.d_min - p2p_dists > 0) * (self.d_min - p2p_dists)).pow(2).sum() / (self.n_proto * self.n_proto - self.n_proto)
        if self.n_ct is not None and ct is not None:
            ct_logits = self.ct_clf1(1 / (c2p_dists+0.5))
            ct_loss = self.ce_(ct_logits, ct)
        else:
            ct_loss = 0
        total_loss = self.lambda_1 * recon_loss +\
                     self.lambda_2 * c2p_loss +\
                     self.lambda_3 * p2c_loss +\
                     self.lambda_4 * p2p_loss +\
                     self.lambda_5 * ct_loss
        if ct is not None:
            return total_loss, ct_logits
        return total_loss


class OurDataset(Dataset):
    def __init__(self, X, y, cell_id=None, gene_id=None, class_id=None, ct=None, ct_id=None):
        self.X = X
        self.y = y
        self.cell_id = cell_id
        self.gene_id = gene_id
        self.class_id = class_id
        self.ct = ct
        self.ct_id = ct_id
    def __getitem__(self, i):
        if self.ct_id is not None:
            return self.X[i], self.y[i], self.ct[i]
        return self.X[i], self.y[i]
    def __len__(self):
        return len(self.y)







path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/lupus/'
adata = sc.read_h5ad(path + 'lupus_processed.h5ad')
# AnnData object with n_obs × n_vars = 834096 × 2000
#     obs: 'disease_cov', 'ct_cov', 'pop_cov', 'ind_cov', 'well', 'batch_cov', 'batch'
#     var: 'gene_ids-0-0-0-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0', 'gene_ids-1-0-0-0-0', 'gene_ids-1-0-0-0', 'gene_ids-1-0-0', 'gene_ids-1-0', 'gene_ids-1', 'n_cells', 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm'
#     uns: 'hvg', 'log1p'
adata.obs.disease_cov.value_counts()
# disease_cov
# sle        557630
# healthy    276466
adata.obs.pop_cov.value_counts()
# pop_cov
# WHITE    533178
# ASIAN    300918
adata_full = adata.copy()
adata.obs.ind_cov.value_counts()    # 169
adata.obs[['pop_cov','ind_cov']].value_counts().sort_index()



task_name = 'lupus_race_cls'
sample_key = 'ind_cov'
task_key = 'pop_cov'
ct_key = 'ct_cov'

class_num = 2
folds = pd.read_csv(path + 'race_5folds.csv', index_col=0)


n_proto = 4
path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/benchmarking/ProtoCell4P/proto_' + str(n_proto) + '/'


res_stat_dict = {}
for SEED in range(10):
    for fold in [1,2,3,4,5]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folds_trainval = folds[folds['fold'] != fold]
        folds_test = folds[folds['fold'] == fold]
        adata = adata_full[adata_full.obs[sample_key].isin(folds_trainval['samples'])]
        adata_test = adata_full[adata_full.obs[sample_key].isin(folds_test['samples'])]
        barcodes = adata.obs.index.tolist()
        genes = adata.var.index.tolist()
        meta = adata.obs.copy()
        cell_types = meta[ct_key]
        ct_id = sorted(set(cell_types))
        mapping_ct = {c:idx for idx, c in enumerate(ct_id)}
        X = []
        y = []
        ct = []
        for ind in tqdm.tqdm(sorted(set(meta[sample_key]))):
                disease = list(set(meta[task_key][meta[sample_key] == ind]))
                x = adata.X[meta[sample_key] == ind]
                X.append(x)
                y.append(disease[0])
                ct.append([mapping_ct[c] for c in cell_types[meta[sample_key] == ind]])
        class_id = sorted(set(y))
        mapping = {c:idx for idx, c in enumerate(class_id)}
        y = [mapping[c] for c in y]
        len(y)
        # 43
        dataset = OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id, ct=ct, ct_id=ct_id)
        # dataset = OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id)   
        from sklearn.model_selection import train_test_split
        train_set, val_set = train_test_split(dataset, test_size=0.2, random_state=SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lr = 1e-4
        max_epoch = 20
        batch_size = 4
        test_step = 1
        h_dim = 128
        z_dim = 32
        d_min = 1
        n_layers = 2
        n_classes = max(dataset.y) + 1
        if dataset.ct is None:
            n_ct = None
        else:
            n_ct = len(dataset.ct_id)
        ct_id = dataset.ct_id
        class_id = dataset.class_id
        pretrained=False
        max_epoch_pretrain = 0
        lr_pretrain = 1e-2
        lambda_1,lambda_2, lambda_3, lambda_4, lambda_5, lambda_6 = 1,1,1,1,1,1
        lambdas = {
                    "lambda_1": lambda_1,
                    "lambda_2": lambda_2,
                    "lambda_3": lambda_3,
                    "lambda_4": lambda_4,
                    "lambda_5": 0 if n_ct is None else lambda_5,
                    "lambda_6": 0 if n_ct is None else lambda_6
                }
        model_type = "ProtoCell"
        model = ProtoCell(dataset.X[0].shape[1], h_dim, z_dim, n_layers, n_proto, n_classes, lambdas, n_ct, device, d_min) 
        model.to(device)
        def my_collate(batch):
                x = [item[0] for item in batch]
                y = torch.tensor([item[1] for item in batch])
                if len(batch[0]) == 3:
                    ct = [item[2] for item in batch]
                    return x, y, ct
                return x, y
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, collate_fn=my_collate)
        val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False, collate_fn=my_collate)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        best_epoch = 0
        best_metric = 0
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
        for epoch in range(max_epoch):
            train_loss = 0
            y_truth = []
            y_logits = []
            if n_ct is not None:
                ct_truth = []
                ct_logits = []
            for bat in train_loader:
                x = bat[0]
                y = bat[1]
                optim.zero_grad()
                if n_ct is not None:
                    loss, logits, ct_logit = model(*bat, sparse=True)    
                else:
                    loss, logits = model(*bat, sparse=True)
                if not pretrained:
                    if n_ct is not None:
                        loss += model.pretrain(*bat, sparse=True)[0]
                    else:
                        loss += model.pretrain(*bat, sparse=True)                    
                loss.backward()
                optim.step()
                train_loss += loss.item() * len(x)
                y_truth.append(y)
                y_logits.append(torch.softmax(logits, dim=1))
                if n_ct is not None:
                    ct_truth.append(torch.tensor([j for i in bat[2] for j in i]))
                    ct_logits.append(torch.softmax(ct_logit, dim=1))
            y_truth = torch.cat(y_truth)
            y_logits = torch.cat(y_logits)
            y_pred = y_logits.argmax(dim=1)
            train_acc = accuracy_score(y_truth.cpu(), y_pred.cpu())
            if y_logits.shape[1] == 2:
                train_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu().detach()[:,1])
            else:
                train_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu().detach(), multi_class="ovo")
            # train_f1 = f1_score(y_truth.cpu(), y_pred.cpu())
            train_f1 = f1_score(y_truth.cpu(), y_pred.cpu(), average="macro")
            if n_ct is not None:
                ct_truth = torch.cat(ct_truth)
                ct_logits = torch.cat(ct_logits)
                ct_pred = ct_logits.argmax(dim=1)
                ct_acc = accuracy_score(ct_truth.cpu(), ct_pred.cpu())
                print("Avg. Training Loss: {:.2f} | Avg. Training Accuracy: {:.2f} | Avg. ROC AUC Score: {:.2f} | F1 Score: {:.2f} | CT Acc: {:.2f}".format(train_loss / len(train_set), train_acc, train_auc, train_f1, ct_acc))
            else:
                print("Avg. Training Loss: {:.2f} | Avg. Training Accuracy: {:.2f} | Avg. ROC AUC Score: {:.2f} | F1 Score: {:.2f}".format(train_loss / len(train_set), train_acc, train_auc, train_f1))
            if (epoch + 1) % test_step == 0:
                model.eval()
                val_loss = 0
                y_truth = []
                y_logits = []
                if n_ct is not None:
                    ct_truth = []
                    ct_logits = []
                with torch.no_grad():
                    for bat in val_loader:
                        x = bat[0]
                        y = bat[1]
                        if n_ct is not None:
                            loss, logits, ct_logit = model(*bat, sparse=True)
                        else:
                            loss, logits = model(*bat, sparse=True)
                        val_loss += loss.item() * len(x)
                        y_truth.append(y)
                        y_logits.append(torch.softmax(logits, dim=1))
                        if n_ct is not None:
                            ct_truth.append(torch.tensor([j for i in bat[2] for j in i]))
                            ct_logits.append(torch.softmax(ct_logit, dim=1))
                    y_truth = torch.cat(y_truth)
                    y_logits = torch.cat(y_logits)
                    y_pred = y_logits.argmax(dim=1)
                    val_acc = accuracy_score(y_truth.cpu(), y_pred.cpu())
                    if y_logits.shape[1] == 2:
                        val_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu().detach()[:,1])
                    else:
                        val_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu().detach(), multi_class="ovo")
                    # val_f1 = f1_score(y_truth.cpu(), y_pred.cpu())
                    val_f1 = f1_score(y_truth.cpu(), y_pred.cpu(), average="macro")                     
                if n_ct is not None:
                    ct_truth = torch.cat(ct_truth)
                    ct_logits = torch.cat(ct_logits)
                    ct_pred = ct_logits.argmax(dim=1)
                    ct_acc = accuracy_score(ct_truth.cpu(), ct_pred.cpu())
                    # curr_metric = val_f1 + ct_acc / 10
                    curr_metric = val_auc + ct_acc / 10
                    print("Avg. Validation Loss: {:.2f} | Avg. Validation Accuracy: {:.2f} | Avg. ROC AUC Score: {:.2f} | F1 Score: {:.2f} | CT Acc: {:.2f}".format(val_loss / len(val_set), val_acc, val_auc, val_f1, ct_acc))
                else:
                    curr_metric = val_auc
                    print("Avg. Validation Loss: {:.2f} | Avg. Validation Accuracy: {:.2f} | Avg. ROC AUC Score: {:.2f} | F1 Score: {:.2f}".format(val_loss / len(val_set), val_acc, val_auc, val_f1))
                if curr_metric > best_metric:
                    torch.save(model, path + "best_model_" + task_name + '_' + str(fold) + '_seed_' + str(SEED) + ".pt")
                    best_metric = curr_metric
                    best_epoch = epoch
                    print("Model Saved!")
                model.train()
        model = torch.load(path + "best_model_" + task_name + '_' + str(fold) + '_seed_' + str(SEED) + ".pt")
        model.eval()
        train_loader_ = DataLoader(train_set, batch_size = len(train_set), shuffle = True, collate_fn=my_collate)
        val_loader_ = DataLoader(val_set, batch_size = len(val_set), shuffle = False, collate_fn=my_collate)
        for bat in train_loader_:
            x = bat[0]
            y = bat[1]
            if n_ct is not None:
                loss, logits, ct_logit = model(*bat, sparse=True)
            else:
                loss, logits = model(*bat, sparse=True)
            y_truth_tr = y.clone().cpu()
            y_logits_tr = torch.softmax(logits, dim=1).detach().cpu()
            y_pred_tr = y_logits_tr.argmax(dim=1)
        if y_logits_tr.shape[1] == 2:
            auROC_tr = roc_auc_score(y_truth_tr, y_logits_tr[:,1]) # 0.9761904761904763
            auROC_tr = roc_auc_score(y_truth_tr, y_pred_tr) # 0.8432539682539683
        else: 
            auROC_tr = roc_auc_score(y_truth_tr, y_logits_tr, multi_class="ovo")
        macroF1_tr = f1_score(y_truth_tr, y_pred_tr, average="macro")      
        for bat in val_loader_:
            x = bat[0]
            y = bat[1]
            if n_ct is not None:
                loss, logits, ct_logit = model(*bat, sparse=True)
            else:
                loss, logits = model(*bat, sparse=True)
            y_truth_val = y.clone().cpu()
            y_logits_val= torch.softmax(logits, dim=1).detach().cpu()
            y_pred_val = y_logits_val.argmax(dim=1)
        if y_logits_val.shape[1] == 2:
            auROC_val = roc_auc_score(y_truth_val, y_logits_val[:,1]) # 0.9761904761904763
            auROC_val = roc_auc_score(y_truth_val, y_pred_val) # 0.8432539682539683
        else: 
            auROC_val = roc_auc_score(y_truth_val, y_logits_val, multi_class="ovo")
        macroF1_val = f1_score(y_truth_val, y_pred_val, average="macro")      
        barcodes = adata_test.obs.index.tolist()
        genes = adata_test.var.index.tolist()
        meta = adata_test.obs.copy()
        cell_types = meta[ct_key]
        ct_id = sorted(set(cell_types))
        mapping_ct = {c:idx for idx, c in enumerate(ct_id)}
        X = []
        y = []
        ct = []
        for ind in tqdm.tqdm(sorted(set(meta[sample_key]))):
                disease = list(set(meta[task_key][meta[sample_key] == ind]))
                x = adata_test.X[meta[sample_key] == ind]
                X.append(x)
                y.append(disease[0])
                ct.append([mapping_ct[c] for c in cell_types[meta[sample_key] == ind]])
        class_id = sorted(set(y))
        mapping = {c:idx for idx, c in enumerate(class_id)}
        y = [mapping[c] for c in y]
        len(y)
        # 43
        test_set = OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id, ct=ct, ct_id=ct_id)
        # dataset = OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id)   
        test_loader = DataLoader(test_set, batch_size = len(test_set), shuffle = True, collate_fn=my_collate)
        for bat in test_loader:
            x = bat[0]
            y = bat[1]
            if n_ct is not None:
                loss, logits, ct_logit = model(*bat, sparse=True)
            else:
                loss, logits = model(*bat, sparse=True)
            y_truth_te = y.clone().cpu()
            y_logits_te = torch.softmax(logits, dim=1).detach().cpu()
            y_pred_te = y_logits_te .argmax(dim=1)
        if y_logits_te.shape[1] == 2:
            auROC_te = roc_auc_score(y_truth_te, y_logits_te[:,1])
            auROC_te = roc_auc_score(y_truth_te, y_pred_te)
        else: 
            auROC_te = roc_auc_score(y_truth_te, y_logits_te, multi_class="ovo")
        macroF1_te = f1_score(y_truth_te, y_pred_te, average="macro")      
        print(sample_key, '\n',
                    "auROC_tr", auROC_tr,'\n',
                    "auROC_val", auROC_val,'\n',
                    "macroF1_tr", macroF1_tr,'\n',
                    "macroF1_val", macroF1_val,'\n',
                    "auROC_test", auROC_te,'\n',
                    "macroF1_test", macroF1_te,'\n',
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
                    'auROC_test': auROC_te,
                    'macroF1_test': macroF1_te,
                    'fold': fold
                }, index=[0]
            )
        res_df = res_df.T
        res_df.columns = [task_name]
        res_stat_dict['fold_' + str(fold) + '_seed_' + str(SEED)] = res_df
        res_df.to_csv(path + task_name + '_results_fold_' + str(fold) + '_seed_' + str(SEED) +  '.csv')




torch.save(res_stat_dict, path + task_name + '_results_all_folds.pth')



res_stat_dict.keys()


res_test_df = pd.DataFrame(index=list(res_stat_dict.keys()), 
                           columns=['auROC_test','macroF1_test'])


for i in list(res_stat_dict.keys()):
     print(i)
     res_test_df.loc[i,'auROC_test'] = res_stat_dict[i].loc['auROC_test'].values
     res_test_df.loc[i,'macroF1_test'] = res_stat_dict[i].loc['macroF1_test'].values


res_test_df.mean()

# Lupus 4 8 16
# auROC_test      0.784504
# macroF1_test    0.783053

# auROC_test      0.839732
# macroF1_test    0.851749

# auROC_test      0.915493
# macroF1_test    0.921419


# race 4 8 16
# res_test_df.mean()
# auROC_test      0.641647
# macroF1_test    0.608486

# auROC_test      0.669185
# macroF1_test    0.648723

# auROC_test      0.734267
# macroF1_test    0.723932