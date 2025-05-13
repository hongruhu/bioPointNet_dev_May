# (cloudpred) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet/benchmarking/CloudPred$ python

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset




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



# #----------------------------------------------------------#
data = adata.copy()
meta = adata.obs.copy()
import os
import tqdm
import math
import scipy
import pathlib
for state in set(data.obs[task_key]):
    pathlib.Path(os.path.join("data", task_name, state)).mkdir(parents=True, exist_ok=True)


for ind in tqdm.tqdm(set(data.obs[sample_key])):
        state = list(set(data.obs[data.obs[sample_key] == ind][task_key]))
        assert(len(state) == 1)
        state = state[0]
        X = data.X[(data.obs[sample_key] == ind).values, :].transpose()
        scipy.sparse.save_npz(os.path.join("data", task_name, state, "{}.npz".format(ind)), X)
        np.save(os.path.join("data", task_name, state, "ct_{}.npy".format(ind)),
                data.obs[data.obs[sample_key] == ind][ct_key].values)


# #----------------------------------------------------------#
# # run_all.sh
# # centers="5 10 15 20 25"
# # for seed in `seq 25`
# # do
# # python3 -m cloudpred data/lupus      
# #         -t log --logfile log/lupus/cloudpred_${seed}      
# #         --cloudpred  
# #         --linear     
# #         --generative 
# #         --genpat     
# #         --deepset    
# #         --centers ${centers} 
# #         --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_${seed}_
import copy
import random
import logging.config
import traceback
import pickle
import glob

def load_counts(filename):
    counts = scipy.sparse.load_npz(filename)
    counts = counts.astype(float)
    ct_filename = os.path.join(os.path.dirname(filename), "ct_" + os.path.splitext(os.path.basename(filename))[0] + ".npy")
    if os.path.isfile(ct_filename):
        ct = np.load(ct_filename, allow_pickle=True)
    else:
        ct = None
    return counts.transpose(), ct, filename.split('/')[-1].split('.')[0]


def scipy_sparse_to_pytorch(x):
    x = x.tocoo()
    v = torch.Tensor(x.data)
    i = torch.LongTensor([x.row, x.col])
    return torch.sparse.FloatTensor(i, v, x.shape)


dir = './data/' + task_name
Xall = []
state = []
sample_id_all = []
for dirname in sorted(glob.iglob(dir + "/*")):
    X = []
    for filename in tqdm.tqdm(sorted(glob.iglob(dirname + "/*.npz"))):
        X.append(load_counts(filename))
    X = list(map(lambda x: (x[0], len(state), x[1], x[2]), X))
    state.append(os.path.basename(dirname))
    Xall.append(X)


import cloudpred
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
def metric_output(X, model):
    y_true = []
    y_score = []
    for (start, (x, y, *_)) in enumerate(X):
        z = model(torch.Tensor(x))
        y_true.append(y)
        y_score.append(list(z[0].detach().numpy()))
    y_score = softmax(np.array(y_score), axis=1)
    class_num
    y_true = np.eye(class_num)[y_true]
    return np.mean([
            roc_auc_score(y_true[:, c], y_score[:, c])
            for c in range(y_true.shape[1])
            if len(np.unique(y_true[:, c])) > 1
            ]), f1_score(y_true.argmax(1), y_score.argmax(1), average="macro")  


def deepset_train(Xtrain, Xvalid, centers=2, class_num=2, regression=False):
    outputs = class_num
    classifier = torch.nn.Sequential(torch.nn.Linear(Xtrain[0][0].shape[1], centers), 
                                     torch.nn.ReLU(), 
                                     torch.nn.Linear(centers, centers), 
                                     torch.nn.ReLU(), 
                                     cloudpred.utils.Aggregator(), 
                                     torch.nn.Linear(centers, centers), 
                                     torch.nn.ReLU(), 
                                     torch.nn.Linear(centers, outputs))
    reg = None
    return cloudpred.utils.train_classifier(Xtrain, Xvalid, [], classifier, regularize=reg, iterations=100, eta=1e-4, stochastic=True, regression=regression)



project_pc = False
res_stat_dict = {}
for SEED in range(10):
    for fold in [1,2,3,4,5]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folds_trainval = folds[folds['fold'] != fold]
        folds_test = folds[folds['fold'] == fold]
        Xtest = [[x for x in class_group if x[-1] in folds_test.samples.tolist()] for class_group in Xall]
        Xtest = [x for class_group in Xtest for x in class_group]
        Xtrainval = [[x for x in class_group if x[-1] in folds_trainval.samples.tolist()] for class_group in Xall]
        Xtrainval = [x for class_group in Xtrainval for x in class_group]
        from sklearn.model_selection import train_test_split
        random.seed(SEED)
        np.random.seed(SEED)
        Xtrain, Xval = train_test_split(Xtrainval, test_size=0.2, random_state=SEED)
        # Xtrain, Xval, Xtest, state
        if project_pc == True:
            dims = 50
            iterations = 25
            transform = 'none'
            figroot = None
            pc = cloudpred.utils.train_pca_autoencoder(Xtrain = scipy.sparse.vstack(map(lambda x: x[0], Xtrain)), # N x 3000 genes
                                                    Ytrain = None,
                                                    Xtest = scipy.sparse.vstack(map(lambda x: x[0], Xval)),    # N_ x 3000 genes
                                                    Ytest = None,
                                                    dims = dims, transform = transform, iterations=iterations)
            # pc.shape # genes x dims
            pc = pc[:, :dims] # same
            ### Project onto principal components ###
            mu = scipy.sparse.vstack(list(map(lambda x: x[0], Xtrain))).mean(axis=0)            # 1 x genes 
            Xtrain = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xtrain))    # - np.asarray(mu.dot(pc))
            Xval = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xval))      # - np.asarray(mu.dot(pc))
            Xtest  = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xtest))     # - np.asarray(mu.dot(pc))
            full = np.concatenate(list(map(lambda x: x[0], Xtrain))) # N x 50
            mu = np.mean(full, axis=0)                               # 1 x 50   
            sigma = np.sqrt(np.mean(np.square(full - mu), axis=0))   # 1 x 50
            sigma = sigma[0, 0]
            Xtrain = list(map(lambda x: (np.array((x[0] - mu) / sigma), *x[1:]), Xtrain))  # - np.asarray(mu.dot(pc))
            Xval = list(map(lambda x: (np.array((x[0] - mu) / sigma), *x[1:]), Xval))  # - np.asarray(mu.dot(pc))
            Xtest  = list(map(lambda x: (np.array((x[0] - mu) / sigma), *x[1:]), Xtest))   # - np.asarray(mu.dot(pc))
        else:
            Xtrain = list(map(lambda x: (np.array(x[0].todense()), *x[1:]), Xtrain))
            Xval = list(map(lambda x: (np.array(x[0].todense()), *x[1:]), Xval))
            Xtest  = list(map(lambda x: (np.array(x[0].todense()), *x[1:]), Xtest))
        # deepset
        centers = 5
        model, _ = deepset_train(Xtrain, Xval, centers, class_num=class_num, regression=False) 
        auROC_tr, macroF1_tr = metric_output(Xtrain, model=model)
        auROC_val, macroF1_val = metric_output(Xval, model=model)
        auROC_te, macroF1_te = metric_output(Xtest, model=model)
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
        res_df.to_csv('../DeepSet/' + task_name + '/' + task_name + '_results_fold_' + str(fold) + '_seed_' + str(SEED) +  '.csv')


torch.save(res_stat_dict, '../DeepSet/' + task_name + '/' + task_name + '_results_all_folds.pth')
res_stat_dict.keys()
res_test_df = pd.DataFrame(index=list(res_stat_dict.keys()), 
                           columns=['auROC_test','macroF1_test'])
for i in list(res_stat_dict.keys()):
     print(i)
     res_test_df.loc[i,'auROC_test'] = res_stat_dict[i].loc['auROC_test'].values
     res_test_df.loc[i,'macroF1_test'] = res_stat_dict[i].loc['macroF1_test'].values


res_test_df.to_csv('../DeepSet/' + task_name + '/' + task_name + '_results_all_folds.csv')
res_test_df.mean()


project_pc = True
res_stat_dict_pc = {}
for SEED in range(10):
    for fold in [1,2,3,4,5]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folds_trainval = folds[folds['fold'] != fold]
        folds_test = folds[folds['fold'] == fold]
        Xtest = [[x for x in class_group if x[-1] in folds_test.samples.tolist()] for class_group in Xall]
        Xtest = [x for class_group in Xtest for x in class_group]
        Xtrainval = [[x for x in class_group if x[-1] in folds_trainval.samples.tolist()] for class_group in Xall]
        Xtrainval = [x for class_group in Xtrainval for x in class_group]
        from sklearn.model_selection import train_test_split
        random.seed(SEED)
        np.random.seed(SEED)
        Xtrain, Xval = train_test_split(Xtrainval, test_size=0.2, random_state=SEED)
        # Xtrain, Xval, Xtest, state
        if project_pc == True:
            dims = 50
            iterations = 25
            transform = 'none'
            figroot = None
            pc = cloudpred.utils.train_pca_autoencoder(Xtrain = scipy.sparse.vstack(map(lambda x: x[0], Xtrain)), # N x 3000 genes
                                                    Ytrain = None,
                                                    Xtest = scipy.sparse.vstack(map(lambda x: x[0], Xval)),    # N_ x 3000 genes
                                                    Ytest = None,
                                                    dims = dims, transform = transform, iterations=iterations)
            # pc.shape # genes x dims
            pc = pc[:, :dims] # same
            ### Project onto principal components ###
            mu = scipy.sparse.vstack(list(map(lambda x: x[0], Xtrain))).mean(axis=0)            # 1 x genes 
            Xtrain = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xtrain))    # - np.asarray(mu.dot(pc))
            Xval = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xval))      # - np.asarray(mu.dot(pc))
            Xtest  = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xtest))     # - np.asarray(mu.dot(pc))
            full = np.concatenate(list(map(lambda x: x[0], Xtrain))) # N x 50
            mu = np.mean(full, axis=0)                               # 1 x 50   
            sigma = np.sqrt(np.mean(np.square(full - mu), axis=0))   # 1 x 50
            sigma = sigma[0, 0]
            Xtrain = list(map(lambda x: (np.array((x[0] - mu) / sigma), *x[1:]), Xtrain))  # - np.asarray(mu.dot(pc))
            Xval = list(map(lambda x: (np.array((x[0] - mu) / sigma), *x[1:]), Xval))  # - np.asarray(mu.dot(pc))
            Xtest  = list(map(lambda x: (np.array((x[0] - mu) / sigma), *x[1:]), Xtest))   # - np.asarray(mu.dot(pc))
        else:
            Xtrain = list(map(lambda x: (np.array(x[0].todense()), *x[1:]), Xtrain))
            Xval = list(map(lambda x: (np.array(x[0].todense()), *x[1:]), Xval))
            Xtest  = list(map(lambda x: (np.array(x[0].todense()), *x[1:]), Xtest))
        # deepset
        centers = 5
        model, _ = deepset_train(Xtrain, Xval, centers, class_num=class_num, regression=False) 
        auROC_tr, macroF1_tr = metric_output(Xtrain, model=model)
        auROC_val, macroF1_val = metric_output(Xval, model=model)
        auROC_te, macroF1_te = metric_output(Xtest, model=model)
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
        res_stat_dict_pc['fold_' + str(fold) + '_seed_' + str(SEED)] = res_df
        res_df.to_csv('../DeepSet/' + task_name + '/' + 'PC_' + task_name + '_results_fold_' + str(fold) + '_seed_' + str(SEED) +  '.csv')


torch.save(res_stat_dict_pc, '../DeepSet/' + task_name + '/' + 'PC_' + task_name + '_results_all_folds.pth')
res_stat_dict_pc.keys()
res_test_df_pc = pd.DataFrame(index=list(res_stat_dict_pc.keys()), 
                           columns=['auROC_test','macroF1_test'])
for i in list(res_stat_dict_pc.keys()):
     print(i)
     res_test_df_pc.loc[i,'auROC_test'] = res_stat_dict_pc[i].loc['auROC_test'].values
     res_test_df_pc.loc[i,'macroF1_test'] = res_stat_dict_pc[i].loc['macroF1_test'].values


res_test_df_pc.to_csv('../DeepSet/' + task_name + '/' + 'PC_' + task_name + '_results_all_folds.csv')
res_test_df_pc.mean()




task_name
# 'lupus_race_cls'
res_test_df.mean()
# auROC_test      0.834024
# macroF1_test    0.702351
res_test_df_pc.mean()
# auROC_test      0.581455
# macroF1_test     0.38257