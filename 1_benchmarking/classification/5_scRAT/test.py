# (scRAT) hongruhu@farm:/group/gquongrp/workspaces/hongruhu/bioPointNet/benchmarking/scRAT$

import numpy as np
import pickle
import scanpy


def Custom_data(dataset, sample_key='donor_id', task_key='disease_id', ct_key='celltype', task_dict={},
                pca=False):
    id_dict = {}   # {'cancer': 1, 'health': 0}
    data = dataset.copy() # adata
    if pca == True:
        origin = data.obsm['X_pca']
    else:
        origin = data.X
    patient_id = data.obs[sample_key]
    labels = data.obs[task_key]
    cell_type = data.obs[ct_key]
    cell_type_large = None
    # This (high resolution) cell_type is only for attention analysis, not necessary
    # cell_type_large = data.obs['cell_type_large']
    labels_ = np.array(labels.map(id_dict))
    l_dict = {}
    indices = np.arange(origin.shape[0])
    p_ids = sorted(set(patient_id))
    p_idx = []
    for i in p_ids:
        idx = indices[patient_id == i]
        if len(set(labels_[idx])) > 1:      # one patient with more than one labels
            for ii in sorted(set(labels_[idx])):
                if ii > -1:
                    iidx = idx[labels_[idx] == ii]
                    tt_idx = iidx
                    if len(tt_idx) < 500:   # exclude the sample with the number of cells fewer than 500
                        continue
                    p_idx.append(tt_idx)
                    l_dict[labels_[iidx[0]]] = l_dict.get(labels_[iidx[0]], 0) + 1
        else:
            if labels_[idx[0]] > -1:
                tt_idx = idx
                if len(tt_idx) < 500:  # exclude the sample with the number of cells fewer than 500
                    continue
                p_idx.append(tt_idx)
                l_dict[labels_[idx[0]]] = l_dict.get(labels_[idx[0]], 0) + 1
    # print(l_dict)
    return p_idx, labels_, cell_type, patient_id, origin, cell_type_large



from sklearn import metrics
from sklearn.metrics import accuracy_score
import scipy.stats as st
from torch.optim import Adam
from utils import *
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split

from model_baseline import *
from Transformer import TransformerPredictor

