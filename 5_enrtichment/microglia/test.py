# (mil) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet$

path = '/group/gquongrp/workspaces/hongruhu/sciLaMA_graph/MTG_microglia/'

import scanpy as sc
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns


loading_path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/SEAAD_Glia/MTG/MTG_cls/correct_sample/split_from_all/Split_sex_cov_ALT/'
split_list = [0,2,3]
celltype_key = 'Microglia-PVM'

cell_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/SEAAD_Glia/MTG/Microglia-PVM_MTG_sciMultiLaMA_CELL_embedding.csv', index_col=0)
gene_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/SEAAD_Glia/MTG/Microglia-PVM_MTG_sciMultiLaMA_GENE_embedding.csv', index_col=0)
# [35887 rows x 50 columns] [15507 rows x 50 columns]
# cell_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/SEAAD_Glia/MTG/MTG_sciMultiLaMA_CELL_embedding.csv', index_col=0)
# gene_emb = pd.read_csv('/group/gquongrp/workspaces/hongruhu/bioPointNet/SEAAD_Glia/MTG/MTG_sciMultiLaMA_GENE_embedding.csv', index_col=0)
# # [226669 rows x 50 columns] [15909 rows x 50 columns]


obj_all = sc.read_h5ad('/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/MTG_glia_intersection_norm.h5ad')
obj_all = obj_all[cell_emb.index,gene_emb.index]
marker_list = ['IL1B', 'CSF1R', 'STAB1', 'NINJ1', 'JAK3',
'IRF1', 'IRF7', 'IFI16',
'FCGR1A', 'FCGR1B', 'FCGR2A', 'FCGR3B', 
'CD74', 'HLA-DRB5',
'C1QA', 'C1QB',
'CSF1R', 'CTSC', 'C1QA', 'C1QB', 'LY86', 'FCGR3A'
'CTSD', 'CTSS',
'LYZ',
'APOE',
'RUNX1', 'IKZF1', 'NFATC2', 'MAF']
len(marker_list)
# 29

split_id = 0
adata_all = sc.read(loading_path + celltype_key + '/' + 'Adata_Tr_' + celltype_key + '_' + str(split_id)+ '.h5ad')


obj_disease = adata_all[adata_all.obs.disease == 'dementia']
cell_emb_disease = cell_emb.loc[obj_disease.obs.index]
obj_control = adata_all[adata_all.obs.disease == 'normal']
cell_emb_control = cell_emb.loc[obj_control.obs.index]

for i in range(cell_emb.shape[1]):
    print(i)
    fig = plt.figure(figsize=(6,6))
    sns.scatterplot(x=cell_emb_disease.iloc[:,0].tolist(),
                y=cell_emb_disease.iloc[:,0].tolist(),
                hue=obj_disease.obs['attention_score_norm_cellnum'])
    fig.savefig('./disease/umap_disease_Mic_attn_norm_dim_' + str(i) + '_.png')
    fig = plt.figure(figsize=(6,6))
    sns.scatterplot(x=cell_emb_disease.iloc[:,0].tolist(),
                y=cell_emb_disease.iloc[:,0].tolist(),
                hue=obj_disease.obs['Supertype'])
    fig.savefig('./disease/umap_disease_Mic_attn_Subtype_dim_' + str(i) + '_.png')
    fig = plt.figure(figsize=(6,6))
    sns.scatterplot(x=cell_emb_control.iloc[:,0].tolist(),
                y=cell_emb_control.iloc[:,0].tolist(),
                hue=obj_control.obs['attention_score_norm_cellnum'])
    fig.savefig('./healthy/umap_normal_Mic_attn_norm_dim_' + str(i) + '_.png')
    fig = plt.figure(figsize=(6,6))
    sns.scatterplot(x=cell_emb_control.iloc[:,0].tolist(),
                y=cell_emb_control.iloc[:,0].tolist(),
                hue=obj_control.obs['Supertype'])
    fig.savefig('./healthy/umap_normal_Mic_attn_Subtype_dim_' + str(i) + '_.png')






import matplotlib.pyplot as plt
import seaborn as sns

for i in range(cell_emb.shape[1]):
    print(i)
    # Plot disease attention with point size proportional to attention score
    fig = plt.figure(figsize=(6,6))
    sns.scatterplot(
        x=cell_emb_disease.iloc[:, i],
        y=cell_emb_disease.iloc[:, i],
        hue=obj_disease.obs['attention_score_norm_cellnum'],
        size=obj_disease.obs['attention_score_norm_cellnum'],
        sizes=(10, 200),  # control min and max size
        palette="viridis",
        alpha=0.8,
    )
    plt.legend(title='Attention Score', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.savefig(f'./disease/umap_disease_Mic_attn_norm_dim_{i}_.png', bbox_inches='tight')
    plt.close(fig)
    # # Plot disease Supertype with special handling for "Micro-PVM_3-SEAAD"
    # fig = plt.figure(figsize=(6,6))
    # supertype = obj_disease.obs['Supertype']
    # highlight = (supertype == "Micro-PVM_3-SEAAD")
    # sns.scatterplot(
    #     x=cell_emb_disease.iloc[:, 0],
    #     y=cell_emb_disease.iloc[:, i],
    #     hue=supertype,
    #     palette="tab20",
    #     size=highlight.map({True: 200, False: 50}),
    #     sizes=(50, 200),
    #     alpha=highlight.map({True: 1.0, False: 0.3})
    # )
    # plt.legend(title='Supertype', bbox_to_anchor=(1.05, 1), loc='upper left')
    # fig.savefig(f'./disease/umap_disease_Mic_attn_Subtype_dim_{i}_.png', bbox_inches='tight')
    # plt.close(fig)
    # Plot control attention
    fig = plt.figure(figsize=(6,6))
    sns.scatterplot(
        x=cell_emb_control.iloc[:, i],
        y=cell_emb_control.iloc[:, i],
        hue=obj_control.obs['attention_score_norm_cellnum'],
        size=obj_control.obs['attention_score_norm_cellnum'],
        sizes=(10, 200),
        palette="viridis",
        alpha=0.8,
    )
    plt.legend(title='Attention Score', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.savefig(f'./healthy/umap_normal_Mic_attn_norm_dim_{i}_.png', bbox_inches='tight')
    plt.close(fig)
    # Plot control Supertype
    # fig = plt.figure(figsize=(6,6))
    # supertype = obj_control.obs['Supertype']
    # highlight = (supertype == "Micro-PVM_3-SEAAD")
    # sns.scatterplot(
    #     x=cell_emb_control.iloc[:, 0],
    #     y=cell_emb_control.iloc[:, i],
    #     hue=supertype,
    #     palette="tab20",
    #     size=highlight.map({True: 200, False: 50}),
    #     sizes=(50, 200),
    #     alpha=highlight.map({True: 1.0, False: 0.3})
    # )
    # plt.legend(title='Supertype', bbox_to_anchor=(1.05, 1), loc='upper left')
    # fig.savefig(f'./healthy/umap_normal_Mic_attn_Subtype_dim_{i}_.png', bbox_inches='tight')
    # plt.close(fig)



for i in [0,11,14,46]:
    for j in [0,11,14,46]:
        fig = plt.figure(figsize=(6,6))
        sns.scatterplot(
            x=cell_emb_disease.iloc[:, i],
            y=cell_emb_disease.iloc[:, j],
            hue=obj_disease.obs['attention_score_norm_cellnum'],
            size=obj_disease.obs['attention_score_norm_cellnum'],
            sizes=(10, 200),  # control min and max size
            palette="viridis",
            alpha=0.8,
        )
        plt.legend(title='Attention Score', bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.savefig(f'./disease/umap_disease_Mic_attn_norm_dim_{i}_{j}.png', bbox_inches='tight')
        plt.close(fig)
        fig = plt.figure(figsize=(6,6))
        sns.scatterplot(
            x=cell_emb_disease.iloc[:, i],
            y=cell_emb_disease.iloc[:, j],
            hue=obj_disease.obs['Supertype'],
            size=obj_disease.obs['attention_score_norm_cellnum'],
            sizes=(10, 200),  # control min and max size
            alpha=0.8,
        )
        plt.legend(title='Attention Score', bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.savefig(f'./disease/umap_disease_Mic_Supertype_dim_{i}_{j}.png', bbox_inches='tight')
        plt.close(fig)






for i in [11,14,26, 42, 44, 13, 25]:
    for j in [11,14,26, 42, 44, 13, 25]:
        if i !=j:
            fig = plt.figure(figsize=(6,6))
            sns.scatterplot(
                x=cell_emb_control.iloc[:, i],
                y=cell_emb_control.iloc[:, j],
                hue=obj_control.obs['attention_score_norm_cellnum'],
                size=obj_control.obs['attention_score_norm_cellnum'],
                sizes=(10, 200),  # control min and max size
                palette="viridis",
                alpha=0.8,
            )
            plt.legend(title='Attention Score', bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.savefig(f'./healthy/umap__Mic_attn_norm_dim_{i}_{j}.png', bbox_inches='tight')
            plt.close(fig)
            fig = plt.figure(figsize=(6,6))
            sns.scatterplot(
                x=cell_emb_control.iloc[:, i],
                y=cell_emb_control.iloc[:, j],
                hue=obj_control.obs['Supertype'],
                size=obj_control.obs['attention_score_norm_cellnum'],
                sizes=(10, 200),  # control min and max size
                alpha=0.8,
            )
            plt.legend(title='Attention Score', bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.savefig(f'./healthy/umap__Mic_Supertype_dim_{i}_{j}.png', bbox_inches='tight')
            plt.close(fig)