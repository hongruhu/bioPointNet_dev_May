# (sciLaMA_graph) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/sciLaMA_graph$

path = '/group/gquongrp/workspaces/hongruhu/sciLaMA_graph/MTG_microglia/'

import scanpy as sc
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from coembed import *


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
obj_all = obj_all[:,gene_emb.index]
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


for split_id in split_list:
    print(split_id)
    adata_all = sc.read(loading_path + celltype_key + '/' + 'Adata_Tr_' + celltype_key + '_' + str(split_id)+ '.h5ad')
    obj = obj_all[adata_all.obs.index]
    sc.pp.highly_variable_genes(obj, n_top_genes=3000, flavor='cell_ranger')
    obj = obj[:,obj.var.highly_variable==True]
    C = cell_emb.loc[adata_all.obs.index].values
    G = gene_emb.loc[obj.var.index].values
    cell_meta = adata_all.obs.copy()
    adata_C = sc.AnnData(X=C, obs=cell_meta)
    adata_G = sc.AnnData(X=G, obs=obj.var)
    dirname='graph_CG_GG_KNN_cossim' + '_' + celltype_key + '_' + str(split_id)
    pbg_params = dict(
        # These paths are filled dynamically by `gen_graph_cgc_cosine`
        entity_path="",
        edge_paths=[""],
        checkpoint_path="",
        # Graph structure (populated by the graph generator)
        entities={},        # ← gets filled with {'C': ..., 'G': ...}
        relations=[],       # ← gets filled based on enabled edge types
        dynamic_relations=False,
        # Embedding model
        dimension=30,           # Size of embedding space
        global_emb=False,       # No global embeddings
        comparator='dot',       # Can also use 'cosine' or 'l2' if needed
        # Training configuration
        num_epochs=10,
        workers=4,              # Number of CPU threads
        num_batch_negs=50,      # Negative samples per positive
        num_uniform_negs=50,    # Global uniform negatives
        loss_fn='softmax',
        lr=0.1,
        regularization_coef=0.0,
        wd=0.0,                 # Weight decay
        wd_interval=50,
        # Evaluation during training
        eval_fraction=0.05,
        eval_num_batch_negs=50,
        eval_num_uniform_negs=50,
        checkpoint_preservation_interval=None,
    )
    df_edges, entity_alias, graph_stats = gen_graph_cosine_knn(
        cell_emb=C,
        gene_emb=G,
        cell_ids=adata_C.obs.index,
        gene_ids=adata_G.obs.index,
        dirname=dirname,
        sim_threshold=0.3,      # still used for C–G
        cc_mode='knn',
        k_cc=10,
        gg_mode='knn',
        k_gg=10,
        include_cell_cell=True,
        include_gene_gene=True,
        pbg_params=pbg_params
    )
    pbg_train(
        dirname=dirname,
        output='model',
        auto_wd=True,
        save_wd=True,
        graph_stats=graph_stats,
        pbg_params=pbg_params
    )
    path_emb = pbg_params['checkpoint_path']
    path_entity = pbg_params['entity_path']
    num_epochs = pbg_params["num_epochs"]
    prefix = []
    path_entity_alias = Path(path_emb).parent.as_posix()
    df_entity_alias = pd.read_csv(
        os.path.join(path_entity_alias, 'entity_alias.txt'),
        header=0,
        index_col=0,
        sep='\t')
    df_entity_alias['id'] = df_entity_alias.index
    df_entity_alias.index = df_entity_alias['alias'].values
    convert_alias = True
    dict_adata = dict()
    for x in os.listdir(path_emb):
        if x.startswith('embeddings'):
            entity_type = x.split('_')[1]
            if (len(prefix) == 0) or (entity_type in prefix):
                adata = \
                    read_hdf(os.path.join(path_emb,
                                            f'embeddings_{entity_type}_0.'
                                            f'v{num_epochs}.h5'),
                                key="embeddings")
                with open(
                    os.path.join(path_entity,
                                    f'entity_names_{entity_type}_0.json'), "rt")\
                        as tf:
                    names_entity = json.load(tf)
                if convert_alias:
                    names_entity = \
                        df_entity_alias.loc[names_entity, 'id'].tolist()
                adata.obs.index = names_entity
                dict_adata[entity_type] = adata
    adata_C = dict_adata['C']  # embeddings of cells
    adata_G = dict_adata['G']  # embeddings of genes
    adata_C.obs['Supertype'] = obj_all[adata_C.obs.index].obs.Supertype.tolist()
    adata_C.obs['Supertype'].value_counts()
    import simba as si
    si.tl.umap(adata_C,n_neighbors=15,n_components=2)
    si.pl.umap(adata_C,color=['Supertype'],
                fig_size=(6,4),
                drawing_order='random')
    plt.savefig(dirname + '/cell_umap_cossim_KNN.png')
    # embed cells and genes into the same space
    adata_all = si.tl.embed(adata_ref=adata_C,list_adata_query=[adata_G])
    adata_all.obs.head()
    ## add annotations of cells and genes
    adata_all.obs['entity_anno'] = ""
    adata_all.obs.loc[adata_C.obs_names, 'entity_anno'] = adata_all.obs.loc[adata_C.obs_names, 'Supertype']
    adata_all.obs.loc[adata_G.obs_names, 'entity_anno'] = 'gene'
    adata_all.obs.head()
    si.tl.umap(adata_all,n_neighbors=15,n_components=2)
    si.pl.umap(adata_all,color=['id_dataset','entity_anno'],
            drawing_order='original',
            fig_size=(6,5))
    plt.savefig(dirname + '/cell_gene_umap_cossim_KNN.png')
    adata_all_umap = pd.DataFrame(adata_all.obsm['X_umap'], index=adata_all.obs.index)
    si.pl.umap(adata_all[adata_all.obs.id_dataset=='ref'], color=['entity_anno'],
            drawing_order='original',
            fig_size=(6,5))
    plt.savefig(dirname + '/cell_gene_umap_cossim_KNN_cell.png')
    adata_gene = adata_all[adata_all.obs.id_dataset!='ref']
    adata_gene.obs['feature_name'] = obj.var.loc[adata_gene.obs.index]['feature_name']
    adata_gene.obs['marker'] = 'No'
    mask = adata_gene.obs['feature_name'].isin(marker_list)
    adata_gene.obs.loc[mask, 'marker'] = 'Yes'
    si.pl.umap(adata_gene, color=['marker'],
            drawing_order='original',
            fig_size=(6,5))
    plt.savefig(dirname + '/cell_gene_umap_cossim_KNN_gene.png')
    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    umap = adata_gene.obsm['X_umap']
    marker_mask = adata_gene.obs['marker'] == 'Yes'
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot all points in light grey
    ax.scatter(umap[:, 0], umap[:, 1], c='lightgrey', s=5, alpha=0.3)
    # Plot marker=='Yes' points in red
    ax.scatter(umap[marker_mask, 0], umap[marker_mask, 1], c='red', s=30, label='Markers')
    # Prepare text annotations
    texts = []
    for i in adata_gene.obs[marker_mask].index:
        x, y = umap[adata_gene.obs.index.get_loc(i)]
        gene = adata_gene.obs.loc[i, 'feature_name']
        texts.append(ax.text(x, y, gene, fontsize=8, color='black'))
    # Repel overlapping text
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    ax.set_title('UMAP with Marker Genes Highlighted and Labeled')
    ax.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(dirname + '/cell_gene_umap_cossim_KNN_gene_marker.png')

