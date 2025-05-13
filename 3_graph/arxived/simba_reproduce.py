# (sciLaMA_graph) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/sciLaMA_graph$

path = '/group/gquongrp/workspaces/hongruhu/sciLaMA_graph/test/skin/'

import scanpy as sc
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns

from sciLaMA import *
from coembed import *

adata_CG = sc.read(path + 'obj.h5ad') 
# AnnData object with n_obs × n_vars = 6436 × 7006
#     obs: 'celltype', 'train_val'
#     var: 'n_cells', 'mean', 'std', 'n_counts', 'pct_cells', 'Human_orhtolog', 'Human_ID', 'Mouse_ID', 'Mouse_orhtolog'
#     uns: 'log1p'
#     obsm: 'X_pca', 'X_scVI'
#     layers: 'raw'
adata_CG.X.max()
# 880.0
sc.pp.highly_variable_genes(adata_CG,n_top_genes=2000, flavor='seurat_v3')

sc.pp.normalize_total(adata_CG, 1e4)
adata_CG.X.max()
# 6205.493

sc.pp.log1p(adata_CG)
adata_CG.X.max()
# 8.733352

discretize(adata_CG)
plot_discretize(adata_CG, fig_path='graph_plot', save_fig=True)
adata_CG.layers['discretize'].shape
# (6436, 7006) same as original cell x gene


pbg_params = dict(
                # I/O data
                entity_path="",
                edge_paths=["", ],
                checkpoint_path="",
                # Graph structure
                entities={},
                relations=[],
                dynamic_relations=False,
                # Scoring model
                dimension=50,
                global_emb=False,
                comparator='dot',
                # Training
                num_epochs=10,
                workers=4,
                num_batch_negs=50,
                num_uniform_negs=50,
                loss_fn='softmax',
                lr=0.1,
                early_stopping=False,
                regularization_coef=0.0,
                wd=0.0,
                wd_interval=50,
                # Evaluation during training
                eval_fraction=0.05,
                eval_num_batch_negs=50,
                eval_num_uniform_negs=50,
                checkpoint_preservation_interval=None,
            )


dirname='graph'
df_edges, entity_alias, dict_graph_stats = gen_graph(
        adata_CG,
        prefix_C='C',
        prefix_G='G',
        layer='discretize',
        dirname='graph',
        use_highly_variable=True,
        pbg_params = pbg_params
)


pbg_train(dirname=dirname,
              output='model',
              auto_wd=True,
              save_wd=True,
              graph_stats = dict_graph_stats,
              pbg_params=pbg_params)




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

## Add annotation of celltypes (optional)
adata_C.obs['celltype'] = adata_CG[adata_C.obs_names,:].obs['celltype'].copy()
adata_C

palette_celltype={'TAC-1':'#1f77b4',
                  'TAC-2':'#ff7f0e',
                  'IRS':'#279e68',
                  'Medulla':"#aa40fc",
                  'Hair Shaft-cuticle.cortex':'#d62728'}


sc.tl.umap(adata_C,n_neighbors=15,n_components=2)
sc.pl.umap(adata_C,color=['celltype'],
            dict_palette={'celltype': palette_celltype},
            fig_size=(6,4),
            drawing_order='random')
plt.savefig('cell_umap.png')





# embed cells and genes into the same space
adata_all = si.tl.embed(adata_ref=adata_C,list_adata_query=[adata_G])
adata_all.obs.head()


## add annotations of cells and genes
adata_all.obs['entity_anno'] = ""
adata_all.obs.loc[adata_C.obs_names, 'entity_anno'] = adata_all.obs.loc[adata_C.obs_names, 'celltype']
adata_all.obs.loc[adata_G.obs_names, 'entity_anno'] = 'gene'
adata_all.obs.head()

palette_entity_anno = palette_celltype.copy()
palette_entity_anno['gene'] = "#607e95"


si.tl.umap(adata_all,n_neighbors=15,n_components=2)
si.pl.umap(adata_all,color=['id_dataset','entity_anno'],
           dict_palette={'entity_anno': palette_entity_anno},
           drawing_order='original',
           fig_size=(6,5))
plt.savefig('cell_gene_umap.png')









output='model'
graph_stats = dict_graph_stats
auto_wd = True
save_wd = True
filepath = os.path.join('./'+ dirname)
pbg_params['checkpoint_path'] = os.path.join(filepath, output)
if auto_wd:
    print('Auto-estimating weight decay ...')
    # empirical numbers from simulation experiments
    if graph_stats['n_edges'] < 5e7:
        # optimial wd (0.013) for sample size (2725781)
        wd = np.around(
            0.013 * 2725781 / graph_stats['n_edges'],
            decimals=6)
    else:
        # optimial wd (0.0004) for sample size (59103481)
        wd = np.around(
            0.0004 * 59103481 / graph_stats['n_edges'],
            decimals=6)
    pbg_params['wd'] = wd
    if save_wd:
        pbg_params['wd'] = pbg_params['wd']
        print(f"`pbg_params['wd']` has been updated to {wd}")
print(f'Weight decay being used for training is {pbg_params["wd"]}')
# to avoid oversubscription issues in workloads
# that involve nested parallelism
os.environ["OMP_NUM_THREADS"] = "1"
loader = ConfigFileLoader()
config = loader.load_config_simba(pbg_params)
set_logging_verbosity(config.verbose)
list_filenames = [os.path.join(filepath, "pbg_graph.txt")]
input_edge_paths = [Path(name) for name in list_filenames]
print("Converting input data ...")
convert_input_data(
            config.entities,
            config.relations,
            config.entity_path,
            config.edge_paths,
            input_edge_paths,
            TSVEdgelistReader(lhs_col=0, rhs_col=2, rel_col=1),
            dynamic_relations=config.dynamic_relations,
            )

subprocess_init = SubprocessInitializer()
subprocess_init.register(setup_logging, config.verbose)
subprocess_init.register(add_to_sys_path, loader.config_dir.name)
train_config = attr.evolve(config, edge_paths=config.edge_paths)
print("Starting training ...")
train(train_config, subprocess_init=subprocess_init)
print("Finished")


path_emb = pbg_params['checkpoint_path']






si.tl.pbg_train(auto_wd=True, save_wd=True, output='model')




# # modify parameters
# dict_config = si.settings.pbg_params.copy()
# # dict_config['wd'] = 0.015521
# dict_config['wd_interval'] = 10 # we usually set `wd_interval` to 10 for scRNA-seq datasets for a slower but finer training
# dict_config['workers'] = 12 #The number of CPUs.

# ## start training
# si.tl.pbg_train(pbg_params = dict_config, auto_wd=True, save_wd=True, output='model')



si.pl.pbg_metrics(fig_ncol=1)
plt.savefig('pbg_metrics.png')


# si.tl.pbg_train(pbg_params = dict_config, auto_wd=True, save_wd=True, output="model2")
# si.settings.set_pbg_params(dict_config)
# # load back 'graph0'
# si.load_graph_stats()
# # load back 'model' that was trained on 'graph0'
# si.load_pbg_config()

# # load back 'graph1'
# si.load_graph_stats(path='./result_simba_rnaseq/pbg/graph1/')
# # load back 'model2' that was trained on 'graph1'
# si.load_pbg_config(path='./result_simba_rnaseq/pbg/graph1/model2/')


# read in entity embeddings obtained from pbg training.
dict_adata = si.read_embedding()
dict_adata

adata_C = dict_adata['C']  # embeddings of cells
adata_G = dict_adata['G']  # embeddings of genes

## Add annotation of celltypes (optional)
adata_C.obs['celltype'] = adata_CG[adata_C.obs_names,:].obs['celltype'].copy()
adata_C


adata_CG.obs.celltype.value_counts()
# celltype
# TAC-1                        3026
# Hair Shaft-cuticle.cortex     998
# TAC-2                         969
# Medulla                       812
# IRS                           631


palette_celltype={'TAC-1':'#1f77b4',
                  'TAC-2':'#ff7f0e',
                  'IRS':'#279e68',
                  'Medulla':"#aa40fc",
                  'Hair Shaft-cuticle.cortex':'#d62728'}


si.tl.umap(adata_C,n_neighbors=15,n_components=2)
si.pl.umap(adata_C,color=['celltype'],
            dict_palette={'celltype': palette_celltype},
            fig_size=(6,4),
            drawing_order='random')
plt.savefig('cell_umap.png')



# embed cells and genes into the same space
adata_all = si.tl.embed(adata_ref=adata_C,list_adata_query=[adata_G])
adata_all.obs.head()


## add annotations of cells and genes
adata_all.obs['entity_anno'] = ""
adata_all.obs.loc[adata_C.obs_names, 'entity_anno'] = adata_all.obs.loc[adata_C.obs_names, 'celltype']
adata_all.obs.loc[adata_G.obs_names, 'entity_anno'] = 'gene'
adata_all.obs.head()

palette_entity_anno = palette_celltype.copy()
palette_entity_anno['gene'] = "#607e95"


si.tl.umap(adata_all,n_neighbors=15,n_components=2)
si.pl.umap(adata_all,color=['id_dataset','entity_anno'],
           dict_palette={'entity_anno': palette_entity_anno},
           drawing_order='original',
           fig_size=(6,5))
plt.savefig('cell_gene_umap.png')
