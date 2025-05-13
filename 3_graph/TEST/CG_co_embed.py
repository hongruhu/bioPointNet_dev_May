import os
import attr
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import find
from sklearn.cluster import KMeans
from pandas.core.dtypes.common import is_numeric_dtype
from pandas.api.types import (is_string_dtype,is_categorical_dtype,)

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from adjustText import adjust_text

from torchbiggraph.config import (add_to_sys_path,ConfigFileLoader)
from torchbiggraph.converters.importers import (convert_input_data,TSVEdgelistReader)
from torchbiggraph.train import train
from torchbiggraph.util import (set_logging_verbosity,setup_logging,SubprocessInitializer)

from anndata import (
    AnnData,
    read_h5ad,
    read_csv,
    read_excel,
    read_hdf,
    read_loom,
    read_mtx,
    read_text,
    read_umi_tools,
    read_zarr,
)

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def discretize(adata,
               layer=None,
               n_bins=5,
               max_bins=100):
    """Discretize continous values
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    layer: `str`, optional (default: None)
        The layer used to perform discretization
    n_bins: `int`, optional (default: 5)
        The number of bins to produce.
        It must be smaller than `max_bins`.
    max_bins: `int`, optional (default: 100)
        The number of bins used in the initial approximation.
        i.e. the number of bins to cluster.
    Returns
    -------
    updates `adata` with the following fields
    `.layer['discretize']` : `array_like`
        The matrix of discretized values to build SIMBA graph.
    `.uns['disc']` : `dict`
        `bin_edges`: The edges of each bin.
        `bin_count`: The number of values in each bin.
        `hist_edges`: The edges of each bin \
                      in the initial approximation.
        `hist_count`: The number of values in each bin \
                      for the initial approximation.
    """
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]
    nonzero_cont = X.data
    hist_count, hist_edges = np.histogram(
        nonzero_cont,
        bins=max_bins,
        density=False)
    hist_centroids = (hist_edges[0:-1] + hist_edges[1:])/2
    kmeans = KMeans(n_clusters=n_bins, random_state=2021, n_init='auto').fit(
        hist_centroids.reshape(-1, 1),
        sample_weight=hist_count)
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())
    padding = (hist_edges[-1] - hist_edges[0])/(max_bins*10)
    bin_edges = np.array(
        [hist_edges[0]-padding] +
        list((cluster_centers[0:-1] + cluster_centers[1:])/2) +
        [hist_edges[-1]+padding])
    nonzero_disc = np.digitize(nonzero_cont, bin_edges).reshape(-1,)
    bin_count = np.unique(nonzero_disc, return_counts=True)[1]
    adata.layers['discretize'] = X.copy()
    adata.layers['discretize'].data = nonzero_disc
    adata.uns['disc'] = dict()
    adata.uns['disc']['bin_edges'] = bin_edges
    adata.uns['disc']['bin_count'] = bin_count
    adata.uns['disc']['hist_edges'] = hist_edges
    adata.uns['disc']['hist_count'] = hist_count


def plot_discretize(adata,
               kde=None,
               fig_size=(6, 6),
               pad=1.08,
               w_pad=None,
               h_pad=None,
               save_fig=None,
               fig_path=None,
               fig_name='plot_discretize.pdf',
               **kwargs):
    """Plot original data VS discretized data
    Parameters
    ----------
    adata : `Anndata`
        Annotated data matrix.
    kde : `bool`, optional (default: None)
        If True, compute a kernel density estimate to smooth the distribution
        and show on the plot. Invalid as of v0.2.
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    fig_size: `tuple`, optional (default: (5,8))
        figure size.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'plot_discretize.pdf')
        if `save_fig` is True, specify figure name.
    **kwargs: `dict`, optional
        Other keyword arguments are passed through to ``plt.hist()``
    Returns
    -------
    None
    """
    if kde is not None:
        warnings.warn("kde is not supported as of v0.2", DeprecationWarning)
    assert 'disc' in adata.uns_keys(), \
        "please run `discretize()` first"
    if kde is not None:
        warnings.warn("kde is no longer supported as of v1.1",
                      DeprecationWarning)
    hist_edges = adata.uns['disc']['hist_edges']
    hist_count = adata.uns['disc']['hist_count']
    bin_edges = adata.uns['disc']['bin_edges']
    bin_count = adata.uns['disc']['bin_count']
    fig, ax = plt.subplots(2, 1, figsize=fig_size)
    _ = ax[0].hist(hist_edges[:-1],
                   hist_edges,
                   weights=hist_count,
                   linewidth=0,
                   **kwargs)
    _ = ax[1].hist(bin_edges[:-1],
                   bin_edges,
                   weights=bin_count,
                   **kwargs)
    ax[0].set_xlabel('Non-zero values')
    ax[0].set_ylabel('Count')
    ax[0].set_title('Original')
    ax[1].set_xlabel('Non-zero values')
    ax[1].set_ylabel('Count')
    ax[1].set_title('Discretized')
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if save_fig:
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


config = dict(
    # These paths are filled dynamically by `gen_graph`
    entity_path="",
    edge_paths=[""],
    checkpoint_path="",
    # Graph structure (populated by the graph generator)
    entities={},        # ← gets filled with {'C': ..., 'G': ...}
    relations=[],       # ← gets filled based on enabled edge types
    dynamic_relations=False,
    # Embedding model
    dimension=50,           # Size of embedding space
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


def gen_graph(
        adata,
        prefix_C='C',
        prefix_G='G',
        layer='discretize',
        dirname='graph',
        use_highly_variable=True,
        pbg_params=config
):
    filepath = './'+ dirname + '/'
    pbg_params['entity_path'] = \
        os.path.join(filepath, "input/entity")
    pbg_params['edge_paths'] = \
        [os.path.join(filepath, "input/edge"), ]
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    entity_alias = pd.DataFrame(columns=['alias'])
    dict_graph_stats = dict()
    col_names = ["source", "relation", "destination"]
    df_edges = pd.DataFrame(columns=col_names)
    if use_highly_variable:
        adata = adata[:, adata.var['highly_variable']].copy()
    # prefix_C = 'C'
    ids_cells = adata.obs.index
    df_cells = pd.DataFrame(
            index=ids_cells,
            columns=['alias'],
            data=[f'{prefix_C}.{x}' for x in range(len(ids_cells))])
    pbg_params['entities'][prefix_C] = {'num_partitions': 1}
    entity_alias = pd.concat([entity_alias, df_cells],ignore_index=False)
    # entity_alias    # [cell number rows x 1 columns]
    # prefix_G = 'G'
    ids_genes = adata.var.index
    df_genes = pd.DataFrame(
            index=ids_genes,
            columns=['alias'],
            data=[f'{prefix_G}.{x}' for x in range(len(ids_genes))])
    pbg_params['entities'][prefix_G] = {'num_partitions': 1}
    entity_alias = pd.concat([entity_alias, df_genes],ignore_index=False)
    # entity_alias    # [cell + gene number rows x 1 columns]
    id_r = 0
    pbg_params['relations'] = []
    # layer = 'discretize'
    arr_discretize = adata.layers[layer] 
    expr_level = np.unique(arr_discretize.data)                     # unique values of discretize expression array([1, 2, 3, 4, 5])
    expr_weight = np.linspace(start=1, stop=5, num=len(expr_level)) # array([1., 2., 3., 4., 5.])
    for i_lvl, lvl in enumerate(expr_level):
        _row, _col = (arr_discretize == lvl).astype(int).nonzero()
        df_edges_x = pd.DataFrame(columns=col_names)
        df_edges_x['source'] = df_cells.loc[
            adata.obs_names[_row], 'alias'].values
        df_edges_x['relation'] = f'r{id_r}'
        df_edges_x['destination'] = df_genes.loc[
            adata.var_names[_col], 'alias'].values
        pbg_params['relations'].append({
            'name': f'r{id_r}',
            'lhs': f'{prefix_C}',
            'rhs': f'{prefix_G}',
            'operator': 'none',
            'weight': round(expr_weight[i_lvl], 2),
            })
        print(
            f'relation{id_r}: '
            f'source: {prefix_C}, '
            f'destination: {prefix_G}\n'
            f'#edges: {df_edges_x.shape[0]}')
        dict_graph_stats[f'relation{id_r}'] = {
            'source': prefix_C,
            'destination': prefix_G,
            'n_edges': df_edges_x.shape[0]}
        id_r += 1
        df_edges = pd.concat(
            [df_edges, df_edges_x], ignore_index=True)
    adata.obs['pbg_id'] = ""
    adata.var['pbg_id'] = ""
    adata.obs.loc[adata.obs_names, 'pbg_id'] = \
        df_cells.loc[adata.obs_names, 'alias'].copy()
    adata.var.loc[adata.var_names, 'pbg_id'] = \
        df_genes.loc[adata.var_names, 'alias'].copy()
    print(f'Total number of edges: {df_edges.shape[0]}') # Total number of edges: 1183986
    dict_graph_stats['n_edges'] = df_edges.shape[0]
    print(f'Writing graph file "pbg_graph.txt" to "{filepath}" ...')
    df_edges.to_csv(os.path.join(filepath, "pbg_graph.txt"),
                    header=False,
                    index=False,
                    sep='\t')
    entity_alias.to_csv(os.path.join(filepath, 'entity_alias.txt'),
                        header=True,
                        index=True,
                        sep='\t')
    with open(os.path.join(filepath, 'graph_stats.json'), 'w') as fp:
        json.dump(dict_graph_stats,
                    fp,
                    sort_keys=True,
                    indent=4,
                    separators=(',', ': '))
    print("Finished.")
    return df_edges, entity_alias, dict_graph_stats


# def gen_graph_cosine(cell_emb, gene_emb, cell_ids, gene_ids,
#                          dirname,
#                          sim_threshold=0.2,
#                          cc_threshold=0.3,
#                          gg_threshold=0.3,
#                          include_cell_cell=True,
#                          include_gene_gene=True,
#                          pbg_params=None,
#                          prefix_C='C', prefix_G='G'):
#     filepath = './' + dirname + '/'
#     os.makedirs(filepath + "input/edge", exist_ok=True)
#     os.makedirs(filepath + "input/entity", exist_ok=True)
#     # Alias tables
#     df_cells = pd.DataFrame(index=cell_ids, columns=['alias'],
#                             data=[f'{prefix_C}.{i}' for i in range(len(cell_ids))])
#     df_genes = pd.DataFrame(index=gene_ids, columns=['alias'],
#                             data=[f'{prefix_G}.{j}' for j in range(len(gene_ids))])
#     entity_alias = pd.concat([df_cells, df_genes])
#     # Initialize edge list
#     edges = []
#     relations = []
#     # --- 1. Cell ↔ Gene (C-G)
#     sim_cg = cosine_similarity(cell_emb, gene_emb)
#     for i in range(cell_emb.shape[0]):
#         for j in range(gene_emb.shape[0]):
#             if sim_cg[i, j] > sim_threshold:
#                 edges.append((df_cells.iloc[i, 0], "cell-gene", df_genes.iloc[j, 0]))
#     relations.append({'name': 'cell-gene', 'lhs': prefix_C, 'rhs': prefix_G, 'operator': 'none', 'weight': 1.0})
#     # --- 2. Cell ↔ Cell (C-C)
#     if include_cell_cell:
#         sim_cc = cosine_similarity(cell_emb)
#         for i in range(cell_emb.shape[0]):
#             for j in range(i + 1, cell_emb.shape[0]):
#                 if sim_cc[i, j] > cc_threshold:
#                     edges.append((df_cells.iloc[i, 0], "cell-cell", df_cells.iloc[j, 0]))
#         relations.append({'name': 'cell-cell', 'lhs': prefix_C, 'rhs': prefix_C, 'operator': 'none', 'weight': 1.0})
#     # --- 3. Gene ↔ Gene (G-G)
#     if include_gene_gene:
#         sim_gg = cosine_similarity(gene_emb)
#         for i in range(gene_emb.shape[0]):
#             for j in range(i + 1, gene_emb.shape[0]):
#                 if sim_gg[i, j] > gg_threshold:
#                     edges.append((df_genes.iloc[i, 0], "gene-gene", df_genes.iloc[j, 0]))
#         relations.append({'name': 'gene-gene', 'lhs': prefix_G, 'rhs': prefix_G, 'operator': 'none', 'weight': 1.0})
#     # Save edge list and entity aliases
#     df_edges = pd.DataFrame(edges, columns=["source", "relation", "destination"])
#     df_edges.to_csv(os.path.join(filepath, "pbg_graph.txt"),
#                     sep='\t', header=False, index=False)
#     entity_alias.to_csv(os.path.join(filepath, "entity_alias.txt"),
#                         sep='\t', header=True, index=True)
#     # Update PBG config
#     pbg_params['entity_path'] = os.path.join(filepath, "input/entity")
#     pbg_params['edge_paths'] = [os.path.join(filepath, "input/edge")]
#     pbg_params['checkpoint_path'] = os.path.join(filepath, "model")
#     pbg_params['entities'] = {
#         prefix_C: {'num_partitions': 1},
#         prefix_G: {'num_partitions': 1}
#     }
#     pbg_params['relations'] = relations
#     # Print edge counts
#     edge_counts = pd.Series(df_edges['relation']).value_counts().to_dict()
#     print(f"Total edges: {df_edges.shape[0]} | Breakdown: {edge_counts}")
#     return df_edges, entity_alias, {'n_edges': df_edges.shape[0]}


def gen_graph_cosine_knn(cell_emb, gene_emb, cell_ids, gene_ids,
                         dirname='graph_CGC',
                         sim_threshold=0.2,
                         cc_threshold=0.3,
                         gg_threshold=0.3,
                         cc_mode='threshold',  # 'knn' or 'threshold'
                         gg_mode='threshold',
                         k_cc=10,
                         k_gg=10,
                         include_cell_cell=True,
                         include_gene_gene=True,
                         pbg_params=None,
                         prefix_C='C', prefix_G='G'):
    filepath = './' + dirname + '/'
    os.makedirs(filepath + "input/edge", exist_ok=True)
    os.makedirs(filepath + "input/entity", exist_ok=True)
    # Alias tables
    df_cells = pd.DataFrame(index=cell_ids, columns=['alias'],
                            data=[f'{prefix_C}.{i}' for i in range(len(cell_ids))])
    df_genes = pd.DataFrame(index=gene_ids, columns=['alias'],
                            data=[f'{prefix_G}.{j}' for j in range(len(gene_ids))])
    entity_alias = pd.concat([df_cells, df_genes])
    edges = []
    relations = []
    # --- 1. Cell - Gene (C-G)
    sim_cg = cosine_similarity(cell_emb, gene_emb)
    for i in range(cell_emb.shape[0]):
        for j in range(gene_emb.shape[0]):
            if sim_cg[i, j] > sim_threshold:
                edges.append((df_cells.iloc[i, 0], "cell-gene", df_genes.iloc[j, 0]))
    relations.append({'name': 'cell-gene', 'lhs': prefix_C, 'rhs': prefix_G, 'operator': 'none', 'weight': 1.0})
    # --- 2. Cell - Cell (C-C)
    if include_cell_cell:
        if cc_mode == 'threshold':
            sim_cc = cosine_similarity(cell_emb)
            for i in range(cell_emb.shape[0]):
                for j in range(i + 1, cell_emb.shape[0]):
                    if sim_cc[i, j] > cc_threshold:
                        edges.append((df_cells.iloc[i, 0], "cell-cell", df_cells.iloc[j, 0]))
        elif cc_mode == 'knn':
            nn = NearestNeighbors(n_neighbors=k_cc+1, metric='cosine').fit(cell_emb)
            dist, idx = nn.kneighbors(cell_emb)
            for i in range(cell_emb.shape[0]):
                for j in idx[i][1:]:  # skip self
                    edges.append((df_cells.iloc[i, 0], "cell-cell", df_cells.iloc[j, 0]))
        relations.append({'name': 'cell-cell', 'lhs': prefix_C, 'rhs': prefix_C, 'operator': 'none', 'weight': 1.0})
    # --- 3. Gene - Gene (G-G)
    if include_gene_gene:
        if gg_mode == 'threshold':
            sim_gg = cosine_similarity(gene_emb)
            for i in range(gene_emb.shape[0]):
                for j in range(i + 1, gene_emb.shape[0]):
                    if sim_gg[i, j] > gg_threshold:
                        edges.append((df_genes.iloc[i, 0], "gene-gene", df_genes.iloc[j, 0]))
        elif gg_mode == 'knn':
            nn = NearestNeighbors(n_neighbors=k_gg+1, metric='cosine').fit(gene_emb)
            dist, idx = nn.kneighbors(gene_emb)
            for i in range(gene_emb.shape[0]):
                for j in idx[i][1:]:  # skip self
                    edges.append((df_genes.iloc[i, 0], "gene-gene", df_genes.iloc[j, 0]))
        relations.append({'name': 'gene-gene', 'lhs': prefix_G, 'rhs': prefix_G, 'operator': 'none', 'weight': 1.0})
    # Save edges and aliases
    df_edges = pd.DataFrame(edges, columns=["source", "relation", "destination"])
    df_edges.to_csv(os.path.join(filepath, "pbg_graph.txt"),
                    sep='\t', header=False, index=False)
    entity_alias.to_csv(os.path.join(filepath, "entity_alias.txt"),
                        sep='\t', header=True, index=True)
    # Update PBG config
    pbg_params['entity_path'] = os.path.join(filepath, "input/entity")
    pbg_params['edge_paths'] = [os.path.join(filepath, "input/edge")]
    pbg_params['checkpoint_path'] = os.path.join(filepath, "model")
    pbg_params['entities'] = {
        prefix_C: {'num_partitions': 1},
        prefix_G: {'num_partitions': 1}
    }
    pbg_params['relations'] = relations
    print(f"Edge counts by type:\n{pd.Series([e[1] for e in edges]).value_counts()}")
    return df_edges, entity_alias, {'n_edges': df_edges.shape[0]}


def gen_graph_euclidean_knn(cell_emb, gene_emb, cell_ids, gene_ids,
                             dirname='graph_CGC_euclidean',
                             dist_threshold=2.0,
                             cc_threshold=2.0,
                             gg_threshold=2.0,
                             cc_mode='threshold',  # 'knn' or 'threshold'
                             gg_mode='threshold',
                             k_cc=10,
                             k_gg=10,
                             include_cell_cell=True,
                             include_gene_gene=True,
                             pbg_params=None,
                             prefix_C='C', prefix_G='G'):
    filepath = './' + dirname + '/'
    os.makedirs(filepath + "input/edge", exist_ok=True)
    os.makedirs(filepath + "input/entity", exist_ok=True)
    df_cells = pd.DataFrame(index=cell_ids, columns=['alias'],
                            data=[f'{prefix_C}.{i}' for i in range(len(cell_ids))])
    df_genes = pd.DataFrame(index=gene_ids, columns=['alias'],
                            data=[f'{prefix_G}.{j}' for j in range(len(gene_ids))])
    entity_alias = pd.concat([df_cells, df_genes])
    edges = []
    relations = []
    # --- 1. Cell - Gene (C-G)
    dist_cg = euclidean_distances(cell_emb, gene_emb)
    for i in range(cell_emb.shape[0]):
        for j in range(gene_emb.shape[0]):
            if dist_cg[i, j] < dist_threshold:
                edges.append((df_cells.iloc[i, 0], "cell-gene", df_genes.iloc[j, 0]))
    relations.append({'name': 'cell-gene', 'lhs': prefix_C, 'rhs': prefix_G, 'operator': 'none', 'weight': 1.0})
    print()
    # --- 2. Cell - Cell (C-C)
    if include_cell_cell:
        if cc_mode == 'threshold':
            dist_cc = euclidean_distances(cell_emb)
            for i in range(cell_emb.shape[0]):
                for j in range(i + 1, cell_emb.shape[0]):
                    if dist_cc[i, j] < cc_threshold:
                        edges.append((df_cells.iloc[i, 0], "cell-cell", df_cells.iloc[j, 0]))
        elif cc_mode == 'knn':
            nn = NearestNeighbors(n_neighbors=k_cc + 1, metric='euclidean').fit(cell_emb)
            dist, idx = nn.kneighbors(cell_emb)
            for i in range(cell_emb.shape[0]):
                for j in idx[i][1:]:  # skip self
                    edges.append((df_cells.iloc[i, 0], "cell-cell", df_cells.iloc[j, 0]))
        relations.append({'name': 'cell-cell', 'lhs': prefix_C, 'rhs': prefix_C, 'operator': 'none', 'weight': 1.0})
    # --- 3. Gene - Gene (G-G)
    if include_gene_gene:
        if gg_mode == 'threshold':
            dist_gg = euclidean_distances(gene_emb)
            for i in range(gene_emb.shape[0]):
                for j in range(i + 1, gene_emb.shape[0]):
                    if dist_gg[i, j] < gg_threshold:
                        edges.append((df_genes.iloc[i, 0], "gene-gene", df_genes.iloc[j, 0]))
        elif gg_mode == 'knn':
            nn = NearestNeighbors(n_neighbors=k_gg + 1, metric='euclidean').fit(gene_emb)
            dist, idx = nn.kneighbors(gene_emb)
            for i in range(gene_emb.shape[0]):
                for j in idx[i][1:]:  # skip self
                    edges.append((df_genes.iloc[i, 0], "gene-gene", df_genes.iloc[j, 0]))
        relations.append({'name': 'gene-gene', 'lhs': prefix_G, 'rhs': prefix_G, 'operator': 'none', 'weight': 1.0})
    # Save edges and aliases
    df_edges = pd.DataFrame(edges, columns=["source", "relation", "destination"])
    df_edges.to_csv(os.path.join(filepath, "pbg_graph.txt"),
                    sep='\t', header=False, index=False)
    entity_alias.to_csv(os.path.join(filepath, "entity_alias.txt"),
                        sep='\t', header=True, index=True)
    # Update PBG config
    pbg_params['entity_path'] = os.path.join(filepath, "input/entity")
    pbg_params['edge_paths'] = [os.path.join(filepath, "input/edge")]
    pbg_params['checkpoint_path'] = os.path.join(filepath, "model")
    pbg_params['entities'] = {
        prefix_C: {'num_partitions': 1},
        prefix_G: {'num_partitions': 1}
    }
    pbg_params['relations'] = relations
    print(f"Edge counts by type:\n{pd.Series([e[1] for e in edges]).value_counts()}")
    return df_edges, entity_alias, {'n_edges': df_edges.shape[0]}


def pbg_train(dirname=None,
              output='model',
              auto_wd=True,
              save_wd=True,
            #   use_edge_weights=False,
              graph_stats = dict(),
              pbg_params=config):
    """PBG training
    Parameters
    ----------
    dirname: `str`, optional (default: None)
        The name of the directory in which graph is stored
    pbg_params: `dict`, optional (default: None)
        Configuration for pbg training.
        If specified, it will be used instead of the default setting
    output: `str`, optional (default: 'model')
        The name of the directory where training output will be written to.
        It overrides `pbg_params` if `checkpoint_path` is specified in it
    auto_wd: `bool`, optional (default: True)
        If True, it will override `pbg_params['wd']` with a new weight decay
        estimated based on training sample size
        Recommended for relative small training sample size (<1e7)
    save_wd: `bool`, optional (default: False)
        If True, estimated `wd` will be saved to `settings.pbg_params['wd']`
    use_edge_weights: `bool`, optional (default: False)
        If True, the edge weights are used for the training;
        If False, the weights of relation types are used instead,
        and edge weights will be ignored.
    Returns
    -------
    updates `settings.pbg_params` with the following parameter
    checkpoint_path:
        The path to the directory where checkpoints (and thus the output)
        will be written to.
        If checkpoints are found in it, training will resume from them.
    """
    output='model'
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


