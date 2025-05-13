# (mil) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/bioPointNet$ python
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns

from sciMultiLaMA import *


path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/seaad/'
adata = sc.read_h5ad(path + 'MTG_glia_intersection_norm.h5ad')
# AnnData object with n_obs × n_vars = 253684 × 15914
#     obs: 'assay_ontology_term_id', 'suspension_type', 'cell_type_ontology_term_id', 'development_stage_ontology_term_id', 'disease_ontology_term_id', 'self_reported_ethnicity_ontology_term_id', 'organism_ontology_term_id', 'sex_ontology_term_id', 'tissue_ontology_term_id', 'is_primary_data', 'donor_id', 'Neurotypical reference', 'Class', 'Subclass', 'Supertype', 'Age at death', 'Years of education', 'Cognitive status', 'ADNC', 'Braak stage', 'Thal phase', 'CERAD score', 'APOE4 status', 'Lewy body disease pathology', 'LATE-NC stage', 'Microinfarct pathology', 'Specimen ID', 'PMI', 'Number of UMIs', 'Genes detected', 'Fraction mitochrondrial UMIs', 'tissue_type', 'cell_type', 'assay', 'disease', 'organism', 'sex', 'tissue', 'self_reported_ethnicity', 'development_stage', 'observation_joinid'
#     var: 'feature_is_filtered', 'feature_name', 'feature_reference', 'feature_biotype', 'feature_length', 'feature_type', 'n_cells', 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm'
#     uns: 'ADNC_colors', 'APOE4 status_colors', 'Age at death_colors', 'Braak stage_colors', 'CERAD score_colors', 'Cognitive status_colors', 'Great Apes Metadata', 'LATE-NC stage_colors', 'Lewy body disease pathology_colors', 'Microinfarct pathology_colors', 'PMI_colors', 'Subclass_colors', 'Supertype_colors', 'Thal phase_colors', 'UW Clinical Metadata', 'Years of education_colors', 'batch_condition', 'citation', 'default_embedding', 'hvg', 'neighbors', 'schema_reference', 'schema_version', 'sex_ontology_term_id_colors', 'title', 'umap'
#     obsm: 'X_scVI', 'X_umap'
#     obsp: 'connectivities', 'distances'

adata.var.feature_type.value_counts()
# feature_type
# protein_coding    15914

adata.obs.Subclass.value_counts()
# Subclass
# Oligodendrocyte    111187
# Astrocyte           70008
# Microglia-PVM       39998
# OPC                 32491

adata = adata[adata.obs.Supertype.isin(['Micro-PVM_3-SEAAD'])]



adata = adata[adata.obs.Subclass.isin(['Microglia-PVM'])]
adata.var.index = adata.var.feature_name
adata.raw = adata  
sc.pl.umap(adata, color=['LPL', 'ITGAX'], 
           save='_intersection_marker_list_.pdf')








sc.pp.filter_genes(adata, min_cells=5)


adata.write('MTG_sciMultiLaMA_norm.h5ad')

sc.pp.scale(adata)
metadata = adata.obs.copy()
metadata.disease.value_counts()
metadata.donor_id.value_counts() # 87
metadata.sex.value_counts()
metadata.Subclass.value_counts()

sample_key = 'donor_id'
task_key = 'disease'
ct_key = 'Subclass'
cov_key = 'sex'

batch_key = task_key
split_key = [ct_key, sample_key]

from sklearn.model_selection import train_test_split
train, val = train_test_split(metadata, test_size=0.2, stratify=metadata[split_key], 
                              random_state=0)
metadata['train_val'] = 'val'
metadata.loc[train.index, 'train_val'] = 'train'

trainY = metadata.loc[metadata.train_val == 'train']
trainX = adata[trainY.index].X
trainX.shape  
valY = metadata.loc[metadata.train_val == 'val']
valX = adata[valY.index].X
valX.shape     
metadata = pd.concat([trainY, valY])


dummy_df_cov = pd.get_dummies(metadata[[batch_key]]).astype(int)
trainC = dummy_df_cov.loc[trainY.index].values
valC = dummy_df_cov.loc[valY.index].values


# Gene raw embeddings
genept = pd.read_csv(path + 'genept_embedding.csv', index_col=0)
esm = pd.read_csv(path + 'esm2_embedding.csv', index_col=0)
scgpt = pd.read_csv(path + 'scgpt_embedding.csv', index_col=0)

np.sum(genept.index == adata.var.feature_name)
np.sum(esm.index == adata.var.feature_name)
np.sum(scgpt.index == adata.var.feature_name)

genept = genept.loc[adata.var.feature_name]
esm = esm.loc[adata.var.feature_name]
scgpt = scgpt.loc[adata.var.feature_name]


gene_embed_list = [genept, esm, scgpt]
gene_embed_names = ['genept','esm','scgpt']

print('Gene embeddings:', [gene_embed.shape for gene_embed in gene_embed_list])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gene_emb_list = [torch.FloatTensor(gene_embed.values).to(device) for gene_embed in gene_embed_list]
print('Gene embeddings:', [gene_emb.shape for gene_emb in gene_embed_list])
# Gene embeddings: [(13979, 1536), (13979, 5120), (13979, 512)]

# cell VAE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainData = torch.FloatTensor(trainX)
valData = torch.FloatTensor(valX)
trainCov = torch.FloatTensor(trainC)
valCov = torch.FloatTensor(valC)
train_dataset = TensorDataset(trainData,trainCov)
val_dataset = TensorDataset(valData,valCov)
feature_dim = trainData.shape[1]
feature_dim  
batch_dim = trainCov.shape[1]
batch_dim    


time_dict = {}
SEED = 0
hidden_dim = [1500, 300]          
latent_dim = 50
dropout_rate = 0.2
batchnorm = False    
layernorm = True      
learning_rate = 1e-3
batch_size = 1000
L2_lambda = 0
beta_increasing_rate = 0.05 
beta_warmup_steps = 20
epochs = 1000
sample_z = int(1)
activation =nn.LeakyReLU() 
gm = 0.05
min_val_loss = np.Inf
epochs_no_improve = 0
early_stop = False
patience = 20


# cell VAE
setup_seed(SEED)
cellvae_encoder = RNA_ENCODER(feature_dim, batch_dim, hidden_dim, latent_dim, 
                             batchnorm=batchnorm, layernorm=layernorm,
                             activation=activation, dropout_rate=dropout_rate)
cellvae_encoder.apply(init_weights)
cellvae_encoder.to(device)
setup_seed(SEED)
cellvae_decoder = RNA_DECODER(feature_dim, batch_dim, hidden_dim, latent_dim, 
                             batchnorm=batchnorm, layernorm=layernorm,
                             activation=activation, dropout_rate=dropout_rate)
cellvae_decoder.apply(init_weights)
cellvae_decoder.to(device)
# feature VAE
setup_seed(SEED)
genevae_encoder = MultiModalEncoder(feature_dims=[f.shape[1] for f in gene_emb_list], # [1536, 1024]
                                                    hidden_dims=hidden_dim, 
                                                    latent_dim=latent_dim,
                                                    fuse='average',
                                                    batchnorm=batchnorm,
                                                    layernorm=layernorm,
                                                    activation=activation,
                                                    dropout_rate=dropout_rate)
genevae_encoder.apply(init_weights)
genevae_encoder.to(device)
setup_seed(SEED)
genevae_decoder = RNA_DECODER(1, 0, hidden_dim, latent_dim, 
                             batchnorm=False, layernorm=layernorm,
                             activation=activation, dropout_rate=dropout_rate)
genevae_decoder.apply(init_weights)
genevae_decoder.to(device)
# Freeze the layers with names 'final' or 'output'
for name, param in genevae_decoder.named_parameters():
    if 'output' in name or 'final' in name:
        param.requires_grad = False


# non-weight-decay layers
non_decay_layers = ["latent"]
# cell vae
decay_param_encoder, nodecay_param_encoder, decay_name_encoder, nodecay_name_encoder = add_weight_decay(cellvae_encoder, output_layer=non_decay_layers)
decay_param_decoder, nodecay_param_decoder, decay_name_decoder, nodecay_name_decoder = add_weight_decay(cellvae_decoder, output_layer=non_decay_layers)
# gene vae
decay_param_encoder_g, nodecay_param_encoder_g, decay_name_encoder_g, nodecay_name_encoder_g = add_weight_decay(genevae_encoder, output_layer=non_decay_layers)
decay_param_decoder_g, nodecay_param_decoder_g, decay_name_decoder_g, nodecay_name_decoder_g = add_weight_decay(genevae_decoder, output_layer=non_decay_layers)
# optimizer
optimizer = torch.optim.AdamW([{'params': decay_param_encoder, 'weight_decay':L2_lambda, 'lr': learning_rate},
                                 {'params': decay_param_decoder, 'weight_decay':L2_lambda, 'lr': learning_rate},
                                 {'params': nodecay_param_encoder, 'weight_decay':0, 'lr': learning_rate},
                                 {'params': nodecay_param_decoder, 'weight_decay':0, 'lr': learning_rate},
                                 {'params': decay_param_encoder_g, 'weight_decay':L2_lambda, 'lr': learning_rate},
                                 {'params': decay_param_decoder_g, 'weight_decay':L2_lambda, 'lr': learning_rate},
                                 {'params': nodecay_param_encoder_g, 'weight_decay':0, 'lr': learning_rate},
                                 {'params': nodecay_param_decoder_g, 'weight_decay':0, 'lr': learning_rate}])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99, last_epoch=-1)

setup_seed(SEED)
from torch.utils.data import DataLoader, Dataset
DataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

tr_curve = []
val_curve = []
from tqdm import tqdm
for epoch in range(epochs):
    if epoch <= beta_warmup_steps:
        kl_weight = 0
    else:
        kl_weight = min(1, beta_increasing_rate * (epoch - beta_warmup_steps))
    # Switch to training mode once
    vae_training_mode(genevae_encoder, genevae_decoder)
    vae_training_mode(cellvae_encoder, cellvae_decoder)
    # Training loop with tqdm progress bar
    for idx, (x, c) in tqdm(enumerate(DataLoader), total=len(DataLoader), desc=f"Epoch {epoch + 1}/{epochs} Training", leave=False):
        # Move data to device and train the model
        x, c = x.to(device), c.to(device)
        train_si_rnaVAE(gene_emb_list, x, c,
                        cellvae_encoder, cellvae_decoder,
                        genevae_encoder, genevae_decoder,
                        optimizer, kl_weight=kl_weight, gamma_=gm)
    # Switch to evaluation mode once
    vae_evaluating_mode(genevae_encoder, genevae_decoder)
    vae_evaluating_mode(cellvae_encoder, cellvae_decoder)
    val_total_loss = 0.0  # To accumulate the total loss
    with torch.no_grad():
        # Validation loop with tqdm progress bar
        for i in tqdm(range(0, len(valData), batch_size), desc=f"Epoch {epoch + 1}/{epochs} Validation", leave=False):
            # Get the current batch
            batch_data = valData[i:i + batch_size].to(device)
            batch_cov = valCov[i:i + batch_size].to(device)
            # Call your evaluation function
            vtotal_loss, vneg_log_likelihood, vmean_square_error, vkl_divergence = \
                eval_si_rnaVAE(gene_emb_list, batch_data, batch_cov,
                               cellvae_encoder, cellvae_decoder,
                               genevae_encoder, genevae_decoder,
                               kl_weight=kl_weight, gamma_=gm)
            # Accumulate the loss
            val_total_loss += vtotal_loss.item()
    # Append the total validation loss
    val_curve.append(val_total_loss)
    # Print the validation results
    print(f"Epoch [{epoch + 1}/{epochs}], validation: total Loss: {val_total_loss:.4f}")
    # Early stopping logic
    if epoch > beta_warmup_steps + 1 / beta_increasing_rate if beta_increasing_rate != 0 else beta_warmup_steps:
        if val_total_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_total_loss
            # Save checkpoints of models
            genevae_encoder_ckpt = copy.deepcopy(genevae_encoder)
            genevae_decoder_ckpt = copy.deepcopy(genevae_decoder)
            cellvae_encoder_ckpt = copy.deepcopy(cellvae_encoder)
            cellvae_decoder_ckpt = copy.deepcopy(cellvae_decoder)
        else:
            epochs_no_improve += 1
            print(f"Early stopping triggered! {patience - epochs_no_improve} epochs remaining")
    if epoch > beta_warmup_steps + 1 / beta_increasing_rate if beta_increasing_rate != 0 else beta_warmup_steps:
        if epochs_no_improve == patience:
            print("Stopped!")
            early_stop = True
            break
    else:
        continue


cellvae_encoder_ckpt.eval()
cellvae_decoder_ckpt.eval()
torch.save(cellvae_encoder_ckpt, 'MTG_sciMultiLaMA_cell_encoder_ckpt.pt')
torch.save(cellvae_decoder_ckpt, 'MTG_sciMultiLaMA_cell_decoder_ckpt.pt')
genevae_encoder_ckpt.eval()
genevae_decoder_ckpt.eval()
torch.save(genevae_encoder_ckpt, 'MTG_sciMultiLaMA_gene_encoder_ckpt.pt')
torch.save(genevae_decoder_ckpt, 'MTG_sciMultiLaMA_gene_decoder_ckpt.pt')


# Placeholder for accumulated embeddings
train_embeddings = []
val_embeddings = []
# Calculate the embeddings for training data in batches
with torch.no_grad():
    for i in range(0, len(trainData), batch_size):
        batch_data = trainData[i:i + batch_size].to(device)
        batch_cov = trainCov[i:i + batch_size].to(device)
        # Compute the embeddings for the batch
        batch_embeddings = cellvae_encoder_ckpt(batch_data, batch_cov)[0]
        # Store the embeddings
        train_embeddings.append(batch_embeddings.cpu())  # Move back to CPU to avoid memory issues
    # Concatenate all batches
    train_embeddings = torch.cat(train_embeddings, dim=0)


# Calculate the embeddings for validation data in batches
with torch.no_grad():
    for i in range(0, len(valData), batch_size):
        batch_data = valData[i:i + batch_size].to(device)
        batch_cov = valCov[i:i + batch_size].to(device)
        # Compute the embeddings for the batch
        batch_embeddings = cellvae_encoder_ckpt(batch_data, batch_cov)[0]
        # Store the embeddings
        val_embeddings.append(batch_embeddings.cpu())  # Move back to CPU to avoid memory issues
    # Concatenate all batches
    val_embeddings = torch.cat(val_embeddings, dim=0)


cell_embedding = np.concatenate([train_embeddings.cpu().detach().numpy(), val_embeddings.cpu().detach().numpy()])
cell_embedding_df = pd.DataFrame(cell_embedding, index=metadata.index)
cell_embedding_df = cell_embedding_df.loc[adata.obs.index]
cell_embedding_df.to_csv('./MTG_sciMultiLaMA_CELL_embedding.csv')
gene_embedding_df = pd.DataFrame(genevae_encoder_ckpt(gene_emb_list,None)[0].detach().cpu().numpy(), index=adata.var.index)
gene_embedding_df.to_csv('./MTG_sciMultiLaMA_GENE_embedding.csv')


adata.obsm['X_sciMultiLaMA'] = cell_embedding_df.loc[adata.obs.index].values
sc.pp.neighbors(adata, n_neighbors=25, use_rep='X_sciMultiLaMA')
sc.tl.umap(adata)
sc.pl.umap(adata, color=[sample_key, task_key, cov_key, 'Supertype'], legend_loc='on data',
            ncols=2, wspace=0.5, save='_MTG_sciMultiLaMA_CELL.png')
import umap
import matplotlib.pyplot as plt
import seaborn as sns
U = umap.UMAP(random_state=0)
umap_coords = U.fit_transform(gene_embedding_df)
umap_coords = pd.DataFrame(umap_coords, index=gene_embedding_df.index)
fig = plt.figure(figsize=(9, 9))
sns.scatterplot(x=umap_coords.iloc[:, 0], y=umap_coords.iloc[:, 1])
sns.despine()
fig.savefig('./UMAP_MTG_sciMultiLaMA_GENE.png')

adata.write('./MTG_sciMultiLaMA.h5ad')

metadata = adata.obs.copy()
metadata.to_csv('./MTG_glia_metadata.csv')


umap_df = pd.DataFrame(umap_coords.values, index=umap_coords.index, columns=["UMAP1", "UMAP2"])


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



marker_list = [ 'IL1B', 'CSF1R', 'STAB1', 'NINJ1', 'JAK3',  # inflammatory process
                'IRF1', 'IRF7', 'IFI16',                    # interferon response
                'FCGR1A', 'FCGR2A', 'FCGR3B',     # Fc receptors | FCGR1B is not in the data
                'CD74', 'HLA-DRB5',                         # MHC
                'C1QA', 'C1QB']                             # complement components
len(marker_list) # 16 - 1 = 15

marker_df = adata.var[adata.var.feature_name.isin(marker_list)]
marker_id_list = marker_df.index.tolist()
marker_name_list = marker_df.feature_name.tolist()
gene_emb_dist = pd.DataFrame(np.sqrt(np.sum(gene_embedding_df**2, 1)), index=gene_embedding_df.index, columns=['dist'])
gene_emb_dist.sort_values('dist')
gene_emb_dist.mean() # dist    1.327988
gene_emb_dist.loc[marker_id_list].sort_values('dist')


# Step 2: Plot
fig, ax = plt.subplots(figsize=(10, 8))
# Plot all genes (faint)
ax.scatter(umap_df['UMAP1'], umap_df['UMAP2'], c='lightgrey', s=10, label='All genes', alpha=0.5)
# Plot marker genes (larger, red)
marker_mask = umap_df.index.isin(marker_id_list)
ax.scatter(umap_df.loc[marker_mask, 'UMAP1'],
           umap_df.loc[marker_mask, 'UMAP2'],
           c='red', s=50, label='Marker genes')
# Step 3: Add text with repel
texts = []
for gene in marker_id_list:
    if gene in umap_df.index:
        x, y = umap_df.loc[gene, ['UMAP1', 'UMAP2']]
        texts.append(ax.text(x, y, marker_df.loc[gene].feature_name, fontsize=8, color='black'))


from adjustText import adjust_text

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
# Styling
ax.set_title('UMAP of Gene Embeddings with Highlighted Marker Genes')
ax.legend()
plt.tight_layout()
plt.show()
fig.savefig('gene_embedding.pdf')


