# (sciLaMA_graph) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/sciLaMA_graph$

saving_path = '/group/gquongrp/workspaces/hongruhu/sciLaMA_graph/test/'

import scanpy as sc
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns

from sciLaMA import *




adata = sc.read(saving_path + 'sample_fetal_liver_atlas_dataset.h5ad') 
# AnnData object with n_obs × n_vars = 500 × 2000
metadata = adata.obs.copy()

# random split train and val
from sklearn.model_selection import train_test_split
metadata['train_val'] = 'train'
train_idx, val_idx = train_test_split(metadata.index, test_size=0.15, random_state=0)
metadata.loc[val_idx, 'train_val'] = 'val'
metadata['train_val'].value_counts()
# train_val
# train    425
# val       75



trainY = metadata.loc[metadata.train_val == 'train']
trainX = adata[trainY.index].X
valY = metadata.loc[metadata.train_val == 'val']
valX = adata[valY.index].X
metadata = pd.concat([trainY, valY])

# cell VAE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainData = torch.FloatTensor(trainX).to(device)
valData = torch.FloatTensor(valX).to(device)
train_dataset = TensorDataset(trainData,trainData)
val_dataset = TensorDataset(valData,valData)
feature_dim = trainData.shape[1]
feature_dim  # 2000
cell_dim = trainData.shape[0] + valData.shape[0]
cell_dim     # 500
batch_dim = 0



# Gene raw embeddings
si = adata.X.T
gene_embed_list = [si]
gene_embed_names = ['si']
print('Gene embeddings:', [gene_embed.shape for gene_embed in gene_embed_list])
# Gene embeddings: [(2000, 500)]


dim_k = 3

gene_embed_id = 0
gene_embed_name = gene_embed_names[gene_embed_id]
gene_embed_df = gene_embed_list[gene_embed_id]
SEED = 0
hidden_dim = [150]          
latent_dim = dim_k             
dropout_rate = 0.1
batchnorm = False    
layernorm = True      
learning_rate = 1e-4
batch_size = 100
L2_lambda = 0
beta_increasing_rate = 0.05 
beta_warmup_steps = 25
epochs = 1000
sample_z = int(1)
activation =nn.LeakyReLU() 
gm = 0.05
min_val_loss = np.Inf
epochs_no_improve = 0
early_stop = False
patience = 25
cell_dim = gene_embed_df.shape[1]
batch_dim = 0
#
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
genevae_encoder = RNA_ENCODER(cell_dim, 0, hidden_dim, latent_dim, 
                            batchnorm=False, layernorm=layernorm,
                            activation=activation, dropout_rate=dropout_rate)
genevae_encoder.apply(init_weights)
genevae_encoder.to(device)
setup_seed(SEED)
genevae_decoder = RNA_DECODER(cell_dim, 0, hidden_dim, latent_dim, 
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
setup_seed(1234)
from torch.utils.data import DataLoader, Dataset
DataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
Xt = torch.FloatTensor(gene_embed_df).to(device)         # C x E
# tr_curve = []
# val_curve = []


for epoch in range(epochs):
    if epoch <= beta_warmup_steps:
        kl_weight = 0
    else:
        kl_weight = min(1, beta_increasing_rate*(epoch-beta_warmup_steps))
    vae_training_mode(genevae_encoder, genevae_decoder)
    vae_training_mode(cellvae_encoder, cellvae_decoder)
    for idx, (x, _) in enumerate(DataLoader):
        train_si_rnaVAE(Xt, x, None,
                        cellvae_encoder, cellvae_decoder,
                        genevae_encoder, genevae_decoder,
                        optimizer, kl_weight=kl_weight, gamma_=gm)
    vae_evaluating_mode(genevae_encoder, genevae_decoder)
    vae_evaluating_mode(cellvae_encoder, cellvae_decoder)
    with torch.no_grad():
        ttotal_loss, tneg_log_likelihood, tmean_square_error, tkl_divergence = \
        eval_si_rnaVAE(Xt, trainData, None,
                        cellvae_encoder, cellvae_decoder,
                        genevae_encoder, genevae_decoder,
                        kl_weight=kl_weight, gamma_=gm)
        # tr_curve.append(ttotal_loss.item())
        print("Epoch [{}/{}], train     : RNA Loss: {:.4f}, RNA MSE: {:.4f}, RNA Negative Log Likelihood: {:.4f}, beta: {:.4f}, RNA KL Divergence: {:.4f}".format(
                epoch+1, epochs,
                ttotal_loss.item(),
                tmean_square_error.item(),
                tneg_log_likelihood.item(),
                kl_weight,
                tkl_divergence.item()))
        vtotal_loss, vneg_log_likelihood, vmean_square_error, vkl_divergence = \
        eval_si_rnaVAE(Xt, valData, None,
                        cellvae_encoder, cellvae_decoder,
                        genevae_encoder, genevae_decoder,
                        kl_weight=kl_weight, gamma_=gm)
        # val_curve.append(vtotal_loss.item())
        print("Epoch [{}/{}], validation: RNA Loss: {:.4f}, RNA MSE: {:.4f}, RNA Negative Log Likelihood: {:.4f}, beta: {:.4f}, RNA KL Divergence: {:.4f}".format(
                epoch+1, epochs,
                vtotal_loss.item(),
                vmean_square_error.item(),
                vneg_log_likelihood.item(),
                kl_weight,
                vkl_divergence.item()))
    if epoch > beta_warmup_steps + 1/beta_increasing_rate if beta_increasing_rate != 0 else beta_warmup_steps:
        if vtotal_loss.item() < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = vtotal_loss.item()
            genevae_encoder_ckpt = copy.deepcopy(genevae_encoder)
            genevae_decoder_ckpt = copy.deepcopy(genevae_decoder)
            cellvae_encoder_ckpt = copy.deepcopy(cellvae_encoder)
            cellvae_decoder_ckpt = copy.deepcopy(cellvae_decoder)
        else:
            epochs_no_improve += 1
            print("Early stopping triggered!", patience-epochs_no_improve)
    else:
        continue
    if epoch > beta_warmup_steps + 1/beta_increasing_rate if beta_increasing_rate != 0 else beta_warmup_steps: 
        if epochs_no_improve == patience:
            print("Stopped!")
            early_stop = True
            break
    else:
        continue



genevae_encoder_ckpt.eval()
genevae_decoder_ckpt.eval()
torch.save(genevae_encoder_ckpt, saving_path + str(dim_k) + '_direct_sciLaMA_genevae_encoder_ckpt_' + gene_embed_name + '.pt')
torch.save(genevae_decoder_ckpt, saving_path + str(dim_k) + '_direct_sciLaMA_genevae_decoder_ckpt_' + gene_embed_name + '.pt')
cellvae_encoder_ckpt.eval()
cellvae_decoder_ckpt.eval()
torch.save(cellvae_encoder_ckpt, saving_path + str(dim_k) + '_direct_sciLaMA_cellvae_encoder_ckpt_' + gene_embed_name + '.pt')
torch.save(cellvae_decoder_ckpt, saving_path + str(dim_k) + '_direct_sciLaMA_cellvae_decoder_ckpt_' + gene_embed_name + '.pt')




# cell VAE embedding
train_embeddings = cellvae_encoder_ckpt(trainData, None)[0]
val_embeddings = cellvae_encoder_ckpt(valData, None)[0]
embedding = np.concatenate([train_embeddings.cpu().detach().numpy(), val_embeddings.cpu().detach().numpy()])
embedding_df = pd.DataFrame(embedding, index=metadata.index)
embedding_df.loc[adata.obs.index].to_csv(saving_path + str(dim_k) + '_direct_sciLaMA_CELL_embedding_' + gene_embed_name + '.csv')
adata.obsm['X_direct_' + str(dim_k) + '_' + gene_embed_name] = embedding_df.loc[adata.obs.index].values




fig = plt.figure(figsize=(10, 10))
sns.scatterplot(x=adata.obsm['X_direct_' + str(dim_k) + '_' + gene_embed_name][:, 0], 
                y=adata.obsm['X_direct_' + str(dim_k) + '_' + gene_embed_name][:, 1], hue=adata.obs['Cell Type'], s=100)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title(str(dim_k) + '_direct_umap_' + gene_embed_name)
plt.tight_layout()
fig.savefig(saving_path + str(dim_k) + '_sciLaMA_CELL_direct_embedding_' + gene_embed_name + '.png')


siVAE_gene_embedding = pd.DataFrame(genevae_encoder_ckpt(Xt,None)[0].detach().cpu().numpy(), index=adata.var.index)
siVAE_gene_embedding.to_csv(saving_path + str(dim_k) + '_direct_sciLaMA_GENE_embedding_' + gene_embed_name + '.csv')
fig = plt.figure(figsize=(10, 10))
sns.scatterplot(x=siVAE_gene_embedding.values[:, 0], y=siVAE_gene_embedding.values[:, 1])
fig.savefig(saving_path + str(dim_k) + '_sciLaMA_GENE_direct_embedding_' + gene_embed_name + '.png')









# 3d viz for adata.obsm['X_direct_' + str(dim_k) + '_' + gene_embed_name], and siVAE_gene_embedding
siVAE_gene_embedding
#                  0         1         2
# ISG15     0.331508 -0.331430  1.206936
# AURKAIP1  0.133783 -0.656925  1.236156
# MRPL20    0.151755 -0.591393  1.007188
# GNB1      0.590482 -0.671169  1.496043
# FAAP20    0.015309 -0.580057  1.309415
# ...            ...       ...       ...
# MT-ND4L   0.473370 -0.394720  0.968414
# MT-ND4    0.663644 -0.337246  0.944337
# MT-ND5    0.591501 -0.348023  0.863208
# MT-ND6    0.315608 -0.646202  1.398799
# MT-CYB    0.711448 -0.337500  0.935650

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract data from AnnData object
x = adata.obsm['X_direct_3_si'][:, 0]  # X-axis data
y = adata.obsm['X_direct_3_si'][:, 1]  # Y-axis data
z = adata.obsm['X_direct_3_si'][:, 2]  # Z-axis data

# Use Seaborn to map 'Cell Type' to colors
palette = sns.color_palette("Set1", as_cmap=True)  # Choose a color palette
cell_types = adata.obs['Cell Type']

# Scatter plot with color coding based on 'Cell Type'
scatter = ax.scatter(x, y, z, c=cell_types.astype('category').cat.codes, cmap=palette, s=10)

# Add labels to the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Add a color bar to indicate the cell types
fig.colorbar(scatter, ax=ax, label='Cell Type')

# Show the plot
plt.show()

# Save the plot as a PNG file
fig.savefig(saving_path + '3d_viz.png')


fig = plt.figure(figsize=(10, 10))
sns.scatterplot(x=siVAE_gene_embedding.values[:, 0], y=siVAE_gene_embedding.values[:, 1])
fig.savefig(saving_path + str(dim_k) + '_sciLaMA_direct_GENE_embedding_' + gene_embed_name + '12.png')

fig = plt.figure(figsize=(10, 10))
sns.scatterplot(x=siVAE_gene_embedding.values[:, 1], y=siVAE_gene_embedding.values[:, 2])
fig.savefig(saving_path + str(dim_k) + '_sciLaMA_direct_GENE_embedding_' + gene_embed_name + '23.png')

fig = plt.figure(figsize=(10, 10))
sns.scatterplot(x=siVAE_gene_embedding.values[:, 0], y=siVAE_gene_embedding.values[:, 2])
fig.savefig(saving_path + str(dim_k) + '_sciLaMA_direct_GENE_embedding_' + gene_embed_name + '13.png')

fig = plt.figure(figsize=(10, 10))
sns.scatterplot(x=adata.obsm['X_direct_' + str(dim_k) + '_' + gene_embed_name][:, 0], 
                y=adata.obsm['X_direct_' + str(dim_k) + '_' + gene_embed_name][:, 1], hue=adata.obs['Cell Type'], s=100)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title(str(dim_k) + '_direct_umap_' + gene_embed_name)
plt.tight_layout()
fig.savefig(saving_path + str(dim_k) + '_sciLaMA_CELL_direct_embedding_' + gene_embed_name + '12.png')

fig = plt.figure(figsize=(10, 10))
sns.scatterplot(x=adata.obsm['X_direct_' + str(dim_k) + '_' + gene_embed_name][:, 1], 
                y=adata.obsm['X_direct_' + str(dim_k) + '_' + gene_embed_name][:, 2], hue=adata.obs['Cell Type'], s=100)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title(str(dim_k) + '_direct_umap_' + gene_embed_name)
plt.tight_layout()
fig.savefig(saving_path + str(dim_k) + '_sciLaMA_CELL_direct_embedding_' + gene_embed_name + '23.png')

fig = plt.figure(figsize=(10, 10))
sns.scatterplot(x=adata.obsm['X_direct_' + str(dim_k) + '_' + gene_embed_name][:, 0], 
                y=adata.obsm['X_direct_' + str(dim_k) + '_' + gene_embed_name][:, 2], hue=adata.obs['Cell Type'], s=100)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title(str(dim_k) + '_direct_umap_' + gene_embed_name)
plt.tight_layout()
fig.savefig(saving_path + str(dim_k) + '_sciLaMA_CELL_direct_embedding_' + gene_embed_name + '13.png')





adata.write(saving_path + 'obj.h5ad')



