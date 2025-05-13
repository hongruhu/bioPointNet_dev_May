# (scpair) hongruhu@gpu-4-56:/group/gquongrp/workspaces/hongruhu/MIL/$

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scPointNet_Attn_March10 import *

import pickle
# /group/gquongrp/collaborations/Birds_201/bird_ESM_pointNet.pkl
with open('/group/gquongrp/collaborations/Birds_201/bird_ESM_pointNet.pkl', 'rb') as f:
    obj = pickle.load(f)



len(list(obj.keys())) # 201

phenotypes = pd.read_csv('/group/gquongrp/collaborations/Birds_201/bird_AVONET_pointNet.csv', index_col=0)
# Index(['Species1', 'Family1', 'Order1', 'Avibase.ID1', 'Total.individuals',
#        'Female', 'Male', 'Unknown', 'Complete.measures', 'Beak.Length_Culmen',
#        'Beak.Length_Nares', 'Beak.Width', 'Beak.Depth', 'Tarsus.Length',
#        'Wing.Length', 'Kipps.Distance', 'Secondary1', 'Hand-Wing.Index',
#        'Tail.Length', 'Mass', 'Mass.Source', 'Mass.Refs.Other', 'Inference',
#        'Traits.inferred', 'Reference.species', 'Habitat', 'Habitat.Density',
#        'Migration', 'Trophic.Level', 'Trophic.Niche', 'Primary.Lifestyle',
#        'Min.Latitude', 'Max.Latitude', 'Centroid.Latitude',
#        'Centroid.Longitude', 'Range.Size'],
#       dtype='object')

phenotypes.index = [i.replace(' ', '_') for i in phenotypes.Species1]

intersecting_species = set(phenotypes.index).intersection(set(obj.keys()))
len(intersecting_species) # 198

obj = {k: obj[k] for k in intersecting_species}
len(list(obj.keys())) # 198
phenotypes = phenotypes.loc[list(obj.keys())]


phenotypes['Trophic.Niche'].value_counts() 
# Trophic.Niche
# Invertivore              99
# Omnivore                 37
# Aquatic predator         28
# Frugivore                19
# Granivore                 8
# Nectarivore               3
# Vertivore                 2
# Herbivore aquatic         1
# Herbivore terrestrial     1


# https://www.nature.com/articles/s41467-020-16313-6
phenotypes['Hand-Wing.Index'].value_counts() 


phenotypes['Mass'].describe()


phenotypes[['Hand-Wing.Index', 'Mass']].corr()
#                  Hand-Wing.Index    Mass
# Hand-Wing.Index           1.0000  0.2245
# Mass                      0.2245  1.0000

phenotypes[['Beak.Width', 'Mass']].corr()
#             Beak.Width      Mass
# Beak.Width    1.000000  0.651447
# Mass          0.651447  1.000000

phenotypes[['Beak.Width', 'Beak.Depth']].corr()
#             Beak.Width  Beak.Depth
# Beak.Width    1.000000    0.935943
# Beak.Depth    0.935943    1.000000

phenotypes[['Beak.Width', 'Beak.Length_Nares']].corr()
#                    Beak.Width  Beak.Length_Nares
# Beak.Width           1.000000           0.779208
# Beak.Length_Nares    0.779208           1.000000

phenotypes[['Beak.Length_Culmen', 'Beak.Length_Nares']].corr()
#                     Beak.Length_Culmen  Beak.Length_Nares
# Beak.Length_Culmen            1.000000           0.967626
# Beak.Length_Nares             0.967626           1.000000






obj['Arenaria_interpres']['embeddings'].shape
# torch.Size([13872, 320])

target_pheno = 'Hand-Wing.Index'







obj_pheno = {k: obj[k]['embeddings'] for k in obj.keys()}
len(obj_pheno) # 198
df_pheno = phenotypes[[target_pheno]]

list(obj_pheno.keys()) == list(df_pheno.index) # True


# plot the distribution of "log_brain_mass"
fig = plt.figure(figsize=(10, 5))
sns.histplot(df_pheno[target_pheno], bins=20)
plt.xlabel(target_pheno)
plt.ylabel('Frequency')
plt.title('Distribution of ' + target_pheno)
plt.show()
fig.savefig('bird_' + target_pheno + '_distribution.png')


# split the data into 10 folds randomly
np.random.seed(0)
df_pheno['fold'] = np.random.randint(0, 10, size=len(df_pheno))
df_pheno.to_csv('bird_' + target_pheno + '_labels.csv')

torch.save(obj_pheno, 'embeddings_bird_' + target_pheno + '.pt')