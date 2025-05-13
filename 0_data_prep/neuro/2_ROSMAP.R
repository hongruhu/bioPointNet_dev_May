# (R_env) hongruhu@gpu-5-50:/group/gquongrp/workspaces/hongruhu/bioPointNet$ R
library(Seurat)

path = '/group/gquongrp/workspaces/hongruhu/bioPointNet/data/ROSMAP/rds_files'

all_names <- list.files(path, full.names = FALSE)

#  [1] "Astrocytes.rds"
#  [2] "Excitatory_neurons_set1.rds"
#  [3] "Excitatory_neurons_set2.rds"
#  [4] "Excitatory_neurons_set3.rds"
#  [5] "Immune_cells.rds"
#  [6] "Inhibitory_neurons.rds"
#  [7] "Oligodendrocytes.rds"
#  [8] "OPCs.rds"
#  [9] "ROSMAP.ImmuneCells.6regions.snRNAseq.counts.rds"
# [10] "Vasculature_cells.rds"


all_names = c("Astrocytes.rds", "Immune_cells.rds",
            "Oligodendrocytes.rds", "OPCs.rds", "ROSMAP.ImmuneCells.6regions.snRNAseq.counts.rds")





obj_list = list()
for (file.name in all_names) {
	print(file.name)
    obj = readRDS(file.path(path, file.name))
    obj_list[file.name] = obj
}

# ROSMAP.ImmuneCells.6regions.snRNAseq.counts.rds
# dim(obj_list[file.name][[1]])
# [1]  16228 174420




all_names = c("Astrocytes.rds", "Immune_cells.rds",
            "Oligodendrocytes.rds", "OPCs.rds")


obj_list = list()
for (file.name in all_names) {
	print(file.name)
    obj = readRDS(file.path(path, file.name))
    obj_list[file.name] = obj
}


# > obj_list
# $Astrocytes.rds
# An object of class Seurat
# 33538 features across 149558 samples within 1 assay
# Active assay: RNA (33538 features, 0 variable features)
#  2 layers present: counts, data
#  1 dimensional reduction calculated: umap

# $Immune_cells.rds
# An object of class Seurat
# 33538 features across 83889 samples within 1 assay
# Active assay: RNA (33538 features, 0 variable features)
#  2 layers present: counts, data
#  1 dimensional reduction calculated: umap

# $Oligodendrocytes.rds
# An object of class Seurat
# 33538 features across 645142 samples within 1 assay
# Active assay: RNA (33538 features, 0 variable features)
#  2 layers present: counts, data

# $OPCs.rds
# An object of class Seurat
# 33538 features across 90502 samples within 1 assay
# Active assay: RNA (33538 features, 0 variable features)
#  2 layers present: counts, data
#  1 dimensional reduction calculated: umap



merged_obj <- Reduce(function(x, y) merge(x, y), obj_list)


merged_obj
# An object of class Seurat
# 33538 features across 969091 samples within 1 assay
# Active assay: RNA (33538 features, 0 variable features)
#  2 layers present: counts, data

metadata = as.data.frame(merged_obj@meta.data)
head(metadata)

counts = merged_obj@assays$RNA@counts
cells = colnames(counts)
genes = rownames(counts)

library(data.table)
library(Matrix)
fwrite(metadata, file = "ROSMAP_meta.csv", quote=F, row.names=T, col.names=T)

writeLines(genes, "genes.txt", sep="\n")
writeLines(cells, "barcodes.txt", sep="\n")
writeMM(t(counts), "raw_count.mtx")
