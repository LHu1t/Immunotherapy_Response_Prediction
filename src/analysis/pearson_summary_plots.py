import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Gene sets
APM_GENES = ["PSMB5", "PSMB6", "PSMB7", "PSMB8", "PSMB9", "PSMB10", "TAP1", "TAP2",
             "ERAP1", "ERAP2", "TAPBP", "CANX", "CALR", "PDIA3", "B2M", "HLA-A",
             "HLA-B", "HLA-C"]

TIS_GENES = ["PSMB10", "HLA-DQA1", "HLA-DRB1", "CMKLR1", "HLA-E", "NKG7", "CD8A",
             "CCL5", "CXCL9", "CD27", "CXCR6", "IDO1", "STAT1", "CD274", "CD276",
             "LAG3", "PDCD1LG2", "TIGIT"]


def load_fold_npz(fold_index):
    npz_path = f"CV/LUAD/test_predictions_best_pearson_fold{fold_index}.npz"
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing file: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return data["preds_z"], data["targets_raw"], data["gene_cols"]


def gene_to_col_index(gene_cols, gene_name):
    gene_full_name = f"{gene_name}_fpkm_uq"
    matches = np.where(gene_cols == gene_full_name)[0]
    if matches.size == 0:
        return None
    return int(matches[0])


def compute_genewise_pearson(preds, targets, gene_cols, genes):
    gene_r = {}
    for gene in genes:
        idx = gene_to_col_index(gene_cols, gene)
        if idx is None:
            continue
        r_value, _ = pearsonr(preds[:, idx], targets[:, idx])
        gene_r[gene] = r_value
    return gene_r


def main():
    os.makedirs("CV/LUAD/pearson_plots2", exist_ok=True)

    fold_indices = list(range(5))
    all_genes = sorted(set(APM_GENES + TIS_GENES))

    # Matrix: genes x folds
    pearson_matrix = np.full((len(all_genes), len(fold_indices)), np.nan, dtype=float)

    for j, fold in enumerate(fold_indices):
        preds, targets_raw, gene_cols = load_fold_npz(fold)
        gene_r = compute_genewise_pearson(preds, targets_raw, gene_cols, all_genes)
        for i, gene in enumerate(all_genes):
            if gene in gene_r:
                pearson_matrix[i, j] = gene_r[gene]

    # Print mean and standard deviation for individual genes
    print("\n" + "=" * 60)
    print("Gene-wise Pearson R (mean ± std across folds)")
    print("=" * 60)
    gene_means = np.nanmean(pearson_matrix, axis=1)
    gene_stds = np.nanstd(pearson_matrix, axis=1)
    for gene, mean_val, std_val in zip(all_genes, gene_means, gene_stds):
        print(f"{gene}: {mean_val:.4f} ± {std_val:.4f}")

    # Heatmap of Pearson correlations
    plt.figure(figsize=(10, 0.35 * len(all_genes) + 2))
    im = plt.imshow(pearson_matrix, aspect="auto", cmap="Reds", vmin=0, vmax=0.6)

    cbar = plt.colorbar(im)
    cbar.set_label("Pearson R", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    plt.yticks(
        ticks=np.arange(len(all_genes)),
        labels=all_genes,
        fontsize=14
    )

    plt.xticks(
        ticks=np.arange(len(fold_indices)),
        labels=[f"F{f}" for f in fold_indices],
        fontsize=14
    )

    plt.title("LUAD Gene-wise Pearson Correlations", fontsize=18)
    plt.xlabel("Fold", fontsize=16)
    plt.ylabel("Gene", fontsize=16)

    plt.tight_layout()
    plt.savefig("CV/LUAD/pearson_plots2/luad_pearson_heatmap.png", dpi=150)
    plt.close()

    # Boxplot of gene-wise Pearson by panel
    apm_indices = [i for i, g in enumerate(all_genes) if g in APM_GENES]
    tis_indices = [i for i, g in enumerate(all_genes) if g in TIS_GENES]

    apm_values = pearson_matrix[apm_indices, :].flatten()
    tis_values = pearson_matrix[tis_indices, :].flatten()

    # Remove NaNs if any genes are missing
    apm_values = apm_values[~np.isnan(apm_values)]
    tis_values = tis_values[~np.isnan(tis_values)]

    # Print mean and standard deviation for panels
    print("\n" + "=" * 60)
    print("Panel-wise Pearson R (mean ± std across genes/folds)")
    print("=" * 60)
    print(f"APM: {np.mean(apm_values):.4f} ± {np.std(apm_values):.4f}")
    print(f"T-cell: {np.mean(tis_values):.4f} ± {np.std(tis_values):.4f}")

    plt.figure(figsize=(6, 5))
    plt.boxplot([apm_values, tis_values], labels=["APM", "T-cell"])
    plt.ylabel("Pearson R")
    plt.title("LUAD Gene-wise Pearson by Panel", fontsize=18)
    plt.tight_layout()
    plt.savefig("CV/LUAD/pearson_plots2/luad_pearson_boxplot.png", dpi=150)
    plt.close()

    print("Saved: CV/LUAD/pearson_plots2/luad_pearson_heatmap.png")
    print("Saved: CV/LUAD/pearson_plots2/luad_pearson_boxplot.png")
if __name__ == "__main__":
    main()
