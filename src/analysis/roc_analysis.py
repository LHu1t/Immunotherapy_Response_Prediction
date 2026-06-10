import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import pearsonr
import os

# Gene sets
APM_GENES = ["PSMB5", "PSMB6", "PSMB7", "PSMB8", "PSMB9", "PSMB10", "TAP1", "TAP2", 
             "ERAP1", "ERAP2", "TAPBP", "CANX", "CALR", "PDIA3", "B2M", "HLA-A", 
             "HLA-B", "HLA-C"]

TIS_GENES = ["PSMB10", "HLA-DQA1", "HLA-DRB1", "CMKLR1", "HLA-E", "NKG7", "CD8A", 
             "CCL5", "CXCL9", "CD27", "CXCR6", "IDO1", "STAT1", "CD274", "CD276", 
             "LAG3", "PDCD1LG2", "TIGIT"]


def compute_signature_scores(preds, targets, gene_cols, geneset):
    """
    Compute mean signature scores for a given geneset.
    
    Args:
        preds: predicted expression matrix (n_samples, n_genes)
        targets: target expression matrix (n_samples, n_genes)
        gene_cols: list of gene names corresponding to columns
        geneset: list of genes in the signature
    
    Returns:
        preds_sig: mean predicted signature scores (n_samples,)
        targets_sig: mean target signature scores (n_samples,)
    """
    # Find indices of genes in the geneset
    gene_indices = []
    for gene in geneset:
        # Gene names in gene_cols have format: GENENAME_fpkm_uq
        gene_full_name = f"{gene}_fpkm_uq"
        if gene_full_name in gene_cols:
            gene_indices.append(np.where(gene_cols == gene_full_name)[0][0])
        else:
            print(f"Warning: Gene {gene_full_name} not found in gene_cols")
    
    if len(gene_indices) == 0:
        raise ValueError(f"No genes from {geneset} found in gene_cols")
    
    # Compute mean signature scores
    preds_sig = np.mean(preds[:, gene_indices], axis=1)
    targets_sig = np.mean(targets[:, gene_indices], axis=1)
    
    return preds_sig, targets_sig


def main():
    # Load the NPZ file
    npz_file = "CV/LUSC/test_predictions_best_pearson_fold1.npz"
    
    if not os.path.exists(npz_file):
        print(f"Error: {npz_file} not found")
        return
    
    data = np.load(npz_file, allow_pickle=True)
    
    preds_all = data['preds_z']
    targets_all = data['targets_z']
    targets_all_raw = data['targets_raw']
    gene_cols = data['gene_cols']

    print("First 20 gene_cols:")
    print(gene_cols[:20])
    print(type(gene_cols), gene_cols.dtype)

    if "TMB" in gene_cols:
        tmb_index = np.where(gene_cols == "TMB")[0][0]
        print(f"TMB column index: {tmb_index}")
        print("TMB values (targets_raw):")
        print(targets_all_raw[:, tmb_index])
    else:
        print("TMB column not found in gene_cols")
    
    print(f"Loaded NPZ file: {npz_file}")
    print(f"  preds_z shape: {preds_all.shape}")
    print(f"  targets_z shape: {targets_all.shape}")
    print(f"  targets_raw shape: {targets_all_raw.shape}")
    print(f"  Number of genes: {len(gene_cols)}")
    
    # Print mean and standard deviation for each gene and TMB
    print("\n" + "="*60)
    print("Gene Expression and TMB Statistics (targets_raw):")
    print("="*60)
    for i, gene in enumerate(gene_cols):
        mean_val = np.mean(targets_all_raw[:, i])
        std_val = np.std(targets_all_raw[:, i])
        r_value, p_value = pearsonr(preds_all[:, i], targets_all_raw[:, i])
        print(f"{gene}: Mean = {mean_val:.4f}, Std Dev = {std_val:.4f}, Pearson R = {r_value:.4f}")
    
    # Create output directory
    os.makedirs("CV/LUSC/ROC", exist_ok=True)
    
    # Generate ROC plots for both genesets
    for name, geneset in [("TIS", TIS_GENES), ("APM", APM_GENES)]:
        print(f"\nGenerating ROC plot for {name}...")
        
        try:
            preds_sig, targets_sig = compute_signature_scores(
                preds_all, targets_all_raw, gene_cols, geneset
            )
            
            # Define top quartile based on TRUE signature
            threshold = 2.75 # np.percentile(targets_sig, 75)     # LUAD: TIS:3.10, APM:4.54        LUSC: TIS: 2.75 APM: 4.49
            y_true = (targets_sig >= threshold).astype(int)
            
            # Use predicted signature as score
            fpr, tpr, _ = roc_curve(y_true, preds_sig)
            roc_auc = auc(fpr, tpr)
            
            # Create ROC plot
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], "k--", linewidth=1)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{name} Top-Quartile Classification ROC")
            plt.legend(loc="lower right")
            plt.tight_layout()
            
            output_file = f"CV/LUSC/ROC/{name}_ROC_fold1.png"
            plt.savefig(output_file, dpi=150)
            plt.close()
            
            print(f"  ROC AUC: {roc_auc:.4f}")
            print(f"  Saved to: {output_file}")
            print(f"  Top quartile threshold: {threshold:.4f}")
            print(f"  Samples in top quartile: {np.sum(y_true)}")
            
        except Exception as e:
            print(f"  Error generating ROC for {name}: {str(e)}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
