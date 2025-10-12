import os
import pandas as pd

# === Step 1: Config ===
print(os.listdir(".") )  # Debugging line to check current directory contents

clinical_csv = "LUAD_LUSC_Data/nsclc_clinical_data_ALL.csv"  # CHANGE THIS to your clincial data file
rna_folders = ["LUAD_LUSC_Data/rna_organized_nsclc/TCGA-LUAD",
               "LUAD_LUSC_Data/rna_organized_nsclc/TCGA-LUSC"]  # CHANGE THIS to your folder path containing patient subfolders

for folder in rna_folders:
    print(f"\n📁 Listing folders in: {folder}")
    subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    print(subfolders[:10])  # show first 10 folder names

target_genes = ["KRAS", "EGFR", "ALK", "BRAF", "ERBB2", "STK11", "TP53", "CDKN2A", "NFE2L2", "KEAP1", "PIK3CA", "SOX2"]  # Genes of interest
expression_column = "fpkm_uq_unstranded"

# === Step 2: Load clinical dataset ===
clinical_df = pd.read_csv(clinical_csv)

expression_data = {gene: [] for gene in target_genes}

# === Step 3: Loop through each patient ===
for sid in clinical_df["submitter_id"]:
    gene_values = {gene: None for gene in target_genes}
    found = False

    # Search through both folders
    for folder in rna_folders:
        patient_folder = os.path.join(folder, sid)
        if os.path.exists(patient_folder):
            tsv_files = [f for f in os.listdir(patient_folder) if f.endswith(".tsv")] # Find TSV file inside patient RNA folder
            if tsv_files:
                tsv_path = os.path.join(patient_folder, tsv_files[0])
                try:
                    rna_df = pd.read_csv(tsv_path, sep="\t", skiprows=1) # Skip first row since it contains metadata
                    
                    # Extract expression values for target genes
                    for gene in target_genes:
                        match = rna_df[rna_df["gene_name"].str.upper() == gene]
                        if not match.empty and expression_column in match.columns:
                            gene_values[gene] = match[expression_column].values[0]
                    found = True
                    break  # stop searching once found
                except Exception as e:
                    print(f"⚠️ Error reading {tsv_path} for {sid}: {e}")
            else:
                print(f"⚠️ No TSV found for {sid} in {folder}")

    if not found:
        print(f"⚠️ No RNA data found for {sid} in any folder")

    for gene in target_genes:
        expression_data[gene].append(gene_values.get(gene))

# === Step 4: Merge with clinical ===
for gene in target_genes:
    clinical_df[f"{gene}_fpkm_uq"] = expression_data[gene]

clinical_df.to_csv("combined_clinical_expression_ALL2.csv", index=False)
print("Combined clinical + expression data saved as 'combined_clinical_expression_ALL2.csv'")
