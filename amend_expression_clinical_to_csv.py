import os
import pandas as pd

# === Step 1: Config ===
print(os.listdir(".") )  # Debugging line to check current directory contents

clinical_csv = "LUAD_LUSC_Data/nsclc_clinical_data_LUAD.csv"  # CHANGE THIS to your clincial data file
rna_folder = "LUAD_LUSC_Data/rna_organized_nsclc/TCGA-LUAD"  # CHANGE THIS to your folder path containing patient subfolders

# Genes of interest (for lung adenocarcinoma)
target_genes = ["KRAS", "EGFR", "ALK", "BRAF", "ERBB2", "STK11"]

# The expected column in RNA files
expression_column = "fpkm_uq_unstranded"

# === Step 2: Load clinical dataset ===
clinical_df = pd.read_csv(clinical_csv)

# Ensure submitter_id exists
if "submitter_id" not in clinical_df.columns:
    raise ValueError("Column 'submitter_id' not found in clinical dataset!")

# Initialize a dictionary to hold gene expression data
expression_data = {gene: [] for gene in target_genes}

# === Step 3: Loop through patients ===
for sid in clinical_df["submitter_id"]:
    patient_folder = os.path.join(rna_folder, sid)
    gene_values = {}

    if os.path.exists(patient_folder):
        # Find the TSV file inside the patient folder
        tsv_files = [f for f in os.listdir(patient_folder) if f.endswith(".tsv")]
        if tsv_files:
            tsv_path = os.path.join(patient_folder, tsv_files[0])
            try:
                rna_df = pd.read_csv(tsv_path, sep="\t")
            except Exception as e:
                print(f"⚠️ Could not read TSV for {sid}: {e}")
                rna_df = pd.DataFrame()

            # Extract expression values for target genes
            for gene in target_genes:
                val = None
                if not rna_df.empty:
                    # Find matching gene row
                    match = rna_df[rna_df["gene_name"].str.upper() == gene]
                    if not match.empty and expression_column in match.columns:
                        val = match[expression_column].values[0]
                gene_values[gene] = val
        else:
            print(f"⚠️ No TSV found in folder for {sid}")
            gene_values = {gene: None for gene in target_genes}
    else:
        print(f"⚠️ No folder found for {sid}")
        gene_values = {gene: None for gene in target_genes}

    # Append expression values (even if None)
    for gene in target_genes:
        expression_data[gene].append(gene_values.get(gene))

# === Step 4: Merge expression data with clinical data ===
for gene in target_genes:
    clinical_df[f"{gene}_fpkm_uq"] = expression_data[gene]

# === Step 5: Save combined dataset ===
output_csv = "combined_clinical_expression_adenocarcinoma.csv"
clinical_df.to_csv(output_csv, index=False)

print(f"\n✅ Combined clinical + RNA-seq data saved to '{output_csv}'")
print(f"Total patients processed: {len(clinical_df)}")
print("Preview of merged data:")
print(clinical_df.head())
