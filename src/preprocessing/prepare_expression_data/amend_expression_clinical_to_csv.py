import os
import numpy as np
import pandas as pd

# Configuration
print(os.listdir(".") )  # Debugging line to check current directory contents
clinical_csv = "LUAD_LUSC_Data/Immunotherapy_Prediction/nsclc_clinical_data_ALL.csv"  # CHANGE THIS to your clincial data file
rna_folders = ["LUAD_LUSC_Data/rna_organized_nsclc/TCGA-LUAD",
               "LUAD_LUSC_Data/rna_organized_nsclc/TCGA-LUSC"]  # CHANGE THIS to your folder path containing patient subfolders
tmb_csvs = ["LUAD_LUSC_Data/Immunotherapy_Prediction/LUSC_TMB_Output.csv",
               "LUAD_LUSC_Data/Immunotherapy_Prediction/LUAD_TMB_Output.csv"]  # CHANGE THIS to your folder paths containing TMB data

for folder in rna_folders:
    subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    print(subfolders[:3])  # show first 3 folder names

target_genes = ["PSMB5", "PSMB6", "PSMB7", "PSMB8", "PSMB9", "PSMB10", "TAP1", "TAP2", "ERAP1", "ERAP2", "TAPBP", "CANX", "CALR", "PDIA3", "B2M", "HLA-A", "HLA-B", "HLA-C", "HLA-DQA1", "HLA-DRB1", "CMKLR1", "HLA-E", "NKG7", "CD8A", "CCL5", "CXCL9", "CD27", "CXCR6", "IDO1", "STAT1", "CD274", "CD276", "LAG3", "PDCD1LG2", "TIGIT"] # Genes of interest
apm_genes = ["PSMB5", "PSMB6", "PSMB7", "PSMB8", "PSMB9", "PSMB10", "TAP1", "TAP2", "ERAP1", "ERAP2", "TAPBP", "CANX", "CALR", "PDIA3", "B2M", "HLA-A", "HLA-B", "HLA-C"]
tis_genes = ["PSMB10", "HLA-DQA1", "HLA-DRB1", "CMKLR1", "HLA-E", "NKG7", "CD8A", "CCL5", "CXCL9", "CD27", "CXCR6", "IDO1", "STAT1", "CD274", "CD276", "LAG3", "PDCD1LG2", "TIGIT"]
expression_column = "fpkm_uq_unstranded"

# Load clinical dataset
clinical_df = pd.read_csv(clinical_csv)
expression_data = {gene: [] for gene in target_genes}

# Loop through each patient
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
                    print(f"Error reading {tsv_path} for {sid}: {e}")
            else:
                print(f"No TSV found for {sid} in {folder}")

    if not found:
        print(f"No RNA data found for {sid} in any folder")

    for gene in target_genes:
        expression_data[gene].append(gene_values.get(gene))

# Load TMB data
tmb_records = []

for tmb in tmb_csvs:
    tmb_df = pd.read_csv(tmb)

    # Normalize TCGA_ID → submitter_id (TCGA-XX-YYYY)
    tmb_df["submitter_id"] = tmb_df["TCGA_ID"].str.split("-").str[:3].str.join("-")

    tmb_records.append(tmb_df[["submitter_id", "TMB"]])

if tmb_records:
    tmb_df_all = pd.concat(tmb_records, ignore_index=True)
    tmb_df_all = tmb_df_all.drop_duplicates(subset="submitter_id")
else:
    tmb_df_all = pd.DataFrame(columns=["submitter_id", "TMB"])
    
# Merge TMB data with clinical
clinical_df = clinical_df.merge(
    tmb_df_all,
    on="submitter_id",
    how="left"
)

for gene in target_genes:
    clinical_df[f"{gene}_fpkm_uq"] = expression_data[gene]

# Compute APM: mean(log2(expression + 1)) across APM genes
apm_cols = [f"{g}_fpkm_uq" for g in apm_genes]
clinical_df["APM"] = np.log2(clinical_df[apm_cols] + 1).mean(axis=1, skipna=True)

# Compute TIS: mean of z-scored TIS gene expressions (per gene across patients)
tis_cols = [f"{g}_fpkm_uq" for g in tis_genes]
tis_df = clinical_df[tis_cols]
tis_means = tis_df.mean(axis=0, skipna=True)
tis_stds = tis_df.std(axis=0, skipna=True).replace(0, np.nan)
tis_z = (tis_df - tis_means) / tis_stds
clinical_df["TIS"] = tis_z.mean(axis=1, skipna=True)

clinical_df.to_csv("clinical_expression_TMBTISAPM_LUADandLUSC.csv", index=False)
print("Combined clinical + expression data saved as 'clinical_expression_TMBTISAPM_LUADandLUSC.csv'")
