import pandas as pd

# Load the clinical dataset (replace 'clinical_data.csv' with your actual file path)
clinical_df = pd.read_csv('LUAD_LUSC_Data/nsclc_clinical_data_clean.csv')

print(f"Unfiltered columns: {len(clinical_df.columns)}")

# Select biologically relevant columns ===
selected_columns = [
    'submitter_id',
    'demographic.gender',
    'diagnoses.0.age_at_diagnosis',
    'diagnoses.0.primary_diagnosis',
    'diagnoses.0.ajcc_pathologic_stage',
    'diagnoses.0.ajcc_pathologic_t',
    'diagnoses.0.ajcc_pathologic_n',
    'diagnoses.0.ajcc_pathologic_m',
    'diagnoses.0.prior_treatment',
    'diagnoses.0.treatments.0.treatment_or_therapy'
]

# Keep only existing columns
selected_columns = [col for col in selected_columns if col in clinical_df.columns]
clinical_df = clinical_df[selected_columns].copy()

print(f"Using {len(selected_columns)} selected clinical features.")
print("Columns retained:", selected_columns)

# Clean and standardize ===

# --- Convert age from days to years ---
if 'diagnoses.0.age_at_diagnosis' in clinical_df.columns:
    clinical_df['age_years'] = clinical_df['diagnoses.0.age_at_diagnosis'] / 365.25
    clinical_df.drop(columns=['diagnoses.0.age_at_diagnosis'], inplace=True)

# --- Normalize gender values ---
if 'demographic.gender' in clinical_df.columns:
    clinical_df['demographic.gender'] = (
        clinical_df['demographic.gender']
        .astype(str)
        .str.lower()
        .replace({
            'f': 'female', 'female': 'female',
            'm': 'male', 'male': 'male',
            'nan': None
        })
    )

# --- Simplify AJCC pathologic stage ---
if 'diagnoses.0.ajcc_pathologic_stage' in clinical_df.columns:
    clinical_df['diagnoses.0.ajcc_pathologic_stage'] = (
        clinical_df['diagnoses.0.ajcc_pathologic_stage']
        .astype(str)
        .str.replace(r'(Stage\s*[IVX]+).*', r'\1', regex=True)
    )

# --- Drop duplicate patients (if any) ---
clinical_df = clinical_df.drop_duplicates(subset=['submitter_id'])

# Ensure consistent lowercase text for diagnosis names
if 'diagnoses.0.primary_diagnosis' in clinical_df.columns:
    clinical_df['diagnoses.0.primary_diagnosis'] = (
        clinical_df['diagnoses.0.primary_diagnosis']
        .astype(str)
        .str.lower()
        .str.strip()
    )
else:
    raise ValueError("Column 'diagnoses.0.primary_diagnosis' not found in dataset.")

# Define keyword filters for each cancer subtype
adenocarcinoma_keywords = ['adenocarcinoma']
squamous_keywords = ['squamous']

# Create subsets
adenocarcinoma_df = clinical_df[
    clinical_df['diagnoses.0.primary_diagnosis'].str.contains('|'.join(adenocarcinoma_keywords), na=False)
].copy()

squamous_df = clinical_df[
    clinical_df['diagnoses.0.primary_diagnosis'].str.contains('|'.join(squamous_keywords), na=False)
].copy()

# Save each subset to separate CSV files
adenocarcinoma_df.to_csv("nsclc_clinical_data_LUAD.csv", index=False)
squamous_df.to_csv("nsclc_clinical_data_LUSC.csv", index=False)

# === Print summary ===
print("✅ Split complete.")
print(f"Adenocarcinoma patients: {len(adenocarcinoma_df)}")
print(f"Squamous cell carcinoma patients: {len(squamous_df)}")