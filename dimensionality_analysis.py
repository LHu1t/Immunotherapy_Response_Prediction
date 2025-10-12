import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import umap

# === Step 1: Load data ===
df = pd.read_csv("LUAD_LUSC_Data/combined_clinical_expression_ALL2.csv")

# === Step 2: Detect the cancer subtype ===
diagnosis_col = None
for col in df.columns:
    if "primary_diagnosis" in col:
        diagnosis_col = col
        break

if diagnosis_col is None:
    raise ValueError("Could not find a column containing 'primary_diagnosis'")

def classify_subtype(x):
    if isinstance(x, str):
        x = x.lower()
        if "adeno" in x or "luad" in x:
            return "Adenocarcinoma"
        elif "squamous" in x or "lusc" in x:
            return "Squamous"
    return None

df["Cancer_Subtype"] = df[diagnosis_col].apply(classify_subtype)
df = df[df["Cancer_Subtype"].notnull()]  # keep only valid subtypes

# === Step 3: Select gene expression columns ===
expr_cols = [c for c in df.columns if c.endswith("_fpkm_uq")]
if not expr_cols:
    raise ValueError("No *_fpkm_uq columns found in the CSV — please check column names.")

# Drop rows with missing expression values
df_filtered = df.dropna(subset=expr_cols)

# === Step 4: Prepare data ===
X = np.log2(df_filtered[expr_cols] + 1) # Log-transform expression values

y = df_filtered["Cancer_Subtype"].values

# Standardize features
X_scaled = StandardScaler().fit_transform(X)

# === Step 5: PCA (for baseline visualization) ===
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df_filtered["PCA1"], df_filtered["PCA2"] = pca_result[:, 0], pca_result[:, 1]

# Top genes driving PC1/PC2
loadings = pd.DataFrame(
    pca.components_.T,
    index=expr_cols,
    columns=["PC1", "PC2"]
)
top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(10)
top_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(10)

print("\nTop 10 genes driving PC1:")
print(top_pc1)
print("\nTop 10 genes driving PC2:")
print(top_pc2)

# === Step 6: Linear Discriminant Analysis (supervised) ===
lda = LDA(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)
df_filtered["LDA1"] = X_lda[:, 0]

lda_loadings = pd.Series(lda.coef_[0], index=expr_cols)
top_lda = lda_loadings.abs().sort_values(ascending=False).head(10)
print("\nTop 10 genes driving LDA separation:")
print(top_lda)

# === Step 7: t-SNE (nonlinear) ===
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)
df_filtered["tSNE1"], df_filtered["tSNE2"] = X_tsne[:, 0], X_tsne[:, 1]

# Top correlated genes with tSNE1
tsne_corr = X.corrwith(df_filtered["tSNE1"]).abs().sort_values(ascending=False).head(10)
print("\nTop 10 genes correlated with tSNE1:")
print(tsne_corr)

# === Step 8: UMAP (nonlinear, often clearer than t-SNE) ===
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X_scaled)
df_filtered["UMAP1"], df_filtered["UMAP2"] = X_umap[:, 0], X_umap[:, 1]

umap_corr = X.corrwith(df_filtered["UMAP1"]).abs().sort_values(ascending=False).head(10)
print("\nTop 10 genes correlated with UMAP1:")
print(umap_corr)

# === Step 9: Visualization ===
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plots = [
    ("PCA1", "PCA2", "PCA (unsupervised)"),
    ("LDA1", None, "LDA (supervised)"),
    ("tSNE1", "tSNE2", "t-SNE (nonlinear)"),
    ("UMAP1", "UMAP2", "UMAP (nonlinear)")
]

for ax, (xcol, ycol, title) in zip(axes.flatten(), plots):
    if ycol:
        for subtype in df_filtered["Cancer_Subtype"].unique():
            subset = df_filtered[df_filtered["Cancer_Subtype"] == subtype]
            ax.scatter(subset[xcol], subset[ycol], label=subtype, alpha=0.7)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
    else:
        # 1D LDA plot
        for subtype in df_filtered["Cancer_Subtype"].unique():
            subset = df_filtered[df_filtered["Cancer_Subtype"] == subtype]
            ax.hist(subset[xcol], bins=20, alpha=0.6, label=subtype)
        ax.set_xlabel(xcol)
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()

# Show number of patients (non-missing rows) used in the analysis
print(f"\nNumber of patients included in analysis: {len(df_filtered)}")