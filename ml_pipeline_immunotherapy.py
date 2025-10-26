import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
import openslide

# === CONFIG ===
svs_root = r"/Volumes/SeagateBas/Immunotherapy/LUAD"
gene_csv = "LUAD_LUSC_Data/Immunotherapy_Prediction/combined_clinical_expression_ALL2.csv"
img_size = 224
batch_size = 8
num_epochs = 15
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === STEP 1: Load gene expression data ===
df_expr = pd.read_csv(gene_csv)
df_expr.columns = [c.strip() for c in df_expr.columns]

# Find sample_id column
id_col = None
for c in df_expr.columns:
    if any(k in c.lower() for k in ["sample", "submitter", "case", "id"]):
        id_col = c
        break
if id_col is None:
    raise ValueError("No ID column found in gene expression CSV")

df_expr["sample_id"] = df_expr[id_col].str.upper().str.strip()

# Extract gene expression columns
gene_cols = [c for c in df_expr.columns if c.endswith("_fpkm_uq")]
if not gene_cols:
    raise ValueError("No gene expression columns ending with '_fpkm_uq' found")

print(f"Using {len(gene_cols)} genes: {gene_cols}")
print("Gene expression value ranges:")

# --- Clean and normalize gene expression values ---
# Convert to numeric
df_expr[gene_cols] = df_expr[gene_cols].apply(pd.to_numeric, errors='coerce')
df_expr = df_expr.dropna(subset=gene_cols, how='any')

# Add +1 and log-transform (compress dynamic range)
df_expr[gene_cols] = np.log1p(df_expr[gene_cols])

# Standardize per gene (zero mean, unit variance)
df_expr[gene_cols] = (df_expr[gene_cols] - df_expr[gene_cols].mean()) / df_expr[gene_cols].std()

print("✅ Cleaned and normalized gene expression summary:")
print(df_expr[gene_cols].describe().round(3))

# === STEP 2: Find SVS files and match by TCGA ID ===
def extract_tcga_id(filename):
    match = re.search(r"(TCGA-\w\w-\w\w\w\w)", filename)
    return match.group(1) if match else None

svs_map = {}
for root, _, files in os.walk(svs_root):
    for f in files:
        if f.lower().endswith(".svs"):
            case_id = extract_tcga_id(f)
            if case_id:
                svs_map[case_id] = os.path.join(root, f)

df_expr["svs_path"] = df_expr["sample_id"].map(svs_map)
df_matched = df_expr.dropna(subset=["svs_path"])
print(f"✅ {len(df_matched)} samples matched with SVS files")

if len(df_matched) == 0:
    raise SystemExit("No matching cases found between CSV and SVS files")

# === STEP 3: Custom Dataset ===
class WSIExpressionDataset(Dataset):
    def __init__(self, df, genes, transform=None, tile_size=512):
        self.df = df.reset_index(drop=True)
        self.genes = genes
        self.transform = transform
        self.tile_size = tile_size

    def __len__(self):
        return len(self.df)

    def _load_svs_tile(self, path):
        try:
            slide = openslide.OpenSlide(path)
            level = 0
            w, h = slide.level_dimensions[level]
            # extract a central tile (you can randomize this later)
            x = max(0, (w - self.tile_size) // 2)
            y = max(0, (h - self.tile_size) // 2)
            region = slide.read_region((x, y), level, (self.tile_size, self.tile_size)).convert("RGB")
            slide.close()
            return region
        except Exception as e:
            print(f"⚠️ Error reading {path}: {e}")
            return Image.new("RGB", (self.tile_size, self.tile_size), (0, 0, 0))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_svs_tile(row["svs_path"])
        if self.transform:
            img = self.transform(img)
        expr = torch.tensor(row[self.genes].values.astype(np.float32))
        return img, expr

# === STEP 4: Define transforms ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = WSIExpressionDataset(df_matched, gene_cols, transform)

# Split into train/val
val_ratio = 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])
print(f"Training: {train_size}, Validation: {val_size}")

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# === STEP 5: Model (ResNet for regression) ===
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, len(gene_cols))
model = resnet.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# === STEP 6: Training loop ===
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * imgs.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# === STEP 7: Save model ===
torch.save(model.state_dict(), "resnet_gene_expression.pt")
print("✅ Model saved as resnet_gene_expression.pt")
