import os
import re
import random
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
from scipy.stats import pearsonr

# === CONFIG ===
svs_root = r"/Volumes/SeagateBas/Immunotherapy/LUAD"
gene_csv = "LUAD_LUSC_Data/Immunotherapy_Prediction/combined_clinical_expression_ALL2.csv"
img_size = 224
batch_size = 8
num_epochs = 50
lr = 1e-4
tiles_per_slide = 5
tile_size = 512
val_ratio = 0.2

# === DEVICE ===
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Metal (MPS) GPU")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU (MPS not available)")

# === STEP 1: Load and clean gene expression data ===
df_expr = pd.read_csv(gene_csv)
df_expr.columns = [c.strip() for c in df_expr.columns]

id_col = None
for c in df_expr.columns:
    if any(k in c.lower() for k in ["sample", "submitter", "case", "id"]):
        id_col = c
        break
if id_col is None:
    raise ValueError("No ID column found in gene expression CSV")

df_expr["sample_id"] = df_expr[id_col].str.upper().str.strip()

gene_cols = [c for c in df_expr.columns if c.endswith("_fpkm_uq")]
if not gene_cols:
    raise ValueError("No gene expression columns ending with '_fpkm_uq' found")

df_expr[gene_cols] = df_expr[gene_cols].apply(pd.to_numeric, errors='coerce')
df_expr = df_expr.dropna(subset=gene_cols, how='any')
df_expr[gene_cols] = np.log1p(df_expr[gene_cols])
df_expr[gene_cols] = (df_expr[gene_cols] - df_expr[gene_cols].mean()) / df_expr[gene_cols].std()

print(f"✅ Cleaned expression data, {len(gene_cols)} genes")

# === STEP 2: Find SVS files and match ===
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

# === STEP 3: Dataset with multi-tile sampling ===
class WSIExpressionDataset(Dataset):
    def __init__(self, df, genes, transform=None, tile_size=512, tiles_per_slide=5):
        self.df = df.reset_index(drop=True)
        self.genes = genes
        self.transform = transform
        self.tile_size = tile_size
        self.tiles_per_slide = tiles_per_slide

    def __len__(self):
        return len(self.df)

    def _random_tile_coords(self, w, h):
        x = random.randint(0, max(0, w - self.tile_size))
        y = random.randint(0, max(0, h - self.tile_size))
        return x, y

    def _is_blank(self, img):
        """Heuristic: reject tiles with low saturation / tissue content."""
        hsv = img.convert("HSV")
        h, s, v = np.array(hsv).transpose(2, 0, 1)
        return (s.mean() < 15) or (v.mean() > 230)

    def _load_random_tile(self, path):
        slide = openslide.OpenSlide(path)
        w, h = slide.level_dimensions[0]
        for _ in range(10):  # up to 10 tries to find a non-blank tile
            x, y = self._random_tile_coords(w, h)
            tile = slide.read_region((x, y), 0, (self.tile_size, self.tile_size)).convert("RGB")
            if not self._is_blank(tile):
                slide.close()
                return tile
        slide.close()
        return Image.new("RGB", (self.tile_size, self.tile_size), (255, 255, 255))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_list = []
        for _ in range(self.tiles_per_slide):
            tile = self._load_random_tile(row["svs_path"])
            if self.transform:
                tile = self.transform(tile)
            img_list.append(tile)
        imgs = torch.stack(img_list)  # shape: [tiles_per_slide, 3, H, W]
        expr = torch.tensor(row[self.genes].values.astype(np.float32))
        return imgs, expr

# === STEP 4: Transforms ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = WSIExpressionDataset(df_matched, gene_cols, transform, tile_size=tile_size, tiles_per_slide=tiles_per_slide)

# Split
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])
print(f"Training: {train_size}, Validation: {val_size}")

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)  # batch=1 per slide (multi-tile inside)

# === STEP 5: Model (ResNet50 full fine-tuning) ===
resnet = models.resnet50(pretrained=True)
# Unfreeze all layers for full fine-tuning
for param in resnet.parameters():
    param.requires_grad = True
resnet.fc = nn.Linear(resnet.fc.in_features, len(gene_cols))
model = resnet.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
# OR:
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

# === STEP 6: Training loop ===
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for imgs_batch, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        imgs_batch, targets = imgs_batch.to(device), targets.to(device)
        # imgs_batch: [B, tiles_per_slide, 3, H, W]
        preds_list = []
        for t in range(imgs_batch.size(1)):
            preds = model(imgs_batch[:, t, :, :, :])
            preds_list.append(preds)
        preds_mean = torch.stack(preds_list, dim=0).mean(dim=0)
        loss = criterion(preds_mean, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs_batch.size(0)
    train_loss /= len(train_loader.dataset)

    # === Validation ===
    model.eval()
    val_loss = 0.0
    per_gene_sq_sum = torch.zeros(len(gene_cols), device=device)
    per_gene_count = 0
    preds_all, targets_all = [], []

    with torch.no_grad():
        for imgs_batch, targets in val_loader:
            imgs_batch, targets = imgs_batch.to(device), targets.to(device)
            preds_list = []
            for t in range(imgs_batch.size(1)):
                preds = model(imgs_batch[:, t, :, :, :])
                preds_list.append(preds)
            preds_mean = torch.stack(preds_list, dim=0).mean(dim=0)
            loss = criterion(preds_mean, targets)
            val_loss += loss.item()
            per_gene_sq_sum += ((preds_mean - targets) ** 2).sum(dim=0)
            per_gene_count += 1
            preds_all.append(preds_mean.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    preds_all = np.vstack(preds_all)
    targets_all = np.vstack(targets_all)

    print(preds_all.shape, targets_all.shape)
    print(preds_all[:5, :5])
    print(targets_all[:5, :5])
    print("First 10 genes:", gene_cols[:10])
    print("Preds mean/std:", preds_all.mean(), preds_all.std())
    print("Targets mean/std:", targets_all.mean(), targets_all.std())

    # Per-gene stats
    per_gene_mse = (per_gene_sq_sum / per_gene_count).cpu().numpy()
    pearsons = []
    for i in range(len(gene_cols)):
        r, _ = pearsonr(preds_all[:, i], targets_all[:, i])
        pearsons.append(r)

    results_df = pd.DataFrame({"gene": gene_cols, "val_mse": per_gene_mse, "pearson_r": pearsons}).sort_values("val_mse")

    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(results_df.to_string(index=False))
    print("-" * 60)

torch.save(model.state_dict(), "resnet50_gene_expression_multitile.pt")
print("✅ Model saved as resnet50_gene_expression_multitile.pt")
