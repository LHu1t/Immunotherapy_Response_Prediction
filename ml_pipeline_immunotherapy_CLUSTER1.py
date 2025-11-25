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
import matplotlib.pyplot as plt
import time

# === CONFIG ===
svs_roots = [
    r"/Volumes/SeagateBas/Immunotherapy/LUAD2", # "/home/zcemlhu/Scratch/LUAD",
    r"/Volumes/SeagateBas/Immunotherapy/LUAD" # "/home/zcemlhu/Scratch/LUAD2"
]
gene_csv = "LUAD_LUSC_Data/Immunotherapy_Prediction/combined_clinical_expression_ALL2.csv" # "/home/zcemlhu/Scratch/TestFile1_15_11_2025/combined_clinical_expression_ALL2.csv"
img_size = 224
num_epochs = 1000

# NOTE: lower LR when fine-tuning whole backbone; feel free to try 1e-5 -> 1e-4
lr = 1e-5

# increase tiles_per_slide for better variance (try 10-15 if you have time)
tiles_per_slide = 10
tile_size = 512
val_ratio = 0.2

# early stopping by mean Pearson
early_stop_patience = 500

# === DEVICE SELECTION FOR CLUSTER + MAC ===
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU: {}".format(torch.cuda.get_device_name(0)))
elif torch.backends.mps.is_available():  # Mac M1/M2
    device = torch.device("mps")
    print("Using Apple Metal (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU (no GPU available)")

print("Device =", device)

torch.backends.cudnn.benchmark = True # enable cudnn autotuner for fixed-size inputs

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

# Normalize targets (z-score) - ensures training happens in stable numeric range
df_expr[gene_cols] = df_expr[gene_cols].apply(pd.to_numeric, errors='coerce')
df_expr = df_expr.dropna(subset=gene_cols, how='any')

# log + z-score
df_expr[gene_cols] = np.log1p(df_expr[gene_cols])
y_means = df_expr[gene_cols].mean()
y_stds = df_expr[gene_cols].std()
df_expr[gene_cols] = (df_expr[gene_cols] - y_means) / y_stds

print("Extracted, Z scored, and logged expression data, {} genes".format(len(gene_cols)))

# === Add clinical covariates ===
# Ensure age and gender exist
if 'age_years' not in df_expr.columns or 'demographic.gender' not in df_expr.columns:
    raise ValueError("Expected 'age_years' and 'demographic.gender' in clinical CSV")

# Normalize and encode
df_expr['age_years'] = pd.to_numeric(df_expr['age_years'], errors='coerce').fillna(df_expr['age_years'].median())
df_expr['age_years_z'] = (df_expr['age_years'] - df_expr['age_years'].mean()) / df_expr['age_years'].std()

# Encode gender as numeric: male=0, female=1 (or vice versa)
df_expr['gender_encoded'] = df_expr['demographic.gender'].str.lower().map({'male': 0, 'female': 1}).fillna(0)

# List of covariate columns
clinical_cols = ['age_years_z', 'gender_encoded']


# === STEP 2: Find SVS files and match ===
def extract_tcga_id(filename):
    match = re.search(r"(TCGA-\w\w-\w\w\w\w)", filename)
    return match.group(1) if match else None

svs_map = {}
for svs_root in svs_roots:
    for root, _, files in os.walk(svs_root):
        for f in files:
            if f.startswith("._"):
                continue  # skip hidden files on Mac
            if f.lower().endswith(".svs"):
                case_id = extract_tcga_id(f)
                if case_id:
                    # if duplicates exist, keep the first occurrence
                    if case_id not in svs_map:
                        svs_map[case_id] = os.path.join(root, f)

df_expr["svs_path"] = df_expr["sample_id"].map(svs_map)
df_matched = df_expr.dropna(subset=["svs_path"])
print("{} samples matched with SVS files".format(len(df_matched)))

if len(df_matched) == 0:
    raise SystemExit("No matching cases found between CSV and SVS files")

# === STEP 3: Dataset with multi-tile sampling ===
# === STEP 3: Dataset with caching ===
class WSIExpressionDataset(Dataset):
    def __init__(self, df, genes, transform=None, tile_size=512, tiles_per_slide=100, cache=True, max_cache_tiles=300):
        self.df = df.reset_index(drop=True)
        self.genes = genes
        self.clinical_cols = clinical_cols or []
        self.transform = transform
        self.tile_size = tile_size
        self.tiles_per_slide = tiles_per_slide
        self.cache = cache
        self.max_cache_tiles = max_cache_tiles
        self.tile_cache = {}  # {svs_path: [PIL.Image, ...]}

    def __len__(self):
        return len(self.df)

    def _random_tile_coords(self, w, h):
        x = random.randint(0, max(0, w - self.tile_size))
        y = random.randint(0, max(0, h - self.tile_size))
        return x, y

    def _is_blank(self, img):
        """Heuristic: reject tiles with low saturation / tissue content."""
        hsv = np.array(img.convert("HSV"))
        s = hsv[..., 1]
        v = hsv[..., 2]
        tissue_pixels = ((s > 20) & (v < 240)).sum()
        return tissue_pixels < (0.10 * s.size)

    def _load_random_tile(self, slide):
        w, h = slide.level_dimensions[0]
        for _ in range(30):
            x, y = self._random_tile_coords(w, h)
            tile = slide.read_region((x, y), 0, (self.tile_size, self.tile_size)).convert("RGB")
            if not self._is_blank(tile):
                return tile
        return Image.new("RGB", (self.tile_size, self.tile_size), (255, 255, 255))
    
    def _tile_full_slide(self, slide):
        TARGET = 16000  # target largest dimension

        dims = slide.level_dimensions
        level = min(
            range(len(dims)),
            key=lambda i: abs(max(dims[i]) - TARGET)
        )
        w, h = dims[level]
        ds = slide.level_downsamples[level]

        tiles = []
        for y in range(0, h, self.tile_size):
            for x in range(0, w, self.tile_size):
                tile = slide.read_region(
                    (int(x * ds), int(y * ds)),
                    level,
                    (self.tile_size, self.tile_size)
                ).convert("RGB")

                if not self._is_blank(tile):
                    tiles.append(tile)

        if len(tiles) == 0:
            tiles.append(Image.new("RGB", (self.tile_size, self.tile_size), (255,255,255)))

        return tiles

    def _load_or_cache_tiles(self, svs_path):
        if self.cache and svs_path in self.tile_cache:
            return self.tile_cache[svs_path]

        slide = openslide.OpenSlide(svs_path)
        tiles = self._tile_full_slide(slide)
        slide.close()

        if self.cache:
            self.tile_cache[svs_path] = tiles
        return tiles

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        svs_path = row["svs_path"]
        tiles = self._load_or_cache_tiles(svs_path)

        # Sample fixed number of tiles NOTE: I NEED TO CHANGE THIS FOR A MORE CONSISTENT WAY TO GET A FIXED NUMBER OF TILES
        if len(tiles) >= self.tiles_per_slide:
            tiles = random.sample(tiles, self.tiles_per_slide)
        else:
            # pad with blank tiles if not enough
            tiles += [Image.new("RGB", (self.tile_size, self.tile_size), (255,255,255))] * (self.tiles_per_slide - len(tiles))


        # Transform ALL tiles
        img_list = []
        for tile in tiles:
            if self.transform:
                tile = self.transform(tile)
            img_list.append(tile)

        imgs = torch.stack(img_list)  # [T, 3, H, W]

        expr = torch.tensor(row[self.genes].values.astype(np.float32))
        clinical = torch.tensor(row[self.clinical_cols].values.astype(np.float32))

        return imgs, expr, clinical

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
print("Training: {}, Validation: {}".format(train_size, val_size))

train_loader = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=True, num_workers=8)
val_loader = DataLoader(val_set, batch_size=4, shuffle=False, pin_memory=True, num_workers=8)

# === STEP 5: Model (ResNet50 + clinical features) ===
resnet = models.resnet50(pretrained=True)
in_features = resnet.fc.in_features
resnet.fc = nn.Identity()  # removes final FC; now resnet outputs the embedding vector

class ImageClinicalModel(nn.Module):
    def __init__(self, base_model, n_genes, n_clinical):
        super().__init__()
        self.base = base_model
        self.fc = nn.Sequential(
            nn.Linear(in_features + n_clinical, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_genes)
        )

    def forward(self, imgs, clinical):
        feats = self.base(imgs)  # [B, in_features]
        if clinical.nelement() > 0:  # safety check for empty covariate vector
            feats = torch.cat([feats, clinical], dim=1)
        out = self.fc(feats)
        return out

model = ImageClinicalModel(resnet, len(gene_cols), len(clinical_cols))

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)

model = model.to(device)

# Freeze batchnorm layers (stable training on small batch sizes)
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

# Optionally you can partially unfreeze layers (comment/uncomment as needed)
# for name, param in model.named_parameters():
#     if "layer4" in name or "fc" in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

# === Loss / optimizer / scheduler ===
criterion = nn.SmoothL1Loss(beta=0.1, reduction='mean')  # more stable than MSE
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)

# adaptive scheduler: reduce LR when val loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)


# pearson loss (differentiable proxy)
def pearson_loss(preds, targets, eps=1e-8):
    # preds, targets: [B, G]
    preds_c = preds - preds.mean(dim=0, keepdim=True)
    targ_c = targets - targets.mean(dim=0, keepdim=True)
    num = (preds_c * targ_c).sum(dim=0)
    den = torch.sqrt((preds_c**2).sum(dim=0) * (targ_c**2).sum(dim=0) + eps)
    r = num / den
    # return 1 - mean(r)  (we minimize this)
    return (1.0 - r).mean()

# === STEP 6: Training loop with batched multi-tile inference ===
best_mean_pearson = -999.0
patience_counter = 0

for epoch in range(num_epochs):

    t_epoch_start = time.perf_counter()

    model.train()
    train_loss = 0.0

    t_data = 0.0
    t_forward = 0.0
    t_backward = 0.0

    for imgs_batch, targets, clinical_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        t0 = time.perf_counter()

        B, T, C, H, W = imgs_batch.shape

        imgs_batch = imgs_batch.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        clinical_batch = clinical_batch.to(device, non_blocking=True)

        t_data += time.perf_counter() - t0

        optimizer.zero_grad()

        # ===============================
        # D. MULTI-TILE BATCHED INFERENCE
        # ===============================
        t1 = time.perf_counter()
        imgs_flat = imgs_batch.view(B*T, C, H, W)
        clinical_expanded = clinical_batch.repeat_interleave(T, dim=0)

        preds_flat = model(imgs_flat, clinical_expanded)          # [B*T, G]
        preds_mean = preds_flat.view(B, T, -1).mean(dim=1)        # [B, G]

        # ===============================
        # Loss (same as before)
        # ===============================
        mse_loss = criterion(preds_mean, targets)
        pred_var = preds_mean.var(dim=0).mean()
        targ_var = targets.var(dim=0).mean()
        var_loss = (pred_var - targ_var).abs()
        p_loss = pearson_loss(preds_mean, targets)

        if epoch < 25:
            mse_w, var_w, pearson_w = 0.1, 0.02, 0.25
        else:
            mse_w, var_w, pearson_w = 0.1, 0.02, 0.25

        loss = mse_w*mse_loss + var_w*var_loss + pearson_w*p_loss
        t_forward += time.perf_counter() - t1

        t2 = time.perf_counter()
        loss.backward()
        optimizer.step()
        t_backward += time.perf_counter() - t2

        train_loss += loss.item() * B

    train_loss /= len(train_loader.dataset)

    # =====================================
    # VALIDATION — ALSO BATCHED PER-TILE
    # =====================================
    model.eval()
    val_loss = 0.0
    preds_all, targets_all = [], []

    with torch.no_grad():
        for imgs_batch, targets, clinical_batch in val_loader:

            B, T, C, H, W = imgs_batch.shape

            imgs_batch = imgs_batch.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            clinical_batch = clinical_batch.to(device, non_blocking=True)

            # batched tile fusion
            imgs_flat = imgs_batch.view(B*T, C, H, W)
            clinical_expanded = clinical_batch.repeat_interleave(T, dim=0)

            preds_flat = model(imgs_flat, clinical_expanded)   # [B*T, G]
            preds_mean = preds_flat.view(B, T, -1).mean(dim=1)

            mse = criterion(preds_mean, targets)
            val_loss += mse.item()

            per_gene_sq_sum += ((preds_mean - targets) ** 2).sum(dim=0)
            per_gene_count  += preds_mean.size(0)

            preds_all.append(preds_mean.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    val_loss /= len(val_loader)

    scheduler.step(val_loss)
    preds_all = np.vstack(preds_all)
    targets_all = np.vstack(targets_all)

    # quick diagnostics:
    print(preds_all.shape, targets_all.shape)
    print("Preds mean/std:", preds_all.mean(), preds_all.std())
    print("Targets mean/std:", targets_all.mean(), targets_all.std())

    # Per-gene stats
    per_gene_mse = (per_gene_sq_sum / per_gene_count).detach().cpu().numpy()
    pearsons = []
    for i in range(len(gene_cols)):
        try:
            r, _ = pearsonr(preds_all[:, i], targets_all[:, i])
        except Exception:
            r = np.nan
        pearsons.append(r)

    results_df = pd.DataFrame({"gene": gene_cols, "val_mse": per_gene_mse, "pearson_r": pearsons}).sort_values("val_mse")

    mean_pearson = np.nanmean(pearsons)
    t_epoch = time.perf_counter() - t_epoch_start
    print(f"Epoch {epoch+1} times: total {t_epoch:.1f}s, data {t_data:.1f}s, forward {t_forward:.1f}s, backward {t_backward:.1f}s")
    print("Epoch [{}/{}] Train Loss: {:.4f} | Val Loss: {:.4f} | Mean Pearson: {:.4f}".format(epoch+1, num_epochs, train_loss, val_loss, mean_pearson))
    print(results_df.to_string(index=False))
    print("-" * 60)

    # === Visualization: predicted vs ground truth for genes ===

    genes_to_plot = ["STK11_fpkm_uq", "NFE2L2_fpkm_uq", "BRAF_fpkm_uq"]  # choose any subset of gene_cols

    os.makedirs("plots", exist_ok=True)
    n_patients = min(20, len(preds_all))
    idxs = np.random.choice(len(preds_all), n_patients, replace=False)
    x = np.arange(n_patients)

    for target_gene in genes_to_plot:
        if target_gene not in gene_cols:
            print("Gene {} not found in gene_cols; skipping.".format(target_gene))
            continue

        gene_idx = gene_cols.index(target_gene)
        preds_gene = preds_all[idxs, gene_idx]
        targets_gene = targets_all[idxs, gene_idx]
        gene_mean = targets_all[:, gene_idx].mean()  # population mean (≈0 for z-scored data)

        plt.figure(figsize=(10, 5))
        plt.bar(x - 0.2, targets_gene, width=0.4, label="Ground Truth", color="skyblue")
        plt.bar(x + 0.2, preds_gene, width=0.4, label="Predicted", color="orange")
        plt.axhline(y=gene_mean, color="red", linestyle="--", linewidth=1.5, label=f"Population Mean ({gene_mean:.2f})")
        plt.xticks(x, [f"P{i}" for i in range(n_patients)], rotation=45)
        plt.ylabel("Z-score FPKM-UQ")
        plt.title(f"{target_gene}: Prediction vs Ground Truth (Epoch {epoch+1})")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/{}_epoch{}.png".format(target_gene, epoch+1), dpi=150)
        plt.close()

    # Early stopping & save best by mean Pearson
    if mean_pearson > best_mean_pearson:
        best_mean_pearson = mean_pearson
        torch.save(model.state_dict(), "best_resnet50_by_mean_pearson.pt")
        print("Saved new best model (mean Pearson {:.4f})".format(best_mean_pearson))
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

torch.save(model.state_dict(), "resnet50_gene_expression_multitile_final.pt")
print("Final model saved as resnet50_gene_expression_multitile_final.pt")