# Import required libraries
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil
import platform
from tqdm import tqdm
from PIL import Image

from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import torch.multiprocessing as mp

# Configuration
gene_csv = "/home/zcemlhu/Scratch/clinical_expression_filtered_TMBTISAPM_LUADandLUSC_complete.csv" # Enter CSV path here
base = "/home/zcemlhu/Scratch/LUAD_Tiles" # Enter path to parent folder containing all tiles here

# Model parameters
img_size = 512
num_epochs = 1000
lr = 1e-5
tiles_per_slide = 32
max_cache_tiles = 0
train_ratio = 0.8
val_ratio   = 0.1
test_ratio  = 0.1
early_stop_patience = 10

def main():
    # Device selection
    if torch.cuda.is_available():
        if "LOCAL_RANK" in os.environ:
            torch.distributed.init_process_group("nccl")
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            print(f"Using CUDA GPU {local_rank}")
        else:
            local_rank = None
            device = torch.device("cuda")
            print("Using single-GPU CUDA")

    torch.backends.cudnn.benchmark = True # enable cudnn autotuner (for fixed-size inputs)

    # Function to trouleshoot memory leaks
    def print_memory_usage(tag=""):
        process = psutil.Process(os.getpid())
        used_gb = process.memory_info().rss / (1024**3)
        vm = psutil.virtual_memory()
        print(f"\n Memory usage: {tag}")
        print(f"Process RAM used: {used_gb:.2f} GB") # CPU RAM
        print(f"RAM used: {vm.used/1e9:.2f} GB / {vm.total/1e9:.2f} GB")

        # SHM (Linux only)
        if platform.system() == "Linux":
            try:
                shm_stat = os.statvfs("/dev/shm")
                shm_total = shm_stat.f_frsize * shm_stat.f_blocks
                shm_free = shm_stat.f_frsize * shm_stat.f_bfree
                print(f"SHM used: {(shm_total - shm_free)/1e9:.2f} GB / {shm_total/1e9:.2f} GB")
            except:
                print("SHM: not accessible")

        # GPU(s)
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_allocated(i) / 1e9
            cached = torch.cuda.memory_reserved(i) / 1e9
            print(f"GPU {i}: allocated={mem:.2f} GB  reserved={cached:.2f} GB")

        print("\n")

    # STEP 1: Load and clean gene expression data
    df_expr = pd.read_csv(gene_csv)
    df_expr.columns = [c.strip() for c in df_expr.columns]

    id_col = None
    for c in df_expr.columns:
        if any(k in c.lower() for k in ["sample", "submitter", "case", "id"]):
            id_col = c
            break
    if id_col is None:
        raise ValueError("No ID column found in gene expression CSV")

    df_expr["submitter_id"] = df_expr[id_col].str.upper().str.strip()
    gene_cols = [c for c in df_expr.columns if c.endswith("_fpkm_uq") or c == "TMB"]
    if not gene_cols:
        raise ValueError("No gene expression columns ending with '_fpkm_uq' found")

    gene_symbol_to_col = {g.replace("_fpkm_uq", ""): g for g in gene_cols}
    APM_GENES = ["PSMB5", "PSMB6", "PSMB7", "PSMB8", "PSMB9", "PSMB10", "TAP1", "TAP2", "ERAP1", "ERAP2", "TAPBP", "CANX", "CALR", "PDIA3", "B2M", "HLA-A", "HLA-B", "HLA-C"]
    TIS_GENES = ["PSMB10", "HLA-DQA1", "HLA-DRB1", "CMKLR1", "HLA-E", "NKG7", "CD8A", "CCL5", "CXCL9", "CD27", "CXCR6", "IDO1", "STAT1", "CD274", "CD276", "LAG3", "PDCD1LG2", "TIGIT"]

    # Normalize targets (z-score) to stabilise training
    df_expr[gene_cols] = df_expr[gene_cols].apply(pd.to_numeric, errors='coerce')
    df_expr = df_expr.dropna(subset=gene_cols, how='any')
    df_expr[gene_cols] = np.log1p(df_expr[gene_cols])
    for g in gene_cols: # Keep a copy for later biological evaluation (APM / TIS)
        df_expr[g + "_raw"] = df_expr[g]
    y_means = df_expr[gene_cols].mean()
    y_stds = df_expr[gene_cols].std()
    df_expr[gene_cols] = (df_expr[gene_cols] - y_means) / y_stds
    print("Extracted, Z scored, and logged expression data, {} genes".format(len(gene_cols)))

    # Add clinical covariates
    if 'age_years' not in df_expr.columns or 'demographic.gender' not in df_expr.columns: # Ensure age and gender exist
        raise ValueError("Expected 'age_years' and 'demographic.gender' in clinical CSV")

    # Normalize and encode
    df_expr['age_years'] = pd.to_numeric(df_expr['age_years'], errors='coerce').fillna(df_expr['age_years'].median())
    df_expr['age_years_z'] = (df_expr['age_years'] - df_expr['age_years'].mean()) / df_expr['age_years'].std()
    df_expr['gender_encoded'] = df_expr['demographic.gender'].str.lower().map({'male': 0, 'female': 1}).fillna(0).astype(np.float32) # Encode gender as numeric: male=0, female=1 (or vice versa)
    clinical_cols = ['age_years_z', 'gender_encoded'] # List of covariate columns

    # STEP 2: Find tile files and match
    tile_records = []

    for uuid in os.listdir(base):
        uuid_dir = os.path.join(base, uuid)
        if not os.path.isdir(uuid_dir):
            continue

        for slide in os.listdir(uuid_dir):
            slide_upper = slide.upper().strip()

            # patient barcode is first 12 chars: TCGA-XX-YYYY
            patient_id = slide_upper[:12]

            tile_records.append({
                "submitter_id": patient_id,
                "tiles_root": os.path.join(uuid_dir, slide)
            })

    tiles_df = pd.DataFrame(tile_records)
    df_matched = df_expr.merge(tiles_df, on="submitter_id", how="inner")
    df_matched[clinical_cols] = df_matched[clinical_cols].astype(np.float32)
    df_matched = df_matched.dropna(subset=gene_cols).reset_index(drop=True) # Drop any rows that still have bad gene values
    bad = df_matched[gene_cols].dtypes[df_matched[gene_cols].dtypes == "object"]
    if len(bad) > 0:
        raise RuntimeError(f"Non-numeric gene columns remain: {bad}")
    
    print("Gene dtypes OK:", df_matched[gene_cols].dtypes.unique())
    print("Expression samples:", len(df_expr))
    print("Slides with tiles:", len(tiles_df))
    print("Matched rows:", len(df_matched))
    print(df_matched[["submitter_id", "tiles_root"]].head())

    # STEP 3: Create important dataset functions
    class WSIExpressionDataset(Dataset):
        def __init__(self, df, genes, genes_raw=None, clinical_cols=None, transform=None, tiles_per_slide=32, cache=False, max_cache_tiles=0, deterministic=False):
            self.df = df.reset_index(drop=True)
            self.genes = genes
            self.genes_raw = genes_raw
            self.clinical_cols = clinical_cols or []
            self.transform = transform
            self.tiles_per_slide = tiles_per_slide
            self.cache = cache
            self.max_cache_tiles = max_cache_tiles
            self.tile_cache = {}  # {tiles_root: [PIL.Image, ...]}
            self.deterministic = deterministic

        def __len__(self):
            return len(self.df)

        def _load_tiles_from_dir(self, tiles_root): # Function to load tiles from directory
            if self.deterministic:
                random.seed(hash(tiles_root) % (2**32))
            pngs = [
                os.path.join(tiles_root, f)
                for f in os.listdir(tiles_root)
                if f.lower().endswith(".png")
            ]

            if len(pngs) == 0:
                return [Image.new("RGB", (512, 512), (255, 255, 255))]

            if len(pngs) >= self.tiles_per_slide:
                pngs = random.sample(pngs, self.tiles_per_slide)
            else:
                pngs = pngs + random.choices(pngs, k=self.tiles_per_slide - len(pngs))

            tiles = []
            for p in pngs:
                with Image.open(p) as im:
                    tiles.append(im.convert("RGB"))

            return tiles

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            tiles_root = row["tiles_root"]

            tiles = self._load_tiles_from_dir(tiles_root)

            imgs = torch.stack([
                self.transform(tile) if self.transform else tile
                for tile in tiles
            ])

            expr = torch.from_numpy(row[self.genes].to_numpy(dtype=np.float32))

            if self.genes_raw is not None:
                expr_raw = torch.from_numpy(row[self.genes_raw].to_numpy(dtype=np.float32))
            else:
                expr_raw = torch.empty(0)

            clinical = torch.from_numpy(row[self.clinical_cols].to_numpy(dtype=np.float32))

            return imgs, expr, expr_raw, clinical
        
    def collate_list(batch):
        imgs_list = [item[0] for item in batch] # list of tensors (T, C, H, W)
        targets = torch.stack([item[1] for item in batch])
        targets_raw = torch.stack([item[2] for item in batch])
        clinical = torch.stack([item[3] for item in batch])
        return imgs_list, targets, targets_raw, clinical

    # STEP 4: Transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = WSIExpressionDataset(
        df_matched,
        genes=gene_cols, # z scored
        genes_raw=[g + "_raw" for g in gene_cols], # raw/log copy
        clinical_cols=clinical_cols,
        transform=transform,
        tiles_per_slide=tiles_per_slide,
        max_cache_tiles=max_cache_tiles,
        cache=False
    )

    # Split for KFold
    N = len(df_matched)
    indices = np.random.RandomState(42).permutation(N)
    test_size = int(0.2 * N)
    test_idx = indices[:test_size]
    trainval_idx = indices[test_size:]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results_mean_pearson = []
    fold_results_apm = []
    fold_results_tis = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(trainval_idx)))):
        print(f"\n FOLD {fold}")
        train_idx = trainval_idx[train_idx]
        val_idx   = trainval_idx[val_idx]
        print(f"Fold {fold}: "
            f"train={len(train_idx)}, "
            f"val={len(val_idx)}, "
            f"test={len(test_idx)}")
        df_train = df_matched.iloc[train_idx].reset_index(drop=True)
        df_val   = df_matched.iloc[val_idx].reset_index(drop=True)
        df_test  = df_matched.iloc[test_idx].reset_index(drop=True)
        print(f"Training: {len(df_train)}, Validation: {len(df_val)}, Test: {len(df_test)}")

        train_set = WSIExpressionDataset(
            df_train,
            genes=gene_cols,
            genes_raw=[g + "_raw" for g in gene_cols],
            clinical_cols=clinical_cols,
            transform=transform,
            tiles_per_slide=tiles_per_slide,
            max_cache_tiles=max_cache_tiles,
            cache=False,
            deterministic=False,   # stochastic tiles
        )

        val_set = WSIExpressionDataset(
            df_val,
            genes=gene_cols,
            genes_raw=[g + "_raw" for g in gene_cols],
            clinical_cols=clinical_cols,
            transform=transform,
            tiles_per_slide=tiles_per_slide,
            max_cache_tiles=max_cache_tiles,
            cache=False,
            deterministic=True,    # frozen tiles
        )

        test_set = WSIExpressionDataset(
            df_test,
            genes=gene_cols,
            genes_raw=[g + "_raw" for g in gene_cols],
            clinical_cols=clinical_cols,
            transform=transform,
            tiles_per_slide=tiles_per_slide,
            max_cache_tiles=max_cache_tiles,
            cache=False,
            deterministic=True,    # frozen tiles
        )

        is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()

        if is_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(
            train_set,
            batch_size=4,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            drop_last=True,
            collate_fn=collate_list,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=1
        )

        val_loader = DataLoader(
            val_set,
            batch_size=4,
            sampler=val_sampler,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_list,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=1
        )

        test_loader = DataLoader(
            test_set,
            batch_size=4,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_list,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=1
        )

        if len(train_loader) == 0:
            raise RuntimeError(
                f"Train loader is empty! "
                f"Train set size={len(train_set)}, batch_size=4, world_size={torch.distributed.get_world_size() if is_ddp else 1}"
            )
        
        # STEP 5: Model (ResNet50 + clinical features)

        class ImageClinicalModel(nn.Module):
            def __init__(self, base_model, in_features, n_genes, n_clinical):
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
                if clinical.nelement() == 0:
                    clinical = torch.zeros((feats.size(0), 0), device=feats.device)
                feats = torch.cat([feats, clinical], dim=1)
                out = self.fc(feats)
                return out
            
        weights = ResNet50_Weights.IMAGENET1K_V1 # Preweighted using ImageNet1K for faster learning
        weights_url = weights.url
        resnet = models.resnet50(weights=None)
        state_dict = torch.hub.load_state_dict_from_url(weights_url, map_location="cpu", progress=True)
        resnet.load_state_dict(state_dict)
        in_features = resnet.fc.in_features
        resnet.fc = torch.nn.Identity()
        model = ImageClinicalModel(resnet, in_features, len(gene_cols), len(clinical_cols))
        model.to(device)

        if device.type == "cuda":
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            
        for m in model.modules(): # Freeze batchnorm layers (can also unfreeze in future)
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

        # Loss / optimizer / scheduler
        criterion = nn.SmoothL1Loss(beta=0.1, reduction='mean')  # more stable than MSE
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True) # adaptive scheduler: reduce LR when val loss plateaus

        def pearson_loss(preds, targets, eps=1e-8): # pearson loss
            preds_c = preds - preds.mean(dim=0, keepdim=True) # preds, targets: [B, G]
            targ_c = targets - targets.mean(dim=0, keepdim=True)
            num = (preds_c * targ_c).sum(dim=0)
            den = torch.sqrt((preds_c**2).sum(dim=0) * (targ_c**2).sum(dim=0) + eps)
            r = num / den
            return (1.0 - r).mean() # return 1 - mean(r) (we want to minimize this)

        # STEP 6: Training loop with batched multi-tile inference
        best_mean_pearson = -999.0 # set low so first score is always an improvement
        patience_counter = 0
        use_amp = torch.cuda.is_available()

        if use_amp:
            scaler = torch.amp.GradScaler()

        for epoch in range(num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            t_epoch_start = time.perf_counter()
            model.train()
            train_loss = 0.0
            t_data = 0.0
            t_forward = 0.0
            t_backward = 0.0

            for imgs_list, targets, targets_raw, clinical_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                t0 = time.perf_counter()
                #print_memory_usage(f"Epoch {epoch+1} in batch")
                imgs_batch = torch.stack(imgs_list, dim=0) # [B, T, C, H, W]
                B, T, C, H, W = imgs_batch.shape
                imgs_batch = imgs_batch.to(device, non_blocking=True) # Move to device
                targets = targets.to(device, non_blocking=True)
                clinical_batch = clinical_batch.to(device, non_blocking=True)
                t_data += time.perf_counter() - t0
                optimizer.zero_grad()
                
                # Multi-tile batched interference
                t1 = time.perf_counter()
                imgs_flat = imgs_batch.view(B*T, C, H, W)
                clinical_expanded = clinical_batch.repeat_interleave(T, dim=0)

                if scaler is not None:
                    with torch.amp.autocast(device_type="cuda"):
                        preds_flat = model(imgs_flat, clinical_expanded) # [B*T, G]
                        preds_mean = preds_flat.view(B, T, -1).mean(dim=1) # [B, G]

                        # Loss
                        mse_loss = criterion(preds_mean, targets)
                        pred_var = preds_mean.var(dim=0).mean()
                        targ_var = targets.var(dim=0).mean()
                        var_loss = (pred_var - targ_var).abs()
                        p_loss = pearson_loss(preds_mean, targets)

                        if epoch < 25:
                            mse_w, var_w, pearson_w = 1, 0, 0
                        else:
                            mse_w, var_w, pearson_w = 0.4, 0.2, 0.4

                        loss = mse_w*mse_loss + var_w*var_loss + pearson_w*p_loss
                        t_forward += time.perf_counter() - t1
                        t2 = time.perf_counter()
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        t_backward += time.perf_counter() - t2

                train_loss += loss.item() * B

            train_loss /= len(train_loader.dataset)

            # Validation - also batched per-tile
            per_gene_sq_sum = torch.zeros(len(gene_cols), device=device)
            per_gene_count  = 0
            model.eval()
            val_loss = 0.0
            preds_all, targets_all, targets_all_raw = [], [], []

            with torch.no_grad():
                for imgs_list, targets, targets_raw, clinical_batch in val_loader:
                    imgs_batch = torch.stack(imgs_list, dim=0)   # [B, T, C, H, W]
                    B, T, C, H, W = imgs_batch.shape
                    imgs_batch = imgs_batch.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    clinical_batch = clinical_batch.to(device, non_blocking=True)
                    imgs_flat = imgs_batch.view(B*T, C, H, W) # batched tile fusion
                    clinical_expanded = clinical_batch.repeat_interleave(T, dim=0)
                    preds_flat = model(imgs_flat, clinical_expanded)   # [B*T, G]
                    preds_mean = preds_flat.view(B, T, -1).mean(dim=1)
                    mse = criterion(preds_mean, targets)
                    val_loss += mse.item()
                    per_gene_sq_sum += ((preds_mean - targets) ** 2).sum(dim=0)
                    per_gene_count  += preds_mean.size(0)
                    preds_all.append(preds_mean.cpu().numpy())
                    targets_all_raw.append(targets_raw.cpu().numpy())
                    targets_all.append(targets.cpu().numpy())

            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            preds_all = np.vstack(preds_all)
            targets_all = np.vstack(targets_all)
            targets_all_raw = np.vstack(targets_all_raw)

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

            # Post-hoc TIS/APM analysis
            def compute_signature(preds_z, targets_raw, gene_cols, gene_set, eps=1e-8):
                idxs = [gene_cols.index(gene_symbol_to_col[g]) for g in gene_set if g in gene_symbol_to_col]
                if len(idxs) < 3:
                    return np.nan, np.nan

                # signature means
                preds_sig = preds_z[:, idxs].mean(axis=1)
                targets_sig = targets_raw[:, idxs].mean(axis=1)

                # z-score BOTH vectors (critical)
                preds_sig = (preds_sig - preds_sig.mean()) / (preds_sig.std() + eps)
                targets_sig = (targets_sig - targets_sig.mean()) / (targets_sig.std() + eps)

                # guard against zero variance
                if np.std(preds_sig) < 1e-6 or np.std(targets_sig) < 1e-6:
                    return np.nan, np.nan

                r, p = pearsonr(preds_sig, targets_sig)
                return r, p

            apm_r, apm_p = compute_signature(preds_all, targets_all_raw, gene_cols, APM_GENES)
            tis_r, tis_p = compute_signature(preds_all, targets_all_raw, gene_cols, TIS_GENES)
            print(f"APM score: Pearson r = {apm_r:.3f}, p = {apm_p:.2e}")
            print(f"TIS score: Pearson r = {tis_r:.3f}, p = {tis_p:.2e}")
            results_df = pd.DataFrame({"gene": gene_cols, "val_mse": per_gene_mse, "pearson_r": pearsons}).sort_values("val_mse")
            mean_pearson = np.nanmean(pearsons)
            t_epoch = time.perf_counter() - t_epoch_start
            print(f"Epoch {epoch+1} times: total {t_epoch:.1f}s, data {t_data:.1f}s, forward {t_forward:.1f}s, backward {t_backward:.1f}s")
            print("Epoch [{}/{}] Train Loss: {:.4f} | Val Loss: {:.4f} | Mean Pearson: {:.4f}".format(epoch+1, num_epochs, train_loss, val_loss, mean_pearson))
            print(results_df.to_string(index=False))
            print("-" * 60)

            # Visualisation: predicted vs ground truth for genes
            genes_to_plot = ["STAT1_fpkm_uq", "ERAP1_fpkm_uq", "B2M_fpkm_uq"]  # choose any subset of gene_cols
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

            def compute_signature_scores(preds_z, targets_raw, gene_cols, gene_set):
                idxs = [gene_cols.index(gene_symbol_to_col[g]) for g in gene_set if g in gene_symbol_to_col]
                if len(idxs) < 3:
                    raise ValueError("Too few genes found for signature")

                preds_sig = preds_z[:, idxs].mean(axis=1)
                targets_sig = targets_raw[:, idxs].mean(axis=1)

                return preds_sig, targets_sig

            # Signature scatter plots
            os.makedirs("plots/signatures", exist_ok=True)

            for name, geneset in [("TIS", TIS_GENES), ("APM", APM_GENES)]: 
                preds_sig, targets_sig = compute_signature_scores(preds_all, targets_all_raw, gene_cols, geneset)
                r, p = pearsonr((preds_sig - preds_sig.mean()) / preds_sig.std(),(targets_sig - targets_sig.mean()) / targets_sig.std())

                plt.figure(figsize=(6, 6))
                plt.scatter(targets_sig, preds_sig, alpha=0.6, edgecolor="k")
                plt.xlabel(f"True {name} score (log expression mean)")
                plt.ylabel(f"Predicted {name} score")
                plt.title(f"{name}: Predicted vs True\nPearson r={r:.2f}, p={p:.1e}")
                lims = [min(targets_sig.min(), preds_sig.min()), max(targets_sig.max(), preds_sig.max())]
                plt.plot(lims, lims, "r--", linewidth=1)
                plt.tight_layout()
                plt.savefig(f"plots/signatures/{name}_scatter_epoch{epoch+1}.png", dpi=150)
                plt.close()

            # ROC: Top quartile signatures
            os.makedirs("plots/roc", exist_ok=True)

            for name, geneset in [("TIS", TIS_GENES), ("APM", APM_GENES)]: 
                preds_sig, targets_sig = compute_signature_scores(preds_all, targets_all_raw, gene_cols, geneset)

                # define top quartile based on TRUE signature
                threshold = np.percentile(targets_sig, 75)
                y_true = (targets_sig >= threshold).astype(int)

                # use predicted signature as score
                fpr, tpr, _ = roc_curve(y_true, preds_sig)
                roc_auc = auc(fpr, tpr)

                plt.figure(figsize=(6, 6))
                plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.2f}")
                plt.plot([0, 1], [0, 1], "k--", linewidth=1)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"{name} Top-Quartile Classification ROC")
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(f"plots/roc/{name}_ROC_epoch{epoch+1}.png", dpi=150)
                plt.close()

        to_save = model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()
        torch.save(to_save, "best_resnet50_by_mean_pearson.pt")

        print("\n FINAL TEST EVALUATION (BEST PEARSON MODEL)")
        ckpt = torch.load("best_resnet50_by_mean_pearson.pt", map_location=device) # Load best checkpoint

        if isinstance(model, nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(ckpt)
        else:
            model.load_state_dict(ckpt)

        model.eval()
        preds_all, targets_all, targets_all_raw = [], [], []

        with torch.no_grad():
            for imgs_list, targets, targets_raw, clinical_batch in test_loader:
                imgs_batch = torch.stack(imgs_list, dim=0)
                B, T, C, H, W = imgs_batch.shape
                imgs_batch = imgs_batch.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                clinical_batch = clinical_batch.to(device, non_blocking=True)
                imgs_flat = imgs_batch.view(B*T, C, H, W)
                clinical_expanded = clinical_batch.repeat_interleave(T, dim=0)
                preds_flat = model(imgs_flat, clinical_expanded)
                preds_mean = preds_flat.view(B, T, -1).mean(dim=1)
                preds_all.append(preds_mean.cpu().numpy())
                targets_all.append(targets.cpu().numpy())
                targets_all_raw.append(targets_raw.cpu().numpy())

        preds_all = np.vstack(preds_all)
        targets_all = np.vstack(targets_all)
        targets_all_raw = np.vstack(targets_all_raw)

        pearsons = []
        for i in range(len(gene_cols)):
            try:
                r, _ = pearsonr(preds_all[:, i], targets_all[:, i])
            except:
                r = np.nan
            pearsons.append(r)

        mean_pearson = np.nanmean(pearsons)
        print(f"TEST Mean Pearson: {mean_pearson:.4f}")

        apm_r, apm_p = compute_signature(preds_all, targets_all_raw, gene_cols, APM_GENES)
        tis_r, tis_p = compute_signature(preds_all, targets_all_raw, gene_cols, TIS_GENES)
        print(f"[TEST] APM: Pearson r = {apm_r:.3f}, p = {apm_p:.2e}")
        print(f"[TEST] TIS: Pearson r = {tis_r:.3f}, p = {tis_p:.2e}")

        np.savez(
            f"test_predictions_best_pearson_fold{fold}.npz",
            preds_z=preds_all,
            targets_z=targets_all,
            targets_raw=targets_all_raw,
            gene_cols=gene_cols
        )

        fold_results_mean_pearson.append(mean_pearson)
        fold_results_apm.append(apm_r)
        fold_results_tis.append(tis_r)

    print("\n CROSS-VALIDATION SUMMARY")
    print(f"Mean Pearson across folds: {np.mean(fold_results_mean_pearson):.4f} ± {np.std(fold_results_mean_pearson):.4f}")
    print(f"APM Pearson across folds: {np.mean(fold_results_apm):.3f} ± {np.std(fold_results_apm):.3f}")
    print(f"TIS Pearson across folds: {np.mean(fold_results_tis):.3f} ± {np.std(fold_results_tis):.3f}")

if __name__ == "__main__":
    main()
