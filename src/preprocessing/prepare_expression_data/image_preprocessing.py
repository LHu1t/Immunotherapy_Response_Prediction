#!/usr/bin/env python3

import math
import os
import random
import csv
import traceback
from pathlib import Path
from multiprocessing import Pool
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import openslide
except:
    openslide = None

from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

try:
    import staintools
    STAINTOOLS_AVAILABLE = True
except:
    STAINTOOLS_AVAILABLE = False

# -----------------------------
# CONFIG (unchanged)
# -----------------------------
INPUT_ROOT = r"/Volumes/SeagateBas/Immunotherapy" # "/home/zcemlhu/Scratch"
CANCER = "LUAD" # "LUSC"
OUTPUT_ROOT = r"/Volumes/SeagateBas/Immunotherapy/LUAD_Test_Tiles2" # "/home/zcemlhu/Scratch/LUSC_Tiles"

TILE_SIZE = 512
TILES_PER_WSI = 5000
OCCUPANCY_THRESH = 0.1
TARGET_MPP = 0.5
WORKERS = 4

NORMALIZE = True
STAIN_REF_PATH = r"/Volumes/SeagateBas/Immunotherapy/LUAD_Test/8f7cdbca-32a7-4bd3-be11-6cf7b7b03d13/TCGA-50-6591-01Z-00-DX1.12e1050b-75e9-4059-b945-291995b3e93c.svs"# "/home/zcemlhu/Scratch/LUSC/0a780b0b-fce9-4999-92ba-a8b277578a1d/TCGA-60-2724-01Z-00-DX1.98f9e48f-9c09-4969-aa34-05666616ee9a.svs"

MANIFEST_NAME = 'manifest.csv'
THUMB_MAX_DIM = 512
SAVE_FORMAT = 'png'
PNG_COMPRESSION = 6
WEBP_QUALITY = 80
# -----------------------------

# ---- Global stain reference shared by workers ----
GLOBAL_STAIN_REF = None

def init_worker(stain_ref):
    """
    Runs once per worker process.
    Store the stain reference image globally so it is not pickled per task.
    """
    global GLOBAL_STAIN_REF
    GLOBAL_STAIN_REF = stain_ref

def get_mpp(slide):
    keys = ['openslide.mpp-x', 'aperio.MPP', 'openslide.mpp-y']
    for k in keys:
        if k in slide.properties:
            try:
                return float(slide.properties[k])
            except:
                pass
    return None


def iter_svs_files(root_dir, cancer):
    root = Path(root_dir) / cancer
    for patient in root.iterdir():
        if not patient.is_dir():
            continue
        for svs in patient.glob("*.svs"):
            yield svs


def otsu_mask(arr, max_dim=THUMB_MAX_DIM):
    """Compute fast Otsu mask on a downscaled thumbnail."""
    h, w = arr.shape[:2]
    scale = max_dim / max(h, w)

    if scale < 1:
        thumb = Image.fromarray(arr).resize(
            (int(w * scale), int(h * scale)),
            resample=Image.BILINEAR
        )
        small = np.array(thumb)
    else:
        small = arr

    gray = rgb2gray(small)
    thresh = threshold_otsu((gray * 255).astype(np.uint8)) / 255.0
    mask_small = gray < thresh

    # upscale mask back
    mask = np.array(
        Image.fromarray((mask_small * 255).astype(np.uint8)).resize(
            (w, h),
            resample=Image.NEAREST
        )
    ) > 0

    return mask

def process_wsi_no_pyvips(args):
    (
        wsi_path,
        output_root,
        tiles_per_wsi,
        tile_size,
        occupancy_thresh,
        target_mpp,
        normalize,
    ) = args

    global GLOBAL_STAIN_REF

    wsi_path = Path(wsi_path)
    wsi_id = wsi_path.stem
    out_dir = Path(output_root) / wsi_path.parent.name / wsi_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        slide = openslide.OpenSlide(str(wsi_path))
        slide_mpp = get_mpp(slide)

        if slide_mpp is None:
            print(f"[WARN] {wsi_id}: No MPP → no rescale.")
            scale = 1.0
        else:
            scale = slide_mpp / target_mpp

        # choose appropriate level for reading whole slide
        if scale >= 1:
            best = slide.get_best_level_for_downsample(scale)
            down = slide.level_downsamples[best]
            w, h = slide.level_dimensions[best]
            region = slide.read_region((0, 0), best, (w, h)).convert("RGB")
            img = np.array(region)
            final_w = int(img.shape[1] * (down / scale))
            final_h = int(img.shape[0] * (down / scale))
            img_rescaled = np.array(
                Image.fromarray(img).resize(
                    (final_w, final_h),
                    resample=Image.BILINEAR
                )
            )
        else:
            w, h = slide.dimensions
            region = slide.read_region((0, 0), 0, (w, h)).convert("RGB")
            img = np.array(region)
            final_w = int(w / scale)
            final_h = int(h / scale)
            img_rescaled = np.array(
                Image.fromarray(img).resize(
                    (final_w, final_h),
                    resample=Image.BILINEAR
                )
            )

        # Otsu tissue mask
        mask = otsu_mask(img_rescaled, max_dim=THUMB_MAX_DIM)
        H, W = img_rescaled.shape[:2]

        # tile coords
        coords = []
        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                sub = mask[y:y+tile_size, x:x+tile_size]
                if sub.size == 0:
                    continue
                if sub.mean() >= occupancy_thresh:
                    coords.append((x, y))

        print(f"[INFO] {wsi_id}: {len(coords)} tissue tiles found.")

        # sample/pad
        if len(coords) >= tiles_per_wsi:
            selected = random.sample(coords, tiles_per_wsi)
        else:
            need = tiles_per_wsi - len(coords)
            selected = coords + [None] * need
            print(f"[INFO] {wsi_id}: padded {need} tiles.")

        # stain normalizer (created once per WSI)
        normalizer = None
        if normalize and STAINTOOLS_AVAILABLE and GLOBAL_STAIN_REF is not None:
            try:
                normalizer = staintools.StainNormalizer(method="vahadane")
                normalizer.fit(GLOBAL_STAIN_REF)
            except:
                normalizer = None

        rows = []

        # save tiles
        for i, coord in enumerate(selected):
            if coord is None:
                tile = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
            else:
                x, y = coord
                tile_arr = img_rescaled[y:y+tile_size, x:x+tile_size]
                tile = Image.fromarray(tile_arr)
                if tile.size != (tile_size, tile_size):
                    pad = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
                    pad.paste(tile, (0, 0))
                    tile = pad

            # stain normalization
            if normalize and normalizer is not None:
                try:
                    arr = normalizer.transform(np.array(tile))
                    tile = Image.fromarray(arr)
                except:
                    pass

            fname = out_dir / f"{wsi_id}_tile_{i:05d}.{SAVE_FORMAT}"

            if SAVE_FORMAT == "png":
                tile.save(fname, compress_level=PNG_COMPRESSION)
            else:
                tile.save(fname, quality=WEBP_QUALITY, format="WEBP")

            rows.append((str(wsi_path), str(fname), i))

        return (str(wsi_path), True, rows)

    except Exception:
        return (str(wsi_path), False, traceback.format_exc())
    
def chunk_list(lst, chunk_size):
    """Yield successive chunk_size-sized chunks."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    # 1. Get all slides
    svs_files = list(iter_svs_files(INPUT_ROOT, CANCER))
    print(f"Found {len(svs_files)} SVS files under {INPUT_ROOT}/{CANCER}")

    # 2. Compute batch using slicing (avoids materializing all batches)
    start = args.batch_index * args.batch_size
    end = start + args.batch_size
    batch = svs_files[start:end]

    if not batch:
        print(f"Batch {args.batch_index} is empty.")
        return

    print(f"Processing batch {args.batch_index} "
          f"({len(batch)} slides)")

    # 3. Prepare stain reference once
    stain_ref = None
    if NORMALIZE and STAINTOOLS_AVAILABLE and STAIN_REF_PATH is not None:
        try:
            stain_ref = np.array(Image.open(STAIN_REF_PATH).convert("RGB"))
        except Exception:
            stain_ref = None

    # 4. Build tasks for the batch (smallest possible payload)
    tasks = []
    for svs in batch:
        tasks.append((
            str(svs),
            OUTPUT_ROOT,
            TILES_PER_WSI,
            TILE_SIZE,
            OCCUPANCY_THRESH,
            TARGET_MPP,
            NORMALIZE,
        ))

    # 5. Prepare manifest
    manifest_path = Path(OUTPUT_ROOT) / f"manifest_batch_{args.batch_index}.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, 'w', newline='') as mf:
        csv.writer(mf).writerow(['wsi_path', 'tile_path', 'tile_index'])

    # 6. multiprocessing (with global stain reference)
    if WORKERS > 1 and len(tasks) > 0:
        from multiprocessing import Pool
        with Pool(
            processes=WORKERS,
            initializer=init_worker,
            initargs=(stain_ref,)
        ) as pool:
            for res in tqdm(pool.imap_unordered(process_wsi_no_pyvips, tasks),
                            total=len(tasks)):
                wsi, ok, payload = res
                if ok:
                    with open(manifest_path, 'a', newline='') as mf:
                        writer = csv.writer(mf)
                        for row in payload:
                            writer.writerow(row)
                else:
                    print(f"WSI failed: {wsi} -> {payload}")
    else:
        # single-thread fallback
        with open(manifest_path, 'a', newline='') as mf:
            writer = csv.writer(mf)
            for t in tqdm(tasks):
                wsi, ok, payload = process_wsi_no_pyvips(t)
                if ok:
                    for row in payload:
                        writer.writerow(row)
                else:
                    print(f"WSI failed: {wsi} -> {payload}")

    print("Done. Manifest at:", manifest_path)

if __name__ == "__main__":
    main()
