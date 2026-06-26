#!/usr/bin/env python3
"""
WSI preprocessing pipeline (NO full-resolution WSI loads).

Key properties:
- Never loads the entire WSI into RAM
- Uses OpenSlide pyramids correctly
- Otsu tissue mask computed from thumbnail only
- Tiles read one-at-a-time from OpenSlide
- Optional stain normalisation
- Supports batching (--batch-index / --batch-size)
"""

import os
import math
import csv
import random
import argparse
import traceback
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import openslide
except Exception:
    openslide = None

from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

try:
    import staintools
    STAINTOOLS_AVAILABLE = True
except Exception:
    STAINTOOLS_AVAILABLE = False

# ================= CONFIG =================
INPUT_ROOT = "/home/zcemlhu/Scratch"
CANCER = "LUAD"
OUTPUT_ROOT = "/home/zcemlhu/Scratch/LUAD_Tiles"

TILE_SIZE = 512
TILES_PER_WSI = 5000
OCCUPANCY_THRESH = 0.1
TARGET_MPP = 0.5
WORKERS = 4

NORMALIZE = True
STAIN_REF_PATH = "/home/zcemlhu/Scratch/LUAD/1ddc6a98-e855-4ca9-8809-f20725fe3120/TCGA-55-6983-01Z-00-DX1.8f940a64-1f1b-4e6e-99ea-418175be2b3f.svs"

THUMB_MAX_DIM = 512
SAVE_FORMAT = "png"
PNG_COMPRESSION = 6
# ==========================================


def get_mpp(slide):
    keys = ["openslide.mpp-x", "aperio.MPP", "openslide.mpp-y"]
    for k in keys:
        if k in slide.properties:
            try:
                return float(slide.properties[k])
            except Exception:
                pass
    return None


def iter_svs_files(root, cancer):
    base = Path(root) / cancer
    for patient in base.iterdir():
        if patient.is_dir():
            yield from patient.glob("*.svs")


def otsu_mask_from_thumbnail(slide):
    thumb = slide.get_thumbnail((THUMB_MAX_DIM, THUMB_MAX_DIM))
    arr = np.array(thumb.convert("RGB"))
    gray = rgb2gray(arr)
    t = threshold_otsu((gray * 255).astype(np.uint8)) / 255.0
    return gray < t, arr.shape[1], arr.shape[0]


def process_wsi(args):
    (
        wsi_path,
        output_root,
        tiles_per_wsi,
        tile_size,
        occupancy_thresh,
        target_mpp,
        normalize,
        stain_ref
    ) = args

    wsi_path = Path(wsi_path)
    wsi_id = wsi_path.stem
    out_dir = Path(output_root) / wsi_path.parent.name / wsi_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        slide = openslide.OpenSlide(str(wsi_path))
        slide_mpp = get_mpp(slide)

        if slide_mpp is None:
            scale = 1.0
        else:
            scale = slide_mpp / target_mpp

        best_level = slide.get_best_level_for_downsample(scale)
        level_down = slide.level_downsamples[best_level]
        level_w, level_h = slide.level_dimensions[best_level]

        # thumbnail mask
        mask_thumb, tw, th = otsu_mask_from_thumbnail(slide)

        # map thumb -> level coords
        sx = level_w / tw
        sy = level_h / th

        coords = []
        for y in range(0, level_h, tile_size):
            for x in range(0, level_w, tile_size):
                tx0 = int(x / sx)
                ty0 = int(y / sy)
                tx1 = int(min((x + tile_size) / sx, tw))
                ty1 = int(min((y + tile_size) / sy, th))
                if tx1 <= tx0 or ty1 <= ty0:
                    continue
                occ = mask_thumb[ty0:ty1, tx0:tx1].mean()
                if occ >= occupancy_thresh:
                    coords.append((x, y))

        if len(coords) >= tiles_per_wsi:
            coords = sorted(coords, key=lambda p: (p[1], p[0]))  # y, x
            selected = coords[:tiles_per_wsi]
        else:
            selected = coords + [None] * (tiles_per_wsi - len(coords))

        normalizer = None
        if normalize and STAINTOOLS_AVAILABLE and stain_ref is not None:
            try:
                normalizer = staintools.StainNormalizer(method="vahadane")
                normalizer.fit(stain_ref)
            except Exception:
                normalizer = None

        rows = []
        for i, coord in enumerate(selected):
            if coord is None:
                tile = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
            else:
                lx, ly = coord
                x0 = int(lx * level_down)
                y0 = int(ly * level_down)

                # Create white canvas
                tile = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))

                # Compute how much is actually inside the slide
                max_w, max_h = slide.level_dimensions[best_level]
                read_w = min(tile_size, max_w - lx)
                read_h = min(tile_size, max_h - ly)

                if read_w > 0 and read_h > 0:
                    region = slide.read_region(
                    (x0, y0),
                    best_level,
                    (int(read_w), int(read_h))
                    ).convert("RGB")

                    tile.paste(region, (0, 0))
            
            if normalize and normalizer is not None:
                try:
                    tile = Image.fromarray(normalizer.transform(np.array(tile)))
                except Exception:
                    pass

            fname = out_dir / f"{wsi_id}_tile_{i:05d}.{SAVE_FORMAT}"
            tile.save(fname, compress_level=PNG_COMPRESSION)
            rows.append((str(wsi_path), str(fname), i))

        return str(wsi_path), True, rows

    except Exception:
        return str(wsi_path), False, traceback.format_exc()


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    slides = list(iter_svs_files(INPUT_ROOT, CANCER))
    batches = list(chunk_list(slides, args.batch_size))

    if args.batch_index >= len(batches):
        print("Batch index out of range")
        return

    batch = batches[args.batch_index]
    print(f"Processing batch {args.batch_index}/{len(batches)-1} ({len(batch)} slides)")

    stain_ref = None
    if NORMALIZE and STAINTOOLS_AVAILABLE and STAIN_REF_PATH:
        try:
            stain_ref = np.array(Image.open(STAIN_REF_PATH).convert("RGB"))
        except Exception:
            pass

    tasks = [
        (str(s), OUTPUT_ROOT, TILES_PER_WSI, TILE_SIZE, OCCUPANCY_THRESH,
         TARGET_MPP, NORMALIZE, stain_ref)
        for s in batch
    ]

    manifest = Path(OUTPUT_ROOT) / f"manifest_batch_{args.batch_index}.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest, "w", newline="") as f:
        csv.writer(f).writerow(["wsi", "tile", "index"])

    with Pool(WORKERS) as pool:
        for wsi, ok, payload in tqdm(pool.imap_unordered(process_wsi, tasks), total=len(tasks)):
            if ok:
                with open(manifest, "a", newline="") as f:
                    csv.writer(f).writerows(payload)
            else:
                print(f"FAILED {wsi}\n{payload}")

    print("Done", manifest)


if __name__ == "__main__":
    main()
