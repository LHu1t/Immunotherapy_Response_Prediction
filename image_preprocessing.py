#!/usr/bin/env python3
"""
Rescale-before-tiling pipeline using pyvips.

Requirements:
  conda install -c conda-forge pyvips openslide-python pillow numpy scikit-image tqdm
  (optional) pip install staintools

Edit CONFIG section below before running.
"""
import math
import os
import random
import csv
import traceback
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

# imports that may be missing on some systems
try:
    import pyvips
except Exception:
    pyvips = None

try:
    import openslide
except Exception:
    openslide = None

from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

# Optional: staintools for Macenko/Vahadane
try:
    import staintools
    STAINTOOLS_AVAILABLE = True
except Exception:
    STAINTOOLS_AVAILABLE = False

# -----------------------------
# CONFIG - Edit these values
# -----------------------------
INPUT_ROOT = r"/Volumes/SeagateBas/Immunotherapy"          # top-level root with cancer folder
CANCER = "LUAD_TEST"                       # folder under INPUT_ROOT
OUTPUT_ROOT = r"/Volumes/SeagateBas/Immunotherapy/LUAD_Tiles_test"  # where tiles + manifest are written
TILE_SIZE = 512
TILES_PER_WSI = 5000                # target number of tiles per WSI (pads with blanks)
OCCUPANCY_THRESH = 0.1                # fraction of tile area considered tissue
TARGET_MPP = 0.5                      # desired microns per pixel (0.5 = 20x)
WORKERS = 4
NORMALIZE = True
STAIN_REF_PATH = r"/Volumes/SeagateBas/Immunotherapy/LUAD_TEST/8f7cdbca-32a7-4bd3-be11-6cf7b7b03d13/TCGA-50-6591-01Z-00-DX1.12e1050b-75e9-4059-b945-291995b3e93c.svs" # optional reference tile for stain normalization
MANIFEST_NAME = 'manifest.csv'
THUMB_MAX_DIM = 1024                  # thumbnail size for Otsu mask
SAVE_FORMAT = 'png'                   # 'png' or 'webp' (png is most portable)
PNG_COMPRESSION = 6                   # Pillow compress level, 0-9 (only for PNG)
WEBP_QUALITY = 80                     # if SAVE_FORMAT == 'webp'
# -----------------------------

def get_mpp(slide):
    """Return microns-per-pixel if available, else None"""
    if slide is None:
        return None
    props = slide.properties
    keys = ['openslide.mpp-x', 'openslide.mpp-y', 'aperio.MPP', 'svs.mpp']
    for k in keys:
        if k in props:
            try:
                return float(props[k])
            except Exception:
                continue
    return None

def make_thumbnail_array(vips_img, max_dim=1024):
    """Create a thumbnail (vips) and return an RGB numpy array."""
    # pyvips: thumbnail_image returns a small image; ensure within max_dim
    thumb = vips_img.thumbnail_image(max_dim)
    h = thumb.height
    w = thumb.width
    bands = thumb.bands  # likely 3
    mem = thumb.write_to_memory()
    arr = np.frombuffer(mem, dtype=np.uint8)
    arr = arr.reshape((h, w, bands))
    # if alpha channel present, drop it
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr

def otsu_mask_from_rgb_arr(arr):
    gray = rgb2gray(arr)
    thresh = threshold_otsu((gray * 255).astype('uint8')) / 255.0
    mask = gray < thresh
    return mask

def iter_svs_files(root_dir, cancer):
    root = Path(root_dir)
    base = root / cancer
    if not base.exists():
        return
    for patient in base.iterdir():
        if not patient.is_dir():
            continue
        for svs in patient.glob('*.svs'):
            yield svs

def process_wsi_pyvips(args):
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
    """
    Rescale slide to target_mpp using pyvips then tile. Returns (wsi_path, success, payload)
    """
    wsi_path = Path(wsi_path)
    wsi_id = wsi_path.stem
    out_dir = Path(output_root) / wsi_path.parent.name / wsi_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # open slide via openslide for metadata (MPP). Use pyvips for image ops.
    slide = None
    if openslide is not None:
        try:
            slide = openslide.OpenSlide(str(wsi_path))
        except Exception:
            slide = None

    try:
        # determine slide mpp
        slide_mpp = get_mpp(slide) if slide is not None else None
        if slide_mpp is None:
            print(f"[WARN] {wsi_id}: slide MPP not found in metadata; will not rescale (extracted at native resolution).")
            desired_scale = 1.0
        else:
            desired_scale = slide_mpp / target_mpp
            # desired_scale may be <1 (we want to shrink); that's fine.
            if desired_scale <= 0:
                desired_scale = 1.0

        if pyvips is None:
            return (str(wsi_path), False, "pyvips not available; install pyvips (see instructions).")

        # open whole-slide via pyvips (streaming). Let vips pick efficient loader.
        vimg = pyvips.Image.new_from_file(str(wsi_path), access='sequential')

        # compute target width/height (floating)
        tgt_w = int(math.ceil(vimg.width * desired_scale))
        tgt_h = int(math.ceil(vimg.height * desired_scale))

        if desired_scale != 1.0:
            # use shrink/resize for high-quality (vips.resize is fast and streaming)
            # vips.resize takes scale factor = output/input
            vimg_small = vimg.resize(desired_scale)
        else:
            vimg_small = vimg

        # now we are operating at the target MPP (or at native if MPP missing)

        # create thumbnail from the rescaled image for mask
        thumb_arr = make_thumbnail_array(vimg_small, max_dim=THUMB_MAX_DIM)
        mask_thumb = otsu_mask_from_rgb_arr(thumb_arr)
        thumb_h, thumb_w = mask_thumb.shape

        # map from rescaled pixels to thumb coords
        scale_x = vimg_small.width / float(thumb_w)
        scale_y = vimg_small.height / float(thumb_h)

        # build candidate tile list (non-overlapping grid)
        tile_coords = []
        for y in range(0, vimg_small.height, tile_size):
            for x in range(0, vimg_small.width, tile_size):
                # map tile region to thumbnail coords
                tx0 = int(x / scale_x)
                ty0 = int(y / scale_y)
                tx1 = int(min((x + tile_size) / scale_x, thumb_w))
                ty1 = int(min((y + tile_size) / scale_y, thumb_h))
                if tx1 <= tx0 or ty1 <= ty0:
                    continue
                sub = mask_thumb[ty0:ty1, tx0:tx1]
                occupancy = float(sub.mean())
                if occupancy >= occupancy_thresh:
                    tile_coords.append((x, y))

        n_candidates = len(tile_coords)
        print(f"[INFO] {wsi_id}: found {n_candidates} non-overlapping tissue tiles at target MPP.")

        # decide selected tiles and padding
        if tiles_per_wsi is None:
            selected = tile_coords
            padded = 0
        else:
            if n_candidates >= tiles_per_wsi:
                selected = random.sample(tile_coords, tiles_per_wsi)
                padded = 0
                print(f"[INFO] {wsi_id}: using {tiles_per_wsi} tiles (no padding).")
            else:
                need = tiles_per_wsi - n_candidates
                selected = tile_coords.copy()
                selected.extend([None] * need)
                padded = need
                print(f"[INFO] {wsi_id}: padding with {need} blank tiles (total target {tiles_per_wsi}).")

        # prepare normalizer if requested
        normalizer = None
        if normalize and STAINTOOLS_AVAILABLE:
            try:
                normalizer = staintools.StainNormalizer(method='vahadane')
                if stain_ref is not None:
                    normalizer.fit(stain_ref)
            except Exception:
                normalizer = None

        saved_rows = []
        # iterate and save tiles; crop returns a vips image; write via pillow for portability
        for i, coord in enumerate(selected):
            if coord is None:
                # blank white tile
                im = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
            else:
                x, y = coord
                # ensure cropping inside bounds: width,height may be smaller at edges
                w_crop = min(tile_size, vimg_small.width - x)
                h_crop = min(tile_size, vimg_small.height - y)
                tile_v = vimg_small.crop(x, y, w_crop, h_crop)
                # if edge tile smaller, expand/pad to tile_size with white
                mem = tile_v.write_to_memory()
                arr = np.frombuffer(mem, dtype=np.uint8)
                arr = arr.reshape((h_crop, w_crop, tile_v.bands))
                if arr.shape[2] > 3:
                    arr = arr[:, :, :3]
                im = Image.fromarray(arr)
                if im.size != (tile_size, tile_size):
                    # pad on right/bottom with white to reach tile_size
                    new = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
                    new.paste(im, (0, 0))
                    im = new

            # optional normalization (PIL->ndarray->staintools->PIL)
            if normalize and STAINTOOLS_AVAILABLE and normalizer is not None:
                try:
                    arr = np.array(im)
                    arr = normalizer.transform(arr)
                    im = Image.fromarray(arr)
                except Exception:
                    pass

            # save tile
            fname = out_dir / f"{wsi_id}_tile_{i:05d}.{SAVE_FORMAT}"
            if SAVE_FORMAT == 'png':
                im.save(fname, compress_level=PNG_COMPRESSION)
            else:
                # webp
                im.save(fname, quality=WEBP_QUALITY, format='WEBP')

            saved_rows.append((str(wsi_path), str(fname), i))

        return (str(wsi_path), True, saved_rows)

    except Exception as e:
        tb = traceback.format_exc()
        return (str(wsi_path), False, f"Processing error: {e}\\n{tb}")

def main():
    svs_files = list(iter_svs_files(INPUT_ROOT, CANCER))
    print(f"Found {len(svs_files)} SVS files under {INPUT_ROOT}/{CANCER}")

    stain_ref = None
    if NORMALIZE and STAINTOOLS_AVAILABLE and STAIN_REF_PATH is not None:
        try:
            stain_ref = np.array(Image.open(STAIN_REF_PATH).convert('RGB'))
        except Exception:
            stain_ref = None

    tasks = []
    for svs in svs_files:
        tasks.append((str(svs), OUTPUT_ROOT, TILES_PER_WSI, TILE_SIZE, OCCUPANCY_THRESH,
                      TARGET_MPP, NORMALIZE, stain_ref))

    manifest_path = Path(OUTPUT_ROOT) / MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', newline='') as mf:
        writer = csv.writer(mf)
        writer.writerow(['wsi_path', 'tile_path', 'tile_index'])

    if WORKERS > 1 and len(tasks) > 0:
        with Pool(processes=WORKERS) as pool:
            for res in tqdm(pool.imap_unordered(process_wsi_pyvips, tasks), total=len(tasks)):
                wsi, ok, payload = res
                if ok:
                    with open(manifest_path, 'a', newline='') as mf:
                        writer = csv.writer(mf)
                        for row in payload:
                            writer.writerow(row)
                else:
                    print(f"WSI failed: {wsi} -> {payload}")
    else:
        for t in tqdm(tasks):
            res = process_wsi_pyvips(t)
            wsi, ok, payload = res
            if ok:
                with open(manifest_path, 'a', newline='') as mf:
                    writer = csv.writer(mf)
                    for row in payload:
                        writer.writerow(row)
            else:
                print(f"WSI failed: {wsi} -> {payload}")

    print('Done. Manifest at', manifest_path)

if __name__ == '__main__':
    main()
