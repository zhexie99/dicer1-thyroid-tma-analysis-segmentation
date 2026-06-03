#!/usr/bin/env python3
"""
Cellpose Segmentation — Batch Mode (parallelized I/O + saving)

For each .tif in input_dir, produces:
  <stem>_overlay.png   — original RGB image with cell outlines drawn on top
  <stem>_masks.tif     — uint16 label mask (each cell = unique integer)
  <stem>_flows.png     — Cellpose predicted flow field visualization
  <stem>_polygons.json — cell outlines as GeoJSON-style polygons (via shapely)

Parallelism strategy:
  - Cellpose inference runs sequentially (GPU/CPU saturated per image)
  - I/O (imread) and save (imwrite, polygon write) run in a ThreadPoolExecutor
    so disk ops overlap with inference on the next image

Usage:
    python cellpose_segment.py <input_dir> <output_dir>
    python cellpose_segment.py crops/ results/ --model cpsam --diameter 30 --gpu
    python cellpose_segment.py crops/ results/ --workers 8

Requirements:
    pip install cellpose tifffile numpy matplotlib scikit-image shapely
"""

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cellpose import models
from cellpose.utils import outlines_list   # fast C-level contour extraction
from skimage import io
from shapely.geometry import Polygon       # fastest polygon serialisation available

print_lock = Lock()

def tprint(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# I/O helpers (run in thread pool)
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path))
    # Ensure HxWx3 uint8
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] > 3:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).clip(0, 255).astype(np.uint8)
    return img


def save_overlay(image: np.ndarray, masks: np.ndarray, path: Path) -> None:
    """
    Side-by-side PNG: left = original RGB, right = RGB + green cell outlines.
    Saved at 600 DPI so the physical size matches a reasonable print dimension.
    """
    from skimage.segmentation import find_boundaries

    overlay = image.copy()
    boundary = find_boundaries(masks, mode="outer")
    overlay[boundary] = [0, 255, 0]   # green outlines

    # Stack horizontally: [original | overlay]
    side_by_side = np.concatenate([image, overlay], axis=1)  # H x 2W x 3

    h, w = side_by_side.shape[:2]
    dpi = 600
    # Figure size in inches so that 1 pixel = 1/dpi inches
    fig, ax = plt.subplots(1, 1, figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(side_by_side)
    ax.axis("off")
    # Add a subtle dividing line between the two panels
    ax.axvline(x=image.shape[1] - 0.5, color="white", linewidth=0.5)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(path), dpi=dpi, format="png",
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_masks(masks: np.ndarray, path: Path) -> None:
    tifffile.imwrite(str(path), masks.astype(np.uint16), compression="lzw")


def save_flows(flows, path: Path) -> None:
    """Save the RGB flow-field image (flows[0] is the RGB overlay from Cellpose)."""
    flow_rgb = flows[0]   # H x W x 3 uint8
    if isinstance(flow_rgb, np.ndarray) and flow_rgb.ndim == 3:
        tifffile.imwrite(str(path), flow_rgb.astype(np.uint8),
                         photometric="rgb", compression="lzw")
    else:
        # Fallback: render via matplotlib
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(flow_rgb)
        ax.axis("off")
        fig.savefig(str(path), dpi=100, bbox_inches="tight")
        plt.close(fig)


def save_polygons(masks: np.ndarray, path: Path) -> None:
    """
    Extract cell outlines and write as GeoJSON FeatureCollection.
    outlines_list() (Cellpose C extension) is the fastest way to get contours;
    Shapely handles polygon simplification and JSON serialisation.
    """
    outlines = outlines_list(masks)   # list of Nx2 arrays (row, col)

    features = []
    for cell_id, outline in enumerate(outlines, start=1):
        if len(outline) < 3:
            continue
        # outline is (row, col) → convert to (x, y) = (col, row)
        coords = [(float(pt[1]), float(pt[0])) for pt in outline]
        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)   # auto-repair self-intersections
            features.append({
                "type": "Feature",
                "id": cell_id,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(poly.exterior.coords)],
                },
                "properties": {"cell_id": cell_id, "area": poly.area},
            })
        except Exception:
            pass

    geojson = {"type": "FeatureCollection", "features": features}
    with open(path, "w") as f:
        json.dump(geojson, f, separators=(",", ":"))   # compact JSON


# ---------------------------------------------------------------------------
# Per-image worker (called from thread pool for save phase)
# ---------------------------------------------------------------------------

def save_all(image, masks, flows, stem, output_dir):
    overlay_path  = output_dir / f"{stem}_overlay.png"    # side-by-side 600dpi PNG
    mask_path     = output_dir / f"{stem}_masks.tif"      # uint16 label mask
    flow_path     = output_dir / f"{stem}_flows.tif"      # flow field
    polygon_path  = output_dir / f"{stem}_polygons.json"  # GeoJSON polygons

    save_overlay(image, masks, overlay_path)
    save_masks(masks, mask_path)
    save_flows(flows, flow_path)
    save_polygons(masks, polygon_path)

    n_cells = int(masks.max())
    tprint(f"    saved: overlay.png | masks.tif | flows.tif | polygons.json  ({n_cells} cells)")


# ---------------------------------------------------------------------------
# Cellpose inference (sequential — saturates CPU/GPU)
# ---------------------------------------------------------------------------

def run_cellpose(model, image, diameter, flow_threshold, cellprob_threshold):
    # Cellpose 4 API:
    #   - `channels` argument removed; CP4 auto-detects grayscale vs RGB
    #   - eval() returns exactly (masks, flows, diameters) — 3 values
    #   - flows[0] = RGB image, flows[1] = flow components, flows[2] = cell prob map
    masks, flows, diameters = model.eval(
        image,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    return masks, flows


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def batch_segment(input_dir: Path, output_dir: Path, model_type: str,
                  diameter, gpu: bool, workers: int,
                  flow_threshold: float, cellprob_threshold: float) -> None:

    tif_files = sorted(
        list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    )
    if not tif_files:
        sys.exit(f"[ERROR] No .tif/.tiff files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input   : {input_dir}  ({len(tif_files)} file(s))")
    print(f"Output  : {output_dir}")
    print(f"Model   : {model_type}  gpu={gpu}  diameter={diameter}")
    print(f"Workers : {workers} (I/O threads)\n")

    print(f"Loading Cellpose model: {model_type}...")
    model = models.CellposeModel(gpu=gpu, model_type=model_type)
    print("  Model ready.\n")

    ok = fail = 0

    # Pre-load first image while model is loading; then pipeline:
    # thread reads image[i+1] while inference runs on image[i],
    # and a thread saves image[i-1] results.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Submit all image loads upfront
        load_futures = {pool.submit(load_image, p): p for p in tif_files}
        # Track save futures so we can wait for them
        save_futures = []

        loaded = {}   # path → image (filled as futures complete)

        # Process in order; block on each image's load future
        for idx, tif_path in enumerate(tif_files, 1):
            tprint(f"[{idx}/{len(tif_files)}] {tif_path.name}")
            fut = load_futures[tif_path]
            try:
                image = fut.result()
            except Exception as e:
                tprint(f"    [ERROR] load failed: {e}")
                fail += 1
                continue

            tprint(f"    shape: {image.shape}  dtype: {image.dtype}")

            try:
                masks, flows = run_cellpose(
                    model, image, diameter, flow_threshold, cellprob_threshold
                )
            except Exception as e:
                tprint(f"    [ERROR] inference failed: {e}")
                traceback.print_exc()
                fail += 1
                continue

            # Fire-and-forget save (runs in thread while next inference starts)
            sf = pool.submit(save_all, image, masks, flows, tif_path.stem, output_dir)
            save_futures.append((tif_path.stem, sf))
            ok += 1

        # Wait for all saves to finish and report any errors
        for stem, sf in save_futures:
            try:
                sf.result()
            except Exception as e:
                tprint(f"    [ERROR] save failed for {stem}: {e}")
                traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"Done.  Success: {ok}   Failed: {fail}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch Cellpose segmentation on a directory of TIFF crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_dir",  help="Directory containing input .tif files")
    p.add_argument("output_dir", help="Directory to write results")
    p.add_argument("--model",    default="cyto3",
                   help="Cellpose 4 model type: cyto3 (default), nuclei, cpsam, "
                        "or a path to a custom model")
    p.add_argument("--diameter", type=float, default=None,
                   help="Cell diameter in pixels (None = auto-detect)")
    p.add_argument("--gpu",      action="store_true",
                   help="Use GPU acceleration")
    p.add_argument("--workers",  type=int, default=4,
                   help="Number of I/O threads for loading and saving")
    p.add_argument("--flow-threshold",    type=float, default=0.4)
    p.add_argument("--cellprob-threshold",type=float, default=0.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_segment(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model_type=args.model,
        diameter=args.diameter,
        gpu=args.gpu,
        workers=args.workers,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
    )
