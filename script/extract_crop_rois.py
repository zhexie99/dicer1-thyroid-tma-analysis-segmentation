#!/usr/bin/env python3
"""
Extract full-resolution crops from an SVS using ROIs drawn on a downsampled image.

Workflow:
  1. You drew ROIs in Fiji on a level-N downsampled OME-TIFF
  2. You saved them via ROI Manager → More → Save → RoiSet.zip
  3. This script reads the zip, scales coordinates back to full-res,
     and crops each ROI out of the SVS at level 0.

Requirements:
    pip install openslide-python tifffile numpy

Usage:
    python extract_roi_crops.py slide.svs RoiSet.zip --level 2
    python extract_roi_crops.py slide.svs RoiSet.zip --level 2 --outdir crops/
    python extract_roi_crops.py slide.svs RoiSet.zip --factor 16.0   # manual factor
"""

import argparse
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import tifffile
import openslide


# ---------------------------------------------------------------------------
# Fiji .roi binary parser (pure stdlib — no roifile dependency)
# Spec: https://imagej.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html
# ---------------------------------------------------------------------------

ROI_MAGIC = b"Iout"   # first 4 bytes of every .roi file

TYPE_POLYGON  = 0
TYPE_RECT     = 1
TYPE_OVAL     = 2
TYPE_LINE     = 3
TYPE_FREELINE = 4
TYPE_POLYLINE = 5
TYPE_NOROI    = 6
TYPE_FREEHAND = 7
TYPE_TRACED   = 8
TYPE_ANGLE    = 9
TYPE_POINT    = 10

TYPE_NAMES = {
    TYPE_POLYGON:  "polygon",
    TYPE_RECT:     "rect",
    TYPE_OVAL:     "oval",
    TYPE_LINE:     "line",
    TYPE_FREELINE: "freeline",
    TYPE_POLYLINE: "polyline",
    TYPE_NOROI:    "noroi",
    TYPE_FREEHAND: "freehand",
    TYPE_TRACED:   "traced",
    TYPE_ANGLE:    "angle",
    TYPE_POINT:    "point",
}

def _r16(data, off):  return struct.unpack_from(">h", data, off)[0]
def _r32(data, off):  return struct.unpack_from(">i", data, off)[0]
def _ru16(data, off): return struct.unpack_from(">H", data, off)[0]


def parse_roi(data: bytes) -> dict:
    """Parse a single Fiji .roi binary blob; returns a dict with bounding box
    and (for polygon types) coordinate arrays."""
    if data[:4] != ROI_MAGIC:
        raise ValueError("Not a valid Fiji ROI (bad magic bytes)")

    version   = _ru16(data, 4)
    roi_type  = data[6]
    top       = _r16(data, 8)
    left      = _r16(data, 10)
    bottom    = _r16(data, 12)
    right     = _r16(data, 14)
    n_coords  = _ru16(data, 16)

    # Sub-pixel / large-coordinate support (offsets stored at byte 18/22/26/30)
    x_float = struct.unpack_from(">f", data, 18)[0]   # not always present
    y_float = struct.unpack_from(">f", data, 22)[0]

    name_offset = _r32(data, 48)
    name_length = _r32(data, 52)
    name = ""
    if name_offset > 0 and name_length > 0:
        try:
            name = data[name_offset: name_offset + name_length * 2].decode("utf-16-be")
        except Exception:
            pass

    roi = {
        "type":      TYPE_NAMES.get(roi_type, f"unknown_{roi_type}"),
        "top":       top,
        "left":      left,
        "bottom":    bottom,
        "right":     right,
        "n_coords":  n_coords,
        "name":      name,
        "x_coords":  None,
        "y_coords":  None,
    }

    # For polygon/freehand/point types read coordinate arrays
    if roi_type in (TYPE_POLYGON, TYPE_FREEHAND, TYPE_TRACED,
                    TYPE_POLYLINE, TYPE_FREELINE, TYPE_ANGLE, TYPE_POINT):
        coord_offset = 64
        if n_coords > 0 and len(data) >= coord_offset + n_coords * 4:
            xs = [_r16(data, coord_offset + i * 2)             for i in range(n_coords)]
            ys = [_r16(data, coord_offset + n_coords * 2 + i * 2) for i in range(n_coords)]
            # Coordinates are relative to bounding box top-left
            roi["x_coords"] = [x + left for x in xs]
            roi["y_coords"] = [y + top  for y in ys]

    return roi


def load_roi_zip(zip_path: str) -> list[dict]:
    """Read all .roi entries from a Fiji RoiSet.zip."""
    rois = []
    with zipfile.ZipFile(zip_path) as zf:
        for entry in sorted(zf.namelist()):
            if not entry.lower().endswith(".roi"):
                continue
            data = zf.read(entry)
            try:
                roi = parse_roi(data)
                # Use zip entry name as fallback ROI name
                if not roi["name"]:
                    roi["name"] = Path(entry).stem
                rois.append(roi)
            except Exception as e:
                print(f"  [WARN] Could not parse {entry}: {e}")
    return rois


# ---------------------------------------------------------------------------
# Bounding-box helper
# ---------------------------------------------------------------------------

def roi_bbox(roi: dict) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) bounding box for any ROI type."""
    x = roi["left"]
    y = roi["top"]
    w = roi["right"]  - roi["left"]
    h = roi["bottom"] - roi["top"]
    return x, y, w, h


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_crops(
    svs_path: str,
    roi_zip_path: str,
    outdir: str,
    svs_level: int,        # level the ROIs were drawn on
    downsample_factor: float | None,
    compression: str,
    padding: int,
) -> None:

    svs_path = Path(svs_path)
    roi_zip_path = Path(roi_zip_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not svs_path.exists():
        sys.exit(f"[ERROR] SVS not found: {svs_path}")
    if not roi_zip_path.exists():
        sys.exit(f"[ERROR] ROI zip not found: {roi_zip_path}")

    print(f"Opening slide : {svs_path}")
    slide = openslide.OpenSlide(str(svs_path))

    full_w, full_h = slide.level_dimensions[0]

    # Determine downsample factor
    if downsample_factor is not None:
        factor = downsample_factor
        print(f"Downsample    : {factor:.4f}x  (manual)")
    else:
        factor = slide.level_downsamples[svs_level]
        print(f"Downsample    : {factor:.4f}x  (level {svs_level})")

    print(f"Full-res dims : {full_w} x {full_h}")
    print(f"Output dir    : {outdir}")
    print()

    rois = load_roi_zip(str(roi_zip_path))
    print(f"Found {len(rois)} ROI(s) in {roi_zip_path.name}\n")

    for i, roi in enumerate(rois):
        name = roi["name"] or f"roi_{i:04d}"
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

        # Scale bounding box from downsampled → full-res coordinates
        x_ds, y_ds, w_ds, h_ds = roi_bbox(roi)

        x_full = int(round(x_ds * factor))
        y_full = int(round(y_ds * factor))
        w_full = int(round(w_ds * factor))
        h_full = int(round(h_ds * factor))

        # Apply padding
        x_full = max(0, x_full - padding)
        y_full = max(0, y_full - padding)
        w_full = min(full_w - x_full, w_full + padding * 2)
        h_full = min(full_h - y_full, h_full + padding * 2)

        if w_full <= 0 or h_full <= 0:
            print(f"  [{i+1}/{len(rois)}] {name} — skipped (zero-size after scaling)")
            continue

        print(f"  [{i+1}/{len(rois)}] {name}")
        print(f"    ROI (downsampled) : x={x_ds} y={y_ds} w={w_ds} h={h_ds}")
        print(f"    Crop (full-res)   : x={x_full} y={y_full} w={w_full} h={h_full}")

        # Read full-res region from SVS (level 0)
        region = slide.read_region((x_full, y_full), 0, (w_full, h_full))
        img = np.array(region)[..., :3]   # RGBA → RGB

        out_path = outdir / f"{safe_name}.tif"
        tifffile.imwrite(
            str(out_path),
            img,
            compression=compression,
            photometric="rgb",
            metadata=None,
        )
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"    Saved → {out_path.name}  ({size_mb:.1f} MB)\n")

    slide.close()
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract full-resolution ROI crops from an SVS using a "
                    "Fiji RoiSet.zip drawn on a downsampled image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("svs",     help="Path to the .svs file")
    p.add_argument("roiset",  help="Path to the Fiji RoiSet.zip")
    p.add_argument("--level", type=int, default=2,
                   help="SVS pyramid level the ROIs were drawn on (0=full res)")
    p.add_argument("--factor", type=float, default=None,
                   help="Override downsample factor instead of using --level "
                        "(e.g. 16.0). Use if your OME-TIFF was exported at a "
                        "non-standard factor.")
    p.add_argument("--outdir", default="crops",
                   help="Output directory for cropped TIFFs")
    p.add_argument("--compression", default="lzw",
                   choices=["lzw", "jpeg", "deflate", "zstd", "none"],
                   help="Compression for output TIFFs")
    p.add_argument("--padding", type=int, default=0,
                   help="Extra pixels to add around each ROI bounding box "
                        "(in full-res pixels)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_crops(
        svs_path=args.svs,
        roi_zip_path=args.roiset,
        outdir=args.outdir,
        svs_level=args.level,
        downsample_factor=args.factor,
        compression=args.compression,
        padding=args.padding,
    )