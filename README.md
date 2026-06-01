# SVS to TIF Conversion & Cellpose Segmentation Workflow

This setup provides a complete pipeline for:
1. Reading SVS (Whole Slide Image) files
2. Converting to TIF format
3. Extracting ROIs with Fiji
4. Running Cellpose 4 for cell segmentation

## Installation Summary

✓ Virtual environment created
✓ All dependencies installed:
  - **openslide-python**: Read SVS files
  - **tifffile**: Write high-quality TIF files
  - **cellpose**: AI-powered cell segmentation
  - **torch**: Deep learning backend
  - **scikit-image**: Image processing utilities
  - **opencv-python**: Additional image processing
  - **numpy, scipy, Pillow**: Supporting libraries

## Workflow

### Step 1: Convert SVS to TIF

```bash
python svs_to_tif_converter.py
```

This script will:
- Scan for all `.svs` files in the workspace
- Display file information (dimensions, magnification, etc.)
- Convert each SVS to high-resolution TIF format
- Save as `.tif` files in the same directory

**Options** (edit the script to customize):
- `level`: 0 = highest resolution (default), higher numbers = lower resolution
- `region`: Extract specific regions instead of full image

### Step 2: Extract ROIs with Fiji

1. Open the generated `.tif` file in Fiji
2. Use Fiji tools to select and extract Regions of Interest (ROIs):
   - Use rectangle/polygon selection tools
   - Save ROIs as individual `.tif` files
   - Organize them in a folder (e.g., `fiji_rois/`)

### Step 3: Run Cellpose Segmentation

```bash
python cellpose_segmentation.py
```

This script will:
- Load the Cellpose nuclei model (or customize in script)
- Process all extracted ROI images
- Generate segmentation masks
- Create visualization overlays
- Save results to `cellpose_results/` folder

**Customize model type** (edit script):
- `'nuclei'`: For nuclear staining (default)
- `'cyto'`: For cytoplasm
- `'cyto2'`: For improved cytoplasm
- `diameter`: Cell size in pixels (None = auto-detect)

## Output Files

After segmentation, you'll have:
- `*_masks.tif`: Segmentation mask with labeled cells
- `*_overlay.png`: RGB visualization of segmentation on original image
- `*_flows.png`: Flow field visualization

## Advanced Usage

### Segment specific ROI folder
Edit the script to point to your ROI directory:
```python
roi_dir = workspace_dir / 'your_roi_folder'
```

### Adjust Cellpose parameters
In `cellpose_segmentation.py`:
```python
masks, flows, styles, diameters = self.model.eval(
    image,
    diameter=30,  # Change cell diameter
    flow_threshold=0.4,  # Adjust flow threshold
    cellprob_threshold=0.0  # Adjust probability threshold
)
```

### Enable GPU (if available)
```python
segmenter = CellposeSegmenter(model_type='nuclei', gpu=True)
```

## Python Environment

To activate the virtual environment in the terminal:
```bash
source .venv/bin/activate
```

Then run the scripts normally:
```bash
python svs_to_tif_converter.py
```

## Troubleshooting

**SVS file not readable**: Ensure openslide is properly installed. On macOS, you may need:
```bash
brew install openslide
```

**Out of memory on large images**: Extract smaller regions at lower pyramid levels:
```python
# Use lower resolution (level > 0)
convert_svs_to_tif(svs_file, level=1)
```

**Cellpose slow**: Use a GPU or run on smaller ROI images first for testing.

## Next Steps

1. Run `python svs_to_tif_converter.py` to convert your SVS files
2. Open the TIF files in Fiji and extract your ROIs
3. Run `python cellpose_segmentation.py` when ready to segment
4. Check the `cellpose_results/` folder for your segmentation outputs
