"""
Cellpose Segmentation
Runs Cellpose 4 on extracted ROI images for cell segmentation.
Supports both full TIF images and Fiji ROI sets.
"""

import os
from pathlib import Path
import numpy as np
from cellpose import models
from cellpose import plot
from skimage import io
import matplotlib.pyplot as plt
import tifffile
import zipfile
import struct
from PIL import Image


class FijiROILoader:
    """Load ROI coordinates from Fiji ROI ZIP files."""
    
    @staticmethod
    def read_roi_zip(zip_path):
        """
        Extract ROI coordinates from Fiji ROI ZIP file.
        
        Args:
            zip_path: Path to the .zip file containing .roi files
            
        Returns:
            Dictionary with ROI names as keys and coordinate dicts as values
        """
        rois = {}
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                for filename in z.namelist():
                    if filename.endswith('.roi'):
                        try:
                            roi_name = Path(filename).stem
                            roi_data = z.read(filename)
                            
                            # Try to parse binary format first
                            coords = None
                            
                            # Check for Iout magic number (ImageJ/Fiji rectangular ROI)
                            if roi_data[:4] == b'Iout':
                                # Iout format: bytes 4-6 = version, then coordinates
                                # Extract bounding box coordinates
                                try:
                                    top = struct.unpack('>H', roi_data[8:10])[0]
                                    left = struct.unpack('>H', roi_data[10:12])[0]
                                    bottom = struct.unpack('>H', roi_data[12:14])[0]
                                    right = struct.unpack('>H', roi_data[14:16])[0]
                                    
                                    coords = {
                                        'x': left,
                                        'y': top,
                                        'width': right - left,
                                        'height': bottom - top,
                                        'top': top,
                                        'left': left,
                                        'bottom': bottom,
                                        'right': right
                                    }
                                except:
                                    pass
                            
                            # Fallback: try to parse coordinates from filename
                            # Filenames like "0230-0370" mean x=230, y=370
                            if coords is None:
                                try:
                                    parts = roi_name.split('-')
                                    if len(parts) >= 2:
                                        # Try to interpret as x-y coordinates
                                        x = int(parts[0])
                                        y = int(parts[1])
                                        # Use reasonable default size (can be adjusted)
                                        width = 100
                                        height = 100
                                        coords = {
                                            'x': x,
                                            'y': y,
                                            'width': width,
                                            'height': height
                                        }
                                except:
                                    pass
                            
                            if coords:
                                rois[filename] = coords
                            else:
                                print(f"Warning: Could not parse ROI file {filename}")
                                
                        except Exception as e:
                            print(f"Warning: Could not parse ROI file {filename}: {e}")
        except Exception as e:
            print(f"Error reading ROI ZIP file {zip_path}: {e}")
        
        return rois
    
    @staticmethod
    def extract_roi_region(image_path, roi_coords):
        """
        Extract a single ROI region from an image.
        
        Args:
            image_path: Path to the image file (TIF or otherwise)
            roi_coords: Dictionary with x, y, width, height keys
            
        Returns:
            numpy array of the ROI region
        """
        # Read image
        if str(image_path).lower().endswith(('.tif', '.tiff')):
            full_image = tifffile.imread(str(image_path))
        else:
            full_image = io.imread(str(image_path))
        
        # Handle multi-channel images
        if len(full_image.shape) == 3 and full_image.shape[2] > 3:
            # Multiple channels, take first one
            full_image = full_image[:, :, 0]
        
        # Extract ROI
        x = int(roi_coords['x'])
        y = int(roi_coords['y'])
        width = int(roi_coords['width'])
        height = int(roi_coords['height'])
        
        # Ensure we don't go out of bounds
        x = max(0, min(x, full_image.shape[1]))
        y = max(0, min(y, full_image.shape[0]))
        x_end = min(x + width, full_image.shape[1])
        y_end = min(y + height, full_image.shape[0])
        
        roi_image = full_image[y:y_end, x:x_end]
        
        return roi_image


class CellposeSegmenter:
    def __init__(self, model_type='nuclei', gpu=False):
        """
        Initialize Cellpose model.
        
        Args:
            model_type: 'nuclei', 'cyto', 'cyto2', or custom model path
            gpu: Whether to use GPU acceleration
        """
        print(f"Loading Cellpose {model_type} model...")
        self.model = models.CellposeModel(gpu=gpu, model_type=model_type)
        print("✓ Model loaded successfully")
    
    def segment_image(self, image_path, diameter=None, channels=[0, 0], 
                      flow_threshold=0.4, cellprob_threshold=0.0):
        """
        Run segmentation on a single image.
        
        Args:
            image_path: Path to the image file
            diameter: Cell diameter in pixels (None for automatic)
            channels: [cytoplasm, nuclei] channel indices
            flow_threshold: Flow field confidence threshold
            cellprob_threshold: Cell probability threshold
            
        Returns:
            masks, flows, styles, diameters
        """
        print(f"\nProcessing: {Path(image_path).name}")
        
        # Read image
        if str(image_path).lower().endswith(('.tif', '.tiff')):
            image = tifffile.imread(str(image_path))
        else:
            image = io.imread(str(image_path))
        
        print(f"Image shape: {image.shape}")
        
        # Run segmentation
        masks, flows, styles, diameters = self.model.eval(
            image,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )
        
        print(f"✓ Segmentation complete - found {len(np.unique(masks))-1} cells")
        
        return masks, flows, styles, diameters
    
    def segment_roi(self, image_path, roi_coords, roi_name=None, diameter=None, 
                    channels=[0, 0], flow_threshold=0.4, cellprob_threshold=0.0):
        """
        Run segmentation on an ROI region extracted from an image.
        
        Args:
            image_path: Path to the full image file
            roi_coords: Dictionary with x, y, width, height keys (from FijiROILoader)
            roi_name: Name of the ROI (for logging)
            diameter: Cell diameter in pixels (None for automatic)
            channels: [cytoplasm, nuclei] channel indices
            flow_threshold: Flow field confidence threshold
            cellprob_threshold: Cell probability threshold
            
        Returns:
            masks, flows, styles, diameters, roi_image
        """
        if roi_name is None:
            roi_name = f"ROI at ({roi_coords['x']}, {roi_coords['y']})"
        
        print(f"\nProcessing ROI: {roi_name}")
        
        # Extract ROI region from image
        roi_image = FijiROILoader.extract_roi_region(image_path, roi_coords)
        
        print(f"ROI image shape: {roi_image.shape}")
        
        # Run segmentation on ROI
        masks, flows, styles, diameters = self.model.eval(
            roi_image,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )
        
        print(f"✓ ROI segmentation complete - found {len(np.unique(masks))-1} cells")
        
        return masks, flows, styles, diameters, roi_image
    
    def save_results(self, image_path, masks, flows, output_dir=None, 
                     save_overlay=True, save_masks=True, save_flows=True):
        """
        Save segmentation results.
        
        Args:
            image_path: Original image path
            masks: Segmentation masks
            flows: Flow field
            output_dir: Directory to save results
            save_overlay: Save RGB overlay
            save_masks: Save mask file
            save_flows: Save flow visualization
        """
        if output_dir is None:
            output_dir = Path(image_path).parent / 'cellpose_results'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(image_path).stem
        
        # Save masks
        if save_masks:
            mask_path = output_dir / f'{base_name}_masks.tif'
            tifffile.imwrite(str(mask_path), masks.astype(np.uint16))
            print(f"✓ Saved masks: {mask_path}")
        
        # Save overlay image
        if save_overlay:
            # Read original image
            if str(image_path).lower().endswith(('.tif', '.tiff')):
                image = tifffile.imread(str(image_path))
            else:
                image = io.imread(str(image_path))
            
            # Ensure grayscale or convert
            if len(image.shape) == 3 and image.shape[2] == 3:
                img_gray = np.mean(image, axis=2)
            elif len(image.shape) == 3:
                img_gray = image[:, :, 0]
            else:
                img_gray = image
            
            # Create overlay
            fig = plt.figure(figsize=(12, 10))
            plot.show_segmentation(fig, img_gray, masks, flows[0], channels=[0,0])
            
            overlay_path = output_dir / f'{base_name}_overlay.png'
            plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Saved overlay: {overlay_path}")
        
        # Save flows visualization
        if save_flows:
            fig = plt.figure(figsize=(12, 5))
            plot.show_flows(flows, file_name=None, imshow=False)
            flow_path = output_dir / f'{base_name}_flows.png'
            plt.savefig(flow_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Saved flows: {flow_path}")
        
        return output_dir
    
    def batch_segment(self, input_dir, pattern='*.tif', diameter=None, 
                      output_dir=None):
        """
        Segment all matching images in a directory.
        
        Args:
            input_dir: Directory containing images
            pattern: File pattern to match
            diameter: Cell diameter
            output_dir: Output directory for results
        """
        input_dir = Path(input_dir)
        image_files = sorted(list(input_dir.glob(pattern)))
        
        if not image_files:
            print(f"No images matching '{pattern}' found in {input_dir}")
            return
        
        print(f"\nFound {len(image_files)} image(s) to process")
        
        for idx, image_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}]")
            
            try:
                masks, flows, styles, diameters = self.segment_image(
                    image_path, 
                    diameter=diameter
                )
                
                self.save_results(
                    image_path, 
                    masks, 
                    flows,
                    output_dir=output_dir
                )
            except Exception as e:
                print(f"✗ Error processing {image_path.name}: {e}")
                import traceback
                traceback.print_exc()
    
    def batch_segment_roi(self, image_path, roi_zip_path, diameter=None, 
                         output_dir=None):
        """
        Segment all ROIs from a Fiji ROI ZIP file within an image.
        
        Args:
            image_path: Path to the full image file (TIF or otherwise)
            roi_zip_path: Path to the Fiji ROI ZIP file
            diameter: Cell diameter in pixels
            output_dir: Output directory for results
        """
        image_path = Path(image_path)
        roi_zip_path = Path(roi_zip_path)
        
        # Load ROIs from ZIP
        print(f"Loading ROIs from: {roi_zip_path.name}")
        rois = FijiROILoader.read_roi_zip(roi_zip_path)
        
        if not rois:
            print(f"No ROIs found in {roi_zip_path}")
            return
        
        print(f"Found {len(rois)} ROI(s)")
        
        if output_dir is None:
            output_dir = image_path.parent / 'cellpose_results_roi'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = image_path.stem
        
        for idx, (roi_filename, roi_coords) in enumerate(rois.items(), 1):
            roi_name = Path(roi_filename).stem
            print(f"\n[{idx}/{len(rois)}] Processing ROI: {roi_name}")
            
            try:
                masks, flows, styles, diameters, roi_image = self.segment_roi(
                    image_path,
                    roi_coords,
                    roi_name=roi_name,
                    diameter=diameter
                )
                
                # Save ROI-specific results
                roi_output_dir = output_dir / roi_name
                roi_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save ROI image
                roi_img_path = roi_output_dir / f'{roi_name}_roi.tif'
                if roi_image.dtype in [np.float32, np.float64]:
                    roi_image_uint8 = (roi_image / roi_image.max() * 255).astype(np.uint8)
                else:
                    roi_image_uint8 = roi_image.astype(np.uint8)
                tifffile.imwrite(str(roi_img_path), roi_image_uint8)
                print(f"✓ Saved ROI image: {roi_img_path}")
                
                # Save masks
                mask_path = roi_output_dir / f'{roi_name}_masks.tif'
                tifffile.imwrite(str(mask_path), masks.astype(np.uint16))
                print(f"✓ Saved masks: {mask_path}")
                
                # Save overlay
                fig = plt.figure(figsize=(12, 10))
                plot.show_segmentation(fig, roi_image, masks, flows[0], channels=[0,0])
                overlay_path = roi_output_dir / f'{roi_name}_overlay.png'
                plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"✓ Saved overlay: {overlay_path}")
                
            except Exception as e:
                print(f"✗ Error processing ROI {roi_name}: {e}")
                import traceback
                traceback.print_exc()

def main():
    """Main execution function."""
    
    print("=" * 60)
    print("Cellpose Cell Segmentation")
    print("=" * 60)
    
    # Configuration
    MODEL_TYPE = 'nuclei'  # Change to 'cyto' or 'cyto2' for cytoplasm
    GPU = False  # Set to True if you have GPU
    DIAMETER = None  # Auto-detect if None
    
    # Segmentation mode: 'full', 'roi', or 'both'
    MODE = 'roi'  # 'full' for full TIF, 'roi' for ROI extraction, 'both' for both
    
    # Get workspace directory
    workspace_dir = Path(__file__).parent
    
    print(f"\nWorkspace: {workspace_dir}")
    
    # Initialize segmenter
    segmenter = CellposeSegmenter(model_type=MODEL_TYPE, gpu=GPU)
    
    # Find TIF files and ROI ZIP files
    tif_files = sorted(list(workspace_dir.glob('*.tif')))
    roi_zips = sorted(list(workspace_dir.glob('*ROI*.zip')))
    
    print(f"\nFound {len(tif_files)} TIF file(s)")
    print(f"Found {len(roi_zips)} ROI ZIP file(s)")
    
    # Segment full TIF images
    if MODE in ['full', 'both'] and tif_files:
        print("\n" + "=" * 60)
        print("SEGMENTING FULL TIF IMAGES")
        print("=" * 60)
        
        for tif_file in tif_files:
            print(f"\nSegmenting: {tif_file.name}")
            try:
                masks, flows, styles, diameters = segmenter.segment_image(
                    tif_file,
                    diameter=DIAMETER
                )
                
                segmenter.save_results(
                    tif_file,
                    masks,
                    flows,
                    output_dir=workspace_dir / 'cellpose_results_full'
                )
            except Exception as e:
                print(f"✗ Error processing {tif_file.name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Segment ROIs
    if MODE in ['roi', 'both'] and roi_zips:
        print("\n" + "=" * 60)
        print("SEGMENTING ROI REGIONS")
        print("=" * 60)
        
        for roi_zip in roi_zips:
            # Find corresponding TIF file
            # Extract base name from ROI ZIP (remove -ROI, _ROI, ROI patterns)
            roi_base = roi_zip.stem
            for pattern in ['-ROI', '_ROI', 'ROI']:
                if pattern in roi_base:
                    roi_base = roi_base.replace(pattern, '').strip()
                    break
            
            # Try to find matching TIF
            matching_tif = None
            for tif in tif_files:
                # Check if ROI base is in TIF name or vice versa
                if (roi_base.lower() in tif.stem.lower() or 
                    tif.stem.lower().split()[0] in roi_base.lower()):
                    matching_tif = tif
                    break
            
            # If no match found, try first TIF file as fallback
            if matching_tif is None and tif_files:
                matching_tif = tif_files[0]
                print(f"⚠ Using first TIF as fallback: {matching_tif.name}")
            
            print(f"\nSegmenting ROIs from: {roi_zip.name}")
            print(f"Using TIF: {matching_tif.name}")
            
            try:
                segmenter.batch_segment_roi(
                    matching_tif,
                    roi_zip,
                    diameter=DIAMETER,
                    output_dir=workspace_dir / 'cellpose_results_roi'
                )
            except Exception as e:
                print(f"✗ Error processing ROI ZIP {roi_zip.name}: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✓ Segmentation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
