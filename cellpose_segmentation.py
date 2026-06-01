"""
Cellpose Segmentation
Runs Cellpose 4 on extracted ROI images for cell segmentation.
"""

import os
from pathlib import Path
import numpy as np
from cellpose import models
from cellpose import plot
from skimage import io
import matplotlib.pyplot as plt
import tifffile

class CellposeSegmenter:
    def __init__(self, model_type='nuclei', gpu=False):
        """
        Initialize Cellpose model.
        
        Args:
            model_type: 'nuclei', 'cyto', 'cyto2', or custom model path
            gpu: Whether to use GPU acceleration
        """
        print(f"Loading Cellpose {model_type} model...")
        self.model = models.Cellpose(gpu=gpu, model_type=model_type)
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

def main():
    """Main execution function."""
    
    print("=" * 50)
    print("Cellpose Cell Segmentation")
    print("=" * 50)
    
    # Configuration
    MODEL_TYPE = 'nuclei'  # Change to 'cyto' or 'cyto2' for cytoplasm
    GPU = False  # Set to True if you have GPU
    DIAMETER = None  # Auto-detect if None
    
    # Get workspace directory
    workspace_dir = Path(__file__).parent
    
    # Look for TIF files (from Fiji ROI extraction)
    roi_dir = workspace_dir / 'fiji_rois'  # or adjust path where Fiji exports ROIs
    
    if not roi_dir.exists():
        print(f"\nNote: Expected ROI directory at {roi_dir}")
        print("Please extract ROIs from your TIF files using Fiji first.")
        print("\nAlternatively, you can use TIF files directly:")
        
        tif_files = list(workspace_dir.glob('*.tif'))
        if tif_files:
            print(f"Found {len(tif_files)} TIF file(s)")
            roi_dir = workspace_dir
        else:
            print("No TIF files found.")
            return
    
    # Initialize segmenter
    segmenter = CellposeSegmenter(model_type=MODEL_TYPE, gpu=GPU)
    
    # Run batch segmentation
    print(f"\nSegmenting images with '{MODEL_TYPE}' model...")
    segmenter.batch_segment(
        roi_dir,
        pattern='*.tif',
        diameter=DIAMETER,
        output_dir=roi_dir / 'cellpose_results'
    )
    
    print("\n" + "=" * 50)
    print("✓ Segmentation complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
