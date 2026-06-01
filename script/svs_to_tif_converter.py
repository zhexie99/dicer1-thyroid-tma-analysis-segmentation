"""
SVS to TIF Converter
Converts Whole Slide Image (SVS) files to TIFF format for further processing.
"""

import os
from pathlib import Path
import numpy as np
from openslide import OpenSlide, PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y
from tifffile import imwrite
from PIL import Image

def get_svs_info(svs_path):
    """Get information about the SVS file."""
    try:
        slide = OpenSlide(str(svs_path))
        print(f"\n=== SVS File Information ===")
        print(f"File: {svs_path.name}")
        print(f"Dimensions: {slide.dimensions}")
        print(f"Number of levels: {slide.level_count}")
        print(f"Level dimensions: {slide.level_dimensions}")
        
        # Get pixel size if available
        try:
            mpp_x = float(slide.properties.get(PROPERTY_NAME_MPP_X, 0))
            mpp_y = float(slide.properties.get(PROPERTY_NAME_MPP_Y, 0))
            if mpp_x > 0:
                print(f"Microns per pixel: {mpp_x:.4f}")
        except:
            pass
        
        slide.close()
        return True
    except Exception as e:
        print(f"Error reading SVS file: {e}")
        return False

def convert_svs_to_tif(svs_path, output_path=None, level=0, region=None):
    """
    Convert SVS file to TIF format.
    
    Args:
        svs_path: Path to the SVS file
        output_path: Path for output TIF file (defaults to same name with .tif extension)
        level: Pyramid level to extract (0 = highest resolution)
        region: Tuple of (x, y, width, height) to extract specific region, or None for full image
    """
    try:
        svs_path = Path(svs_path)
        
        if not svs_path.exists():
            print(f"Error: SVS file not found: {svs_path}")
            return False
        
        if output_path is None:
            output_path = svs_path.with_suffix('.tif')
        else:
            output_path = Path(output_path)
        
        print(f"\nConverting {svs_path.name} to TIF...")
        print(f"Reading from level {level}...")
        
        # Open the slide
        slide = OpenSlide(str(svs_path))
        
        # Get image data
        if region:
            x, y, width, height = region
            image = slide.read_region((x, y), level, (width, height))
        else:
            # Get full image at specified level
            width, height = slide.level_dimensions[level]
            image = slide.read_region((0, 0), level, (width, height))
        
        # Convert RGBA to RGB
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image
        
        # Convert to numpy array and save
        image_array = np.array(image)
        
        print(f"Image shape: {image_array.shape}")
        print(f"Saving to {output_path}...")
        
        imwrite(str(output_path), image_array, compression='lzw')
        
        slide.close()
        print(f"✓ Successfully converted to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error converting SVS to TIF: {e}")
        import traceback
        traceback.print_exc()
        return False

def batch_convert_svs_files(directory, output_dir=None, level=0):
    """
    Convert all SVS files in a directory to TIF format.
    
    Args:
        directory: Directory containing SVS files
        output_dir: Output directory for TIF files (defaults to same directory)
        level: Pyramid level to extract
    """
    directory = Path(directory)
    if output_dir is None:
        output_dir = directory
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    svs_files = list(directory.glob('*.svs'))
    
    if not svs_files:
        print(f"No SVS files found in {directory}")
        return
    
    print(f"\nFound {len(svs_files)} SVS file(s)")
    
    for svs_file in svs_files:
        output_path = output_dir / svs_file.with_suffix('.tif').name
        success = convert_svs_to_tif(svs_file, output_path, level=level)
        if not success:
            print(f"✗ Failed to convert {svs_file.name}")

if __name__ == "__main__":
    # Example usage
    print("SVS to TIF Converter")
    print("=" * 60)
    
    # Get the directory containing this script
    workspace_dir = Path(__file__).parent
    
    # First, let's get info about all SVS files
    print("\nScanning for SVS files...")
    svs_files = list(workspace_dir.glob('*.svs'))
    
    if svs_files:
        print(f"Found {len(svs_files)} SVS file(s):\n")
        for svs_file in svs_files:
            get_svs_info(svs_file)
        
        print("\n" + "=" * 60)
        print("NOTE: These are very large whole slide images!")
        print("=" * 60)
        print("\nRecommended workflow:")
        print("1. Use Fiji to view and select ROI regions from the SVS files")
        print("2. Extract specific ROIs as smaller TIF files")
        print("3. Run Cellpose on the extracted ROIs")
        print("\nTo convert specific regions, modify the script:")
        print("  region = (x, y, width, height)  # in pixels")
        print("  convert_svs_to_tif(svs_file, region=region, level=2)")
        print("\nTo convert at lower resolution (faster):")
        print("  batch_convert_svs_files(workspace_dir, level=2)")
        print("  (level 0=highest res, level 1/2/3=lower res)")
        print("=" * 60)
        
        # Option 1: Convert at lower pyramid level (faster)
        print("\nConverting SVS files to TIF at FULL RESOLUTION (level 0)...")
        print("This will create high-quality files for detailed ROI extraction...")
        print("(May take 10-20 minutes per file with your 64GB RAM)")
        batch_convert_svs_files(workspace_dir, level=0)
    else:
        print("No SVS files found in the workspace directory")
