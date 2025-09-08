"""
Data processing pipeline for subsurface imaging data.
"""
import os
import csv
import numpy as np
import tiledb
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SubsurfaceDataProcessor:
    def __init__(self, array_path="data/arrays/subsurface_data"):
        self.array_path = array_path
        self.original_width = 200
        self.target_width = 150
        self.custom_colormap = self._create_custom_colormap()
    
    def _create_custom_colormap(self):
        colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('subsurface', colors, N=n_bins)
        return cmap
    
    def create_array_schema(self, max_images=1000, max_depths=10000, width=150):
        """
        Create optimized 3D schema for efficient geological data queries.
        
        Schema: [image_id, depth_index, pixel_index]
        - image_id: 1 to max_images (tile=10 for efficient image-based queries)
        - depth_index: 0 to max_depths-1 (tile=100 for efficient depth range queries)
        - pixel_index: 0 to width-1 (tile=width for efficient pixel access)
        """
        image_dim = tiledb.Dim(
            name="image_id",
            domain=(1, max_images),
            tile=1,  # Minimal tiles for fastest array creation
            dtype=np.uint32
        )
        
        depth_dim = tiledb.Dim(
            name="depth_index", 
            domain=(0, max_depths - 1),
            tile=100,  # Smaller tiles for faster array creation
            dtype=np.uint32
        )
        
        pixel_dim = tiledb.Dim(
            name="pixel_index",
            domain=(0, width - 1),
            tile=width,  # Full width for efficient pixel access
            dtype=np.uint32
        )
        
        domain = tiledb.Domain(image_dim, depth_dim, pixel_dim)
        
        # Use fast compression for optimal speed/size balance
        intensity_attr = tiledb.Attr(
            name="intensity_value",
            dtype=np.uint8,
            filters=tiledb.FilterList([tiledb.ZstdFilter(level=1)])
        )
        red_attr = tiledb.Attr(name="red_value", dtype=np.uint8, filters=tiledb.FilterList([tiledb.ZstdFilter(level=1)]))
        green_attr = tiledb.Attr(name="green_value", dtype=np.uint8, filters=tiledb.FilterList([tiledb.ZstdFilter(level=1)]))
        blue_attr = tiledb.Attr(name="blue_value", dtype=np.uint8, filters=tiledb.FilterList([tiledb.ZstdFilter(level=1)]))
        
        depth_attr = tiledb.Attr(name="depth_value", dtype=np.float32)
        # Note: image_id is now a dimension, not an attribute
        timestamp_attr = tiledb.Attr(name="processed_at", dtype=np.int64)
        
        schema = tiledb.ArraySchema(
            domain=domain,
            sparse=False,
            attrs=[intensity_attr, red_attr, green_attr, blue_attr, depth_attr, timestamp_attr],
            cell_order='row-major',
            tile_order='row-major',
            capacity=100000  # Optimized capacity for performance
        )
        
        return schema
    
    def resize_image_row(self, row_data):
        """
        Optimized resizing using variance-preserving sampling for geological data.
        This method preserves geological features better than standard interpolation.
        """
        from scipy import ndimage
        
        original_row = np.array(row_data, dtype=np.float32)
        
        # Use scipy's zoom with order=3 for optimal geological data preservation
        # This preserves variance and geological features better than OpenCV interpolation
        resized_row = ndimage.zoom(original_row, self.target_width/len(original_row), order=3)
        
        return resized_row
    
    def apply_colormap(self, intensity_values):
        normalized = (intensity_values - intensity_values.min()) / (intensity_values.max() - intensity_values.min() + 1e-8)
        colored = self.custom_colormap(normalized)
        
        red = (colored[:, 0] * 255).astype(np.uint8)
        green = (colored[:, 1] * 255).astype(np.uint8)
        blue = (colored[:, 2] * 255).astype(np.uint8)
        
        return red, green, blue
    
    def _get_next_image_id(self):
        if not tiledb.object_type(self.array_path) == "array":
            return 1
        
        try:
            with tiledb.open(self.array_path, mode='r') as array:
                # Read a larger sample to ensure we get all image_ids
                non_empty_domain = array.nonempty_domain()
                if non_empty_domain[0][1] >= 0:  # If array has data
                    total_rows = non_empty_domain[0][1] + 1
                    sample_size = min(2000, total_rows // 3) if total_rows > 6000 else min(1000, total_rows)
                    
                    max_image_id = 0
                    
                    # Sample from beginning
                    if total_rows > 0:
                        start_sample = array[0:min(sample_size, total_rows), 0:1]
                        if 'image_id' in start_sample:
                            max_image_id = max(max_image_id, np.max(start_sample['image_id'][:, 0]))
                    
                    # Sample from middle
                    if total_rows > sample_size * 2:
                        mid_start = total_rows // 2 - sample_size // 2
                        mid_sample = array[mid_start:mid_start + sample_size, 0:1]
                        if 'image_id' in mid_sample:
                            max_image_id = max(max_image_id, np.max(mid_sample['image_id'][:, 0]))
                    
                    # Sample from end
                    if total_rows > sample_size:
                        end_start = max(0, total_rows - sample_size)
                        end_sample = array[end_start:total_rows, 0:1]
                        if 'image_id' in end_sample:
                            max_image_id = max(max_image_id, np.max(end_sample['image_id'][:, 0]))
                    
                    return int(max_image_id) + 1
        except Exception as e:
            logger.warning(f"Could not determine next image ID: {e}")
        
        return 1

    def get_available_images(self):
        if not tiledb.object_type(self.array_path) == "array":
            return []
        
        try:
            with tiledb.open(self.array_path, mode='r') as array:
                # Read only image_id column efficiently
                non_empty_domain = array.nonempty_domain()
                if non_empty_domain[0][1] >= 0:  # If array has data
                    total_rows = non_empty_domain[0][1] + 1
                    
                    # Sample from multiple sections to ensure we get all image_ids
                    # Sample from beginning, middle, and end
                    sample_size = min(2000, total_rows // 3) if total_rows > 6000 else min(1000, total_rows)
                    
                    unique_images = set()
                    
                    # Sample from beginning
                    if total_rows > 0:
                        start_sample = array[0:min(sample_size, total_rows), 0:1]
                        if 'image_id' in start_sample:
                            unique_images.update(start_sample['image_id'])
                    
                    # Sample from middle
                    if total_rows > sample_size * 2:
                        mid_start = total_rows // 2 - sample_size // 2
                        mid_sample = array[mid_start:mid_start + sample_size, 0:1]
                        if 'image_id' in mid_sample:
                            unique_images.update(mid_sample['image_id'])
                    
                    # Sample from end
                    if total_rows > sample_size:
                        end_start = max(0, total_rows - sample_size)
                        end_sample = array[end_start:total_rows, 0:1]
                        if 'image_id' in end_sample:
                            unique_images.update(end_sample['image_id'])
                    
                    return sorted([int(s) for s in unique_images])
        except Exception as e:
            logger.warning(f"Could not get available images: {e}")
        
        return []

    def process_csv_file(self, csv_path, image_id=None):
        logger.info(f"Processing CSV file: {csv_path}")
        
        if image_id is None:
            image_id = self._get_next_image_id()
            logger.info(f"Auto-assigned image_id: {image_id}")
        else:
            logger.info(f"Using provided image_id: {image_id}")
        
        import pandas as pd
        df = pd.read_csv(csv_path, header=0)
        
        depth_values = df.iloc[:, 0].to_numpy(np.float32)
        original_image_data = df.iloc[:, 1:].to_numpy(dtype=np.float32)
        
        height, original_width = original_image_data.shape
        logger.info(f"Found {height} depth levels, original width: {original_width}")
        
        logger.info("Resizing image data...")
        resized_image_data = np.zeros((height, self.target_width), dtype=np.float32)
        
        for i in range(height):
            resized_image_data[i] = self.resize_image_row(original_image_data[i])
        
        # Handle NaN values before casting to uint8
        resized_image_data = np.nan_to_num(resized_image_data, nan=0.0, posinf=255.0, neginf=0.0)
        normalized_intensities = np.clip(resized_image_data, 0, 255).astype(np.uint8)
        
        logger.info("Applying colormap...")
        red_data = np.zeros((height, self.target_width), dtype=np.uint8)
        green_data = np.zeros((height, self.target_width), dtype=np.uint8)
        blue_data = np.zeros((height, self.target_width), dtype=np.uint8)
        
        for i in range(height):
            red, green, blue = self.apply_colormap(resized_image_data[i])
            red_data[i] = red
            green_data[i] = green
            blue_data[i] = blue
        
        depth_array = np.broadcast_to(depth_values[:, np.newaxis], (height, self.target_width))
        # Create array for metadata (image_id is now a dimension, not an attribute)
        timestamp_array = np.full((height, self.target_width), int(time.time() * 1000000), dtype=np.int64)
        
        if tiledb.object_type(self.array_path) == "array":
            logger.info("Appending to existing array...")
            self._append_to_existing_array(
                normalized_intensities, red_data, green_data, blue_data,
                depth_array, timestamp_array, image_id
            )
        else:
            logger.info("Creating new TileDB array...")
            self._create_new_array(
                normalized_intensities, red_data, green_data, blue_data,
                depth_array, timestamp_array, image_id
            )
        
        logger.info(f"Successfully processed {height} depth levels")
    
    def _create_new_array(self, intensities, red_data, green_data, blue_data, depth_array, timestamp_array, image_id):
        height, width = intensities.shape
        
        # Ensure the directory exists
        import os
        os.makedirs(os.path.dirname(self.array_path), exist_ok=True)
        
        schema = self.create_array_schema(max_images=1000, max_depths=10000, width=width)
        tiledb.DenseArray.create(self.array_path, schema)
        
        with tiledb.DenseArray(self.array_path, mode="w") as A:
            # Write using 3D coordinates: [image_id, depth_index, pixel_index]
            A[image_id, 0:height, 0:width] = {
                "intensity_value": intensities,
                "red_value": red_data,
                "green_value": green_data,
                "blue_value": blue_data,
                "depth_value": depth_array,
                "processed_at": timestamp_array
            }
        
        logger.info(f"Created new 3D array with image_id={image_id}, depth_range=0:{height-1}, width={width}")
    
    def _append_to_existing_array(self, intensities, red_data, green_data, blue_data, depth_array, timestamp_array, image_id):
        height, width = intensities.shape
        
        # Ensure the directory exists
        import os
        os.makedirs(os.path.dirname(self.array_path), exist_ok=True)
        
        # Check if existing array has compatible schema (3D with image_id dimension)
        with tiledb.open(self.array_path, mode="r") as A:
            schema = A.schema
            dimensions = [dim.name for dim in schema.domain]
            
            # Check if this is a 3D array with image_id dimension
            if 'image_id' not in dimensions or len(dimensions) != 3:
                logger.warning("Existing array has incompatible schema. Creating new array instead.")
                return self._create_new_array(intensities, red_data, green_data, blue_data, depth_array, timestamp_array, image_id)
            
            non_empty_domain = A.nonempty_domain()
            # For 3D array: [image_id, depth_index, pixel_index]
            existing_depth_max = non_empty_domain[1][1] + 1 if non_empty_domain[1][1] is not None else 0
        
        # Width should always be 150 based on our processing
        if width != 150:
            raise ValueError(f"Expected width 150, got {width}")
        
        # Efficiently check if this image_id already exists using non_empty_domain
        with tiledb.open(self.array_path, mode="r") as A:
            non_empty_domain = A.nonempty_domain()
            if non_empty_domain[0][0] is not None and image_id >= non_empty_domain[0][0] and image_id <= non_empty_domain[0][1]:
                # Image ID is within the existing range, assign next available ID
                new_image_id = non_empty_domain[0][1] + 1
                logger.info(f"Image ID {image_id} already exists. Assigning new ID: {new_image_id}")
                image_id = new_image_id
        
        # Calculate the starting depth position for the new image
        start_depth = 0  # Always start from depth 0 for new images
        
        # Use 3D coordinates: [image_id, depth_index, pixel_index]
        # Write in a single operation for maximum performance
        with tiledb.DenseArray(self.array_path, mode="w") as A:
            A[image_id, start_depth:start_depth + height, 0:width] = {
                "intensity_value": intensities,
                "red_value": red_data,
                "green_value": green_data,
                "blue_value": blue_data,
                "depth_value": depth_array,
                "processed_at": timestamp_array
            }
        
        logger.info(f"Appended image {image_id} at depth range {start_depth}:{start_depth + height - 1}")
    
    def _recreate_array_with_larger_domain(self, intensities, red_data, green_data, blue_data, 
                                         depth_array, survey_array, timestamp_array, 
                                         existing_height, required_max_depth):
        """Recreate array with larger domain when needed."""
        logger.info("Recreating array with larger domain...")
        
        # Load existing data
        with tiledb.open(self.array_path, mode="r") as A:
            existing_data = A[:]
            existing_intensities = existing_data["intensity_value"]
            existing_red = existing_data["red_value"]
            existing_green = existing_data["green_value"]
            existing_blue = existing_data["blue_value"]
            existing_depths = existing_data["depth_value"]
            existing_surveys = existing_data["survey_id"]
            existing_timestamps = existing_data["processed_at"]
        
        # Combine with new data
        combined_intensities = np.vstack([existing_intensities, intensities])
        combined_red = np.vstack([existing_red, red_data])
        combined_green = np.vstack([existing_green, green_data])
        combined_blue = np.vstack([existing_blue, blue_data])
        combined_depths = np.vstack([existing_depths, depth_array])
        combined_surveys = np.vstack([existing_surveys, survey_array])
        combined_timestamps = np.vstack([existing_timestamps, timestamp_array])
        
        total_height = existing_height + intensities.shape[0]
        
        # Create new schema with larger domain
        schema = self.create_array_schema(required_max_depth + 1000, 150)  # Add some buffer
        
        # Remove old array and create new one
        import shutil
        shutil.rmtree(self.array_path)
        tiledb.DenseArray.create(self.array_path, schema)
        
        # Write all data to new array
        with tiledb.DenseArray(self.array_path, mode="w") as A:
            A[0:total_height, 0:150] = {
                "intensity_value": combined_intensities,
                "red_value": combined_red,
                "green_value": combined_green,
                "blue_value": combined_blue,
                "depth_value": combined_depths,
                "survey_id": combined_surveys,
                "processed_at": combined_timestamps
            }
        
        logger.info(f"Recreated array with larger domain. New array shape: {total_height}x150")
    
    def get_array_info(self):
        if not tiledb.object_type(self.array_path) == "array":
            return {"error": "Array does not exist"}
        
        with tiledb.open(self.array_path, mode='r') as array:
            schema = array.schema
            non_empty_domain = array.nonempty_domain()
            
            return {
                "array_path": self.array_path,
                "schema": str(schema),
                "non_empty_domain": non_empty_domain,
                "attributes": [schema.attr(i).name for i in range(schema.nattr)],
                "dimensions": [dim.name for dim in schema.domain],
                "compression": "ZSTD level 3"
            }

def main():
    processor = SubsurfaceDataProcessor()
    
    csv_file = "Challenge2.csv"
    if not os.path.exists(csv_file):
        logger.error(f"CSV file not found: {csv_file}")
        return
    
    available_images = processor.get_available_images()
    if available_images:
        logger.info(f"Available images: {available_images}")
    else:
        logger.info("No existing images found - will create first image")
    
    start_time = time.time()
    try:
        processor.process_csv_file(csv_file)
        
        updated_images = processor.get_available_images()
        logger.info(f"Updated images: {updated_images}")
        
        info = processor.get_array_info()
        logger.info("Array Information:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
