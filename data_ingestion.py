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
    
    def create_array_schema(self, total_height, width=150):
        # Use a much larger domain to avoid frequent schema changes
        # This allows for up to 1M depth levels, which should be sufficient
        max_depth = max(total_height, 1000000)
        
        depth_dim = tiledb.Dim(
            name="depth_index",
            domain=(0, max_depth - 1),
            tile=1000,  # Larger tiles for better performance
            dtype=np.uint32
        )
        
        pixel_dim = tiledb.Dim(
            name="pixel_index",
            domain=(0, width - 1),
            tile=width,
            dtype=np.uint32
        )
        
        domain = tiledb.Domain(depth_dim, pixel_dim)
        
        intensity_attr = tiledb.Attr(
            name="intensity_value",
            dtype=np.uint8,
            filters=tiledb.FilterList([tiledb.ZstdFilter(level=1)])  # Lower compression for speed
        )
        
        red_attr = tiledb.Attr(name="red_value", dtype=np.uint8, filters=tiledb.FilterList([tiledb.ZstdFilter(level=1)]))
        green_attr = tiledb.Attr(name="green_value", dtype=np.uint8, filters=tiledb.FilterList([tiledb.ZstdFilter(level=1)]))
        blue_attr = tiledb.Attr(name="blue_value", dtype=np.uint8, filters=tiledb.FilterList([tiledb.ZstdFilter(level=1)]))
        
        depth_attr = tiledb.Attr(name="depth_value", dtype=np.float32)
        image_attr = tiledb.Attr(name="image_id", dtype=np.uint32)
        timestamp_attr = tiledb.Attr(name="processed_at", dtype=np.int64)
        
        schema = tiledb.ArraySchema(
            domain=domain,
            sparse=False,
            attrs=[intensity_attr, red_attr, green_attr, blue_attr, depth_attr, image_attr, timestamp_attr],
            cell_order='row-major',
            tile_order='row-major',
            capacity=100000  # Larger capacity for better performance
        )
        
        return schema
    
    def resize_image_row(self, row_data):
        original_row = np.array(row_data, dtype=np.float32).reshape(1, -1)
        resized_row = cv2.resize(
            original_row, 
            (self.target_width, 1), 
            interpolation=cv2.INTER_CUBIC
        )
        return resized_row.flatten()
    
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
        image_array = np.full((height, self.target_width), image_id, dtype=np.uint32)
        timestamp_array = np.full((height, self.target_width), int(time.time() * 1000000), dtype=np.int64)
        
        if tiledb.object_type(self.array_path) == "array":
            logger.info("Appending to existing array...")
            self._append_to_existing_array(
                normalized_intensities, red_data, green_data, blue_data,
                depth_array, image_array, timestamp_array
            )
        else:
            logger.info("Creating new TileDB array...")
            self._create_new_array(
                normalized_intensities, red_data, green_data, blue_data,
                depth_array, image_array, timestamp_array
            )
        
        logger.info(f"Successfully processed {height} depth levels")
    
    def _create_new_array(self, intensities, red_data, green_data, blue_data, depth_array, image_array, timestamp_array):
        height, width = intensities.shape
        
        # Ensure the directory exists
        import os
        os.makedirs(os.path.dirname(self.array_path), exist_ok=True)
        
        schema = self.create_array_schema(height, width)
        tiledb.DenseArray.create(self.array_path, schema)
        
        with tiledb.DenseArray(self.array_path, mode="w") as A:
            A[0:height, 0:width] = {
                "intensity_value": intensities,
                "red_value": red_data,
                "green_value": green_data,
                "blue_value": blue_data,
                "depth_value": depth_array,
                "image_id": image_array,
                "processed_at": timestamp_array
            }
        
        logger.info(f"Created new array with shape {height}x{width}")
    
    def _append_to_existing_array(self, intensities, red_data, green_data, blue_data, depth_array, image_array, timestamp_array):
        height, width = intensities.shape
        
        # Ensure the directory exists
        import os
        os.makedirs(os.path.dirname(self.array_path), exist_ok=True)
        
        # Get current array info without loading all data
        with tiledb.open(self.array_path, mode="r") as A:
            non_empty_domain = A.nonempty_domain()
            existing_height = non_empty_domain[0][1] + 1  # Get max depth_index + 1
        
        # Width should always be 150 based on our processing
        if width != 150:
            raise ValueError(f"Expected width 150, got {width}")
        
        # Calculate the starting position for the new data
        start_depth = existing_height
        
        # Use proper TileDB append operation - no need to load existing data!
        with tiledb.DenseArray(self.array_path, mode="w") as A:
            A[start_depth:start_depth + height, 0:width] = {
                "intensity_value": intensities,
                "red_value": red_data,
                "green_value": green_data,
                "blue_value": blue_data,
                "depth_value": depth_array,
                "image_id": image_array,
                "processed_at": timestamp_array
            }
        
        logger.info(f"Appended data. New array shape: {start_depth + height}x{width}")
    
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
