import os
import csv
import numpy as np
import tiledb
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
from app.logging_config import get_logger

logger = get_logger(__name__)

class ImageDataProcessor:
    """Processes CSV survey data into TileDB arrays"""
    def __init__(self, array_path="db/arrays/image_data"):
        self.array_path = array_path
        self.original_width = 200
        self.target_width = 150
        self.custom_colormap = self._create_custom_colormap()
    
    def _create_custom_colormap(self):
        """Create geological colormap for visualization"""
        colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('geological', colors, N=n_bins)
        return cmap
    
    def create_array_schema(self, max_images=1000, max_depths=10000, width=150):
        """Create TileDB schema for 3D image data storage"""
        # Define dimensions: image_id, depth_index, pixel_index
        image_dim = tiledb.Dim(
            name="image_id",
            domain=(1, max_images),
            tile=1,
            dtype=np.uint32
        )
        
        depth_dim = tiledb.Dim(
            name="depth_index", 
            domain=(0, max_depths - 1),
            tile=100,
            dtype=np.uint32
        )
        
        pixel_dim = tiledb.Dim(
            name="pixel_index",
            domain=(0, width - 1),
            tile=width,
            dtype=np.uint32
        )
        
        domain = tiledb.Domain(image_dim, depth_dim, pixel_dim)
        
        # Define attributes for intensity, depth values, and timestamps
        
        intensity_attr = tiledb.Attr(
            name="intensity_value",
            dtype=np.uint8,
            filters=tiledb.FilterList([tiledb.ZstdFilter(level=1)])
        )
        
        depth_attr = tiledb.Attr(name="depth_value", dtype=np.float32)
        timestamp_attr = tiledb.Attr(name="processed_at", dtype=np.int64)
        
        schema = tiledb.ArraySchema(
            domain=domain,
            sparse=False,
            attrs=[intensity_attr, depth_attr, timestamp_attr],
            cell_order='row-major',
            tile_order='row-major',
            capacity=100000
        )
        
        return schema
    
    def resize_image_row(self, row_data):
        """Resize image row from 200 to 150 pixels using interpolation"""
        from scipy.ndimage import zoom
        
        # Handle unexpected row sizes
        if len(row_data) != self.original_width:
            logger.warning(f"Expected {self.original_width} pixels, got {len(row_data)}")
            if len(row_data) > self.original_width:
                row_data = row_data[:self.original_width]
            else:
                padded = np.zeros(self.original_width)
                padded[:len(row_data)] = row_data
                row_data = padded
        
        row_data = np.array(row_data, dtype=np.float32)
        
        # Skip empty or invalid rows
        if np.all(np.isnan(row_data)) or np.all(row_data == 0):
            return np.zeros(self.target_width, dtype=np.float32)
        
        # Calculate scaling factor and apply zoom interpolation
        scaling_factor = self.target_width / self.original_width
        
        resized_row = zoom(row_data, scaling_factor, order=3, mode='nearest', prefilter=True)
        
        if len(resized_row) != self.target_width:
            if len(resized_row) > self.target_width:
                resized_row = resized_row[:self.target_width]
            else:
                padded = np.zeros(self.target_width)
                padded[:len(resized_row)] = resized_row
                resized_row = padded
        
        return resized_row
    
    def apply_colormap(self, intensity_values):
        """Apply geological colormap to intensity values"""
        normalized = (intensity_values - intensity_values.min()) / (intensity_values.max() - intensity_values.min() + 1e-8)
        colored = self.custom_colormap(normalized)
        
        red = (colored[:, 0] * 255).astype(np.uint8)
        green = (colored[:, 1] * 255).astype(np.uint8)
        blue = (colored[:, 2] * 255).astype(np.uint8)
        
        return red, green, blue
    
    def _get_next_image_id(self):
        """Get next available image ID from TileDB array"""
        if not tiledb.object_type(self.array_path) == "array":
            return 1
        
        try:
            with tiledb.open(self.array_path, mode="r") as A:
                non_empty_domain = A.nonempty_domain()
                if non_empty_domain[0][1] is not None:
                    return non_empty_domain[0][1] + 1
                else:
                    return 1
        except Exception as e:
            logger.warning(f"Could not determine next image ID: {e}")
            return 1
    
    def get_available_images(self):
        """Get list of available image IDs"""
        if not tiledb.object_type(self.array_path) == "array":
            return []
        
        try:
            with tiledb.open(self.array_path, mode="r") as A:
                non_empty_domain = A.nonempty_domain()
                if non_empty_domain[0][0] is not None and non_empty_domain[0][1] is not None:
                    min_id = non_empty_domain[0][0]
                    max_id = non_empty_domain[0][1]
                    return list(range(min_id, max_id + 1))
                else:
                    return []
        except Exception as e:
            logger.error(f"Error getting available images: {e}")
            return []
    
    def process_csv_file(self, csv_file_path):
        """Process CSV file and store in TileDB array"""
        logger.info(f"Starting CSV processing: {os.path.basename(csv_file_path)}")
        
        depth_values = []
        image_data = []
        
        # Read CSV data
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            
            for row in reader:
                if len(row) < 201:
                    logger.warning(f"Row has insufficient data: {len(row)} columns")
                    continue
                
                try:
                    # Extract depth and pixel values
                    depth = float(row[0])
                    pixel_values = [float(val) if val.strip() != '' else 0.0 for val in row[1:201]]
                    
                    depth_values.append(depth)
                    image_data.append(pixel_values)
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping row due to data conversion error: {e}")
                    continue
        
        if not depth_values:
            logger.error("No valid data found in CSV file")
            return
        
        depth_values = np.array(depth_values, dtype=np.float32)
        image_data = np.array(image_data, dtype=np.float32)
        
        logger.info(f"Loaded data: {len(depth_values)} depth levels, {image_data.shape[1]} pixels per level")
        logger.info(f"Depth range: {depth_values.min():.1f}m to {depth_values.max():.1f}m")
        
        image_id = self._get_next_image_id()
        logger.info(f"Assigning image ID: {image_id}")
        
        self._process_and_store_data(image_data, depth_values, image_id)
    
    def _process_and_store_data(self, image_data, depth_values, image_id):
        """Process and store image data in TileDB"""
        logger.debug("Resizing image data from 200 to 150 pixels...")
        height, width = image_data.shape
        resized_image_data = np.zeros((height, self.target_width), dtype=np.float32)
        
        # Resize each row from 200 to 150 pixels
        for i in range(height):
            resized_image_data[i] = self.resize_image_row(image_data[i])
        
        # Clean and normalize intensity data
        resized_image_data = np.nan_to_num(resized_image_data, nan=0.0, posinf=255.0, neginf=0.0)
        normalized_intensities = np.clip(resized_image_data, 0, 255).astype(np.uint8)
        
        logger.debug("Storing normalized intensity values in TileDB...")
        
        # Prepare depth and timestamp arrays
        depth_array = np.broadcast_to(depth_values[:, np.newaxis], (height, self.target_width))
        timestamp_array = np.full((height, self.target_width), int(time.time() * 1000000), dtype=np.int64)
        
        if tiledb.object_type(self.array_path) == "array":
            logger.debug("Appending data to existing TileDB array...")
            self._append_to_existing_array(
                normalized_intensities, depth_array, timestamp_array, image_id
            )
        else:
            logger.info("Creating new TileDB array for first image...")
            self._create_new_array(
                normalized_intensities, depth_array, timestamp_array, image_id
            )
        
        logger.info(f"Successfully stored {height} depth levels in TileDB")
    
    def _create_new_array(self, intensities, depth_array, timestamp_array, image_id):
        """Create new TileDB array and store data"""
        height, width = intensities.shape
        
        import os
        os.makedirs(os.path.dirname(self.array_path), exist_ok=True)
        
        schema = self.create_array_schema(max_images=1000, max_depths=10000, width=width)
        tiledb.DenseArray.create(self.array_path, schema)
        
        with tiledb.DenseArray(self.array_path, mode="w") as A:
            A[image_id, 0:height, 0:width] = {
                "intensity_value": intensities,
                "depth_value": depth_array,
                "processed_at": timestamp_array
            }
        
        logger.info(f"Created TileDB array: image_id={image_id}, depths={height}, width={width}")
    
    def _append_to_existing_array(self, intensities, depth_array, timestamp_array, image_id):
        """Append data to existing TileDB array"""
        height, width = intensities.shape
        
        import os
        os.makedirs(os.path.dirname(self.array_path), exist_ok=True)
        
        with tiledb.open(self.array_path, mode="r") as A:
            schema = A.schema
            dimensions = [dim.name for dim in schema.domain]
            
            if 'image_id' not in dimensions or len(dimensions) != 3:
                logger.warning("Existing array has incompatible schema. Creating new array instead.")
                return self._create_new_array(intensities, depth_array, timestamp_array, image_id)
            
            non_empty_domain = A.nonempty_domain()
            existing_depth_max = non_empty_domain[1][1] + 1 if non_empty_domain[1][1] is not None else 0
        
        if width != 150:
            raise ValueError(f"Expected width 150, got {width}")
        
        # Check if image ID already exists
        with tiledb.open(self.array_path, mode="r") as A:
            non_empty_domain = A.nonempty_domain()
            if non_empty_domain[0][0] is not None and image_id >= non_empty_domain[0][0] and image_id <= non_empty_domain[0][1]:
                new_image_id = non_empty_domain[0][1] + 1
                logger.info(f"Image ID {image_id} already exists. Assigning new ID: {new_image_id}")
                image_id = new_image_id
        
        start_depth = 0
        
        with tiledb.DenseArray(self.array_path, mode="w") as A:
            A[image_id, start_depth:start_depth + height, 0:width] = {
                "intensity_value": intensities,
                "depth_value": depth_array,
                "processed_at": timestamp_array
            }
        
        logger.info(f"Appended image {image_id}: {height} depth levels starting at index {start_depth}")
    
    def _recreate_array_with_larger_domain(self, intensities, depth_array, timestamp_array, 
                                         existing_height, required_max_depth):
        """Recreate array with larger domain when needed"""
        logger.info("Recreating array with larger domain...")
        
        backup_path = f"{self.array_path}_backup_{int(time.time())}"
        logger.info(f"Backing up existing array to: {backup_path}")
        
        import shutil
        shutil.move(self.array_path, backup_path)
        
        try:
            height, width = intensities.shape
            new_max_depths = max(required_max_depth + 1000, existing_height + height + 1000)
            
            schema = self.create_array_schema(max_images=1000, max_depths=new_max_depths, width=width)
            tiledb.DenseArray.create(self.array_path, schema)
            
            logger.info("Copying existing data...")
            with tiledb.open(backup_path, mode="r") as old_A:
                with tiledb.DenseArray(self.array_path, mode="w") as new_A:
                    old_data = old_A[:]
                    if len(old_data) > 0:
                        new_A[0:existing_height, 0:width] = old_data
            
            logger.info("Adding new data...")
            with tiledb.DenseArray(self.array_path, mode="w") as A:
                A[existing_height:existing_height + height, 0:width] = {
                    "intensity_value": intensities,
                    "depth_value": depth_array,
                    "processed_at": timestamp_array
                }
            
            logger.info(f"Successfully recreated array with new domain. Backup saved at: {backup_path}")
            
        except Exception as e:
            logger.error(f"Failed to recreate array: {e}")
            logger.info("Restoring backup...")
            if os.path.exists(self.array_path):
                shutil.rmtree(self.array_path)
            shutil.move(backup_path, self.array_path)
            raise
    
    def get_array_info(self):
        """Get TileDB array information and stats"""
        if not tiledb.object_type(self.array_path) == "array":
            return {"status": "Array does not exist"}
        
        try:
            with tiledb.open(self.array_path, mode="r") as A:
                non_empty_domain = A.nonempty_domain()
                schema = A.schema
                
                return {
                    "array_path": self.array_path,
                    "non_empty_domain": str(non_empty_domain),
                    "attributes": [attr.name for attr in schema],
                    "dimensions": [dim.name for dim in schema.domain],
                    "compression": "ZSTD level 3"
                }
        except Exception as e:
            return {"error": str(e)}

def main():
    """Main entry point for standalone data processing"""
    processor = ImageDataProcessor()
    
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